import torch
import torchaudio
from transformers import AutoTokenizer
import time
import argparse
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import json
import sys
import re
import jiwer
from tqdm import tqdm
from datetime import datetime
import os

# Add src folder to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file

@dataclass
class WhisfusionBenchmarkResult:
    # File info
    file_path: str
    audio_duration_s: float
    
    # Timing breakdown (ms) - means - WHISFUSION ONLY
    preprocessing_ms: float
    model_inference_ms: float
    postprocessing_ms: float
    candidate_selection_ms: float
    full_generation_ms: float
    
    # Timing statistics - stds
    preprocessing_std: float
    model_inference_std: float
    postprocessing_std: float
    candidate_selection_std: float
    full_generation_std: float
    
    # Time distribution percentages
    preprocessing_pct: float
    model_inference_pct: float
    postprocessing_pct: float
    candidate_selection_pct: float
    
    # Performance metrics
    rtf: float
    tokens_per_second: float
    ms_per_token: float
    num_tokens_generated: int
    num_candidates: int
    num_steps: int
    
    # Accuracy
    wer: float
    cer: float
    ground_truth: str
    hypothesis: str
    
    # Duration category
    duration_category: str
    
    # Step timings
    step_timings_mean: List[float]
    step_timings_std: List[float]

class WhisfusionBenchmark:
    def __init__(self, base_model_path, adapter_path, model_name="Diff_LLaMA_170M", 
                 tokenizer_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                 device="cuda"):
        self.device = device
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        
        # Load models (Whisfusion ONLY)
        self._load_models()
    
    def _load_models(self):
        print("Loading Whisfusion model...")
        self.config = Config.from_name(self.model_name)
        self.model = self._load_whisfusion_model()
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Whisfusion loaded: {total_params/1e6:.1f}M parameters")
    
    def _load_whisfusion_model(self):
        """Load Whisfusion model with adapter"""
        model = TransEncoder(self.config).to(self.device)
        
        # Load base weights
        if self.base_model_path.endswith('.safetensors'):
            base_weights = load_file(self.base_model_path)
            model.load_state_dict(base_weights, strict=False)
        else:
            base_weights = torch.load(self.base_model_path, map_location=self.device)
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']
            model.load_state_dict(base_weights, strict=False)
        
        # Load adapter weights
        adapter_weights = torch.load(self.adapter_path, map_location=self.device)
        if isinstance(adapter_weights, dict) and 'state_dict' in adapter_weights:
            adapter_weights = adapter_weights['state_dict']
        model.load_state_dict(adapter_weights, strict=False)
        
        model = model.to(torch.bfloat16)
        model.eval()
        
        return model
    
    def warmup(self, num_iterations=5):
        print("\nWarming up Whisfusion...")
        
        dummy_condition = torch.randn(1, 100, self.config.n_embd).to(self.device, dtype=torch.bfloat16)
        dummy_input = torch.ones(1, 100, dtype=torch.long).to(self.device)
        
        for iteration in tqdm(range(num_iterations), desc="Warmup", unit="iter"):
            with torch.no_grad():
                _ = self.model(idx=dummy_input, condition=dummy_condition)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("✅ Warmup complete!")
    
    def get_duration_category(self, duration_s):
        """Categorize audio by duration"""
        if duration_s <= 10:
            return "0-10s"
        elif duration_s <= 20:
            return "10-20s"
        elif duration_s <= 30:
            return "20-30s"
        else:
            return "30s+"
    
    def get_ground_truth(self, audio_path):
        """Get ground truth text from LibriSpeech trans.txt file"""
        audio_path = Path(audio_path)
        speaker_id = audio_path.parent.parent.name
        chapter_id = audio_path.parent.name
        trans_file = audio_path.parent / f"{speaker_id}-{chapter_id}.trans.txt"
        
        if not trans_file.exists():
            return None
        
        utterance_id = audio_path.stem
        with open(trans_file, 'r') as f:
            for line in f:
                if line.startswith(utterance_id):
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        return parts[1]
        return None
    
    def normalize_text(self, text):
        """Normalize text for WER calculation"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def calculate_metrics(self, ground_truth, hypothesis):
        """Calculate WER and CER"""
        gt_norm = self.normalize_text(ground_truth)
        hyp_norm = self.normalize_text(hypothesis)
        
        wer = jiwer.wer(gt_norm, hyp_norm) * 100
        cer = jiwer.cer(gt_norm, hyp_norm) * 100
        
        return wer, cer
    
    @torch.no_grad()
    def profile_single_file(self, audio_path, num_runs=5, n_candidates=15, n_steps=4,
                          mask_ratio_schedule=[1.0, 0.9, 0.85, 0.8]):
        """Profile a single audio file - WHISFUSION ONLY"""
        # Load audio (just for duration/ground truth)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        audio_numpy = waveform.squeeze().numpy()
        audio_duration = len(audio_numpy) / sample_rate
        
        # Get ground truth
        ground_truth = self.get_ground_truth(audio_path)
        if not ground_truth:
            return None
        
        # Storage for multiple runs
        all_timings = []
        all_step_timings = []
        hypothesis = None
        
        for run in range(num_runs):
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Create dummy condition (same shape as Whisper hidden states)
            # [1, 1500, 768] for whisper-small
            condition = torch.randn(1, 1500, 768, device=self.device, dtype=torch.bfloat16)
            
            # Create target_ids (same as original)
            bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
            target_ids = torch.full((1, 256), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            target_ids[0, 0] = bos_token_id
            
            # Run profiling - WHISFUSION ONLY
            timings, text, step_timings = self._profile_single_run(
                target_ids, condition, n_candidates, n_steps, mask_ratio_schedule
            )
            
            all_timings.append(timings)
            all_step_timings.append(step_timings)
            
            if run == 0:
                hypothesis = text
        
        # Calculate statistics
        timing_keys = list(all_timings[0].keys())
        means = {k: np.mean([t[k] for t in all_timings]) for k in timing_keys}
        stds = {k: np.std([t[k] for t in all_timings]) for k in timing_keys}
        
        # Step timings statistics
        step_means = []
        step_stds = []
        for step in range(n_steps):
            step_values = [st[step] for st in all_step_timings]
            step_means.append(np.mean(step_values))
            step_stds.append(np.std(step_values))
        
        # Calculate percentages
        total_time = means['full_generation']
        percentages = {
            'preprocessing_pct': means['preprocessing'] / total_time * 100,
            'model_inference_pct': means['model_inference'] / total_time * 100,
            'postprocessing_pct': means['postprocessing'] / total_time * 100,
            'candidate_selection_pct': means['candidate_selection'] / total_time * 100
        }
        
        # Calculate metrics
        wer, cer = self.calculate_metrics(ground_truth, hypothesis)
        rtf = total_time / 1000 / audio_duration
        
        # Tokens/s calculation
        total_tokens = 256  # seq_len
        tokens_per_second = total_tokens / (means['model_inference'] / 1000)
        ms_per_token = means['model_inference'] / total_tokens
        
        return WhisfusionBenchmarkResult(
            file_path=str(audio_path),
            audio_duration_s=audio_duration,
            
            # Means
            preprocessing_ms=means['preprocessing'],
            model_inference_ms=means['model_inference'],
            postprocessing_ms=means['postprocessing'],
            candidate_selection_ms=means['candidate_selection'],
            full_generation_ms=means['full_generation'],
            
            # Stds
            preprocessing_std=stds['preprocessing'],
            model_inference_std=stds['model_inference'],
            postprocessing_std=stds['postprocessing'],
            candidate_selection_std=stds['candidate_selection'],
            full_generation_std=stds['full_generation'],
            
            # Percentages
            **percentages,
            
            # Performance
            rtf=rtf,
            tokens_per_second=tokens_per_second,
            ms_per_token=ms_per_token,
            num_tokens_generated=total_tokens,
            num_candidates=n_candidates,
            num_steps=n_steps,
            
            # Accuracy
            wer=wer,
            cer=cer,
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            
            # Category
            duration_category=self.get_duration_category(audio_duration),
            
            # Step timings
            step_timings_mean=step_means,
            step_timings_std=step_stds
        )
    
    def _profile_single_run(self, target_ids, condition, n_candidates, n_steps, mask_ratio_schedule):
        """Single profiling run - WHISFUSION ONLY"""
        timings = {}
        
        # 1. Whisfusion preprocessing
        t0 = time.perf_counter()
        candidates, time_breakdown, step_timings = self._generate_with_timing(
            target_ids, condition, n_candidates, n_steps, mask_ratio_schedule
        )
        timings['preprocessing'] = time_breakdown['preprocessing_time']
        timings['model_inference'] = time_breakdown['model_inference_time']
        timings['postprocessing'] = time_breakdown['postprocessing_time']
        
        # 2. Candidate selection
        t0 = time.perf_counter()
        best_candidate = max(candidates, key=lambda x: x['avg_confidence'])
        final_text = best_candidate['text']
        timings['candidate_selection'] = (time.perf_counter() - t0) * 1000
        
        # Calculate full generation time
        timings['full_generation'] = sum([
            timings['preprocessing'],
            timings['model_inference'],
            timings['postprocessing'],
            timings['candidate_selection']
        ])
        
        return timings, final_text, step_timings
    
    def _generate_with_timing(self, initial_input, condition, n_candidates, n_steps, mask_ratio_schedule):
        """Generate with detailed per-step timing (unchanged)"""
        device = condition.device
        mask_token_id = self.config.padded_vocab_size
        
        time_breakdown = {
            'preprocessing_time': 0,
            'model_inference_time': 0,
            'postprocessing_time': 0
        }
        step_timings = []
        
        # Preprocessing
        preprocess_start = time.perf_counter()
        
        batch_size = n_candidates
        input_for_mask = initial_input.squeeze(0) if initial_input.dim() > 1 else initial_input
        if input_for_mask.device != device:
            input_for_mask = input_for_mask.to(device)
        
        seq_len = input_for_mask.size(0)
        
        current_outputs = torch.full(
            (n_candidates, seq_len),
            mask_token_id,
            dtype=input_for_mask.dtype,
            device=device
        )
        current_outputs[:, 0] = input_for_mask[0]
        
        batch_condition = condition.expand(n_candidates, -1, -1)
        
        time_breakdown['preprocessing_time'] = (time.perf_counter() - preprocess_start) * 1000
        
        # Generation steps
        for step in range(n_steps):
            step_start = time.perf_counter()
            
            mask_ratio = mask_ratio_schedule[step] if step < len(mask_ratio_schedule) else 0.7
            
            if mask_ratio > 0:
                rand_probs = torch.rand(batch_size, seq_len, device=device)
                mask_indices_batch = rand_probs < mask_ratio
                mask_indices_batch[:, 0] = False
            else:
                mask_indices_batch = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            
            masked_inputs = current_outputs.clone()
            masked_inputs[mask_indices_batch] = mask_token_id
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            inf_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(idx=masked_inputs, condition=batch_condition)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            inf_time = (time.perf_counter() - inf_start) * 1000
            time_breakdown['model_inference_time'] += inf_time
            
            if step == n_steps - 1:
                probs = torch.softmax(logits, dim=-1)
                max_probs, predicted_ids = torch.max(probs, dim=-1)
                final_confidences = max_probs
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
            
            current_outputs = torch.where(mask_indices_batch, predicted_ids, masked_inputs)
            
            step_time = (time.perf_counter() - step_start) * 1000
            step_timings.append(step_time)
        
        # Postprocessing
        post_start = time.perf_counter()
        
        all_outputs_cpu = current_outputs.cpu()
        all_confidences_cpu = final_confidences.cpu()
        all_texts = self.tokenizer.batch_decode(all_outputs_cpu, skip_special_tokens=True)
        
        pad_id = self.tokenizer.pad_token_id
        valid_mask = (all_outputs_cpu != pad_id)
        masked_confidences = all_confidences_cpu * valid_mask.float()
        avg_confidences = masked_confidences.sum(dim=1) / valid_mask.sum(dim=1).float().clamp(min=1)
        
        candidates = [
            {
                'text': all_texts[i],
                'avg_confidence': float(avg_confidences[i]),
                'tokens': all_outputs_cpu[i]
            }
            for i in range(batch_size)
        ]
        
        time_breakdown['postprocessing_time'] = (time.perf_counter() - post_start) * 1000
        
        return candidates, time_breakdown, step_timings
    
    def benchmark_dataset(self, data_path, output_dir, num_runs=5, max_files=None,
                         n_candidates=15, n_steps=4):
        """Benchmark Whisfusion ONLY on dataset"""
        print("\n" + "="*60)
        print("🚀 WHISFUSION-ONLY BENCHMARK")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Candidates: {n_candidates}, Steps: {n_steps}")
        
        # Warmup with progress bar
        self.warmup()
        
        # Find all audio files
        data_path = Path(data_path)
        audio_files = list(data_path.rglob("*.flac"))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Found {len(audio_files)} audio files")
        
        # Benchmark each file
        results = []
        failed_files = []
        
        for audio_path in tqdm(audio_files, desc="🔍 Processing files"):
            try:
                result = self.profile_single_file(
                    audio_path, num_runs, n_candidates, n_steps
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(f"\n❌ Error processing {audio_path}: {e}")
                failed_files.append(str(audio_path))
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_whisfusion_only_{timestamp}.json"
        
        # Convert results to dict
        results_dict = {
            'model': 'Whisfusion-Only',
            'model_config': {
                'base_model': self.base_model_path,
                'adapter': self.adapter_path,
                'model_name': self.model_name,
                'tokenizer': self.tokenizer_name,
                'n_candidates': n_candidates,
                'n_steps': n_steps
            },
            'dataset': str(data_path),
            'num_files': len(results),
            'num_runs_per_file': num_runs,
            'timestamp': timestamp,
            'failed_files': failed_files,
            'results': [asdict(r) for r in results],
            'summary': self.calculate_summary(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        self.print_summary(results_dict['summary'])
        
        return results_dict
    
    def calculate_summary(self, results: List[WhisfusionBenchmarkResult]):
        """Calculate summary statistics - WHISFUSION ONLY"""
        if not results:
            return {}
        
        # Group by duration category
        categories = {}
        for result in results:
            cat = result.duration_category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        summary = {
            'overall': {
                'num_files': len(results),
                'total_duration_hours': sum(r.audio_duration_s for r in results) / 3600,
                'mean_wer': np.mean([r.wer for r in results]),
                'mean_cer': np.mean([r.cer for r in results]),
                'mean_rtf': np.mean([r.rtf for r in results]),
                'mean_tokens_per_second': np.mean([r.tokens_per_second for r in results]),
                'mean_ms_per_token': np.mean([r.ms_per_token for r in results]),
                'mean_full_generation_ms': np.mean([r.full_generation_ms for r in results]),
                'mean_model_inference_pct': np.mean([r.model_inference_pct for r in results]),
            },
            'by_duration': {}
        }
        
        # Summary by duration category
        for cat in ['0-10s', '10-20s', '20-30s', '30s+']:
            if cat in categories:
                cat_results = categories[cat]
                summary['by_duration'][cat] = {
                    'num_files': len(cat_results),
                    'mean_duration_s': np.mean([r.audio_duration_s for r in cat_results]),
                    'mean_wer': np.mean([r.wer for r in cat_results]),
                    'mean_cer': np.mean([r.cer for r in cat_results]),
                    'mean_rtf': np.mean([r.rtf for r in cat_results]),
                    'mean_tokens_per_second': np.mean([r.tokens_per_second for r in cat_results]),
                    'mean_ms_per_token': np.mean([r.ms_per_token for r in cat_results]),
                    'mean_full_generation_ms': np.mean([r.full_generation_ms for r in cat_results]),
                    'mean_model_inference_pct': np.mean([r.model_inference_pct for r in cat_results]),
                }
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics - WHISFUSION ONLY"""
        print("\n" + "="*60)
        print("📊 WHISFUSION-ONLY SUMMARY")
        print("="*60)
        
        overall = summary['overall']
        print(f"\nOverall Statistics:")
        print(f"  Files processed: {overall['num_files']}")
        print(f"  Total duration: {overall['total_duration_hours']:.2f} hours")
        print(f"  Mean WER: {overall['mean_wer']:.2f}%")
        print(f"  Mean CER: {overall['mean_cer']:.2f}%")
        print(f"  Mean RTF: {overall['mean_rtf']:.4f}")
        print(f"  Mean full generation: {overall['mean_full_generation_ms']:.1f}ms")
        print(f"  Mean tokens/s: {overall['mean_tokens_per_second']:.1f}")
        print(f"  Mean ms/token: {overall['mean_ms_per_token']:.2f}")
        print(f"  Model inference: {overall['mean_model_inference_pct']:.1f}% of total time")
        
        print("\nBy Duration Category:")
        for cat in ['0-10s', '10-20s', '20-30s', '30s+']:
            if cat in summary['by_duration']:
                data = summary['by_duration'][cat]
                print(f"\n  {cat}: {data['num_files']} files")
                print(f"    Mean duration: {data['mean_duration_s']:.1f}s")
                print(f"    WER: {data['mean_wer']:.2f}%")
                print(f"    RTF: {data['mean_rtf']:.4f}")
                print(f"    Full generation: {data['mean_full_generation_ms']:.1f}ms")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisfusion ONLY (no Whisper)")
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to LibriSpeech test-clean directory')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Diff_LLaMA_170M')
    parser.add_argument('--tokenizer_name', type=str, 
                       default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs per file')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--n_candidates', type=int, default=15)
    parser.add_argument('--n_steps', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = WhisfusionBenchmark(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name
    )
    
    # Run benchmark
    results = benchmark.benchmark_dataset(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        max_files=args.max_files,
        n_candidates=args.n_candidates,
        n_steps=args.n_steps
    )
    
    print("\n✅ Whisfusion-only benchmark complete!")

if __name__ == "__main__":
    main()