# benchmark_whisper_models.py
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
import time
import argparse
from pathlib import Path
import numpy as np
import GPUtil
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import json
import jiwer
import re
from tqdm import tqdm
from datetime import datetime
import os

@dataclass
class BenchmarkResult:
    """Store complete benchmark results for a single file"""
    # File info
    file_path: str
    audio_duration_s: float
    
    # Timing breakdown (ms)
    preprocess_cpu_ms: float
    gpu_transfer_ms: float
    encoder_ms: float
    decoder_ms: float
    postprocess_ms: float
    full_generation_ms: float
    
    # Timing statistics
    preprocess_std: float
    gpu_transfer_std: float
    encoder_std: float
    decoder_std: float
    postprocess_std: float
    full_generation_std: float
    
    # Time distribution percentages
    preprocess_pct: float
    gpu_transfer_pct: float
    encoder_pct: float
    decoder_pct: float
    postprocess_pct: float
    
    # Performance metrics
    rtf: float
    tokens_per_second: float
    ms_per_token: float
    num_tokens_generated: int
    
    # Accuracy
    wer: float
    cer: float
    ground_truth: str
    hypothesis: str
    
    # Duration category
    duration_category: str  # "0-10s", "10-20s", "20-30s", "30s+"

class WhisperBenchmark:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
    
    def measure_gpu_memory(self):
        """Get current GPU memory usage"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUsed
        return 0
    
    @torch.no_grad()
    def profile_single_file(self, model, processor, audio_path, num_runs=5):
        """Profile a single audio file with multiple runs"""
        # Load audio
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
        
        # Multiple runs for stable measurements
        timings = {
            'preprocess': [],
            'gpu_transfer': [],
            'encoder': [],
            'decoder': [],
            'postprocess': []
        }
        
        hypothesis = None
        num_tokens = 0
        
        for run in range(num_runs):
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 1. Preprocessing
            t0 = time.perf_counter()
            inputs = processor(audio_numpy, sampling_rate=sample_rate, return_tensors="pt")
            t1 = time.perf_counter()
            timings['preprocess'].append((t1 - t0) * 1000)
            
            # 2. GPU transfer
            t0 = time.perf_counter()
            input_features = inputs.input_features.to(self.device)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings['gpu_transfer'].append((t1 - t0) * 1000)
            
            # 3. Encoder (run once)
            t0 = time.perf_counter()
            encoder_outputs = model.model.encoder(input_features)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings['encoder'].append((t1 - t0) * 1000)
            
            # 4. Decoder (use encoder output, fixed token count)
            t0 = time.perf_counter()
            encoder_outputs_wrapped = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state
            )
            predicted_ids = model.generate(
                encoder_outputs=encoder_outputs_wrapped,
                max_new_tokens=444  # Fixed value
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings['decoder'].append((t1 - t0) * 1000)
            
            # 5. Postprocessing
            t0 = time.perf_counter()
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            t1 = time.perf_counter()
            timings['postprocess'].append((t1 - t0) * 1000)
            
            if run == 0:
                hypothesis = text
                num_tokens = predicted_ids.shape[1] - 1  # Exclude start token
        
        # Calculate statistics
        means = {k: np.mean(v) for k, v in timings.items()}
        stds = {k: np.std(v) for k, v in timings.items()}
        
        # Calculate full generation time as sum of components
        full_generation_time = (means['preprocess'] + means['gpu_transfer'] + 
                               means['encoder'] + means['decoder'] + means['postprocess'])
        
        # Calculate percentages
        percentages = {
            'preprocess_pct': means['preprocess'] / full_generation_time * 100,
            'gpu_transfer_pct': means['gpu_transfer'] / full_generation_time * 100,
            'encoder_pct': means['encoder'] / full_generation_time * 100,
            'decoder_pct': means['decoder'] / full_generation_time * 100,
            'postprocess_pct': means['postprocess'] / full_generation_time * 100
        }
        
        # Calculate metrics
        wer, cer = self.calculate_metrics(ground_truth, hypothesis)
        rtf = full_generation_time / 1000 / audio_duration
        tokens_per_second = num_tokens / (means['decoder'] / 1000)
        ms_per_token = means['decoder'] / num_tokens
        
        return BenchmarkResult(
            file_path=str(audio_path),
            audio_duration_s=audio_duration,
            
            # Means
            preprocess_cpu_ms=means['preprocess'],
            gpu_transfer_ms=means['gpu_transfer'],
            encoder_ms=means['encoder'],
            decoder_ms=means['decoder'],
            postprocess_ms=means['postprocess'],
            full_generation_ms=full_generation_time,
            
            # Stds
            preprocess_std=stds['preprocess'],
            gpu_transfer_std=stds['gpu_transfer'],
            encoder_std=stds['encoder'],
            decoder_std=stds['decoder'],
            postprocess_std=stds['postprocess'],
            full_generation_std=0,  # Not applicable for sum
            
            # Percentages
            **percentages,
            
            # Performance
            rtf=rtf,
            tokens_per_second=tokens_per_second,
            ms_per_token=ms_per_token,
            num_tokens_generated=num_tokens,
            
            # Accuracy
            wer=wer,
            cer=cer,
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            
            # Category
            duration_category=self.get_duration_category(audio_duration)
        )
    
    def benchmark_model(self, model_name, data_path, output_dir, num_runs=5, max_files=None):
        """Benchmark a single Whisper model on dataset"""
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*60}")
        
        # Load model
        print(f"Loading model...")
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        model.eval()
        
        # Warmup
        print("Warming up...")
        dummy_audio = torch.randn(16000).numpy()
        for _ in range(3):
            inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                encoder_outputs = model.model.encoder(input_features)
                encoder_outputs_wrapped = BaseModelOutput(
                    last_hidden_state=encoder_outputs.last_hidden_state
                )
                _ = model.generate(
                    encoder_outputs=encoder_outputs_wrapped,
                    max_new_tokens=50
                )
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Find all audio files
        data_path = Path(data_path)
        audio_files = list(data_path.rglob("*.flac"))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Found {len(audio_files)} audio files")
        
        # Benchmark each file
        results = []
        failed_files = []
        
        for audio_path in tqdm(audio_files, desc="Processing files"):
            try:
                result = self.profile_single_file(model, processor, audio_path, num_runs)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
                failed_files.append(str(audio_path))
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{model_name.replace('/', '_')}_{timestamp}.json"
        
        # Convert results to dict
        results_dict = {
            'model': model_name,
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
        
        print(f"\nResults saved to {output_file}")
        self.print_summary(results_dict['summary'])
        
        # Clear model from memory
        del model
        del processor
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return results_dict
    
    def calculate_summary(self, results: List[BenchmarkResult]):
        """Calculate summary statistics"""
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
                'mean_encoder_pct': np.mean([r.encoder_pct for r in results]),
                'mean_decoder_pct': np.mean([r.decoder_pct for r in results]),
                'mean_overhead_pct': np.mean([r.preprocess_pct + r.gpu_transfer_pct + r.postprocess_pct for r in results])
            },
            'by_duration': {}
        }
        
        # Summary by duration category
        for cat, cat_results in categories.items():
            summary['by_duration'][cat] = {
                'num_files': len(cat_results),
                'mean_duration_s': np.mean([r.audio_duration_s for r in cat_results]),
                'mean_wer': np.mean([r.wer for r in cat_results]),
                'mean_cer': np.mean([r.cer for r in cat_results]),
                'mean_rtf': np.mean([r.rtf for r in cat_results]),
                'mean_tokens_per_second': np.mean([r.tokens_per_second for r in cat_results]),
                'mean_ms_per_token': np.mean([r.ms_per_token for r in cat_results]),
                'mean_full_generation_ms': np.mean([r.full_generation_ms for r in cat_results]),
                'mean_encoder_pct': np.mean([r.encoder_pct for r in cat_results]),
                'mean_decoder_pct': np.mean([r.decoder_pct for r in cat_results]),
                'mean_overhead_pct': np.mean([r.preprocess_pct + r.gpu_transfer_pct + r.postprocess_pct for r in cat_results])
            }
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        overall = summary['overall']
        print(f"\nOverall Statistics:")
        print(f"  Files processed: {overall['num_files']}")
        print(f"  Total duration: {overall['total_duration_hours']:.2f} hours")
        print(f"  Mean WER: {overall['mean_wer']:.2f}%")
        print(f"  Mean CER: {overall['mean_cer']:.2f}%")
        print(f"  Mean RTF: {overall['mean_rtf']:.4f}")
        print(f"  Mean tokens/s: {overall['mean_tokens_per_second']:.1f}")
        print(f"  Mean ms/token: {overall['mean_ms_per_token']:.2f}")
        print(f"  Time distribution: Enc {overall['mean_encoder_pct']:.1f}% | "
              f"Dec {overall['mean_decoder_pct']:.1f}% | "
              f"Ovhd {overall['mean_overhead_pct']:.1f}%")
        
        print("\nBy Duration Category:")
        for cat in ['0-10s', '10-20s', '20-30s', '30s+']:
            if cat in summary['by_duration']:
                data = summary['by_duration'][cat]
                print(f"\n  {cat}: {data['num_files']} files")
                print(f"    Mean duration: {data['mean_duration_s']:.1f}s")
                print(f"    WER: {data['mean_wer']:.2f}%")
                print(f"    RTF: {data['mean_rtf']:.4f}")
                print(f"    Tokens/s: {data['mean_tokens_per_second']:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper models on LibriSpeech")
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to LibriSpeech test-clean directory')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--models', nargs='+', 
                       default=['openai/whisper-small'],
                       help='Models to benchmark')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs per file for stable measurements')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = WhisperBenchmark()
    
    # Benchmark each model
    all_results = {}
    for model_name in args.models:
        results = benchmark.benchmark_model(
            model_name=model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_runs=args.num_runs,
            max_files=args.max_files
        )
        all_results[model_name] = results
    
    # Save combined summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.output_dir) / f"benchmark_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump({
            'models': args.models,
            'dataset': args.data_path,
            'timestamp': timestamp,
            'summaries': {
                model: results['summary'] 
                for model, results in all_results.items()
            }
        }, f, indent=2)
    
    print(f"\n✅ All benchmarks complete! Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
    
