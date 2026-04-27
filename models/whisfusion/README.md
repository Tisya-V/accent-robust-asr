# Whisfusion: Parallel ASR Decoding via a Diffusion Transformer

[![arXiv](https://img.shields.io/badge/arXiv-2025.07048-b31b1b.svg)](https://arxiv.org/abs/2508.07048)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/taeyoun811/whisfusion)


Official implementation of **Whisfusion** - the first Diffusion Transformer ASR framework that fuses a Whisper encoder with a diffusion decoder for faster, non-autoregressive transcription.

<p align="center">
  <img src="assets/inference.gif" width="80%">
</p>

## 🏗️ Architecture

Whisfusion combines three key components:

- **Whisper-small Encoder** (88.2M params): Pre-trained acoustic feature extractor from OpenAI
- **SMDM-170M Decoder**: Masked diffusion model for parallel text generation
- **Cross-Attention Adapter** (42.5M params): Lightweight bridge between modalities


## 🎯 Key Features

- **Parallel decoding architecture** with 14× higher throughput than autoregressive models (3,180 vs 230 tokens/s)
- **Superior accuracy** - lower WER than Whisper-tiny (8.3% vs 9.7%) with comparable latency
- **Scalable inference** - constant time regardless of sequence length (up to 2.6× faster on long audio >20s)



## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/taeyoun811/Whisfusion.git
cd Whisfusion

# Install PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention (required for efficient attention)
git clone --recurse-submodules --branch v2.6.3 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install .
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../../.. && rm -rf flash-attention

# Install other dependencies
pip install -r requirements.txt
```


### Download Pre-trained Models

```python
from huggingface_hub import hf_hub_download

# Download Stage 2 model (full model with decoder fine-tuning)
hf_hub_download(
    repo_id="taeyoun811/whisfusion",
    filename="whisfusion_stage2_decoder.pt",
    local_dir="out/",
)

# Download Stage 1 adapter (optional, adapter-only checkpoint)
hf_hub_download(
    repo_id="taeyoun811/whisfusion",
    filename="whisfusion_stage1_adapter.pt",
    local_dir="out/",
)
```

Then use the downloaded models with the evaluation scripts in `src/evaluation/`.


## 💻 Training Requirements

- **GPUs**: 4× NVIDIA A100 (40GB)
- **Storage**: ~700GB for full LibriSpeech 960h dataset and preprocessed features


## 📚 Training & Evaluation

### Data Preparation

```bash
# 1. Download LibriSpeech (~60GB)
bash scripts/00_download_librispeech.sh

# 2. Preprocess audio to Whisper features
bash scripts/01_preprocess_librispeech.sh

# 3. Download pre-trained SMDM weights
bash scripts/02_download_pretrained_model.sh
```

### Two-Stage Training

```bash
# Stage 1: Train adapter only (frozen encoder/decoder)
bash scripts/03_train_stage1_adapter.sh

# Stage 2: Fine-tune full decoder (update --pretrain_path to Stage 1 checkpoint)
bash scripts/04_train_stage2_decoder_high_ratio.sh
```

### Evaluation

```bash
# Evaluate Whisfusion (update --adapter_path to Stage 2 checkpoint)
bash scripts/05_evaluate_whisfusion.sh

# Compare with Whisper baselines
bash scripts/06_evaluate_whisper.sh
```

Results will be saved in `evaluation_results/` directory as JSON files.

## 📁 Project Structure

```
Whisfusion/
├── scripts/                   # Training and evaluation scripts
├── src/
│   ├── data/                  # Data preprocessing and dataset
│   ├── training/              # Training scripts for both stages
│   ├── evaluation/            # Evaluation and inference code
│   └── lit_gpt/               # Model architecture
├── data/                      # Dataset storage
│   ├── raw/                   # Original LibriSpeech files
│   └── processed/             # Preprocessed Whisper features
├── out/                       # Training outputs and downloaded models
└── pretrained_models/         # Pre-trained (SMDM-170M) checkpoints
```

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{kwon2025whisfusion,
  title={Whisfusion: Parallel ASR Decoding via a Diffusion Transformer},
  author={Kwon, Taeyoun and Ahn, Junhyuk and Yun, Taegeun and Jwa, Heeju and Choi, Yoonchae and Park, Siwon and Kim, Nam-Joon and Kim, Jangchan and Ryu, Hyun Gon and Lee, Hyuk-Jae},
  journal={arXiv preprint arXiv:2508.07048},
  year={2025}
}
```

## 🙏 Acknowledgments

This project builds upon the following excellent works:

### SMDM
- Repository: [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM)
- Paper: [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) (ICLR 2025)
- We utilize the SMDM-170M checkpoint and masked diffusion implementation

### Whisper
- Repository: [openai/whisper](https://github.com/openai/whisper)
- We use the pre-trained Whisper encoder for acoustic feature extraction


## 📧 Contact

For questions, discussions, or contributions:
- Open an issue on GitHub
- Email: [ty8352@snu.ac.kr](mailto:ty8352@snu.ac.kr)