# accent-robust-asr

Final year project workspace for accent-robust speech recognition using the L2-ARCTIC dataset and Whisper STT models.

## Repository Structure

```text
.
├── config/
├── logs/
├── output/
├── requirements.txt
└── src/
	├── eval/
	├── notebooks/
	├── preproc/
	└── training/
```

## Quickstart

1. Create and activate a virtual environment.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install `ffmpeg` on your system (required by Whisper/audio tooling).

## Notes

- Put experiment configs in `config/`.
- Keep generated logs in `logs/` and model artifacts/results in `output/`.
- Commit code and lightweight config files only; large data/artifacts are ignored.