# Conversation_New - GPU-Optimized Voice Recognition Pipeline

Production-ready speech pipeline with fast transcription (faster-whisper by default), speaker diarization, optional speaker enrollment (voice fingerprints), and clean outputs.

## Features
- ASR: faster-whisper (GPU-optimized) and OpenAI Whisper (fallback)
- Diarization: VAD + clustering, robust fallbacks to avoid empty outputs
- Enrollment: Map diarized clusters to known speakers using optimal assignment
- Outputs: JSON analysis + conversation text (clean, deduplicated)
- GPU: Auto-detect and optimize (float16, TF32, batching)

## Requirements
Install Python 3.8+ and FFmpeg on your system, then:
```bash
pip install -r requirements.txt
```

If you need CUDA builds (recommended on GPU):
```bash
# Example for CUDA 11.8 (adjust per your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start
Place your audio files in `src/` and run one of the following:

### Fastest (GPU) - faster-whisper (default)
```bash
python run.py -d src --asr faster-whisper --faster-size base
# Maximum speed (smaller model):
python run.py -d src --asr faster-whisper --faster-size tiny
```

### Single file
```bash
python run.py -f "src/your_audio.mp3" --asr faster-whisper --faster-size base
```

### Whisper-base (accuracy-first)
```bash
python run.py -d src --asr whisper --whisper-size base
```

Outputs are written to `out/`:
- `<name>_streamlined_analysis.json`
- `<name>_conversation.txt`

## Enrollment (Optional)
Provide reference audios to map speakers to names.
- Place files under `Reference/<PersonName>/*.wav|*.mp3` or in `enroll/`
- The pipeline will autodetect and build prototypes per person
- Mapping uses normalized similarities and optimal (Hungarian) assignment

## Project Structure
```
Conversation_New/
  run.py                      # CLI entrypoint
  pipeline.py                 # Orchestrator
  asr_service.py              # ASR engines (faster-whisper, whisper)
  diarization_service.py      # VAD + clustering, robust fallbacks
  speechbrain_service.py      # Speaker embeddings (SpeechBrain / Resemblyzer)
  feature_extraction_service.py
  enrollment_service.py       # Enrollment + optimal mapping
  output_service.py           # JSON + conversation outputs
  gpu_optimization.py         # GPU tuning helpers
  requirements.txt            # Consolidated dependencies
  src/                        # Input audio files
  out/                        # Generated outputs
  Reference/ | enroll/        # Optional enrollment data
```

## CLI Usage
```bash
# Directory mode
python run.py -d src [--asr {faster-whisper,whisper}] [--faster-size {tiny,base,small}] [--whisper-size {tiny,base,small}]

# Single file mode
python run.py -f path/to/audio.ext [--asr ...] [...]
```

## Performance Notes
- Default backend is faster-whisper tuned for speed: float16 on GPU, beam_size=1, chunk_length=30, VAD enabled
- CPU path uses small beam for acceptable quality
- Parallelization: ASR and diarization run concurrently on GPU; multi-file processing parallelized

## Output Details
- JSON: includes segments, speaker mappings, and a summary
- Conversation TXT: human-readable merged lines; adjacent same-speaker turns collapsed; text deduplicated

## Robust Fallbacks
- If VAD hears no speech: produce one full-length segment
- If alignment yields no text but ASR produced segments: emit a single User1 segment spanning ASR time range

## Troubleshooting
- Very slow on CPU: use `--asr faster-whisper --faster-size tiny` or ensure CUDA drivers installed
- No conversation text: check that audio has speech; fallbacks will now emit at least one segment if ASR produced text
- Torch/CUDA wheels: install from PyTorch CUDA index per your driver version
- SpeechBrain import issues: pinned torch/torchaudio versions in `requirements.txt` improve compatibility

## License
This repository provides a production-oriented example pipeline. Ensure you comply with the licenses of third-party models and datasets you use.

