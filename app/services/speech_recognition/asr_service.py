"""
ASR service
- Supports OpenAI Whisper and faster-whisper
- Returns list of segments: {start: float, end: float, text: str, words: list}
"""

from __future__ import annotations

from typing import List, Dict
import numpy as np
import librosa
import logging
import torch
import os

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(
        self,
        backend: str = "whisper",
        whisper_model_size: str = "base",
        faster_whisper_size: str = "base",
    ) -> None:
        self.backend = (backend or "whisper").lower().strip()
        self.whisper_model_size = whisper_model_size
        self.faster_whisper_size = faster_whisper_size
        self.whisper_model = None
        self.fw_model = None
        
        # GPU optimization settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            try:
                torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
            except Exception:
                pass
            logger.info("Using CPU")
            
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize ASR models with GPU optimization."""
        if self.backend == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                if device == "cuda":
                    compute_type = "float16"
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                
                logger.info(f"Loading faster-whisper ({device}, {compute_type})")
                self.fw_model = WhisperModel(
                    self.faster_whisper_size, 
                    device=device, 
                    compute_type=compute_type,
                    cpu_threads=8 if device == "cpu" else 1,
                    num_workers=2
                )
                return
            except Exception as e:
                logger.error("Failed to load faster-whisper: %s", e)
                self.backend = "whisper"

        if self.backend == "whisper":
            try:
                import whisper
                logger.info(f"Loading OpenAI Whisper ({self.whisper_model_size}) on {self.device}")
                self.whisper_model = whisper.load_model(self.whisper_model_size, device=self.device)
            except Exception as e:
                logger.error("Failed to load Whisper: %s", e)

    def transcribe(self, audio_path: str) -> List[Dict]:
        """Run ASR and return list of segments with GPU optimization."""
        segments: List[Dict] = []

        # Faster-whisper (max speed tuning)
        if self.backend == "faster-whisper" and self.fw_model:
            try:
                params = dict(
                    vad_filter=True,
                    word_timestamps=False,
                    language=None,
                    task="transcribe",
                )
                if self.device == "cuda":
                    # Max speed on GPU: low beam, small chunk length, faster CT
                    seg_iter, _ = self.fw_model.transcribe(
                        audio_path,
                        beam_size=1,
                        chunk_length=30,
                        **params
                    )
                else:
                    # CPU: slightly higher beam for quality
                    seg_iter, _ = self.fw_model.transcribe(
                        audio_path,
                        beam_size=2,
                        **params
                    )
                for seg in seg_iter:
                    text = (seg.text or "").strip()
                    if text:
                        segments.append({
                            "start": float(seg.start or 0.0),
                            "end": float(seg.end or 0.0),
                            "text": text,
                            "words": []
                        })
                return segments
            except Exception as e:
                logger.error("faster-whisper failed: %s", e)

        # Whisper fallback (balanced)
        if self.whisper_model:
            try:
                import whisper
                waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
                waveform = waveform.astype(np.float32)
                fp16 = (self.device == "cuda")
                result = self.whisper_model.transcribe(
                    waveform,
                    fp16=fp16,
                    language=None,
                    task="transcribe",
                    beam_size=2 if self.device == "cpu" else 1,
                    condition_on_previous_text=False,
                )
                for segment in result.get("segments", []):
                    text = (segment.get("text", "").strip())
                    if text:
                        segments.append({
                            "start": float(segment.get("start", 0.0)),
                            "end": float(segment.get("end", 0.0)),
                            "text": text,
                            "words": segment.get("words", []),
                        })
            except Exception as e:
                logger.error("Whisper failed: %s", e)

        return segments
