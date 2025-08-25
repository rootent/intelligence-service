"""
Speaker embedding service
- Primary: SpeechBrain ECAPA-TDNN
- Fallback: Resemblyzer
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import librosa
import logging
import torch
import os


logger = logging.getLogger(__name__)


class SpeakerEmbeddingService:
    def __init__(self) -> None:
        self.sb_encoder = None
        self.voice_encoder = None
        
        # GPU optimization settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
        else:
            logger.info("Using CPU for speaker embedding")
            
        self._initialize_encoders()

    def _initialize_encoders(self) -> None:
        try:
            from speechbrain.pretrained import EncoderClassifier
            logger.info("Loading SpeechBrain ECAPA-TDNN...")
            
            # GPU optimization for SpeechBrain
            if self.device == "cuda":
                # Set CUDA device explicitly
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                # Use GPU with half precision for speed
                try:
                    self.sb_encoder = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        run_opts={"device": self.device}
                    )
                    # Move to GPU and enable half precision
                    self.sb_encoder = self.sb_encoder.to(self.device)
                    # Try to enable half precision if supported
                    try:
                        if hasattr(self.sb_encoder, 'mods'):
                            for module in self.sb_encoder.mods.values():
                                if hasattr(module, 'half'):
                                    module.half()
                    except Exception:
                        logger.info("Half precision not supported for SpeechBrain on this GPU")
                except Exception as e:
                    logger.warning(f"GPU SpeechBrain failed, falling back to CPU: {e}")
                    self.sb_encoder = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb"
                    )
            else:
                self.sb_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb"
                )
                
            logger.info(f"SpeechBrain loaded on {self.device}")
        except Exception as e:
            logger.error("SpeechBrain failed: %s", e)
            self.sb_encoder = None
            
        try:
            from resemblyzer import VoiceEncoder
            logger.info("Loading Resemblyzer...")
            
            # GPU optimization for Resemblyzer
            if self.device == "cuda":
                try:
                    self.voice_encoder = VoiceEncoder(device=self.device)
                except Exception:
                    logger.warning("GPU Resemblyzer failed, falling back to CPU")
                    self.voice_encoder = VoiceEncoder()
            else:
                self.voice_encoder = VoiceEncoder()
                
            logger.info(f"Resemblyzer loaded on {self.device}")
        except Exception as e:
            logger.error("Resemblyzer failed: %s", e)
            self.voice_encoder = None

    def compute_embedding(self, waveform: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        try:
            if self.sb_encoder is not None:
                if sample_rate != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                
                # GPU optimization for tensor operations
                wav_t = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
                if self.device == "cuda":
                    try:
                        wav_t = wav_t.to(self.device)
                        # Try half precision if supported
                        if wav_t.dtype != torch.float16 and hasattr(self.sb_encoder, 'mods'):
                            wav_t = wav_t.half()
                    except Exception:
                        logger.warning("GPU tensor operations failed, using CPU")
                        wav_t = wav_t.cpu()
                
                with torch.no_grad():
                    emb = self.sb_encoder.encode_batch(wav_t)
                
                # Move back to CPU and convert to numpy
                if self.device == "cuda":
                    emb = emb.cpu()
                return emb.squeeze().numpy().astype(np.float32)
                
        except Exception as e:
            logger.warning("SpeechBrain embedding failed: %s", e)
            
        try:
            if self.voice_encoder is not None:
                if sample_rate != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                
                # Resemblyzer handles GPU internally
                emb = self.voice_encoder.embed_utterance(waveform.astype(np.float32))
                return emb.astype(np.float32)
                
        except Exception as e:
            logger.warning("Resemblyzer embedding failed: %s", e)
            
        return None


