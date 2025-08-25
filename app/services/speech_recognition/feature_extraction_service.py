"""
Feature extraction service
- Provides MFCC + spectral features vector for diarization fallback
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import librosa
import logging


logger = logging.getLogger(__name__)


class FeatureExtractionService:
    def extract(self, segment_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sample_rate)
            chroma = librosa.feature.chroma_stft(y=segment_audio, sr=sample_rate)

            features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                [np.mean(spectral_centroid)],
                [np.mean(spectral_rolloff)],
                [np.mean(spectral_bandwidth)],
                [np.mean(chroma)],
            ])
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            return features.astype(np.float32)
        except Exception as e:
            logger.error("Feature extraction failed: %s", e)
            return np.zeros(32, dtype=np.float32)


