"""
Diarization service
- VAD (energy-based)
- Segment embedding/features
- Agglomerative clustering with cosine distance
"""

from __future__ import annotations

from typing import List, Dict
import numpy as np
import librosa
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch

from speechbrain_service import SpeakerEmbeddingService
from feature_extraction_service import FeatureExtractionService

logger = logging.getLogger(__name__)


class DiarizationService:
    def __init__(self, embedding_service: SpeakerEmbeddingService, feature_service: FeatureExtractionService) -> None:
        self.embedding_service = embedding_service
        self.feature_service = feature_service
        
        # GPU optimization settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU detected for diarization: {torch.cuda.get_device_name()}")
        else:
            logger.info("Using CPU for diarization")
        
        # Fast mode flag: when True, skip heavy embeddings and use lightweight features
        self.fast_mode = False

    def set_fast_mode(self, enabled: bool = True) -> None:
        self.fast_mode = bool(enabled)
        logger.info("Diarization fast_mode=%s", self.fast_mode)

    def _smooth_vad(self, voice_frames: np.ndarray, min_speech_duration: float = 0.5) -> np.ndarray:
        """Smooth out short bursts of speech/noise."""
        frame_rate = 100
        min_speech_frames = int(min_speech_duration * frame_rate)
        smoothed = voice_frames.copy()
        in_speech, start = False, 0

        for i, v in enumerate(voice_frames):
            if v and not in_speech:
                start, in_speech = i, True
            elif not v and in_speech:
                if i - start < min_speech_frames:
                    smoothed[start:i] = False
                in_speech = False
        return smoothed

    def _vad(self, waveform: np.ndarray, sample_rate: int) -> List[Dict]:
        """Energy-based voice activity detection with GPU optimization."""
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.01 * sample_rate)
        
        # GPU optimization for RMS calculation if available
        if self.device == "cuda" and len(waveform) > 10000:  # Only for longer audio
            try:
                # Convert to tensor for GPU processing
                wav_tensor = torch.from_numpy(waveform).to(self.device)
                # Calculate RMS using PyTorch for GPU acceleration
                frames = torch.nn.functional.unfold(
                    wav_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=(1, frame_length), 
                    stride=(1, hop_length)
                )
                energy = torch.sqrt(torch.mean(frames**2, dim=1)).squeeze().cpu().numpy()
            except Exception:
                # Fallback to librosa if GPU processing fails
                logger.warning("GPU VAD failed, falling back to CPU")
                energy = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        else:
            energy = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]

        # Slightly more permissive threshold and min duration to avoid missing quiet speech
        thr = np.percentile(energy, 20 if self.fast_mode else 30)
        voice_frames = self._smooth_vad(energy > thr, min_speech_duration=0.25 if self.fast_mode else 0.5)
        frame_times = librosa.frames_to_time(np.arange(len(voice_frames)), sr=sample_rate, hop_length=hop_length)

        segments, in_segment, start_time = [], False, 0.0
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_segment:
                start_time, in_segment = float(frame_times[i]), True
            elif not is_voice and in_segment:
                end_time = float(frame_times[i])
                if end_time - start_time >= (0.2 if self.fast_mode else 0.5):
                    segments.append({
                        "speaker_id": "Speaker_Unknown",
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                    })
                in_segment = False

        if in_segment:  # last open segment
            end_time = float(frame_times[-1])
            if end_time - start_time >= (0.2 if self.fast_mode else 0.5):
                segments.append({
                    "speaker_id": "Speaker_Unknown",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                })
        return segments

    def _estimate_n_speakers(self, distance_matrix: np.ndarray) -> int:
        """Estimate speaker count via silhouette score with optimization."""
        try:
            from sklearn.metrics import silhouette_score
            n_samples = distance_matrix.shape[0]
            max_speakers = min(4, n_samples - 1)
            if max_speakers < 2:
                return 1

            best_k, best_score = 2, -1.0
            
            # Optimize clustering for speed
            for k in range(2, max_speakers + 1):
                clustering = AgglomerativeClustering(
                    n_clusters=k, 
                    metric="precomputed", 
                    linkage="average",
                    compute_full_tree=False  # Speed optimization
                )
                labels = clustering.fit_predict(distance_matrix)
                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                if score > best_score:
                    best_score, best_k = score, k
            return best_k
        except Exception:
            return 2

    def diarize(self, audio_path: str) -> List[Dict]:
        """Main diarization pipeline with GPU optimization."""
        import soundfile as sf
        waveform, sample_rate = sf.read(audio_path)
        if waveform.ndim > 1:  # convert stereo to mono
            waveform = np.mean(waveform, axis=1)

        # If near-silent audio, early fallback to full-length segment
        total_dur = float(len(waveform) / max(1, sample_rate))
        if total_dur <= 0.0:
            return []

        voice_segments = self._vad(waveform, sample_rate)
        if len(voice_segments) < 1:
            logger.info("VAD found no speech; using full-length single segment fallback")
            return [{
                "speaker_id": "Speaker_Unknown",
                "start_time": 0.0,
                "end_time": total_dur,
                "duration": total_dur,
            }]
        if len(voice_segments) < 2:
            return voice_segments

        # Enrich segments (fast mode uses only lightweight features)
        enriched = []
        batch_size = 16 if (self.device == "cuda" and self.fast_mode) else (8 if self.device == "cuda" else 1)
        
        for i in range(0, len(voice_segments), batch_size):
            batch_segments = voice_segments[i:i + batch_size]
            batch_enriched = []
            
            for seg in batch_segments:
                start_idx, end_idx = int(seg["start_time"] * sample_rate), int(seg["end_time"] * sample_rate)
                segment_audio = waveform[start_idx:end_idx]
                if len(segment_audio) < sample_rate * (0.2 if self.fast_mode else 0.3):
                    continue

                out = seg.copy()
                if self.fast_mode:
                    out["features"] = self.feature_service.extract(segment_audio, sample_rate)
                else:
                    emb = self.embedding_service.compute_embedding(segment_audio, sample_rate)
                    if emb is not None:
                        out["embedding"] = emb.astype(np.float32)
                    else:
                        out["features"] = self.feature_service.extract(segment_audio, sample_rate)
                batch_enriched.append(out)
            
            enriched.extend(batch_enriched)

        if len(enriched) < 2:
            return voice_segments

        # Build similarity matrix
        vectors, valid = [], []
        for s in enriched:
            if "embedding" in s and not self.fast_mode:
                vectors.append(np.asarray(s["embedding"]))
            elif "features" in s:
                vectors.append(np.asarray(s["features"]))
            valid.append(s)

        if len(vectors) < 2:
            return voice_segments

        V = np.stack(vectors)
        
        # GPU optimization for cosine similarity if available
        if self.device == "cuda" and len(V) > 10:
            try:
                V_tensor = torch.from_numpy(V.astype(np.float32)).to(self.device)
                V_norm = torch.nn.functional.normalize(V_tensor, p=2, dim=1)
                sim_matrix = torch.mm(V_norm, V_norm.t())
                dist = (1.0 - sim_matrix).cpu().numpy()
                dist = np.clip(dist, 0.0, 2.0)
            except Exception:
                logger.warning("GPU cosine similarity failed, falling back to CPU")
                dist = np.clip(1.0 - cosine_similarity(V), 0.0, 2.0)
        else:
            dist = np.clip(1.0 - cosine_similarity(V), 0.0, 2.0)

        # Cluster speakers
        k = self._estimate_n_speakers(dist)
        clustering = AgglomerativeClustering(
            n_clusters=k, 
            metric="precomputed", 
            linkage="average",
            compute_full_tree=False
        )
        labels = clustering.fit_predict(dist)

        clustered = []
        for i, (segment, label) in enumerate(zip(valid, labels)):
            seg = segment.copy()
            seg["clustered_speaker"] = f"User{label + 1}"
            try:
                cluster_vecs = V[labels == label]
                sims = cosine_similarity(V[i:i+1], cluster_vecs)[0]
                seg["confidence"] = float(((np.mean(sims) + 1.0) / 2.0))
            except Exception:
                seg["confidence"] = 0.9
            seg.pop("features", None)
            if "embedding" in seg and not self.fast_mode:
                seg["embedding"] = seg["embedding"].tolist()
            clustered.append(seg)

        return self._normalize_labels_by_talk_time(clustered)

    def _normalize_labels_by_talk_time(self, segments: List[Dict]) -> List[Dict]:
        """Ensure most talkative speaker gets User1, next User2, etc."""
        if not segments:
            return segments

        durations = {}
        for seg in segments:
            label = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker_Unknown"))
            durations[label] = durations.get(label, 0.0) + seg.get("duration", 0.0)

        ordered = sorted(durations.items(), key=lambda x: x[1], reverse=True)
        label_map = {old: f"User{idx+1}" for idx, (old, _) in enumerate(ordered)}

        for seg in segments:
            old = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker_Unknown"))
            seg["clustered_speaker"] = label_map.get(old, old)
        return segments
