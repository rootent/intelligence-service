"""
Enrollment service
- Loads enrollment audio from Reference/ (subfolders per person) or flat directory
- Builds per-person prototype embedding and features
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import librosa
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from speechbrain_service import SpeakerEmbeddingService
from feature_extraction_service import FeatureExtractionService


logger = logging.getLogger(__name__)


class EnrollmentService:
    def __init__(self, embedding_service: SpeakerEmbeddingService, feature_service: FeatureExtractionService) -> None:
        self.embedding_service = embedding_service
        self.feature_service = feature_service
        self.enrollment_profiles: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        self.speaker_name_map: Dict[str, str] = {}

    def autodetect_dir(self, explicit: Optional[str] = None) -> Optional[Path]:
        if explicit:
            p = Path(explicit)
            return p if p.exists() and p.is_dir() else None
        for c in [Path("Reference"), Path("enroll")]:
            if c.exists() and c.is_dir():
                return c
        return None

    def load(self, enroll_dir: Path) -> None:
        audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.m4b', '.aac'}
        profiles: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        logger.info("Loading enrollment from %s", enroll_dir)
        subfolders = [p for p in enroll_dir.iterdir() if p.is_dir()]
        if subfolders:
            for folder in subfolders:
                name = folder.name
                emb_list: List[np.ndarray] = []
                feat_list: List[np.ndarray] = []
                for f in folder.iterdir():
                    if not f.is_file() or f.suffix.lower() not in audio_exts:
                        continue
                    try:
                        wav, _sr = librosa.load(str(f), sr=16000, mono=True)
                        emb = self.embedding_service.compute_embedding(wav, 16000)
                        feat = self.feature_service.extract(wav, 16000)
                        if emb is not None:
                            emb_list.append(np.asarray(emb, dtype=np.float32))
                        if feat is not None:
                            feat_list.append(np.asarray(feat, dtype=np.float32))
                    except Exception as e:
                        logger.error("Failed enrollment file %s: %s", f, e)
                if not emb_list and not feat_list:
                    logger.warning("No valid enrollment for %s", name)
                    continue
                embedding: Optional[np.ndarray] = None
                if emb_list:
                    dims: Dict[int, int] = {}
                    for e in emb_list:
                        dims[e.shape[-1]] = dims.get(e.shape[-1], 0) + 1
                    common_dim = max(dims.items(), key=lambda x: x[1])[0]
                    emb_list = [e for e in emb_list if e.shape[-1] == common_dim]
                    embedding = np.mean(np.stack(emb_list, axis=0), axis=0).astype(np.float32)
                features: Optional[np.ndarray] = None
                if feat_list:
                    features = np.mean(np.stack(feat_list, axis=0), axis=0).astype(np.float32)
                profiles[name] = {"embedding": embedding, "features": features}
                logger.info("Enrolled: %s", name)
        else:
            for f in enroll_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in audio_exts:
                    continue
                name = f.stem
                try:
                    wav, _sr = librosa.load(str(f), sr=16000, mono=True)
                    emb = self.embedding_service.compute_embedding(wav, 16000)
                    feat = self.feature_service.extract(wav, 16000)
                    profiles[name] = {"embedding": emb, "features": feat}
                    logger.info("Enrolled: %s", name)
                except Exception as e:
                    logger.error("Failed enrollment %s: %s", name, e)
        self.enrollment_profiles = profiles
        logger.info("Enrollment loaded for %d speaker(s)", len(self.enrollment_profiles))

    def _normalize(self, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-8
        return v / n

    def map_clusters(self, segments: List[Dict]) -> List[Dict]:
        if not segments or not self.enrollment_profiles:
            return segments
        clusters: Dict[str, List[Dict]] = {}
        for seg in segments:
            label = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker_Unknown"))
            clusters.setdefault(label, []).append(seg)
        cluster_reps: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for label, segs in clusters.items():
            emb_list_raw = [np.asarray(s.get("embedding")) for s in segs if s.get("embedding") is not None]
            emb_list: List[np.ndarray] = []
            if emb_list_raw:
                dims: Dict[int, int] = {}
                for e in emb_list_raw:
                    dims[e.shape[-1]] = dims.get(e.shape[-1], 0) + 1
                common_dim = max(dims.items(), key=lambda x: x[1])[0]
                emb_list = [e for e in emb_list_raw if e.shape[-1] == common_dim]
            feat_list = [np.asarray(s.get("features")) for s in segs if s.get("features") is not None]
            rep: Dict[str, Optional[np.ndarray]] = {"embedding": None, "features": None}
            if emb_list:
                rep["embedding"] = self._normalize(np.mean(np.stack(emb_list, axis=0), axis=0))
            if feat_list:
                rep["features"] = self._normalize(np.mean(np.stack(feat_list, axis=0), axis=0))
            cluster_reps[label] = rep
        enroll_emb = {n: self._normalize(p["embedding"]) for n, p in self.enrollment_profiles.items() if p.get("embedding") is not None}
        enroll_feat = {n: self._normalize(p["features"]) for n, p in self.enrollment_profiles.items() if p.get("features") is not None}

        # Build similarity matrices
        labels = list(cluster_reps.keys())
        names_emb = list(enroll_emb.keys())
        names_feat = list(enroll_feat.keys())
        S_emb = None
        S_feat = None
        if labels and names_emb:
            C = np.stack([cluster_reps[l]["embedding"] if cluster_reps[l]["embedding"] is not None else np.zeros_like(next(iter(enroll_emb.values()))) for l in labels])
            M = np.stack([enroll_emb[n] for n in names_emb])
            S_emb = (C @ M.T).astype(np.float32)
        if labels and names_feat:
            C = np.stack([cluster_reps[l]["features"] if cluster_reps[l]["features"] is not None else np.zeros_like(next(iter(enroll_feat.values()))) for l in labels])
            M = np.stack([enroll_feat[n] for n in names_feat])
            S_feat = (C @ M.T).astype(np.float32)

        # Combine similarities with priority to embeddings
        combined_names = list({*names_emb, *names_feat})
        if not combined_names:
            return segments
        name_to_idx = {n: i for i, n in enumerate(combined_names)}
        S = np.zeros((len(labels), len(combined_names)), dtype=np.float32)
        if S_emb is not None:
            for i, l in enumerate(labels):
                for j, n in enumerate(names_emb):
                    S[i, name_to_idx[n]] = max(S[i, name_to_idx[n]], S_emb[i, j])
        if S_feat is not None:
            for i, l in enumerate(labels):
                for j, n in enumerate(names_feat):
                    S[i, name_to_idx[n]] = max(S[i, name_to_idx[n]], 0.7 * S_feat[i, j])

        # Convert to cost matrix and solve optimal assignment
        cost = 1.0 - S
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned: Dict[str, str] = {}
        thr = 0.62  # production threshold after normalization
        for r, c in zip(row_ind, col_ind):
            if S[r, c] >= thr:
                assigned[labels[r]] = combined_names[c]
                logger.info("Enrollment mapping: %s -> %s (sim=%.3f)", labels[r], combined_names[c], S[r, c])

        self.speaker_name_map = assigned
        for seg in segments:
            lbl = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker_Unknown"))
            if lbl in assigned:
                seg["clustered_speaker"] = assigned[lbl]
        return segments


