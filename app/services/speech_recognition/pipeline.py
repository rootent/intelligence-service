"""
Production-friendly modular pipeline orchestrator
"""

from __future__ import annotations

from typing import Dict, List, Iterator, Tuple
from pathlib import Path
import os
import logging
import librosa
import soundfile as sf
import numpy as np
import torch
import concurrent.futures
from functools import partial
from datetime import datetime

from asr_service import ASRService
from speechbrain_service import SpeakerEmbeddingService
from feature_extraction_service import FeatureExtractionService
from diarization_service import DiarizationService
from enrollment_service import EnrollmentService
from output_service import OutputService
from gpu_optimization import optimize_gpu_settings, get_optimal_batch_size


logger = logging.getLogger(__name__)


class StreamlinedPipeline:
    def __init__(
        self,
        src_folder: str = "src",
        output_folder: str = "out",
        asr_backend: str = "faster-whisper",
        whisper_model_size: str = "base",
        faster_whisper_size: str = "tiny",
        enroll_dir: str | None = None,
        stream: bool = False,
        chunk_seconds: int = 30,
        chunk_overlap: float = 2.0,
    ) -> None:
        self.src_folder = Path(src_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # GPU optimization settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Apply GPU optimizations
            optimize_gpu_settings()
            logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Get optimal batch size for this GPU
            self.optimal_batch_size = get_optimal_batch_size()
            logger.info(f"Optimal batch size: {self.optimal_batch_size}")
        else:
            logger.info("Using CPU")
            self.optimal_batch_size = 1
            
        # Services
        self.asr = ASRService(asr_backend, whisper_model_size, faster_whisper_size)
        self.embedding = SpeakerEmbeddingService()
        self.features = FeatureExtractionService()
        self.diar = DiarizationService(self.embedding, self.features)
        self.enroll = EnrollmentService(self.embedding, self.features)
        self.output = OutputService(output_folder)
        
        # Enrollment
        self.enrollment_dir = self.enroll.autodetect_dir(enroll_dir)
        if self.enrollment_dir is not None:
            self.enroll.load(self.enrollment_dir)
        try:
            self.diar.set_fast_mode(enabled=(len(self.enroll.enrollment_profiles) == 0))
        except Exception:
            pass

        self.audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.m4b', '.aac'}

        # Streaming config
        self.stream = bool(stream)
        self.chunk_seconds = int(max(5, chunk_seconds))
        self.chunk_overlap = float(max(0.0, min(self.chunk_seconds - 1, chunk_overlap)))

    def _preprocess(self, audio_path: str) -> str:
        """Convert to 16 kHz mono WAV with GPU optimization."""
        audio_path = Path(audio_path)
        wav_path = self.output_folder / f"{audio_path.stem}_preprocessed.wav"
        
        # GPU optimization for audio loading and resampling
        if self.device == "cuda":
            try:
                waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
                waveform = waveform.astype(np.float32)
                if np.max(np.abs(waveform)) > 0:
                    waveform = waveform / np.max(np.abs(waveform)) * 0.95
            except Exception:
                waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        else:
            waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
            
        sf.write(str(wav_path), waveform, 16000)
        return str(wav_path)

    def _merge_speaker_turns(self, speaker_segments: List[Dict]) -> List[Dict]:
        if not speaker_segments:
            return []
        segments_sorted = sorted(speaker_segments, key=lambda s: (float(s.get("start_time", 0.0)), float(s.get("end_time", 0.0))))
        merged: List[Dict] = []
        for seg in segments_sorted:
            speaker = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker_Unknown"))
            if not merged:
                merged.append(seg.copy())
                continue
            last = merged[-1]
            last_speaker = last.get("clustered_speaker", last.get("speaker_id", "Speaker_Unknown"))
            if speaker == last_speaker and float(seg["start_time"]) <= float(last["end_time"]) + 0.25:
                last["end_time"] = float(max(last["end_time"], seg["end_time"]))
                last["duration"] = float(last["end_time"] - last["start_time"]) 
            else:
                merged.append(seg.copy())
        return merged

    def _overlap(self, a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        return max(0.0, end - start)

    def _dedupe_texts(self, texts: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for t in texts:
            k = t.strip().lower()
            if not k:
                continue
            if k in seen:
                continue
            if len(k) <= 2:
                continue
            seen.add(k)
            deduped.append(t.strip())
        return deduped

    def align(self, asr_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        if not speaker_segments:
            return []
        turns = self._merge_speaker_turns(speaker_segments)
        aligned: List[Dict] = []
        for turn in turns:
            turn_start = float(turn["start_time"])
            turn_end = float(turn["end_time"])
            texts: List[str] = []
            for asr in asr_segments:
                a_start = float(asr.get("start", 0.0))
                a_end = float(asr.get("end", 0.0))
                if self._overlap(a_start, a_end, turn_start, turn_end) > 0:
                    t = (asr.get("text") or "").strip()
                    if t:
                        texts.append(t)
            texts = self._dedupe_texts(texts)
            joined = " ".join(texts)
            joined = " ".join(joined.split())
            seg = turn.copy()
            seg["text"] = joined if joined else "[No transcription]"
            aligned.append(seg)
        return aligned

    # ---------- Streaming helpers ----------
    def _iterate_chunks(self, wav_path: str) -> Iterator[Tuple[float, float, str]]:
        import soundfile as sf
        waveform, sr = sf.read(wav_path)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        total = float(len(waveform) / max(sr, 1))
        start = 0.0
        while start < total:
            end = min(total, start + self.chunk_seconds)
            yield start, end, wav_path  # We will pass offsets to ASR
            start = end - self.chunk_overlap
            if start < 0:
                start = 0.0

    def _write_conv_header(self, conv_path: Path, audio_file: str) -> None:
        header = f"CONVERSATION: {audio_file}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + ("=" * 60) + "\n\n"
        with open(conv_path, 'w', encoding='utf-8') as f:
            f.write(header)

    def _append_conv_lines(self, conv_path: Path, lines: List[str]) -> None:
        if not lines:
            return
        with open(conv_path, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")

    def _format_ts(self, seconds: float) -> str:
        s = max(0, int(round(seconds)))
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"

    def _transcribe_chunk(self, wav_path: str, chunk_start: float, chunk_end: float) -> List[Dict]:
        """Transcribe a time slice quickly (ASR only, for streaming preview)."""
        try:
            import soundfile as sf
            waveform, sr = sf.read(wav_path)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            s_idx = int(chunk_start * sr)
            e_idx = int(chunk_end * sr)
            segment = waveform[s_idx:e_idx]
            temp_chunk = self.output_folder / "_tmp_chunk.wav"
            sf.write(str(temp_chunk), segment, sr)
            segs = self.asr.transcribe(str(temp_chunk))
            try:
                os.remove(temp_chunk)
            except Exception:
                pass
            # shift timestamps by chunk_start
            for s in segs:
                s["start"] = float(chunk_start) + float(s.get("start", 0.0))
                s["end"] = float(chunk_start) + float(s.get("end", 0.0))
            return segs
        except Exception as e:
            logger.error("Streaming chunk failed: %s", e)
            return []

    def process_file_streaming(self, audio_path: str) -> Dict:
        preprocessed = self._preprocess(audio_path)
        base = os.path.basename(audio_path)
        conv_path = self.output_folder / f"{Path(audio_path).stem}_conversation.txt"
        self._write_conv_header(conv_path, base)

        all_asr: List[Dict] = []
        for c_start, c_end, _ in self._iterate_chunks(preprocessed):
            segs = self._transcribe_chunk(preprocessed, c_start, c_end)
            if segs:
                all_asr.extend(segs)
                # Append preview lines (ASR-only, labeled User1)
                lines: List[str] = []
                for s in segs:
                    text = (s.get("text") or "").strip()
                    if not text:
                        continue
                    ts = self._format_ts(float(s.get("start", 0.0)))
                    lines.append(f"[{ts}] User1: {text}")
                self._append_conv_lines(conv_path, lines)

        # After preview, run full pipeline to generate final aligned outputs
        asr_segments = all_asr if all_asr else self.asr.transcribe(preprocessed)
        diar_segments = self.diar.diarize(preprocessed)
        if self.enroll.enrollment_profiles:
            diar_segments = self.enroll.map_clusters(diar_segments)
        aligned = self.align(asr_segments, diar_segments)
        outputs = self.output.write_outputs(
            aligned, os.path.basename(audio_path),
            str(self.enrollment_dir) if self.enrollment_dir else None,
            self.enroll.speaker_name_map,
        )
        try:
            os.remove(preprocessed)
        except Exception:
            pass
        return {
            "status": "success",
            "audio_file": audio_path,
            "outputs": outputs,
            "segments": aligned,
        }

    def process_file(self, audio_path: str) -> Dict:
        """Process a single audio file with GPU optimization."""
        if self.stream:
            return self.process_file_streaming(audio_path)
        try:
            preprocessed = self._preprocess(audio_path)
            
            # Run ASR and diarization in parallel if GPU is available
            if self.device == "cuda":
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    asr_future = executor.submit(self.asr.transcribe, preprocessed)
                    diar_future = executor.submit(self.diar.diarize, preprocessed)
                    
                    asr_segments = asr_future.result()
                    diar_segments = diar_future.result()
            else:
                asr_segments = self.asr.transcribe(preprocessed)
                diar_segments = self.diar.diarize(preprocessed)
            
            if self.enroll.enrollment_profiles:
                diar_segments = self.enroll.map_clusters(diar_segments)
                
            aligned = self.align(asr_segments, diar_segments)
            # Fallback: if alignment empty but ASR has text, create one full-span segment
            if (not aligned or all((seg.get("text", "").strip() == "[No transcription]" or not seg.get("text", "").strip()) for seg in aligned)) and asr_segments:
                try:
                    first_start = float(min(s.get("start", 0.0) for s in asr_segments))
                    last_end = float(max(s.get("end", 0.0) for s in asr_segments))
                except Exception:
                    first_start, last_end = 0.0, 0.0
                concat_text = " ".join([(s.get("text") or "").strip() for s in asr_segments if (s.get("text") or "").strip()])
                concat_text = " ".join(concat_text.split())
                aligned = [{
                    "speaker_id": "Speaker_Unknown",
                    "clustered_speaker": "User1",
                    "start_time": first_start,
                    "end_time": last_end if last_end > first_start else first_start,
                    "duration": (last_end - first_start) if last_end > first_start else 0.0,
                    "text": concat_text if concat_text else "[No transcription]",
                }]
            outputs = self.output.write_outputs(
                aligned, os.path.basename(audio_path),
                str(self.enrollment_dir) if self.enrollment_dir else None,
                self.enroll.speaker_name_map,
            )
            
            try:
                os.remove(preprocessed)
            except Exception:
                pass
                
            return {
                "status": "success",
                "audio_file": audio_path,
                "outputs": outputs,
                "segments": aligned,
            }
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return {
                "status": "error",
                "audio_file": audio_path,
                "error": str(e)
            }

    def process_directory(self) -> Dict:
        """Process all audio files in directory with GPU optimization."""
        files = [p for p in self.src_folder.iterdir() if p.is_file() and p.suffix.lower() in self.audio_extensions]
        if not files:
            return {"status": "no_files", "message": "No audio files found"}
            
        results: Dict[str, Dict] = {}
        ok = 0
        fail = 0
        
        if self.device == "cuda" and len(files) > 1 and not self.stream:
            max_workers = min(self.optimal_batch_size, len(files))
            logger.info(f"Processing {len(files)} files with {max_workers} parallel workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.process_file, str(f)): f for f in files}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        res = future.result()
                        results[file.name] = res
                        ok += 1 if res.get("status") == "success" else 0
                        fail += 0 if res.get("status") == "success" else 1
                    except Exception as e:
                        results[file.name] = {"status": "error", "error": str(e)}
                        fail += 1
        else:
            for f in files:
                try:
                    res = self.process_file(str(f))
                    results[f.name] = res
                    ok += 1 if res.get("status") == "success" else 0
                    fail += 0 if res.get("status") == "success" else 1
                except Exception as e:
                    results[f.name] = {"status": "error", "error": str(e)}
                    fail += 1
                    
        return {
            "status": "completed",
            "total_files": len(files),
            "successful": ok,
            "failed": fail,
            "results": results,
        }


