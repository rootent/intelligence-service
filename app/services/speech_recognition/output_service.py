"""
Output service
- Generates JSON and conversation TXT
"""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path
import json
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class OutputService:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _format_time(self, seconds: float) -> str:
        s = max(0, int(round(seconds)))
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"

    def _join_turn_texts(self, a: str, b: str) -> str:
        a = (a or '').strip()
        b = (b or '').strip()
        if not a:
            return b
        if not b:
            return a
        if a.endswith(('.', '!', '?')):
            return f"{a} {b}"
        return f"{a}. {b}"

    def _generate_conversation(self, segments: List[Dict], audio_file: str) -> str:
        lines: List[str] = []
        header = f"CONVERSATION: {audio_file}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + ("=" * 60) + "\n\n"
        lines.append(header)
        if not segments:
            return "".join(lines)
        segments_sorted = sorted(segments, key=lambda s: (float(s.get("start_time", 0.0)), float(s.get("end_time", 0.0))))
        collapsed: List[Dict] = []
        for seg in segments_sorted:
            if not collapsed:
                collapsed.append(seg.copy())
                continue
            last = collapsed[-1]
            last_speaker = last.get("clustered_speaker", last.get("speaker_id", "Speaker"))
            cur_speaker = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker"))
            if cur_speaker == last_speaker and float(seg.get("start_time", 0.0)) <= float(last.get("end_time", 0.0)) + 0.25:
                last["end_time"] = float(seg.get("end_time", last.get("end_time", seg.get("start_time", 0.0))))
                last["duration"] = float(last["end_time"] - float(last.get("start_time", 0.0)))
                last["text"] = self._join_turn_texts(last.get("text", ""), seg.get("text", ""))
            else:
                collapsed.append(seg.copy())
        for seg in collapsed:
            speaker = seg.get("clustered_speaker", seg.get("speaker_id", "Speaker"))
            text = (seg.get("text") or "").strip()
            ts = self._format_time(float(seg.get("start_time", 0.0)))
            if text:
                lines.append(f"[{ts}] {speaker}: {text}\n")
        return "".join(lines)

    def write_outputs(self, aligned_segments: List[Dict], audio_file: str, enrollment_dir: str, speaker_name_map: Dict[str, str]) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        base = Path(audio_file).stem
        # JSON (use previous naming for continuity)
        json_data = {
            "audio_file": audio_file,
            "processing_timestamp": datetime.now().isoformat(),
            "segments": aligned_segments,
            "speaker_name_map": speaker_name_map,
            "summary": {
                "total_segments": len(aligned_segments),
                "total_speakers": len(set(seg.get("clustered_speaker", seg["speaker_id"]) for seg in aligned_segments)),
                "total_duration": sum(seg["duration"] for seg in aligned_segments)
            }
        }
        json_path = self.output_dir / f"{base}_streamlined_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        outputs["json"] = str(json_path)
        # Conversation
        conv_text = self._generate_conversation(aligned_segments, audio_file)
        conv_path = self.output_dir / f"{base}_conversation.txt"
        with open(conv_path, 'w', encoding='utf-8') as f:
            f.write(conv_text)
        outputs["conversation_txt"] = str(conv_path)
        logger.info("Output files written for %s", audio_file)
        return outputs


