import os, io, math, srt, datetime as dt
from pathlib import Path
from typing import List, Dict, Any
from pydub import AudioSegment

def to_ms(td: dt.timedelta) -> int:
    return int(td.total_seconds() * 1000)

def srt_to_entries(srt_text: str):
    return list(srt.parse(srt_text))

def entries_to_srt(entries) -> str:
    return srt.compose(entries)

def concat_audio(segments: List[AudioSegment]) -> AudioSegment:
    out = AudioSegment.silent(duration=0)
    for seg in segments:
        out += seg
    return out

def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
