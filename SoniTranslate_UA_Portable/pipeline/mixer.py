import subprocess, os, io, datetime as dt
from pathlib import Path
from typing import List, Dict, Any
from pydub import AudioSegment
import numpy as np

def wav_from_array(wav: np.ndarray, sr: int) -> AudioSegment:
    # float32 [-1,1] -> int16 bytes for pydub
    arr = (wav.clip(-1,1) * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=arr, sample_width=2, frame_rate=sr, channels=1
    )

def render_timeline(entries, synthesized: List[AudioSegment], sr: int) -> AudioSegment:
    # entries: SRT entries (same length as synthesized)
    timeline = AudioSegment.silent(duration=0, frame_rate=sr)
    cursor = 0
    for e, seg in zip(entries, synthesized):
        start_ms = int(e.start.total_seconds()*1000)
        if cursor < start_ms:
            timeline += AudioSegment.silent(duration=(start_ms - cursor), frame_rate=sr)
            cursor = start_ms
        timeline += seg
        cursor += len(seg)
    return timeline

def mux_audio_video(audio_path: str, video_path: str, out_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        out_path
    ]
    subprocess.run(cmd, check=True)
