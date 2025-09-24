"""Збирання дубльованого звуку та мікс із оригінальною доріжкою."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from pydub import AudioSegment

from .align import AlignedChunk

logger = logging.getLogger(__name__)


def _db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def _resample(array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return array
    duration = array.shape[0] / float(original_sr)
    target_length = max(1, int(duration * target_sr))
    indices = np.linspace(0, array.shape[0] - 1, target_length)
    resampled = np.interp(indices, np.arange(array.shape[0]), array)
    return resampled.astype(np.float32)


def _stretch(array: np.ndarray, factor: float) -> np.ndarray:
    if abs(1.0 - factor) < 1e-3:
        return array
    target_length = max(1, int(array.shape[0] * factor))
    indices = np.linspace(0, array.shape[0] - 1, target_length)
    stretched = np.interp(indices, np.arange(array.shape[0]), array)
    return stretched.astype(np.float32)


def _ensure_mono(segment: AudioSegment, target_sr: int) -> AudioSegment:
    seg = segment.set_channels(1)
    if seg.frame_rate != target_sr:
        seg = seg.set_frame_rate(target_sr)
    return seg


def audiosegment_to_array(segment: AudioSegment) -> np.ndarray:
    samples = np.array(segment.get_array_of_samples())
    sample_width = segment.sample_width
    max_val = float(1 << (8 * sample_width - 1))
    return (samples.astype(np.float32) / max_val).copy()


def array_to_audiosegment(array: np.ndarray, sample_rate: int) -> AudioSegment:
    clipped = np.clip(array, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype("<i2")
    return AudioSegment(
        data=int16.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


class AudioMixer:
    def __init__(
        self,
        sample_rate: int = 22050,
        ducking_db: float = -6.0,
        music_gain_db: float = -2.0,
        voice_gain_db: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.ducking_gain = _db_to_gain(ducking_db)
        self.music_gain = _db_to_gain(music_gain_db)
        self.voice_gain = _db_to_gain(voice_gain_db)

    def mix(
        self,
        aligned: Iterable[AlignedChunk],
        original_audio: Optional[Path] = None,
    ) -> np.ndarray:
        segments: List[AlignedChunk] = list(aligned)
        if not segments:
            raise ValueError("Список сегментів порожній")

        total_duration = max(chunk.target_end for chunk in segments)
        total_samples = int(total_duration * self.sample_rate) + self.sample_rate
        voice_track = np.zeros(total_samples, dtype=np.float32)
        ducking_mask = np.ones(total_samples, dtype=np.float32)

        for chunk in segments:
            audio = _resample(chunk.audio, chunk.sample_rate, self.sample_rate)
            audio = _stretch(audio, chunk.stretch_factor)
            start = int(chunk.target_start * self.sample_rate)
            end = start + audio.shape[0]
            if end > voice_track.shape[0]:
                extra = end - voice_track.shape[0]
                voice_track = np.pad(voice_track, (0, extra))
                ducking_mask = np.pad(ducking_mask, (0, extra), constant_values=1.0)
            voice_track[start:end] += audio
            duck_factor = min(self.ducking_gain, 1.0)
            ducking_mask[start:end] = np.minimum(ducking_mask[start:end], duck_factor)

        voice_track *= self.voice_gain

        if original_audio:
            original_path = Path(original_audio)
            logger.info("Завантаження оригінального аудіо %s", original_path)
            seg = AudioSegment.from_file(original_path)
            seg = _ensure_mono(seg, self.sample_rate)
            original = audiosegment_to_array(seg) * self.music_gain
            if original.shape[0] < voice_track.shape[0]:
                original = np.pad(original, (0, voice_track.shape[0] - original.shape[0]))
            else:
                voice_track = np.pad(voice_track, (0, original.shape[0] - voice_track.shape[0]))
                ducking_mask = np.pad(
                    ducking_mask, (0, original.shape[0] - ducking_mask.shape[0]), constant_values=1.0
                )
            original *= ducking_mask
            mixed = original + voice_track
        else:
            mixed = voice_track

        mixed = np.clip(mixed, -1.0, 1.0)
        return mixed.astype(np.float32)

    def save(self, audio: np.ndarray, path: Path) -> None:
        segment = array_to_audiosegment(audio, self.sample_rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        segment.export(path, format="wav")
        logger.info("Готовий мікс: %s", path)


__all__ = ["AudioMixer", "array_to_audiosegment", "audiosegment_to_array"]
