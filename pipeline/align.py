"""Інструменти вирівнювання синтезованих сегментів з вихідним таймлайном."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .utils import Segment

logger = logging.getLogger(__name__)


@dataclass
class AlignedChunk:
    segment: Segment
    audio: np.ndarray
    sample_rate: int
    target_start: float
    target_end: float
    stretch_factor: float

    @property
    def target_duration(self) -> float:
        return max(0.0, self.target_end - self.target_start)

    @property
    def audio_duration(self) -> float:
        return self.audio.shape[0] / float(self.sample_rate)


class AlignmentPlanner:
    """Просте вирівнювання через time-stretch."""

    def __init__(
        self,
        transition_padding: float = 0.1,
        stretch_tolerance: float = 0.08,
    ) -> None:
        self.transition_padding = transition_padding
        self.stretch_tolerance = stretch_tolerance

    def plan(
        self,
        segments: Sequence[Segment],
        audios: Sequence[Tuple[np.ndarray, int]],
    ) -> List[AlignedChunk]:
        if len(segments) != len(audios):
            raise ValueError("Кількість сегментів та аудіо не співпадає")

        aligned: List[AlignedChunk] = []
        prev_end = 0.0
        for segment, (audio, sample_rate) in zip(segments, audios):
            start = max(segment.start - self.transition_padding, prev_end)
            end = max(segment.end + self.transition_padding, start + 0.05)
            duration = max(0.05, end - start)
            audio_duration = audio.shape[0] / float(sample_rate)
            if audio_duration <= 0:
                raise ValueError(f"Порожнє аудіо для сегмента {segment.id}")

            stretch = duration / audio_duration
            if abs(1.0 - stretch) <= self.stretch_tolerance:
                stretch = 1.0

            aligned.append(
                AlignedChunk(
                    segment=segment,
                    audio=audio,
                    sample_rate=sample_rate,
                    target_start=start,
                    target_end=end,
                    stretch_factor=stretch,
                )
            )
            prev_end = end

        logger.debug("Підготовлено %d вирівняних сегментів", len(aligned))
        return aligned


__all__ = ["AlignedChunk", "AlignmentPlanner"]
