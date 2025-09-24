#!/usr/bin/env python
"""CLI для генерації дубльованої доріжки з SRT."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from pipeline.align import AlignmentPlanner
from pipeline.tts import StyleTTS2Synthesizer
from pipeline.utils import (
    Segment,
    load_env,
    read_yaml,
    setup_logging,
    timestamp_to_seconds,
)
from pipeline.mixer import AudioMixer

logger = logging.getLogger(__name__)


def parse_srt(path: Path) -> List[Segment]:
    raw = path.read_text(encoding="utf-8")
    blocks = []
    current: List[str] = []
    for line in raw.splitlines():
        if line.strip():
            current.append(line.rstrip())
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)

    segments: List[Segment] = []
    for block in blocks:
        if len(block) < 2:
            continue
        idx_line = block[0].strip()
        time_line = block[1].strip()
        text_lines = block[2:]
        try:
            start_raw, end_raw = [part.strip() for part in time_line.split("-->")]
            start = timestamp_to_seconds(start_raw)
            end = timestamp_to_seconds(end_raw)
        except Exception as exc:
            raise ValueError(f"Невірний формат таймінгу у блоці: {block}") from exc
        text = " ".join(text_lines).strip()
        if not text:
            continue
        segments.append(
            Segment(
                id=len(segments),
                start=start,
                end=end,
                text=text,
            )
        )
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("srt", type=Path, help="Шлях до SRT субтитрів")
    parser.add_argument("output", type=Path, help="WAV файл для озвучки")
    parser.add_argument("--preset", type=Path, default=Path("config/presets/uk_to_uk.yaml"))
    parser.add_argument("--original", type=Path, help="Оригінальне відео або аудіо для міксу")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)
    load_env()

    preset = read_yaml(args.preset)
    pipeline_cfg = preset.get("pipeline", {})
    tts_cfg = pipeline_cfg.get("tts", {})
    align_cfg = pipeline_cfg.get("aligner", {})
    mixer_cfg = pipeline_cfg.get("mixer", {})

    segments = parse_srt(args.srt)
    logger.info("Завантажено %d сегментів із %s", len(segments), args.srt)

    tts = StyleTTS2Synthesizer(
        repo_id=tts_cfg.get("repo_id"),
        speaker=tts_cfg.get("speaker", "neutral_female"),
        length_scale=float(tts_cfg.get("length_scale", 1.0)),
        noise_scale=float(tts_cfg.get("noise_scale", 0.667)),
        noise_scale_w=float(tts_cfg.get("noise_scale_w", 0.8)),
    )

    rendered = []
    for segment in segments:
        logger.debug("Синтез сегмента %s: %s", segment.id, segment.text)
        audio, sr = tts.synthesize(segment.text)
        rendered.append((audio, sr))

    planner = AlignmentPlanner(
        transition_padding=float(align_cfg.get("transition_padding", 0.1)),
        stretch_tolerance=float(align_cfg.get("stretch_factor", 0.08)),
    )
    aligned = planner.plan(segments, rendered)

    mixer = AudioMixer(
        sample_rate=int(mixer_cfg.get("sample_rate", 22050)),
        ducking_db=float(mixer_cfg.get("ducking_db", -6.0)),
        music_gain_db=float(mixer_cfg.get("music_gain_db", -2.0)),
        voice_gain_db=float(mixer_cfg.get("voice_gain_db", 0.0)),
    )
    mix = mixer.mix(aligned, original_audio=args.original)
    mixer.save(mix, args.output)

    logger.info("Готово. Озвучка: %s", args.output)


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
