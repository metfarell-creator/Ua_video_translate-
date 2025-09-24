#!/usr/bin/env python
"""Завантаження моделей WhisperX та StyleTTS2 у локальний кеш."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.tts import StyleTTS2Synthesizer
from pipeline.utils import detect_device, load_env, read_yaml, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=Path("config/presets/uk_to_uk.yaml"))
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def prefetch_asr(preset: dict) -> None:
    if preset.get("pipeline") is None:
        return
    asr_cfg = preset["pipeline"].get("asr", {})
    language = asr_cfg.get("language", preset["pipeline"].get("language", "uk"))
    model_size = asr_cfg.get("model_size", "large-v3")
    compute_type = asr_cfg.get("compute_type", "float16")
    device = detect_device()

    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover - залежить від оточення
        raise RuntimeError("whisperx не встановлено. Запустіть pip install whisperx") from exc

    logger.info("Завантаження WhisperX (%s) на %s", model_size, device)
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
    del model
    logger.info("Align модель")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    del align_model
    del metadata

    if asr_cfg.get("diarize"):
        logger.info("Підготовка діаризатора")
        diarizer = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
        del diarizer


def prefetch_tts(preset: dict) -> None:
    tts_cfg = preset.get("pipeline", {}).get("tts", {})
    if not tts_cfg.get("repo_id"):
        logger.warning("У пресеті не вказано repo_id для TTS")
        return
    tts = StyleTTS2Synthesizer(
        repo_id=tts_cfg["repo_id"],
        speaker=tts_cfg.get("speaker", "neutral_female"),
    )
    _ = tts.sample_rate
    logger.info("TTS модель %s завантажено", tts.repo_id)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    load_env()

    preset = read_yaml(args.preset)
    if not args.skip_asr:
        prefetch_asr(preset)
    if not args.skip_tts:
        prefetch_tts(preset)


if __name__ == "__main__":
    main()
