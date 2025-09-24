#!/usr/bin/env python
"""CLI для підміни звукової доріжки у відео."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import ffmpeg

from pipeline.utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Оригінальне відео")
    parser.add_argument("audio", type=Path, help="Нове озвучення WAV")
    parser.add_argument("output", type=Path, help="Вихідний файл (mp4/mkv)")
    parser.add_argument("--audio-bitrate", default="256k")
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def mux(args: argparse.Namespace) -> None:
    video = str(args.video)
    audio = str(args.audio)
    output = str(args.output)

    logger.info("FFmpeg: %s + %s -> %s", video, audio, output)
    stream = ffmpeg.output(
        ffmpeg.input(video).video,
        ffmpeg.input(audio).audio,
        output,
        vcodec="copy",
        acodec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        shortest=None,
    )
    if args.overwrite:
        stream = ffmpeg.overwrite_output(stream)
    stream.run()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    mux(args)


if __name__ == "__main__":
    main()
