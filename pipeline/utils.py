"""Загальні утиліти для портативного конвеєра дубляжу."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("sonitranslate")


@dataclasses.dataclass
class Segment:
    """Опис сегмента мовлення."""

    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[Dict[str, float]]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_env(env_path: Optional[Path] = None) -> None:
    """Завантажує `.env` та встановлює стандартний шлях до кешу HF."""

    env_file = env_path or Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        logger.debug("Loaded environment variables from %s", env_file)

    hf_home = os.getenv("HF_HOME")
    if not hf_home:
        default_hf_home = Path.cwd() / ".cache" / "huggingface"
        os.environ["HF_HOME"] = str(default_hf_home)
        logger.debug("HF_HOME set to %s", default_hf_home)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    logger.debug("Loaded YAML config from %s", path)
    return data


def seconds_to_timestamp(value: float) -> str:
    total_ms = int(value * 1000)
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


_TIMESTAMP_PATTERN = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)


def timestamp_to_seconds(value: str) -> float:
    match = _TIMESTAMP_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Невідомий формат тайм-коду: {value}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def slugify(text: str, max_length: int = 80) -> str:
    value = re.sub(r"[^\w\-]+", "-", text.strip().lower())
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:max_length]


@contextlib.contextmanager
def temporary_audio_file(suffix: str = ".wav") -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"temp{suffix}"
        yield tmp_path


def to_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.debug("Saved JSON to %s", path)


def detect_device(prefer_cuda: bool = True) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iter_chunks(iterable: Iterable[Segment], chunk_size: int) -> Iterator[List[Segment]]:
    chunk: List[Segment] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = [
    "Segment",
    "ensure_directory",
    "iter_chunks",
    "load_env",
    "read_yaml",
    "seconds_to_timestamp",
    "setup_logging",
    "slugify",
    "timestamp_to_seconds",
    "temporary_audio_file",
    "to_json",
    "detect_device",
]
