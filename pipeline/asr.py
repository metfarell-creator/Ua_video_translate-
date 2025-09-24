"""WhisperX обгортка для отримання розбивки відео на сегменти."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .utils import Segment, detect_device

logger = logging.getLogger(__name__)


class WhisperXASR:
    """Високорівнева обгортка WhisperX."""

    def __init__(
        self,
        language: str = "uk",
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: str = "float16",
        hf_token: Optional[str] = None,
    ) -> None:
        self.language = language
        self.model_size = model_size
        self.device = device or detect_device()
        self.compute_type = compute_type
        self.hf_token = hf_token

        try:
            import whisperx
        except ImportError as exc:  # pragma: no cover - залежність важка
            raise RuntimeError(
                "Пакет whisperx не встановлено. Виконайте pip install whisperx"
            ) from exc

        logger.info(
            "Завантаження моделі WhisperX %s на %s (%s)",
            model_size,
            self.device,
            compute_type,
        )
        self._whisperx = whisperx
        self._model = whisperx.load_model(
            model_size,
            self.device,
            compute_type=compute_type,
            language=language,
        )
        self._align_model = None
        self._align_metadata = None

    def _ensure_align_model(self) -> None:
        if self._align_model is not None:
            return
        logger.debug("Завантаження align-моделі для мови %s", self.language)
        self._align_model, self._align_metadata = self._whisperx.load_align_model(
            language_code=self.language,
            device=self.device,
            model_dir=None,
        )

    def transcribe(
        self,
        audio_path: Path,
        batch_size: int = 8,
        diarize: bool = False,
    ) -> List[Segment]:
        """Розпізнає мовлення з вирівнюванням по словам."""

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        logger.info("WhisperX: транскрипція %s", audio_path)
        result = self._model.transcribe(
            str(audio_path),
            batch_size=batch_size,
        )

        segments = result.get("segments", [])
        if not segments:
            return []

        self._ensure_align_model()
        aligned = self._whisperx.align(
            segments,
            self._align_model,
            self._align_metadata,
            str(audio_path),
            self.device,
        )
        segments = aligned.get("segments", segments)

        if diarize:
            logger.info("WhisperX: діаризація активована")
            diarizer = self._whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device,
            )
            diarization_map = diarizer(str(audio_path))
            segments = self._whisperx.assign_word_speakers(
                diarization_map, segments
            )["segments"]

        converted: List[Segment] = []
        for idx, seg in enumerate(segments):
            speaker = seg.get("speaker") if isinstance(seg, dict) else None
            words = seg.get("words") if isinstance(seg, dict) else None
            converted.append(
                Segment(
                    id=idx,
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                    speaker=speaker,
                    words=words,
                )
            )

        logger.debug("Отримано %d сегментів", len(converted))
        return converted

    def dump_segments(self, segments: List[Segment], path: Path) -> None:
        data: List[Dict[str, object]] = [asdict(seg) for seg in segments]
        import json

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["WhisperXASR"]
