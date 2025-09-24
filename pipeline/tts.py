"""Генерація синтезованої мови за допомогою StyleTTS2 / Hugging Face."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .utils import detect_device

logger = logging.getLogger(__name__)


class StyleTTS2Synthesizer:
    """Обгортка навколо styletts2 або HF-пайплайну text-to-speech."""

    def __init__(
        self,
        repo_id: str,
        speaker: str = "neutral_female",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.repo_id = repo_id
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_scale_w = noise_scale_w
        self.device = device or detect_device()
        self.hf_token = hf_token

        self._pipeline = None
        self._styletts2 = None
        self._sample_rate = 22050

    # -- internal -----------------------------------------------------------------
    def _load_pipeline(self) -> None:
        if self._pipeline is not None or self._styletts2 is not None:
            return

        try:
            from styletts2.api import StyleTTS2
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=self.repo_id,
                use_auth_token=self.hf_token,
                allow_patterns=["*.pt", "*.yaml", "*.json", "*.pth"],
            )
            logger.info("Завантажено чекпоінт StyleTTS2 у %s", cache_dir)
            self._styletts2 = StyleTTS2.from_pretrained(cache_dir, device=self.device)
            self._sample_rate = self._styletts2.sample_rate
            return
        except ImportError:
            logger.debug("styletts2.api недоступний, пробуємо transformers pipeline")
        except Exception as exc:  # pragma: no cover - поведінка залежить від пакета
            logger.warning("StyleTTS2 native API недоступний: %s", exc)

        try:
            from transformers import pipeline

            device_idx = 0 if self.device.startswith("cuda") else -1
            self._pipeline = pipeline(
                task="text-to-speech",
                model=self.repo_id,
                device=device_idx,
                use_auth_token=self.hf_token,
            )
            meta = getattr(self._pipeline.model, "config", None)
            if meta is not None and hasattr(meta, "sample_rate"):
                self._sample_rate = int(meta.sample_rate)
            logger.info("Пайплайн text-to-speech завантажено (%s)", self.repo_id)
        except ImportError as exc:
            raise RuntimeError(
                "Не знайдено styletts2 або transformers. Встановіть залежності."
            ) from exc

    # -- public -------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        self._load_pipeline()
        return self._sample_rate

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_scale_w: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Повертає масив аудіо та частоту дискретизації."""

        self._load_pipeline()
        speaker = speaker or self.speaker
        length_scale = length_scale or self.length_scale
        noise_scale = noise_scale or self.noise_scale
        noise_scale_w = noise_scale_w or self.noise_scale_w

        if self._styletts2 is not None:
            audio = self._styletts2.tts(
                text=text,
                speaker=speaker,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
            if isinstance(audio, tuple):
                audio = audio[0]
            audio = np.asarray(audio, dtype=np.float32)
            return audio, self._sample_rate

        if self._pipeline is None:
            raise RuntimeError("Не вдалося ініціалізувати жодну з TTS моделей")

        params = {
            "speaker": speaker,
            "length_scale": float(length_scale),
            "noise_scale": float(noise_scale),
            "noise_scale_w": float(noise_scale_w),
        }

        try:
            outputs = self._pipeline(text, forward_params=params)
        except TypeError:
            # Деякі пайплайни підтримують лише voice_preset
            outputs = self._pipeline(text, voice_preset=speaker)
        audio = outputs["audio"]
        sr = int(outputs.get("sampling_rate", self._sample_rate))
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        return audio.astype(np.float32), sr

    def save_wav(self, array: np.ndarray, sample_rate: int, path: Path) -> None:
        import wave

        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(array, -1.0, 1.0)
        int16 = (clipped * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(int16.tobytes())
        logger.debug("Збережено синтезоване аудіо у %s", path)


__all__ = ["StyleTTS2Synthesizer"]
