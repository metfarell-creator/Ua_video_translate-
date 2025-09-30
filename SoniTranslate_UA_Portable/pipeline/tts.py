from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import snapshot_download
import numpy as np

# Підтримка двох варіантів API (styletts2 vs styletts2_inference)
try:
    from styletts2 import tts as _tts_mod
    API = "styletts2"
except Exception:
    _tts_mod = None
    API = None

try:
    from styletts2_inference.models import StyleTTS2 as _StyleTTS2
    API2 = "styletts2_inference"
except Exception:
    _StyleTTS2 = None
    API2 = None

def _load_ua_repo(repo: str) -> Tuple[Path, Path]:
    local = snapshot_download(repo_id=repo, local_dir=None, local_dir_use_symlinks=False)
    p = Path(local)
    # Пошук чекпойнта та конфігу
    ckpt = next((x for x in p.rglob("*.pth")), None)
    cfg  = next((x for x in p.rglob("*config*.yml")), None) or next((x for x in p.rglob("*config*.yaml")), None)
    if not ckpt or not cfg:
        raise RuntimeError("Не знайдено чекпойнт .pth або config.yaml у моделі " + repo)
    return ckpt, cfg

def synthesize(text: str, repo: str, sample_rate: int = 24000, speed: float = 1.0) -> np.ndarray:
    ckpt, cfg = _load_ua_repo(repo)

    if API == "styletts2":
        model = _tts_mod.StyleTTS2(model_checkpoint_path=str(ckpt), config_path=str(cfg))
        wav = model.inference(text, output_sample_rate=sample_rate, speed=speed)
        return wav.astype(np.float32)
    elif API2 == "styletts2_inference":
        model = _StyleTTS2(str(ckpt.parent), device="cuda")
        wav = model.tts(text, speed=speed)
        return wav.astype(np.float32)
    else:
        raise RuntimeError("Не знайдено жодного підтримуваного API StyleTTS2")
