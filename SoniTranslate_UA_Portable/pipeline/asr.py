from pathlib import Path
from typing import Dict, Any
import torch
import whisperx

def transcribe(audio_path: str, model_name: str = "large-v3", language: str = "uk",
               device: str = "cuda", compute_type: str = "float16",
               diarization: bool = False, hf_token: str | None = None) -> Dict[str, Any]:
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options={"language": language})
    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, language=language)
    # Алінмент (краще таймінги)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result["segments"] = whisperx.align(result["segments"], align_model, metadata, audio, device)
    # Діаризація (опційно)
    if diarization and hf_token:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    return result
