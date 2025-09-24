#!/usr/bin/env python
"""Gradio UI для конвеєра дубляжу."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg
import gradio as gr

from pipeline.align import AlignmentPlanner
from pipeline.asr import WhisperXASR
from pipeline.mixer import AudioMixer
from pipeline.tts import StyleTTS2Synthesizer
from pipeline.utils import (
    Segment,
    ensure_directory,
    load_env,
    read_yaml,
    seconds_to_timestamp,
    setup_logging,
    slugify,
    to_json,
)

load_env()
setup_logging()


class DubbingPipeline:
    def __init__(self, preset_path: Path) -> None:
        self.preset_path = preset_path
        config = read_yaml(preset_path)
        pipeline_cfg = config.get("pipeline", {})
        self.config = pipeline_cfg

        hf_token = os.getenv("HF_TOKEN")
        language = pipeline_cfg.get("language", "uk")
        asr_cfg = pipeline_cfg.get("asr", {})
        self.asr_cfg = asr_cfg
        tts_cfg = pipeline_cfg.get("tts", {})
        align_cfg = pipeline_cfg.get("aligner", {})
        mixer_cfg = pipeline_cfg.get("mixer", {})

        self.asr = WhisperXASR(
            language=language,
            model_size=asr_cfg.get("model_size", "large-v3"),
            compute_type=asr_cfg.get("compute_type", "float16"),
            hf_token=hf_token,
        )
        self.tts = StyleTTS2Synthesizer(
            repo_id=tts_cfg.get("repo_id"),
            speaker=tts_cfg.get("speaker", "neutral_female"),
            length_scale=float(tts_cfg.get("length_scale", 1.0)),
            noise_scale=float(tts_cfg.get("noise_scale", 0.667)),
            noise_scale_w=float(tts_cfg.get("noise_scale_w", 0.8)),
            hf_token=hf_token,
        )
        self.planner = AlignmentPlanner(
            transition_padding=float(align_cfg.get("transition_padding", 0.1)),
            stretch_tolerance=float(align_cfg.get("stretch_factor", 0.08)),
        )
        self.mixer = AudioMixer(
            sample_rate=int(mixer_cfg.get("sample_rate", 22050)),
            ducking_db=float(mixer_cfg.get("ducking_db", -6.0)),
            music_gain_db=float(mixer_cfg.get("music_gain_db", -2.0)),
            voice_gain_db=float(mixer_cfg.get("voice_gain_db", 0.0)),
        )

    def _extract_audio(self, media_path: Path, tmpdir: Path) -> Path:
        audio_path = tmpdir / "input.wav"
        (
            ffmpeg.input(str(media_path))
            .output(str(audio_path), ac=1, ar=self.mixer.sample_rate)
            .overwrite_output()
            .run(quiet=True)
        )
        return audio_path

    def _segments_to_srt(self, segments: list[Segment], path: Path) -> None:
        lines = []
        for idx, seg in enumerate(segments, start=1):
            lines.append(str(idx))
            lines.append(
                f"{seconds_to_timestamp(seg.start)} --> {seconds_to_timestamp(seg.end)}"
            )
            lines.append(seg.text)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def run(self, media: Path, progress: Optional[gr.Progress] = None) -> dict:
        media_path = Path(media)
        name = slugify(media_path.stem)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            audio_path = self._extract_audio(media_path, tmpdir)

            if progress:
                progress(0.15, desc="Транскрипція")
            segments = self.asr.transcribe(
                audio_path,
                batch_size=int(self.asr_cfg.get("batch_size", 8)),
                diarize=bool(self.asr_cfg.get("diarize", False)),
            )

            if not segments:
                raise RuntimeError("WhisperX не повернув сегменти. Перевірте вхідний файл.")

            if progress:
                progress(0.4, desc="Синтез мовлення")
            rendered = []
            for seg in segments:
                audio, sr = self.tts.synthesize(seg.text, speaker=seg.speaker)
                rendered.append((audio, sr))

            if progress:
                progress(0.6, desc="Вирівнювання")
            aligned = self.planner.plan(segments, rendered)

            if progress:
                progress(0.75, desc="Мікс")
            mix = self.mixer.mix(aligned, original_audio=audio_path)
            voice_path = tmpdir / f"{name}_voice.wav"
            self.mixer.save(mix, voice_path)

            if progress:
                progress(0.9, desc="Кодування відео")
            output_dir = ensure_directory(Path("outputs"))
            dubbed_audio = output_dir / f"{name}_uk.wav"
            voice_path.replace(dubbed_audio)

            transcript_path = output_dir / f"{name}.srt"
            self._segments_to_srt(segments, transcript_path)
            to_json([seg.__dict__ for seg in segments], output_dir / f"{name}.json")

            if media_path.suffix.lower() in {".wav", ".mp3", ".flac"}:
                dubbed_video = None
            else:
                dubbed_video = output_dir / f"{name}_uk.mp4"
                (
                    ffmpeg.output(
                        ffmpeg.input(str(media_path)).video,
                        ffmpeg.input(str(dubbed_audio)).audio,
                        str(dubbed_video),
                        vcodec="copy",
                        acodec="aac",
                        audio_bitrate="256k",
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )

        return {
            "audio": str(dubbed_audio),
            "video": str(dubbed_video) if dubbed_video else None,
            "transcript": str(transcript_path),
        }


def build_interface() -> gr.Blocks:
    pipeline = DubbingPipeline(Path("config/presets/uk_to_uk.yaml"))

    with gr.Blocks(title="SoniTranslate UA") as demo:
        gr.Markdown("## 🇺🇦 Дубляж українською — WhisperX + StyleTTS2")
        with gr.Row():
            media_input = gr.File(label="Відео або аудіо")
            with gr.Column():
                audio_output = gr.File(label="Озвучення (WAV)")
                video_output = gr.File(label="Відео з дубляжем", interactive=False)
                transcript_output = gr.File(label="Транскрипція (SRT)")
        run_button = gr.Button("Старт")

        def process(file: gr.FileData, progress=gr.Progress(track_tqdm=True)):
            result = pipeline.run(Path(file.name), progress)
            return result["audio"], result["video"], result["transcript"]

        run_button.click(process, inputs=media_input, outputs=[audio_output, video_output, transcript_output])

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
