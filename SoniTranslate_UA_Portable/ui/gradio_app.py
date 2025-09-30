import os, srt, tempfile, shutil
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from pipeline import asr, tts, align, mixer, utils

load_dotenv()

def step_transcribe(file, model, language, device, compute_type, diarization):
    result = asr.transcribe(file, model_name=model, language=language, device=device, compute_type=compute_type,
                            diarization=diarization, hf_token=os.getenv("HF_TOKEN"))
    # Збираємо SRT
    entries = []
    for i, seg in enumerate(result["segments"]):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        entries.append(srt.Subtitle(index=i+1, start=srt.srt_timestamp_to_timedelta(start), end=srt.srt_timestamp_to_timedelta(end), content=text))
    srt_text = srt.compose(entries)
    return srt_text

def step_synthesize(srt_text, ua_repo, sample_rate):
    entries = utils.srt_to_entries(srt_text)
    out_segs = []
    for e in entries:
        wav = tts.synthesize(e.content, repo=ua_repo, sample_rate=sample_rate, speed=1.0)
        seg = mixer.wav_from_array(wav, sample_rate)
        slot = int((e.end - e.start).total_seconds()*1000)
        seg = align.fit_to_slot(seg, slot)
        out_segs.append(seg)
    timeline = utils.concat_audio(out_segs)
    wav_path = tempfile.mktemp(suffix=".wav")
    timeline.export(wav_path, format="wav")
    return wav_path

def step_mux(wav_path, video_file):
    out = tempfile.mktemp(suffix=".mp4")
    mixer.mux_audio_video(wav_path, video_file, out)
    return out

with gr.Blocks(title="SoniTranslate UA Portable") as demo:
    gr.Markdown("# 🇺🇦 SoniTranslate UA — WhisperX + StyleTTS2")
    with gr.Tab("1) Транскрипція"):
        in_file = gr.File(label="Відео/Аудіо файл")
        model = gr.Dropdown(choices=["tiny","base","small","medium","large-v2","large-v3"], value="large-v3", label="ASR модель")
        language = gr.Dropdown(choices=["uk","en","ru","auto"], value="uk", label="Мова")
        device = gr.Dropdown(choices=["cuda","cpu"], value="cuda", label="Пристрій")
        compute_type = gr.Dropdown(choices=["float16","int8"], value="float16", label="WhisperX compute type")
        diar = gr.Checkbox(value=False, label="Діаризація (потрібен HF_TOKEN)")
        srt_out = gr.Code(label="SRT (редагований текст)", language="srt")
        btn_tr = gr.Button("Транскрибувати")
        btn_tr.click(step_transcribe, inputs=[in_file, model, language, device, compute_type, diar], outputs=srt_out)

    with gr.Tab("2) Озвучення (UA StyleTTS2)"):
        ua_repo = gr.Text(value="patriotyk/styletts2-ukrainian", label="HF Repo для UA StyleTTS2")
        sr = gr.Slider(8000, 48000, value=24000, step=1000, label="Sample Rate")
        wav_path = gr.Audio(type="filepath", label="Синтезований WAV")
        btn_syn = gr.Button("Синтезувати WAV")
        btn_syn.click(step_synthesize, inputs=[srt_out, ua_repo, sr], outputs=wav_path)

    with gr.Tab("3) Збірка відео"):
        in_vid = gr.File(label="Оригінальне відео")
        out_vid = gr.File(label="Дубльоване відео (MP4)")
        btn_mux = gr.Button("Зібрати відео")
        btn_mux.click(step_mux, inputs=[wav_path, in_vid], outputs=out_vid)

if __name__ == "__main__":
    demo.launch()
