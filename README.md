# SoniTranslate_UA_Portable (WhisperX + StyleTTS2 Ukrainian)

Портативний конвеєр для транскрипції та дубляжу **українською**:
- **ASR:** WhisperX (оновлені версії, підтримка CUDA 12.1 / cuDNN 9 через CTranslate2 4.5.0)
- **TTS:** StyleTTS2 (українські чекпойнти з Hugging Face)
- **UI:** Gradio 5.x
- **FFmpeg:** portable (Windows-варіант у комплекті)
- **Preset:** `uk→uk дубляж` (без перекладу)

> Моделі не включені в архів — інсталятори все докачають у локальний кеш.

## Швидкий старт

### Windows
```bat
install_portable.bat
run_ui.bat
```

### Linux/macOS
```bash
bash install_portable.sh
bash run_ui.sh
```

Після запуску відкрий Gradio‑посилання в консолі.

---

## Структура

```
SoniTranslate_UA_Portable/
  .cache/huggingface/              # локальний кеш моделей (портативно)
  config/
    presets/uk_to_uk.yaml          # пресет дубляжу uk→uk
  pipeline/
    asr.py                         # WhisperX-обгортка
    tts.py                         # StyleTTS2 (українські чекпойнти)
    align.py                       # підгін озвучки під таймінги
    mixer.py                       # збирання таймлайна та склейка з відео
    utils.py                       # допоміжні функції
  scripts/
    dub_from_srt.py                # CLI: SRT → WAV з таймінгами
    export_video.py                # склеювання WAV з оригінальним відео (FFmpeg)
  tools/
    prefetch_models.py             # докачування моделей (Whisper + StyleTTS2 UA)
    ffmpeg_setup.bat               # portable FFmpeg для Windows
  ui/
    gradio_app.py                  # Gradio‑інтерфейс
  install_portable.bat             # інсталяція (Windows)
  install_portable.sh              # інсталяція (Linux/macOS)
  run_ui.bat                       # запуск UI (Windows)
  run_ui.sh                        # запуск UI (Linux/macOS)
  requirements.txt                 # зафіксовані версії залежностей
  .env.example                     # опційні токени/налаштування
  LICENSE (MIT)
```

## Параметри середовища (`.env`)

- `HF_TOKEN` — якщо потрібна діаризація (pyannote) або приватні моделі HF.
- `HF_HOME` — шлях до кешу HF (за замовчуванням `.cache/huggingface`).
- `PYTORCH_CUDA_ALLOC_CONF` — напр. `max_split_size_mb:512`.

## Нотатки щодо сумісності

- **Torch 2.5.1 + CUDA 12.1** (або 11.8) + **CTranslate2 4.5.0** ⇒ сумісність з cuDNN 9 і свіжими драйверами.
- Якщо GPU/драйвер не підходить — інсталятор автоматично перейде на CPU‑колеса Torch.
- WhisperX ставимо **з PyPI** (стабільний шлях).
- StyleTTS2 беремо як пакет **`styletts2`** + UA чекпойнти з HuggingFace (автозавантаження в кеш).

## Ліцензія

MIT (див. файл `LICENSE`). Дотримуйтеся ліцензій моделей/чекпойнтів на HuggingFace.
