from pydub import AudioSegment

HEAD_PAD_MS = 70
TAIL_PAD_MS = 120
FADE_MS     = 40
MAX_TRIM_OVERFLOW_MS = 220

def fit_to_slot(seg: AudioSegment, slot_ms: int) -> AudioSegment:
    seg = AudioSegment.silent(duration=HEAD_PAD_MS, frame_rate=seg.frame_rate) + seg + AudioSegment.silent(duration=TAIL_PAD_MS, frame_rate=seg.frame_rate)
    if len(seg) > slot_ms + MAX_TRIM_OVERFLOW_MS:
        seg = seg.fade_out(FADE_MS)[:slot_ms]
    elif len(seg) < slot_ms:
        seg += AudioSegment.silent(duration=slot_ms - len(seg), frame_rate=seg.frame_rate)
    return seg
