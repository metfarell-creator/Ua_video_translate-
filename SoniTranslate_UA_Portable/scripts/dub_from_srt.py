#!/usr/bin/env python3
import argparse, srt, datetime as dt
from pathlib import Path
from pydub import AudioSegment
from ukrainian_word_stress import Stressifier
from pipeline import tts, align, mixer, utils

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--repo", default="patriotyk/styletts2-ukrainian")
    ap.add_argument("--out", default="dubbed.wav")
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    entries = utils.srt_to_entries(Path(args.srt).read_text(encoding="utf-8"))
    stress = Stressifier()

    out_segs = []
    for e in entries:
        text = stress.process_text(e.content)
        wav = tts.synthesize(text, repo=args.repo, sample_rate=args.sr, speed=1.0)
        seg = mixer.wav_from_array(wav, args.sr)
        slot = int((e.end - e.start).total_seconds()*1000)
        seg = align.fit_to_slot(seg, slot)
        out_segs.append(seg)

    timeline = utils.concat_audio(out_segs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    timeline.export(args.out, format="wav")
    print(args.out)

if __name__ == "__main__":
    main()
