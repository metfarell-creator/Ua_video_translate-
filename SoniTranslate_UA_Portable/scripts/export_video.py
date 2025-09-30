#!/usr/bin/env python3
import argparse
from pipeline.mixer import mux_audio_video

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="output.mp4")
    args = ap.parse_args()
    mux_audio_video(args.audio, args.video, args.out)
    print(args.out)

if __name__ == "__main__":
    main()
