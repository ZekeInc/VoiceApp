"""
Sanity check for the exported model: runs audio files through the fp32 and the
int8 ONNX models and prints the phonemes each one hears, plus timing.

If the two lines differ a lot, quantization damaged the model.

Usage (from the repo root):
    tools/.venv/bin/python tools/check_model.py path/to/a.wav [more files...]

Any format ffmpeg can read works (wav, m4a, mp3, flac, ogg, webm).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
VOCAB = json.loads((REPO / "voice/data/vocab.json").read_text(encoding="utf-8"))
MODELS = {
    "fp32": REPO / "tools/build/model_fp32.onnx",
    "int8": REPO / "tools/build/model_int8.onnx",
}


def load_audio(path):
    """Decode any audio file to 16 kHz mono float32 using ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    # Same normalisation the browser must apply: zero mean, unit variance.
    return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)


def greedy_decode(logits):
    """Plain CTC decoding: best token per frame, merge repeats, drop blanks."""
    ids = logits.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != VOCAB["blankId"]:
            out.append(VOCAB["tokens"][i])
        prev = i
    return " ".join(out)


def main(paths):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # the browser on GitHub Pages is single-threaded
    sessions = {k: ort.InferenceSession(str(p), opts) for k, p in MODELS.items()}
    for path in paths:
        audio = load_audio(path)
        print(f"\n{path}  ({len(audio) / 16000:.1f} s)")
        results = {}
        for name, sess in sessions.items():
            t0 = time.time()
            logits = sess.run(None, {"input_values": audio[None, :]})[0][0]
            results[name] = logits
            print(f"  {name} {time.time() - t0:5.2f}s  {greedy_decode(logits)}")
        agree = (results["fp32"].argmax(-1) == results["int8"].argmax(-1)).mean()
        print(f"  frames where fp32 and int8 agree: {agree:.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
