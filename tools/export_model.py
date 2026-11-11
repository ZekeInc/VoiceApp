"""
Step 1 of the build plan: turn the Hugging Face phoneme model into files the
browser can load.

What it does, in order:
  1. Downloads bookbot/wav2vec2-ljspeech-gruut (PyTorch, ~360 MB).
  2. Exports it to ONNX (fp32) in tools/build/.
  3. Quantizes the weights to int8, which is what shrinks it to ~95 MB.
  4. Splits the int8 file into chunks under voice/model/ (git rejects any
     single file over 100 MB, and small chunks also resume better on a phone).
  5. Writes voice/data/vocab.json (phoneme <-> output index table).

Run from the repo root:
    tools/.venv/bin/python tools/export_model.py

Re-running is safe; it overwrites its own outputs.
"""

import hashlib
import json
import sys
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCTC, AutoProcessor

MODEL_ID = "bookbot/wav2vec2-ljspeech-gruut"

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "tools" / "build"          # scratch output, git-ignored
MODEL_OUT = REPO / "voice" / "model"      # what the browser downloads
DATA_OUT = REPO / "voice" / "data"

FP32_PATH = BUILD / "model_fp32.onnx"
INT8_PATH = BUILD / "model_int8.onnx"

# Chunk size for the committed model. 20 MB keeps every file far below
# GitHub's 100 MB hard limit and its 50 MB warning.
CHUNK_BYTES = 20 * 1024 * 1024

# Which ONNX operators get int8 weights. The transformer layers (MatMul/Gemm)
# hold ~95% of the weights. The convolutional front end is deliberately left
# as fp32: quantized Conv on wav2vec2 is known to crash or badly hurt accuracy
# in onnxruntime, and it is only ~4M parameters anyway.
QUANTIZE_OPS = ["MatMul", "Gemm"]


class LogitsOnly(torch.nn.Module):
    """Wraps the HF model so the ONNX graph has one input and one output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        # input_values: float32 [batch, samples], 16 kHz mono, already
        # normalised to zero mean / unit variance (the browser does this).
        # returns:      float32 [batch, frames, vocab], one frame per 20 ms.
        return self.model(input_values).logits


def export_fp32():
    print(f"Downloading {MODEL_ID} ...")
    model = AutoModelForCTC.from_pretrained(MODEL_ID).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Facts the browser code depends on; printed so they are easy to check.
    fe = processor.feature_extractor
    print("  sampling_rate       :", fe.sampling_rate)
    print("  do_normalize        :", fe.do_normalize)
    print("  return_attention_mask:", fe.return_attention_mask)
    print("  vocab size          :", model.config.vocab_size)
    print("  pad/blank token id  :", model.config.pad_token_id)

    BUILD.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 16000)  # one second of audio
    print("Exporting to ONNX ...")
    torch.onnx.export(
        LogitsOnly(model),
        (dummy,),
        str(FP32_PATH),
        input_names=["input_values"],
        output_names=["logits"],
        # The audio length (and so the frame count) must stay variable.
        dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                      "logits": {0: "batch", 1: "frames"}},
        opset_version=17,
        dynamo=False,  # the classic exporter; simplest graph for quantizing
        external_data=False,
    )
    return processor


def quantize():
    print("Quantizing to int8 ...")
    quantize_dynamic(
        str(FP32_PATH),
        str(INT8_PATH),
        op_types_to_quantize=QUANTIZE_OPS,
        weight_type=QuantType.QUInt8,
    )
    # Report which operators exist, so it is obvious if the quantizer missed
    # the big layers (the file size would give that away too).
    ops = {}
    for node in onnx.load(str(INT8_PATH), load_external_data=False).graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("  ops in int8 graph:", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]))


def split_into_chunks():
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    for old in MODEL_OUT.glob("model.onnx.part*"):
        old.unlink()

    data = INT8_PATH.read_bytes()
    parts = []
    for i in range(0, len(data), CHUNK_BYTES):
        name = f"model.onnx.part{len(parts)}"
        chunk = data[i:i + CHUNK_BYTES]
        (MODEL_OUT / name).write_bytes(chunk)
        parts.append({"file": name, "bytes": len(chunk)})

    manifest = {
        "source": MODEL_ID,
        "format": "onnx, int8 dynamic quantization of " + "/".join(QUANTIZE_OPS),
        "totalBytes": len(data),
        # The browser checks this after reassembly to catch a broken download.
        "sha256": hashlib.sha256(data).hexdigest(),
        "parts": parts,
    }
    (MODEL_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(parts)} chunks, {len(data) / 1e6:.1f} MB total -> {MODEL_OUT}")


def write_vocab(processor):
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    vocab = processor.tokenizer.get_vocab()  # {"phoneme": index}
    # Keep only the first 45 entries: those are the model's real output
    # columns. The tokenizer lists a couple of malformed extras after them.
    n_out = 45
    by_id = [None] * n_out
    for token, idx in vocab.items():
        if idx < n_out:
            by_id[idx] = token
    out = {
        "source": MODEL_ID,
        # CTC "blank" = the model saying "no new phoneme in this frame".
        "blankId": processor.tokenizer.pad_token_id,
        "tokens": by_id,  # tokens[i] is the phoneme for output column i
    }
    (DATA_OUT / "vocab.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote vocab ({len(by_id)} tokens) -> {DATA_OUT / 'vocab.json'}")


if __name__ == "__main__":
    proc = export_fp32()
    quantize()
    split_into_chunks()
    write_vocab(proc)
    print(f"fp32: {FP32_PATH.stat().st_size / 1e6:.1f} MB   "
          f"int8: {INT8_PATH.stat().st_size / 1e6:.1f} MB")
    sys.exit(0)
