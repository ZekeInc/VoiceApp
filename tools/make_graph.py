"""
Writes voice/model/graph.onnx: a small copy of the model that contains only
the wiring (which operation feeds which), with every large block of weights
replaced by a pointer of the form "bytes N..M of the file called `weights`".

Why: it roughly halves the memory the browser needs to start the model.
When onnxruntime is handed the full 355 MB model.onnx, it copies the file
into its own memory and then unpacks the weights into a second copy, which
is too much for some phones ("std::bad_alloc"). Handed graph.onnx plus the
355 MB file as a separate "weights" blob, it copies each weight exactly once.

The pointers refer to byte positions inside the existing model.onnx, so the
committed model chunks (model.onnx.part*) are used as they are; nothing big
is added to the repo. graph.onnx must be regenerated whenever the chunks
change, which export_model.py does by calling write_graph().

Run on its own (rebuilds graph.onnx from the committed chunks):
    tools/.venv/bin/python tools/make_graph.py
"""

import hashlib
import json
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parent.parent
MODEL_OUT = REPO / "voice" / "model"
GRAPH_NAME = "graph.onnx"

# The name the pointers use for the big file. model.js must hand the
# reassembled model.onnx to onnxruntime under exactly this name.
WEIGHTS_NAME = "weights"

# Weights smaller than this many bytes stay inside graph.onnx. Moving tiny
# ones out saves no memory and only adds bookkeeping.
MIN_EXTERNAL_BYTES = 1024


def _varint(data, pos):
    """Read one protobuf varint at `pos`. Returns (value, next position)."""
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _fields(data, start, end):
    """Yield (field number, payload start, payload end) for every
    length-delimited field of the protobuf message at data[start:end]."""
    pos = start
    while pos < end:
        key, pos = _varint(data, pos)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:                      # varint
            _, pos = _varint(data, pos)
        elif wire_type == 1:                    # 64-bit
            pos += 8
        elif wire_type == 5:                    # 32-bit
            pos += 4
        elif wire_type == 2:                    # length-delimited
            length, pos = _varint(data, pos)
            yield number, pos, pos + length
            pos += length
        else:
            raise ValueError(f"unexpected protobuf wire type {wire_type}")


def _raw_data_positions(data):
    """Find where each weight's bytes sit inside the serialized model.

    Field numbers come from onnx.proto: ModelProto.graph = 7,
    GraphProto.initializer = 5, TensorProto.name = 8, TensorProto.raw_data = 9.

    Returns {weight name: (offset, length)}.
    """
    positions = {}
    for number, g_start, g_end in _fields(data, 0, len(data)):
        if number != 7:
            continue
        for number, t_start, t_end in _fields(data, g_start, g_end):
            if number != 5:
                continue
            name = raw = None
            for number, start, end in _fields(data, t_start, t_end):
                if number == 8:
                    name = bytes(data[start:end]).decode()
                elif number == 9:
                    raw = (start, end - start)
            if name is not None and raw is not None:
                positions[name] = raw
    return positions


def write_graph(data=None):
    """Write graph.onnx next to the chunks and record it in manifest.json.

    `data` is the full model.onnx as bytes; if omitted it is reassembled
    from the committed chunks.
    """
    manifest_path = MODEL_OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if data is None:
        data = b"".join((MODEL_OUT / p["file"]).read_bytes() for p in manifest["parts"])
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise SystemExit("model bytes do not match manifest.json; run export_model.py")

    positions = _raw_data_positions(memoryview(data))
    model = onnx.load_model_from_string(data)

    moved = moved_bytes = 0
    for tensor in model.graph.initializer:
        if len(tensor.raw_data) < MIN_EXTERNAL_BYTES:
            continue
        offset, length = positions[tensor.name]
        # Make sure the pointer really lands on this weight's bytes.
        assert data[offset:offset + length] == tensor.raw_data, tensor.name
        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for key, value in (("location", WEIGHTS_NAME), ("offset", offset), ("length", length)):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, str(value)
        moved += 1
        moved_bytes += length

    graph_bytes = model.SerializeToString()
    (MODEL_OUT / GRAPH_NAME).write_bytes(graph_bytes)

    manifest["graph"] = {
        "file": GRAPH_NAME,
        "bytes": len(graph_bytes),
        # The name the pointers in graph.onnx use for the reassembled chunks.
        "weightsName": WEIGHTS_NAME,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {GRAPH_NAME}: {len(graph_bytes) / 1e6:.2f} MB, pointing at "
          f"{moved} weights ({moved_bytes / 1e6:.1f} MB) inside model.onnx")


if __name__ == "__main__":
    write_graph()
