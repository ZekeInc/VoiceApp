# VoiceApp

Offline pronunciation ("voice") module for a kids' learning web app. Hosted on
GitHub Pages at https://github.com/ZekeInc/VoiceApp. The child hears a target
sound, records themselves, and gets a star or "try again".

Status: step 1 (model spike) is built. The model is exported and `spike.html`
runs it on uploaded audio files. Scorer, recorder, kid UI, debug mode and the
service worker are not written yet; the layout below marks what exists.

## Hard requirements

- Scope is letter sounds and 3-letter words only (/b/, "cat", "dog").
- Fully offline in the browser after first load. No paid APIs, no server, no CDN
  at runtime: vendor every dependency into the repo.
- Must run on a phone (reference device: Galaxy Note 20, Chrome on Android).
- Static site with no build step: plain HTML, CSS and ES modules.
- Comment code thoroughly. The owner tunes the scoring by hand, so every
  threshold needs a comment saying what it does and which way to move it.

## Design decisions

- **Model:** `bookbot/wav2vec2-ljspeech-gruut` (wav2vec2-base, IPA phoneme CTC),
  exported to ONNX and int8-quantized by us: 122 MB, in six chunks. Quantize
  MatMul/Gemm only; quantized Conv layers are known to crash onnxruntime on wav2vec2.
  Unproven on children's voices. Fallback if it fails: a wav2vec2-large phoneme
  model at roughly 320-355 MB.
- **Model output quirk:** it emits a `|` token (id 0) between most phonemes.
  It carries no sound; the scorer must allow an optional `|` between targets.
- **Runtime:** onnxruntime-web called directly, not transformers.js, because
  scoring needs raw per-frame log-probabilities.
- **Scoring is constrained, not open recognition.** For each item, compute the
  CTC likelihood of the target phoneme sequence and of each listed confusion
  (e.g. /k/ vs /t/), then pass or fail on the margin between them.
- **Phoneme targets are precomputed** in `voice/data/items.json`. No espeak or
  any other grapheme-to-phoneme step at runtime.
- **Isolated stops tolerate a trailing vowel.** "buh" counts as a correct /b/:
  each stop item also accepts target + schwa-like vowel as a correct variant.
- **Target audio is prerecorded clips**, not `speechSynthesis`, which is
  unreliable for isolated phonemes.
- **Debug mode** (`?debug=1`) saves each recording and its scores to IndexedDB,
  lets a parent label it right/wrong, exports everything, and can re-score all
  saved recordings against the current config.

## GitHub Pages constraints

- No custom response headers, so no cross-origin isolation and no
  multi-threaded WASM. Inference is single-threaded.
- Git rejects files over 100 MB. The model is committed as ~25 MB chunks with a
  manifest and reassembled on first load.
- The service worker caches the app shell and the model. Bump its cache version
  whenever a cached file changes.
- The microphone needs HTTPS (Pages) or `localhost`.

## Planned layout

```
spike.html              EXISTS. Step 1 test page: upload audio, see phonemes + timing
index.html              kid UI: play, record, star / try again
voice/
  voice.css
  js/
    main.js             UI flow and state
    recorder.js         mic capture to 16 kHz mono, auto-stop on silence
    model.js            EXISTS. Load model chunks, run ONNX, return frame log-probs
    audio.js            EXISTS. Decode any audio file/blob to 16 kHz mono
    scorer.js           constrained CTC scoring
    config.js           every tunable threshold in one place
    debug.js            save, list, label, export and re-score recordings
  data/
    items.json          per item: target phonemes, confusions, audio clip
    vocab.json          EXISTS (generated). Model phoneme-to-id table
  audio/                prerecorded target sounds
  model/                EXISTS (generated). model.onnx.partN + manifest.json
  vendor/ort/           EXISTS. onnxruntime-web 1.31.1, plain WASM build
sw.js                   offline cache for app shell and model
tools/
  export_model.py       EXISTS. Download, export, quantize, chunk the model
  check_model.py        EXISTS. Compare fp32 vs int8 output on audio files
```

## Running locally

Serve the repo root over HTTP (`python3 -m http.server 8000`) and open
`http://localhost:8000/`. Opening the files via `file://` will not work: ES
modules, the service worker and the microphone all need an HTTP origin.

## Model tools

The venv (`tools/.venv`) and scratch output (`tools/build`) are git-ignored.

```
python3 -m venv tools/.venv
tools/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
tools/.venv/bin/pip install transformers onnx onnxruntime onnxscript soundfile numpy
tools/.venv/bin/python tools/export_model.py            # regenerates voice/model + vocab.json
tools/.venv/bin/python tools/check_model.py clip.m4a    # needs ffmpeg
```

## Working agreements

- Plan before coding on anything non-trivial; the owner asked for this.
- Commit or push only when asked.
