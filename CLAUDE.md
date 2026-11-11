# VoiceApp

Offline pronunciation ("voice") module for a kids' learning web app, to be
hosted on GitHub Pages from https://github.com/ZekeInc/VoiceApp (private repo,
branch `main`). The child hears a target sound, records themselves, and gets a
star or "try again".

## Status

Step 1 of 8 (model spike) is built and pushed. Steps 2 and 6 are started
(uncommitted): `index.html` has a record button that records the microphone,
runs the model and shows what it heard in a temporary "testing" panel. It
also shows the current letter (big), how to say it ("buh"), an example word,
the phonemes it should match (temporary, for testing), and a "Next letter"
button that walks a-z from `voice/data/items.json`. No
scoring, target-sound playback, auto-stop on silence or offline cache yet.
Steps 3-5, 7 and 8 are not started.

Done:
- Model chosen, exported, quantized and committed (see "Model" below).
- `spike.html` runs the model in the browser on uploaded audio files and shows
  the phonemes heard, a confidence per phoneme, and the model time.
- Verified in headless Chromium and by the owner in desktop Firefox.

Not done / open:
- GitHub Pages is enabled by the owner (deploys from `main` / root). URLs:
  `https://zekeinc.github.io/VoiceApp/` (kid UI) and
  `https://zekeinc.github.io/VoiceApp/spike.html`.
- **Nothing has been run on a phone yet.** Speed and memory on the Galaxy
  Note 20 are unmeasured.
- **Accuracy on children is unconfirmed.** The model gives confident output on
  the child clips, but the clips are unlabelled, so correctness is unknown.
  Waiting on: labels for `c_02`..`c_18`, and new spoken recordings from the
  owner's kids (one sound or word per file, named after what was said).
- Target-sound playback is assumed to be prerecorded clips, not
  `speechSynthesis`. The owner has not confirmed this.

## Hard requirements

- Scope is letter sounds and 3-letter words only (/b/, "cat", "dog").
- Fully offline in the browser after first load. No paid APIs, no server, no CDN
  at runtime: vendor every dependency into the repo.
- Must run on a phone (reference device: Galaxy Note 20, Chrome on Android).
- Static site with no build step: plain HTML, CSS and ES modules.
- Comment code thoroughly. The owner tunes the scoring by hand, so every
  threshold needs a comment saying what it does and which way to move it.

## Model

- **In use:** `vitouphy/wav2vec2-xls-r-300m-timit-phoneme` (wav2vec2-large,
  315M parameters, TIMIT phoneme CTC, 39 phonemes). Exported to ONNX and
  int8-quantized by `tools/export_model.py`: 355 MB, committed as 17 chunks of
  20 MB in `voice/model/` with a `manifest.json` (sizes and SHA-256).
- **Loading:** `model.js` starts the session from `voice/model/graph.onnx`
  (0.8 MB, wiring only, written by `tools/make_graph.py`) and passes the
  reassembled chunks as onnxruntime `externalData` named `weights`;
  `graph.onnx` points at byte offsets inside them. Passing the 355 MB file as
  the model itself needed an 823 MB WASM heap and failed with
  `std::bad_alloc` on one of the owner's phones; this way needs 358 MB
  (measured in desktop Chromium). Regenerate `graph.onnx` whenever the chunks
  change (`export_model.py` does).
- **Quantization:** MatMul/Gemm only. Quantized Conv layers are known to crash
  onnxruntime on wav2vec2. The int8 model matched fp32 exactly on 14 of the 18
  child clips and differed by one or two phonemes on the rest.
- **Input:** 16 kHz mono float32, normalised to zero mean and unit variance
  (done in `model.js`). No attention mask is needed for a single clip.
- **Output:** 44 columns per 20 ms frame. Blank is id 41. Ids 0 (`|`) and 16
  (` `) are gap tokens that carry no sound; `vocab.json` lists them as
  `separatorIds`. The scorer must allow optional gaps between target phonemes.
- **Speed:** about 0.5 s of model time per second of audio in desktop
  Chromium, single-threaded WASM (0.27 s for a 0.5 s clip). Model start-up
  from a local server: 1.7 s.

### Models tried and rejected

Tested on 18 short clips cut from a 2-year-old singing the alphabet (a worst
case: toddler, sung, unlabelled).

- `bookbot/wav2vec2-ljspeech-gruut` (wav2vec2-base, 122 MB int8). The original
  choice. Returned nothing for 13 of 18 child clips, quantized or not, while
  transcribing 0.6 s adult snippets fine. Padding with silence did not help.
  It was trained on a single adult voice. Do not go back to it to save size.
- `facebook/wav2vec2-lv-60-espeak-cv-ft` (wav2vec2-large, multilingual).
  Low confidence on the child clips and drifted into non-English phonemes.

## Design decisions (for the steps still to build)

- **Runtime:** onnxruntime-web 1.31.1 called directly, not transformers.js,
  because scoring needs raw per-frame log-probabilities.
- **Scoring is constrained, not open recognition.** For each item, compute the
  CTC likelihood of the target phoneme sequence and of each listed confusion
  (e.g. /k/ vs /t/), then pass or fail on the margin between them.
- **Phoneme targets are precomputed** in `voice/data/items.json` using the
  token set in `vocab.json`. No espeak or other grapheme-to-phoneme at runtime.
- **Isolated stops tolerate a trailing vowel.** "buh" counts as a correct /b/:
  each stop item also accepts target + schwa-like vowel as a correct variant.
- **Debug mode** (`?debug=1`) saves each recording and its scores to IndexedDB,
  lets a parent label it right/wrong, exports everything, and can re-score all
  saved recordings against the current config.

## GitHub Pages constraints

- No custom response headers, so no cross-origin isolation and no
  multi-threaded WASM. Inference is single-threaded.
- Git rejects files over 100 MB, hence the chunked model.
- The repo is about 480 MB: the current model plus the rejected 122 MB model,
  which remains in git history. Pages sites are limited to 1 GB.
- The microphone and `crypto.subtle` need HTTPS (Pages) or `localhost`. On a
  plain `http://192.168.x.x` server `model.js` skips the checksum.

## Layout

Exists now:

```
index.html              kid UI so far: record button + "what the model heard"
spike.html              step 1 test page: upload audio, see phonemes + timing
voice/
  js/
    model.js            loadModel(), runModel(), greedyDecode()
    audio.js            decodeTo16k(), resampleTo16k(), encodeWav()
    recorder.js         startRecording(): raw mic capture -> 16 kHz mono;
                        stops on a second tap or at RECORDER.maxSeconds
    recorder-worklet.js audio-thread half of recorder.js
    config.js           every tunable setting (recorder only so far)
    main.js             kid UI states: loading/ready/recording/thinking;
                        letter display and Next letter button
    items.js            loadItems(), checkItems() (tokens vs vocab.json)
  voice.css             kid UI looks, driven by <body data-state>
  data/vocab.json       generated: tokens, blankId, separatorIds
  data/items.json       hand-edited: letters a-z with show, say, example,
                        target phonemes, alsoOk (stop + ə); no confusions yet
  model/                generated: model.onnx.part0..16, graph.onnx, manifest.json
  vendor/ort/           onnxruntime-web, plain WASM build only
tools/
  export_model.py       download, export, quantize, chunk the model
  check_model.py        compare fp32 vs int8 output on audio files
  make_graph.py         write voice/model/graph.onnx from the chunks
  .venv/, build/        git-ignored (Python env; ONNX files and test clips)
```

Planned:

```
index.html              add: play target sound, star / try again
sw.js                   offline cache for app shell and model
voice/
  js/recorder.js        add: auto-stop on silence
  js/scorer.js          constrained CTC scoring
  js/config.js          add: scoring thresholds
  js/debug.js           save, list, label, export and re-score recordings
  data/items.json       add: confusions, audio clip, 3-letter words
  audio/                prerecorded target sounds
```

## Build plan

1. Model spike (go/no-go). **Built; awaiting phone and child-voice results.**
2. Recorder: mic to 16 kHz mono, auto-stop on silence, noise suppression as a
   setting (it can swallow the burst of a stop sound).
3. Scorer: constrained CTC with confusions and trailing-vowel tolerance.
4. Items data: letter sounds and a starter set of 3-letter words.
5. Debug mode, before the kid UI, because it is how thresholds get tuned.
6. Kid UI, with a loading screen for the first model download.
7. Offline: service worker caching the app shell and model chunks. Bump its
   cache version whenever a cached file changes.
8. Tune thresholds from labelled recordings of the kids.

## Running and testing

Serve the repo root over HTTP and open the test page. `file://` does not work.

```
python3 -m http.server 8000        # then http://localhost:8000/spike.html
```

Model tools (need `ffmpeg` on the PATH):

```
python3 -m venv tools/.venv
tools/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
tools/.venv/bin/pip install transformers onnx onnxruntime onnxscript soundfile numpy
tools/.venv/bin/python tools/export_model.py            # regenerates voice/model + vocab.json
tools/.venv/bin/python tools/check_model.py clip.m4a    # fp32 vs int8 on a clip
```

Test clips live in `tools/build/samples/` (git-ignored, they are recordings of
the owner's children; never commit them). The owner drops files there and
asks for them to be run through the model.

## Working agreements

- Plan before coding on anything non-trivial; the owner asked for this.
- Commit or push only when asked.
- Commits use the repo-local identity `ZekeInc` with the GitHub no-reply
  address, not the owner's personal email.
