// model.js — loads the phoneme model and runs it on audio.
//
// The model is wav2vec2 (vitouphy/wav2vec2-xls-r-300m-timit-phoneme),
// exported to ONNX by tools/export_model.py. It takes 16 kHz mono audio and
// returns, for every 20 ms "frame" of audio, a score for each of 44 tokens
// (39 phonemes plus blank, two "gap" tokens and two unused ones).
//
// Nothing in here knows about letters, words or scoring. It only turns audio
// into a grid of per-frame phoneme probabilities; scorer.js (later step)
// decides what those mean.

import * as ort from '../vendor/ort/ort.wasm.min.mjs';

// Folder URLs, resolved relative to this file so the app works from any path
// (GitHub Pages serves it under /VoiceApp/, localhost serves it under /).
const MODEL_DIR = new URL('../model/', import.meta.url);
const VOCAB_URL = new URL('../data/vocab.json', import.meta.url);
const ORT_DIR = new URL('../vendor/ort/', import.meta.url);

// Tell onnxruntime where its .wasm file lives (we vendor it; no CDN).
ort.env.wasm.wasmPaths = ORT_DIR.href;
// GitHub Pages cannot send the headers that multi-threaded WASM needs, so
// asking for more than 1 thread would only produce a console warning.
ort.env.wasm.numThreads = 1;

let session = null; // the loaded ONNX model
let vocab = null;   // { blankId, separatorIds: number[], tokens: string[] }

/**
 * Download the model (once per page load) and get it ready to run.
 *
 * @param {(fraction: number) => void} [onProgress] called with 0..1 while
 *        the model chunks download, so the UI can show a progress bar.
 * @returns {Promise<{tokens: string[], blankId: number}>} the phoneme table.
 */
export async function loadModel(onProgress = () => {}) {
  if (session) return vocab;

  vocab = await (await fetch(VOCAB_URL)).json();
  const manifest = await (await fetch(new URL('manifest.json', MODEL_DIR))).json();

  // The model is stored as several ~20 MB chunks (git refuses files over
  // 100 MB). Download them one after another into a single buffer.
  const bytes = new Uint8Array(manifest.totalBytes);
  let offset = 0;
  for (const part of manifest.parts) {
    const res = await fetch(new URL(part.file, MODEL_DIR));
    if (!res.ok) throw new Error(`Could not download ${part.file} (${res.status})`);
    const chunk = new Uint8Array(await res.arrayBuffer());
    if (chunk.length !== part.bytes) {
      throw new Error(`${part.file} is ${chunk.length} bytes, expected ${part.bytes}`);
    }
    bytes.set(chunk, offset);
    offset += chunk.length;
    onProgress(offset / manifest.totalBytes);
  }

  // Check the reassembled file against the fingerprint in the manifest, so a
  // half-finished or corrupted download fails loudly here instead of giving
  // mysterious wrong scores later.
  // (crypto.subtle only exists on https:// and localhost. On a plain
  // http://192.168.x.x test server the check is skipped.)
  if (crypto.subtle) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
    if (hex !== manifest.sha256) throw new Error('Model download is corrupted (checksum mismatch)');
  } else {
    console.warn('Not a secure page: skipping the model checksum.');
  }

  // Start the model. onnxruntime is given two things:
  //   - graph.onnx: the small "wiring" file, with no weights in it;
  //   - the 355 MB we just downloaded, as a separate blob of weights that
  //     graph.onnx points into (see tools/make_graph.py).
  // Handing it the 355 MB as the model itself also works on a desktop, but
  // it then keeps two copies while starting up, and phones with little
  // memory fail with "std::bad_alloc". This way it copies each weight once.
  const graphRes = await fetch(new URL(manifest.graph.file, MODEL_DIR));
  if (!graphRes.ok) throw new Error(`Could not download ${manifest.graph.file} (${graphRes.status})`);
  const graph = new Uint8Array(await graphRes.arrayBuffer());
  if (graph.length !== manifest.graph.bytes) {
    throw new Error(`${manifest.graph.file} is ${graph.length} bytes, expected ${manifest.graph.bytes}`);
  }
  session = await ort.InferenceSession.create(graph, {
    executionProviders: ['wasm'],
    externalData: [{ path: manifest.graph.weightsName, data: bytes }],
  });
  return vocab;
}

/**
 * Run the model on one recording.
 *
 * @param {Float32Array} samples 16 kHz mono audio, raw (-1..1). Use
 *        decodeTo16k() in audio.js to get this from a file or recording.
 * @returns {Promise<{logProbs: Float32Array, frames: number, vocabSize: number}>}
 *        logProbs is a flat [frames x vocabSize] grid: the log-probability of
 *        token v in frame f is logProbs[f * vocabSize + v]. One frame = 20 ms.
 */
export async function runModel(samples) {
  if (!session) throw new Error('Call loadModel() first');

  // The model was trained on audio scaled to zero mean and unit variance,
  // so every clip must be scaled the same way. This also makes quiet and
  // loud recordings look alike to the model.
  let mean = 0;
  for (const s of samples) mean += s;
  mean /= samples.length;
  let variance = 0;
  for (const s of samples) variance += (s - mean) ** 2;
  const std = Math.sqrt(variance / samples.length + 1e-7);
  const normalised = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) normalised[i] = (samples[i] - mean) / std;

  const input = new ort.Tensor('float32', normalised, [1, samples.length]);
  const { logits } = await session.run({ input_values: input });
  const [, frames, vocabSize] = logits.dims;

  // Turn raw scores ("logits") into log-probabilities, frame by frame
  // (a log-softmax). Scoring adds these up, which needs proper log-probs.
  const logProbs = new Float32Array(logits.data.length);
  for (let f = 0; f < frames; f++) {
    const row = f * vocabSize;
    let max = -Infinity;
    for (let v = 0; v < vocabSize; v++) max = Math.max(max, logits.data[row + v]);
    let sum = 0;
    for (let v = 0; v < vocabSize; v++) sum += Math.exp(logits.data[row + v] - max);
    const logSum = max + Math.log(sum);
    for (let v = 0; v < vocabSize; v++) logProbs[row + v] = logits.data[row + v] - logSum;
  }
  return { logProbs, frames, vocabSize };
}

/**
 * "What did the model hear?" — open-ended decoding, for debugging only.
 * The real app does constrained scoring instead (see scorer.js).
 *
 * Takes the best token in each frame, merges runs of the same token and
 * drops blanks (standard greedy CTC decoding). The model's "gap" tokens
 * ("|" and " ", listed in vocab.separatorIds) are dropped too, since they
 * are not sounds.
 *
 * @returns {{token: string, prob: number, startMs: number}[]} prob is the
 *          average confidence (0..1) over the frames of that token.
 */
export function greedyDecode({ logProbs, frames, vocabSize }) {
  const out = [];
  let prev = -1;
  for (let f = 0; f < frames; f++) {
    let best = 0;
    for (let v = 1; v < vocabSize; v++) {
      if (logProbs[f * vocabSize + v] > logProbs[f * vocabSize + best]) best = v;
    }
    const p = Math.exp(logProbs[f * vocabSize + best]);
    if (best !== vocab.blankId && !vocab.separatorIds.includes(best)) {
      if (best !== prev) {
        out.push({ token: vocab.tokens[best], prob: p, n: 1, startMs: f * 20 });
      } else {
        const last = out[out.length - 1];
        last.prob += p;
        last.n += 1;
      }
    }
    prev = best;
  }
  return out.map(({ token, prob, n, startMs }) => ({ token, prob: prob / n, startMs }));
}
