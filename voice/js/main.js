// main.js — the kid UI's flow: what the record button does in each state.
//
// The page is always in exactly one state (see setState below). The state
// is written to <body data-state="…"> and voice.css changes the look.
//
//   loading -> ready -> starting -> recording -> thinking -> ready -> …
//
// There is no scoring yet, so after "thinking" the page shows what the
// model heard instead of a star / try again.
//
// The letter to say comes from voice/data/items.json; the Next letter
// button steps through them in order.

import { loadModel, runModel, greedyDecode } from './model.js';
import { startRecording } from './recorder.js';
import { encodeWav, SAMPLE_RATE } from './audio.js';
import { RECORDER } from './config.js';
import { loadItems, checkItems } from './items.js';

const $ = (id) => document.getElementById(id);
const button = $('record');
const nextButton = $('next');
const ring = document.querySelector('.level-ring');

let state = 'loading';
let recording = null; // the { stop() } handle from recorder.js while recording
let items = [];       // from voice/data/items.json
let itemIndex = 0;    // which item is on screen

/**
 * Move the page to a new state and update the button and status line.
 * @param {'loading'|'ready'|'starting'|'recording'|'thinking'|'error'} next
 * @param {string} message text for the status line (for the grown-up; the
 *        child goes by the button's colour and icon).
 */
function setState(next, message) {
  state = next;
  document.body.dataset.state = next;
  $('status').textContent = message;
  // The button can only be tapped to start (ready) or to stop (recording).
  button.disabled = !(next === 'ready' || next === 'recording');
  // Changing letter mid-recording would score the sound against the wrong
  // letter, so Next only works while waiting (or if the model failed to
  // load, so the letters can still be looked through).
  nextButton.disabled = !items.length || !(next === 'ready' || next === 'error');
  button.setAttribute('aria-label', {
    loading: 'Loading', ready: 'Record', starting: 'Starting',
    recording: 'Stop', thinking: 'Thinking', error: 'Not working',
  }[next]);
}

// ---- Record button: tap to start, tap again to stop. ----
button.addEventListener('click', () => {
  if (state === 'ready') start();
  else if (state === 'recording') finish();
});

async function start() {
  setState('starting', 'Starting the microphone…');
  try {
    recording = await startRecording({
      onLevel: (level) => ring.style.setProperty('--level', level),
      onLimit: finish, // reached RECORDER.maxSeconds: stop by itself
    });
    setState('recording', 'Listening… tap to stop');
  } catch (err) {
    // A blocked microphone is not fatal: let them fix it and tap again.
    setState('ready', err.message);
  }
}

async function finish() {
  if (state !== 'recording') return; // a tap and the time limit can race
  setState('thinking', 'Thinking…');
  ring.style.setProperty('--level', 0);
  try {
    const samples = await recording.stop();
    recording = null;

    const seconds = samples.length / SAMPLE_RATE;
    if (seconds < RECORDER.minSeconds) {
      setState('ready', 'Too short. Tap, say it, then tap again.');
      return;
    }

    const t0 = performance.now();
    const output = await runModel(samples);
    const modelSeconds = (performance.now() - t0) / 1000;
    showHeard(samples, seconds, modelSeconds, greedyDecode(output));
    setState('ready', 'Tap to try again');
  } catch (err) {
    console.error(err);
    setState('ready', `That did not work (${err.message}). Tap to try again.`);
  }
}

// ---- The letter at the top, and the Next letter button. ----

/** Put items[itemIndex] on screen and clear the last letter's results. */
function showItem() {
  const item = items[itemIndex];
  $('item-show').textContent = item.show;
  $('item-say').textContent = `say “${item.say}”`;
  $('item-example').textContent = `as in ${item.example}`;

  // Temporary, for testing: the phoneme sequences that will count as right,
  // in the same symbols the "what the model heard" panel uses.
  const target = $('item-target');
  target.textContent = 'Listening for: ';
  [item.target, ...item.alsoOk].forEach((sequence, i) => {
    if (i > 0) target.append(' or ');
    const span = document.createElement('span');
    span.className = 'ph';
    span.textContent = sequence.join(' ');
    target.append(span);
  });

  $('heard').hidden = true;
}

nextButton.addEventListener('click', () => {
  itemIndex = (itemIndex + 1) % items.length; // after the last, back to the first
  showItem();
  if (state === 'ready') setState('ready', 'Tap and say the sound');
});

// ---- Temporary results panel (until scorer.js gives a star). ----
function showHeard(samples, clipSeconds, modelSeconds, phonemes) {
  const line = $('phonemes');
  line.textContent = phonemes.length ? '' : '(nothing heard)';
  for (const { token, prob } of phonemes) {
    const span = document.createElement('span');
    // Same colour bands as spike.html: under 50% red, under 80% yellow.
    span.className = 'ph' + (prob < 0.5 ? ' low' : prob < 0.8 ? ' mid' : '');
    span.textContent = token;
    const pct = document.createElement('small');
    pct.textContent = Math.round(prob * 100) + '%';
    span.append(pct);
    line.append(span);
  }

  // Let the grown-up hear exactly what the model was given.
  const player = $('playback');
  if (player.src) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(encodeWav(samples));

  $('meta').textContent =
    `clip ${clipSeconds.toFixed(1)} s · model took ${modelSeconds.toFixed(2)} s`;
  $('heard').hidden = false;
}

// ---- Start-up: show the first letter, download the model, then enable
// the buttons. The letters are tiny, so they appear while the model loads.
try {
  items = await loadItems();
  showItem();
  const vocab = await loadModel((fraction) => { $('progress').value = fraction; });
  checkItems(items, vocab);
  setState('ready', 'Tap and say the sound');
} catch (err) {
  console.error(err);
  // "bad_alloc" is onnxruntime's way of saying the device ran out of memory.
  const reason = /bad_alloc|out of memory/i.test(err.message)
    ? 'this device does not have enough free memory for the model. ' +
      'Close other tabs and apps, then reload.'
    : err.message;
  setState('error', `Could not load: ${reason}`);
}
