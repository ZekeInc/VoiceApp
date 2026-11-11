// recorder.js — records the microphone and returns what the model wants:
// 16 kHz mono Float32Array samples.
//
// It captures the raw, uncompressed samples (through a small audio worklet)
// rather than using MediaRecorder, because MediaRecorder compresses the
// audio (Opus/AAC) and compression smears the short bursts that tell /b/,
// /t/ and /k/ apart.
//
// Not here yet: stopping automatically when the child goes quiet. For now a
// recording ends on a second tap or when it reaches RECORDER.maxSeconds.

import { RECORDER } from './config.js';
import { resampleTo16k } from './audio.js';

const WORKLET_URL = new URL('./recorder-worklet.js', import.meta.url);

/**
 * Ask for the microphone and start recording.
 *
 * Must be called from a tap or click, or the browser will refuse the
 * microphone. The promise resolves once sound is actually being captured,
 * so the UI should only show "recording" after it resolves.
 *
 * @param {object} [callbacks]
 * @param {(level: number) => void} [callbacks.onLevel] called about 50 times
 *        a second with the current loudness, 0 (silent) to 1 (loud), for
 *        drawing a level meter.
 * @param {() => void} [callbacks.onLimit] called once if the recording
 *        reaches RECORDER.maxSeconds. The caller should then call stop().
 * @returns {Promise<{stop: () => Promise<Float32Array>}>} call stop() to end
 *        the recording and get the 16 kHz mono samples.
 */
export async function startRecording({ onLevel = () => {}, onLimit = () => {} } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    // Browsers hide the microphone on plain http:// pages (except localhost).
    throw new Error('The microphone needs an https:// page (or localhost).');
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        noiseSuppression: RECORDER.noiseSuppression,
        autoGainControl: RECORDER.autoGainControl,
        echoCancellation: RECORDER.echoCancellation,
      },
    });
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      throw new Error('The microphone is blocked. Allow it in the browser settings for this page.');
    }
    if (err.name === 'NotFoundError') throw new Error('No microphone was found.');
    throw err;
  }

  // The audio context runs at the microphone's own rate (usually 48 kHz);
  // the recording is converted to 16 kHz at the end. Asking the browser for
  // a 16 kHz context directly is not reliable in every browser.
  const context = new AudioContext();
  let source, node;
  try {
    await context.audioWorklet.addModule(WORKLET_URL);
    await context.resume();
    source = context.createMediaStreamSource(stream);
    node = new AudioWorkletNode(context, 'recorder-processor', {
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: 'explicit', // mix a stereo microphone down to mono
    });
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    context.close();
    throw err;
  }

  const blocks = [];      // every 128-sample block received so far
  let sampleCount = 0;    // total samples in `blocks`
  let peak = 0;           // loudest sample since the last onLevel call
  let blocksSinceLevel = 0;
  let stopped = false;
  const maxSamples = RECORDER.maxSeconds * context.sampleRate;

  node.port.onmessage = ({ data }) => {
    if (stopped || sampleCount >= maxSamples) return;
    blocks.push(data);
    sampleCount += data.length;

    // Level meter: report the loudest sample of every 8 blocks (~20 ms).
    for (const s of data) peak = Math.max(peak, Math.abs(s));
    if (++blocksSinceLevel === 8) {
      onLevel(Math.min(1, peak * RECORDER.levelGain));
      peak = 0;
      blocksSinceLevel = 0;
    }

    if (sampleCount >= maxSamples) onLimit();
  };

  // The worklet only runs if it leads to the speakers, so connect it
  // through a volume of zero: it runs, but nothing is played back.
  const mute = context.createGain();
  mute.gain.value = 0;
  source.connect(node).connect(mute).connect(context.destination);

  return {
    async stop() {
      if (stopped) throw new Error('This recording was already stopped');
      stopped = true;
      node.port.onmessage = null;
      source.disconnect();
      // Release the microphone so the browser's "recording" indicator goes
      // away between attempts.
      stream.getTracks().forEach((track) => track.stop());
      await context.close();

      // Join the blocks into one array, then convert to 16 kHz.
      const samples = new Float32Array(sampleCount);
      let offset = 0;
      for (const block of blocks) {
        samples.set(block, offset);
        offset += block.length;
      }
      return resampleTo16k(samples, context.sampleRate);
    },
  };
}
