// audio.js — turns any audio the browser can play into what the model wants:
// 16 kHz, mono, Float32Array of samples in the range -1..1.
//
// Used for uploaded files (spike.html) and microphone recordings (recorder.js).

export const SAMPLE_RATE = 16000; // the model only understands 16 kHz audio

/**
 * Decode an audio file (m4a, mp3, wav, ogg, webm ... whatever this browser
 * can play) and resample it to 16 kHz mono.
 *
 * @param {Blob} blob a File from <input type="file"> or a recorded Blob.
 * @returns {Promise<Float32Array>}
 */
export async function decodeTo16k(blob) {
  const encoded = await blob.arrayBuffer();

  // Step 1: decode at whatever sample rate the file has (often 44.1/48 kHz).
  // An OfflineAudioContext is used only as a decoder here; its own settings
  // (1 channel, 1 sample, 16 kHz) do not affect the decoded result's length.
  const decoder = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await decoder.decodeAudioData(encoded);

  // Step 2: resample to 16 kHz and mix stereo down to mono.
  return renderAt16k(decoded);
}

/**
 * Resample raw mono samples (e.g. a 48 kHz microphone recording) to 16 kHz.
 *
 * @param {Float32Array} samples mono audio in the range -1..1.
 * @param {number} fromRate the sample rate `samples` was captured at.
 * @returns {Promise<Float32Array>}
 */
export async function resampleTo16k(samples, fromRate) {
  if (!samples.length) return new Float32Array(0);
  const buffer = new AudioBuffer({ length: samples.length, sampleRate: fromRate });
  buffer.copyToChannel(samples, 0);
  return renderAt16k(buffer);
}

// Play an AudioBuffer into a 16 kHz mono offline context. The browser does
// the resampling (and the stereo-to-mono mixdown) for us.
async function renderAt16k(buffer) {
  const length = Math.max(1, Math.ceil(buffer.duration * SAMPLE_RATE));
  const resampler = new OfflineAudioContext(1, length, SAMPLE_RATE);
  const source = resampler.createBufferSource();
  source.buffer = buffer;
  source.connect(resampler.destination);
  source.start();
  const rendered = await resampler.startRendering();
  return rendered.getChannelData(0);
}

/**
 * Wrap 16 kHz mono samples in a WAV file, so a recording can be played back
 * in an <audio> element (and, later, saved by debug mode).
 *
 * @param {Float32Array} samples 16 kHz mono audio in the range -1..1.
 * @returns {Blob} a 16-bit PCM .wav file.
 */
export function encodeWav(samples) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  // Standard 44-byte WAV header: mono, 16 bits per sample.
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true);              // size of the "fmt " section
  view.setUint16(20, 1, true);               // 1 = uncompressed PCM
  view.setUint16(22, 1, true);               // channels
  view.setUint32(24, SAMPLE_RATE, true);     // samples per second
  view.setUint32(28, SAMPLE_RATE * 2, true); // bytes per second
  view.setUint16(32, 2, true);               // bytes per sample
  view.setUint16(34, 16, true);              // bits per sample
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i])); // clip, then scale to int16
    view.setInt16(44 + i * 2, s * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}
