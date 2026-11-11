// audio.js — turns any audio the browser can play into what the model wants:
// 16 kHz, mono, Float32Array of samples in the range -1..1.
//
// Used for uploaded files now, and for microphone recordings later.

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

  // Step 2: play the decoded audio into a 16 kHz mono offline context. The
  // browser does the resampling and the stereo-to-mono mixdown for us.
  const length = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE));
  const resampler = new OfflineAudioContext(1, length, SAMPLE_RATE);
  const source = resampler.createBufferSource();
  source.buffer = decoded;
  source.connect(resampler.destination);
  source.start();
  const rendered = await resampler.startRendering();
  return rendered.getChannelData(0);
}
