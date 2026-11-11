// recorder-worklet.js — runs on the browser's audio thread and hands the raw
// microphone samples to recorder.js.
//
// The browser calls process() about every 3 ms with the next 128 samples.
// All this does is copy them and post them to the page; the page stores
// them. It has to be a separate file because audio worklets are loaded by
// URL (audioContext.audioWorklet.addModule).

class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    // inputs[0] is the microphone; [0] again is its first (only) channel.
    // It is missing for a moment while the microphone is starting up.
    const channel = inputs[0][0];
    if (channel) {
      // The browser reuses `channel` for the next block, so send a copy.
      const copy = channel.slice();
      this.port.postMessage(copy, [copy.buffer]);
    }
    return true; // keep running until the node is disconnected
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
