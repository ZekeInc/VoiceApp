// config.js — every number and switch that is meant to be tuned by hand,
// in one place. Each entry says what it does and which way to move it.
//
// Only the recorder settings exist so far. Scoring thresholds will be added
// here when scorer.js is built.

export const RECORDER = {
  // Longest recording allowed, in seconds. The recording stops by itself
  // when it gets this long, in case the child never taps "stop".
  //   Raise it if words are getting cut off.
  //   Lower it to make the wait for the result shorter: the model needs
  //   roughly 0.5 s of thinking per second of audio on a desktop, and more
  //   on a phone.
  maxSeconds: 4,

  // Shortest recording that is sent to the model, in seconds. Anything
  // shorter is treated as an accidental double tap and thrown away.
  //   Raise it if accidental taps are producing nonsense results.
  //   Lower it if real, quick sounds ("t") are being thrown away.
  minSeconds: 0.3,

  // Browser noise suppression (removes steady background noise).
  //   false = the model hears the raw microphone. Best for short stop
  //           sounds (/b/, /t/, /k/), whose burst can be mistaken for noise
  //           and removed.
  //   true  = try this if the room is noisy and results are poor.
  noiseSuppression: false,

  // Browser automatic gain control (turns quiet voices up).
  //   true  = helps a quiet child or a phone held far away.
  //   false = try this if the start of each sound comes out distorted
  //           while the gain is still adjusting.
  autoGainControl: true,

  // Echo cancellation. Only matters if the target sound is still playing
  // from the speaker while the child records, so it is off: it alters the
  // voice slightly. Turn it on if the app's own audio leaks into recordings.
  echoCancellation: false,

  // How strongly the ring around the record button reacts to loudness.
  // Purely cosmetic.
  //   Raise it if the ring barely moves when the child speaks.
  //   Lower it if the ring is always at full size.
  levelGain: 6,
};
