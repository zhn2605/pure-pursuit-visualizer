/* Playback-timing tests.  Run with:  node web/test_frontend.js
   These cover the bug that made "Run" appear to do nothing: a fractional
   frame index read undefined out of the frame arrays and threw inside the
   requestAnimationFrame callback, killing the animation and the result banner. */

const assert = require('assert');
const { frameIndexAt } = require('./app.js');

const frames = { t: [] };
for (let k = 0; k < 91; k++) frames.t.push(+(k * 0.02).toFixed(3));
const last = frames.t.length - 1;

// every index must be a valid integer position in the array
for (let ms = 0; ms <= 2600; ms += 7) {
  const i = frameIndexAt(frames, ms / 1000);
  assert.ok(Number.isInteger(i), `index ${i} not an integer at ${ms}ms`);
  assert.ok(i >= 0 && i <= last, `index ${i} out of range at ${ms}ms`);
  assert.notStrictEqual(frames.t[i], undefined, `frames.t[${i}] undefined at ${ms}ms`);
}

// advances monotonically and reaches the end
assert.strictEqual(frameIndexAt(frames, 0), 0);
assert.strictEqual(frameIndexAt(frames, 0.02), 1);
assert.strictEqual(frameIndexAt(frames, 0.03), 1);       // mid-frame floors down
assert.strictEqual(frameIndexAt(frames, last * 0.02), last);
assert.strictEqual(frameIndexAt(frames, 999), last);      // clamps past the end

// degenerate inputs must not produce a bad index
assert.strictEqual(frameIndexAt(frames, -5), 0);          // negative time clamps to 0
assert.strictEqual(frameIndexAt(frames, NaN), last);
assert.strictEqual(frameIndexAt(frames, Infinity), last);
assert.strictEqual(frameIndexAt({ t: [] }, 1), -1);       // no frames
assert.strictEqual(frameIndexAt(null, 1), -1);
assert.strictEqual(frameIndexAt(undefined, 1), -1);
assert.strictEqual(frameIndexAt({ t: [0.0] }, 5), 0);     // single frame

// a run whose frames all share one timestamp must not divide by zero
assert.strictEqual(frameIndexAt({ t: [0, 0, 0] }, 1), 2);

// the old accumulate-and-don't-floor approach produced fractional indices;
// confirm the new one never does, even with jittery frame deltas
let simulated = 0;
for (const dtMs of [16.7, 33.4, 8.1, 120.0, 4.2, 16.7]) {
  simulated += dtMs;
  const i = frameIndexAt(frames, simulated / 1000);
  assert.ok(Number.isInteger(i) && frames.t[i] !== undefined);
}

console.log('frontend playback tests: OK');
