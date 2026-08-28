'use strict';

/* Pure Pursuit Challenge - front end.
   The server runs the physics; this file draws a matplotlib-style figure and
   replays the trajectory it was given. */

const SLIDER_META = {
  lookahead: { label: 'lookahead', step: 0.1 },
  max_speed: { label: 'max speed', step: 0.5 },
  heading:   { label: 'heading offset (deg)', step: 1 },
};
const SLIDER_ORDER = ['lookahead', 'max_speed', 'heading'];

const HINT = 'functions: sin cos tan exp log sqrt abs floor sign min max clip where; constants: pi e';

const $ = (id) => document.getElementById(id);

/* ---------- playback timing (pure, unit-tested in test_frontend.js) ---------- */

/** Frame index for a given elapsed playback time. Always a valid index, or
 *  -1 when there are no frames. Time-based rather than accumulated, so it
 *  cannot drift or turn into NaN. */
function frameIndexAt(frames, elapsedSeconds) {
  if (!frames || !frames.t || frames.t.length === 0) return -1;
  const n = frames.t.length;
  if (n === 1) return 0;
  const dt = (frames.t[n - 1] - frames.t[0]) / (n - 1);
  if (!(dt > 0) || !Number.isFinite(elapsedSeconds)) return n - 1;
  const i = Math.floor(elapsedSeconds / dt);
  if (!Number.isFinite(i)) return n - 1;
  return Math.max(0, Math.min(i, n - 1));
}

if (typeof module !== 'undefined' && module.exports) module.exports = { frameIndexAt };

/* ---------- state ---------- */

const canvas = typeof document !== 'undefined' ? $('arena') : null;
const ctx = canvas ? canvas.getContext('2d') : null;

let course = null;
let previewTrack = null;
let currentRun = null;
let playhead = 0;
let playing = false;
let playStart = 0;
let lastEntryId = null;

/* ---------- world -> screen ---------- */

const MARGIN = { left: 38, right: 8, top: 8, bottom: 24 };
let view = { scale: 1, ox: 0, oy: 0 };

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (course) {
    const b = course.bounds;
    const w = rect.width - MARGIN.left - MARGIN.right;
    const h = rect.height - MARGIN.top - MARGIN.bottom;
    // Equal aspect: a square obstacle must look square, or the rotations read wrong.
    view.scale = Math.min(w / (b.xmax - b.xmin), h / (b.ymax - b.ymin));
    view.ox = MARGIN.left + (w - (b.xmax - b.xmin) * view.scale) / 2 - b.xmin * view.scale;
    view.oy = MARGIN.top + (h + (b.ymax - b.ymin) * view.scale) / 2 + b.ymin * view.scale;
  }
  draw();
}

const sx = (x) => view.ox + x * view.scale;
const sy = (y) => view.oy - y * view.scale;

/* ---------- drawing primitives ---------- */

function polyline(xs, ys, color, width, dash, upto) {
  const n = upto === undefined ? (xs ? xs.length : 0) : Math.min(upto, xs.length);
  if (!xs || n < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(sx(xs[0]), sy(ys[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(sx(xs[i]), sy(ys[i]));
  ctx.stroke();
  ctx.restore();
}

function marker(x, y, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(sx(x), sy(y), r, 0, Math.PI * 2);
  ctx.fill();
}

function cross(x, y, r, color, width) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(sx(x) - r, sy(y) - r); ctx.lineTo(sx(x) + r, sy(y) + r);
  ctx.moveTo(sx(x) + r, sy(y) - r); ctx.lineTo(sx(x) - r, sy(y) + r);
  ctx.stroke();
  ctx.restore();
}

/** The car's footprint: a rectangle spanning rear axle to front axle.
 *  Mirrors body_corners() in obstacles.py so what is drawn is exactly what
 *  the server collides. */
function bodyCorners(rx, ry, fx, fy, width) {
  let ax = fx - rx, ay = fy - ry;
  const len = Math.hypot(ax, ay);
  if (len < 1e-12) { ax = 1; ay = 0; } else { ax /= len; ay /= len; }
  const nx = -ay * (width / 2), ny = ax * (width / 2);
  return [[rx + nx, ry + ny], [fx + nx, fy + ny], [fx - nx, fy - ny], [rx - nx, ry - ny]];
}

/* ---------- axes, matplotlib style ---------- */

function drawAxes(b) {
  const x0 = sx(b.xmin), x1 = sx(b.xmax);
  const y0 = sy(b.ymin), y1 = sy(b.ymax);

  ctx.save();
  ctx.font = '10px "DejaVu Sans", Verdana, sans-serif';
  ctx.fillStyle = '#000';

  // grid
  ctx.strokeStyle = '#b0b0b0';
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let x = b.xmin; x <= b.xmax + 1e-9; x += 10) {
    ctx.moveTo(sx(x), y0); ctx.lineTo(sx(x), y1);
  }
  for (let y = b.ymin; y <= b.ymax + 1e-9; y += 8) {
    ctx.moveTo(x0, sy(y)); ctx.lineTo(x1, sy(y));
  }
  ctx.stroke();

  // ticks and labels
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.beginPath();
  for (let x = b.xmin; x <= b.xmax + 1e-9; x += 10) {
    ctx.moveTo(sx(x), y0); ctx.lineTo(sx(x), y0 + 4);
    ctx.fillText(String(x), sx(x), y0 + 6);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let y = b.ymin; y <= b.ymax + 1e-9; y += 8) {
    ctx.moveTo(x0, sy(y)); ctx.lineTo(x0 - 4, sy(y));
    ctx.fillText(String(y), x0 - 6, sy(y));
  }
  ctx.stroke();

  // spines
  ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);
  ctx.restore();
}

function drawLegend(b) {
  const items = [
    ['line', '#0000ff', 'car / path driven', null],
    ['line', '#000000', 'path', [4, 3]],
    ['line', '#ff0000', 'obstacle / wall', null],
    ['line', '#ff0000', 'lookahead', [3, 3]],
  ];
  const w = 96, lh = 13, h = items.length * lh + 8;
  // upper-left: the upper-right corner is where the finish line and its label sit
  const x = sx(b.xmin) + 6, y = sy(b.ymax) + 6;

  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 0.8;
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
  ctx.font = '10px "DejaVu Sans", Verdana, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  items.forEach(([, color, label, dash], k) => {
    const cy = y + 4 + lh * k + lh / 2;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x + 5, cy); ctx.lineTo(x + 24, cy);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 28, cy);
  });
  ctx.restore();
}

/* ---------- figure ---------- */

function draw() {
  if (!ctx) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!course) return;
  const b = course.bounds;

  drawAxes(b);

  // solid walls: ceiling and floor are lethal, so they are drawn like obstacles
  ctx.save();
  ctx.strokeStyle = '#ff0000';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(sx(b.xmin), sy(b.ymax)); ctx.lineTo(sx(b.xmax), sy(b.ymax));
  ctx.moveTo(sx(b.xmin), sy(b.ymin)); ctx.lineTo(sx(b.xmax), sy(b.ymin));
  ctx.stroke();
  ctx.restore();

  // obstacles: red outlines only, drawn from the server's rotated corners.
  // Clipped to the arena so pillars anchored to a wall stop at it.
  ctx.save();
  ctx.beginPath();
  ctx.rect(sx(b.xmin), sy(b.ymax), (b.xmax - b.xmin) * view.scale,
           (b.ymax - b.ymin) * view.scale);
  ctx.clip();
  ctx.strokeStyle = '#ff0000';
  ctx.lineWidth = 1.4;
  for (const o of course.obstacles) {
    const c = o.corners;
    ctx.beginPath();
    ctx.moveTo(sx(c[0][0]), sy(c[0][1]));
    for (let k = 1; k < c.length; k++) ctx.lineTo(sx(c[k][0]), sy(c[k][1]));
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();

  // finish line
  ctx.save();
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(course.finish_x), sy(b.ymin));
  ctx.lineTo(sx(course.finish_x), sy(b.ymax));
  ctx.stroke();
  ctx.restore();
  ctx.save();
  ctx.font = '10px "DejaVu Sans", Verdana, sans-serif';
  ctx.fillStyle = '#000';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText('finish', sx(course.finish_x) - 3, sy(b.ymax) + 3);
  ctx.restore();

  // the path: preview while typing, or the one actually driven
  const track = currentRun ? currentRun.track : previewTrack;
  if (track) polyline(track.xs, track.ys, '#000000', 1.2, [4, 3]);

  drawLegend(b);

  if (!currentRun) return;
  const f = currentRun.frames;
  if (!f || !f.t || f.t.length === 0) return;
  const i = Math.max(0, Math.min(playhead, f.t.length - 1));

  polyline(f.x, f.y, '#0000ff', 1.6, null, i + 1);

  // lookahead
  polyline([f.fx[i], f.lx[i]], [f.fy[i], f.ly[i]], '#ff0000', 1.2, [3, 3]);
  cross(f.lx[i], f.ly[i], 5, '#000000', 1.3);

  // car footprint: the exact rectangle the server collides with
  const body = bodyCorners(f.x[i], f.y[i], f.fx[i], f.fy[i], course.car_width || 1.2);
  polyline(body.map(p => p[0]).concat([body[0][0]]),
           body.map(p => p[1]).concat([body[0][1]]), '#0000ff', 1.4, null);
  // axle line and the two axle positions
  polyline([f.x[i], f.fx[i]], [f.y[i], f.fy[i]], '#000000', 1.6, null);
  marker(f.x[i], f.y[i], 3, '#000000');
  marker(f.fx[i], f.fy[i], 3, '#000000');

  const status = currentRun.result.status;
  if (i >= f.t.length - 1 && (status === 'crashed' || status === 'out_of_bounds')) {
    cross(f.fx[i], f.fy[i], 9, '#ff0000', 2.2);
  }

  $('hudTime').textContent = f.t[i].toFixed(2);
  $('hudSpeed').textContent = f.v[i].toFixed(2);
}

/* ---------- playback ---------- */

function tick(now) {
  if (!playing || !currentRun) return;
  const f = currentRun.frames;
  playhead = frameIndexAt(f, (now - playStart) / 1000);
  draw();
  if (playhead >= f.t.length - 1) {
    playing = false;
    showResult();
    return;
  }
  requestAnimationFrame(tick);
}

function play() {
  if (!currentRun || !currentRun.frames || currentRun.frames.t.length === 0) return;
  playhead = 0;
  playing = true;
  playStart = performance.now();
  $('banner').hidden = true;
  draw();
  requestAnimationFrame(tick);
}

/* ---------- UI ---------- */

function buildSliders(limits) {
  const host = $('sliders');
  host.textContent = '';
  for (const key of SLIDER_ORDER) {
    const lim = limits[key];
    const meta = SLIDER_META[key];
    if (!lim || !meta) continue;

    const wrap = document.createElement('div');
    wrap.className = 'slider';
    const row = document.createElement('div');
    row.className = 'row';
    const name = document.createElement('span');
    name.textContent = meta.label;
    const value = document.createElement('b');
    row.append(name, value);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = lim.min; input.max = lim.max; input.step = meta.step;
    input.value = lim.default;
    input.dataset.key = key;
    const render = () => { value.textContent = (+input.value).toFixed(1); };
    input.addEventListener('input', render);
    render();

    wrap.append(row, input);
    host.append(wrap);
  }
}

function buildPresets(examples) {
  const host = $('presets');
  host.textContent = '';
  for (const { name, equation } of examples) {
    const button = document.createElement('button');
    button.className = 'chip';
    button.type = 'button';
    button.textContent = name;
    button.title = 'y = ' + equation;
    button.addEventListener('click', () => {
      $('equation').value = equation;
      schedulePreview();
    });
    host.append(button);
  }
}

function readParams() {
  const params = {};
  for (const input of document.querySelectorAll('#sliders input[type=range]')) {
    params[input.dataset.key] = parseFloat(input.value);
  }
  return params;
}

function renderBoard(entries, stats) {
  const list = $('board');
  list.textContent = '';
  for (const entry of entries) {
    const li = document.createElement('li');
    if (entry.id === lastEntryId) li.className = 'you';
    if (entry.status !== 'finished') li.classList.add('dnf');
    const rank = document.createElement('span');
    rank.textContent = entry.rank + '.';
    // textContent throughout: player names are never parsed as markup.
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = entry.name;
    who.title = entry.equation + '  (' + entry.status + ')';
    const done = document.createElement('span');
    done.className = 'done';
    done.textContent = entry.completion.toFixed(0) + '%';
    const time = document.createElement('span');
    time.className = 'time';
    time.textContent = entry.time === null ? 'n/a' : entry.time.toFixed(2) + 's';
    li.append(rank, who, done, time);
    list.append(li);
  }
  $('boardEmpty').hidden = entries.length > 0;
  $('boardHead').hidden = entries.length === 0;
  if (stats) {
    $('stats').textContent =
      `${stats.players} players, ${stats.attempts} attempts, ${stats.finishes} finishes`;
  }
}

function showResult() {
  const banner = $('banner');
  const r = currentRun.result;
  banner.textContent = '';
  banner.hidden = false;
  if (r.status === 'finished') {
    const b = document.createElement('b');
    b.textContent = r.time.toFixed(2) + 's';
    banner.append('Finished in ', b);
    if (currentRun.rank) banner.append(' — rank ' + currentRun.rank);
  } else {
    banner.textContent = `${r.message} Reached ${(r.progress * 100).toFixed(0)}% of the way.`;
  }
}

async function api(path, body) {
  const options = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ ok: false, error: 'Bad response from server.' }));
  if (!response.ok || !data.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try {
      const data = await api('/api/preview', { equation: $('equation').value });
      previewTrack = data.track;
      currentRun = null;
      playing = false;
      $('replay').disabled = true;
      $('equation').classList.remove('bad');
      $('eqHint').classList.remove('error');
      $('eqHint').textContent = HINT;
      draw();
    } catch (err) {
      $('equation').classList.add('bad');
      $('eqHint').classList.add('error');
      $('eqHint').textContent = err.message;
    }
  }, 120);
}

async function doRun() {
  const button = $('run');
  button.disabled = true;
  try {
    const data = await api('/api/run', {
      name: $('name').value,
      equation: $('equation').value,
      params: readParams(),
    });
    currentRun = { track: data.track, frames: data.frames, result: data.result, rank: data.rank };
    lastEntryId = data.entry_id;
    renderBoard(data.entries, data.stats);
    $('replay').disabled = false;
    play();
  } catch (err) {
    const banner = $('banner');
    banner.hidden = false;
    banner.textContent = err.message;
  } finally {
    button.disabled = false;
  }
}

/* ---------- boot ---------- */

async function init() {
  const data = await api('/api/course');
  course = data.course;
  $('hudCourse').textContent = course.name;
  $('equation').value = course.default_equation;
  $('eqHint').textContent = HINT;
  buildSliders(data.limits);
  buildPresets(data.examples || []);

  const board = await api('/api/scoreboard');
  renderBoard(board.entries, board.stats);

  resize();
  schedulePreview();
}

if (typeof document !== 'undefined' && canvas) {
  $('equation').addEventListener('input', schedulePreview);
  $('run').addEventListener('click', doRun);
  $('replay').addEventListener('click', play);
  $('name').addEventListener('keydown', (e) => { if (e.key === 'Enter') doRun(); });
  $('equation').addEventListener('keydown', (e) => { if (e.key === 'Enter') doRun(); });
  window.addEventListener('resize', resize);

  init().catch((err) => {
    const banner = $('banner');
    banner.hidden = false;
    banner.textContent = 'Could not reach the server: ' + err.message;
  });
}
