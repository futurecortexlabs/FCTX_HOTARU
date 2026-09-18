/* test/shapes.test.js — contract and quality tests for hotaru/shapes.js.
 *
 *   node test/shapes.test.js
 *
 * Plain node script: it require()s the library for its global side effect.
 * One line per case, non-zero exit on any failure.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'hotaru', 'shapes.js');
require(SRC);
const S = globalThis.HotaruShapes;

const TAU = Math.PI * 2;
const NAMES = ['sphere', 'galaxy', 'torusKnot', 'wave', 'ring', 'cube', 'heart', 'helix'];
const COUNTS = [1, 2, 3, 17, 1000, 250000];
const HARD_RADIUS = 1.15;   /* contract: nothing may ever exceed this */
const TIGHT_RADIUS = 1.02;  /* intent: everything lives inside the unit sphere */

let passed = 0;
let failed = 0;

function ok(label, cond, detail) {
  if (cond) {
    passed++;
    console.log('  ok   ' + label + (detail ? '  ' + detail : ''));
  } else {
    failed++;
    console.log('  FAIL ' + label + (detail ? '  ' + detail : ''));
  }
  return !!cond;
}

function section(title) {
  console.log('\n== ' + title + ' ' + '='.repeat(Math.max(3, 60 - title.length)));
}

/* ── analysis helpers ───────────────────────────────────────────────────── */

function survey(a) {
  const n = a.length / 3;
  let finite = true, maxR = 0;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < a.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = a[i + k];
      if (!Number.isFinite(v)) finite = false;
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
    const d = a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2];
    if (d > maxR) maxR = d;
  }
  return {
    n, finite, maxR: Math.sqrt(maxR),
    span: n ? [max[0] - min[0], max[1] - min[1], max[2] - min[2]] : [0, 0, 0],
    min, max
  };
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  const ba = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
  const bb = Buffer.from(b.buffer, b.byteOffset, b.byteLength);
  return Buffer.compare(ba, bb) === 0;
}

/* Deterministic sampler for the test itself — no Math.random here either, so a
   failing run can be reproduced exactly. */
function lcg(seed) {
  let s = seed >>> 0 || 1;
  return function () {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* Nearest-neighbour distances over the whole cloud, via a uniform spatial hash,
   sampled at `queries` random points (optionally restricted to a region). */
function nearestNeighbours(a, expectedSpacing, queries, filter) {
  const n = a.length / 3;
  const cs = Math.max(expectedSpacing * 2, 1e-6);
  const map = new Map();
  const key = (x, y, z) => x * 8192 * 8192 + y * 8192 + z;
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(a[3 * i] / cs), Math.floor(a[3 * i + 1] / cs), Math.floor(a[3 * i + 2] / cs));
    let b = map.get(k);
    if (!b) { b = []; map.set(k, b); }
    b.push(i);
  }
  const rnd = lcg(0xc0ffee);
  const out = [];
  let tries = 0;
  while (out.length < queries && tries < queries * 60) {
    tries++;
    const i = Math.floor(rnd() * n);
    const x = a[3 * i], y = a[3 * i + 1], z = a[3 * i + 2];
    if (filter && !filter(x, y, z)) continue;
    const ix = Math.floor(x / cs), iy = Math.floor(y / cs), iz = Math.floor(z / cs);
    let best = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const b = map.get(key(ix + dx, iy + dy, iz + dz));
          if (!b) continue;
          for (let q = 0; q < b.length; q++) {
            const j = b[q];
            if (j === i) continue;
            const d = Math.hypot(a[3 * j] - x, a[3 * j + 1] - y, a[3 * j + 2] - z);
            if (d < best) best = d;
          }
        }
      }
    }
    if (Number.isFinite(best)) out.push(best);
  }
  const mean = out.reduce((p, q) => p + q, 0) / Math.max(1, out.length);
  const varr = out.reduce((p, q) => p + (q - mean) * (q - mean), 0) / Math.max(1, out.length);
  return { count: out.length, mean, cv: Math.sqrt(varr) / mean };
}

/* Circular histogram of the xz-plane angle, restricted to an annulus. */
function angularHistogram(a, rLo, rHi, bins) {
  const h = new Float64Array(bins);
  let total = 0;
  for (let i = 0; i < a.length; i += 3) {
    const r = Math.hypot(a[i], a[i + 2]);
    if (r < rLo || r > rHi) continue;
    let th = Math.atan2(a[i + 2], a[i]);
    if (th < 0) th += TAU;
    h[Math.min(bins - 1, Math.floor((th / TAU) * bins))]++;
    total++;
  }
  return { h, total };
}

function countPeaks(h, smooth, threshold) {
  const bins = h.length;
  const sm = new Float64Array(bins);
  const half = (smooth - 1) / 2;
  let total = 0;
  for (let i = 0; i < bins; i++) total += h[i];
  for (let i = 0; i < bins; i++) {
    let s = 0;
    for (let k = -half; k <= half; k++) s += h[(i + k + bins * 2) % bins];
    sm[i] = s / smooth;
  }
  const mean = total / bins;
  let peaks = 0;
  for (let i = 0; i < bins; i++) {
    const prev = sm[(i - 1 + bins) % bins], cur = sm[i], next = sm[(i + 1) % bins];
    if (cur > prev && cur >= next && cur > mean * threshold) peaks++;
  }
  let lo = Infinity, hi = 0;
  for (let i = 0; i < bins; i++) { if (sm[i] < lo) lo = sm[i]; if (sm[i] > hi) hi = sm[i]; }
  return { peaks, contrast: hi / Math.max(lo, 1e-9) };
}

/* ── the heart's implicit surface, re-stated independently of the library ── */

function heartF(x, y, z) {
  const g = x * x + 2.25 * y * y + z * z - 1;
  return g * g * g - x * x * z * z * z - (9 / 80) * y * y * z * z * z;
}
function heartGradLen(x, y, z) {
  const g = x * x + 2.25 * y * y + z * z - 1;
  const g2 = 3 * g * g, z3 = z * z * z, z2 = z * z;
  const gx = g2 * 2 * x - 2 * x * z3;
  const gy = g2 * 4.5 * y - 0.225 * y * z3;
  const gz = g2 * 2 * z - 3 * z2 * (x * x + (9 / 80) * y * y);
  return Math.hypot(gx, gy, gz);
}

/* ══ 1. source hygiene ═══════════════════════════════════════════════════ */

section('source hygiene');
{
  const src = fs.readFileSync(SRC, 'utf8');
  /* comments talk about Math.random and the DOM; the code must not use them */
  const code = src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1 ');
  ok('library never calls Math.random', !/Math\s*\.\s*random/.test(code));
  ok('library has no import/export/require', !/\b(require|import|export)\b\s*[({\s]/.test(code));
  ok('library never touches window or document', !/\b(window|document)\b/.test(code));
  ok('library has no TODO or placeholder markers', !/\b(TODO|FIXME|XXX|PLACEHOLDER)\b/.test(src));
  ok('publishes exactly one global', typeof S === 'object' && S !== null);
}

/* ══ 2. registry ═════════════════════════════════════════════════════════ */

section('registry');
{
  const l = S.list();
  ok('list() order', JSON.stringify(l) === JSON.stringify(NAMES), JSON.stringify(l));
  ok('list() returns a fresh array', S.list() !== S.list());
  const want = {
    sphere: '球', galaxy: '銀河', torusKnot: '結び目', wave: '波',
    ring: '環', cube: '立方体', heart: 'ハート', helix: '螺旋'
  };
  for (const name of NAMES) {
    ok('labels.' + name, S.labels[name] === want[name], S.labels[name]);
    ok('generator ' + name + ' is a function', typeof S[name] === 'function');
  }
  ok('labels has no extra keys', Object.keys(S.labels).length === NAMES.length);
}

/* ══ 3. contract: length, finiteness, bounds, determinism ════════════════ */

const SEED_A = 12345, SEED_B = 98765;

for (const name of NAMES) {
  section(name + ' — contract');
  for (const count of COUNTS) {
    const label = name + '(' + count + ')';
    const a = S[name](count, { seed: SEED_A });
    const s = survey(a);

    ok(label + ' type', a instanceof Float32Array);
    ok(label + ' length === count*3', a.length === count * 3, 'len=' + a.length);
    ok(label + ' all components finite', s.finite);
    ok(label + ' radius <= ' + HARD_RADIUS, s.maxR <= HARD_RADIUS, 'maxR=' + s.maxR.toFixed(5));
    ok(label + ' radius <= ' + TIGHT_RADIUS + ' (fits the unit sphere)', s.maxR <= TIGHT_RADIUS,
      'maxR=' + s.maxR.toFixed(5));

    const again = S[name](count, { seed: SEED_A });
    ok(label + ' same seed is byte-identical', bytesEqual(a, again));
    const other = S[name](count, { seed: SEED_B });
    ok(label + ' different seed differs', !bytesEqual(a, other));
    const strSeeded = S[name](count, { seed: 'hotaru' });
    ok(label + ' string seeds work and differ', strSeeded.length === a.length && !bytesEqual(a, strSeeded));
  }
  /* count 0 and junk counts must not throw */
  ok(name + '(0) is empty', S[name](0, { seed: 1 }).length === 0);
  ok(name + '(-5) is empty', S[name](-5, { seed: 1 }).length === 0);
  ok(name + ' works with no opts', S[name](50).length === 150);
  ok(name + ' unseeded is repeatable', bytesEqual(S[name](50), S[name](50)));
}

section('bad options never poison the buffer');
{
  const junk = [{ arms: 'x' }, { arms: NaN }, { turns: '8' }, { p: {} }, { extent: Infinity },
    { shell: '0.5' }, { seed: {} }, { tube: -1 }, { inner: NaN }, { radius: null }];
  for (const name of NAMES) {
    let clean = true;
    for (const o of junk) {
      const a = S[name](200, Object.assign({ seed: 3 }, o));
      if (a.length !== 600) clean = false;
      for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) clean = false;
    }
    ok(name + ' ignores junk options instead of emitting NaN', clean);
  }
}

/* ══ 4. shape structure ══════════════════════════════════════════════════ */

const BIG = 250000;
const MID = 1000;

/* minimum span per axis at large counts — a shape that collapses on an axis it
   should occupy is degenerate even if every other assertion passes */
const SPAN = {
  sphere: [1.9, 1.9, 1.9],
  galaxy: [1.7, 0.03, 1.7],
  torusKnot: [1.5, 1.5, 0.5],
  wave: [1.3, 0.05, 1.3],
  ring: [1.8, 0.005, 1.8],
  cube: [1.1, 1.1, 1.1],
  heart: [1.4, 1.4, 0.8],
  helix: [0.6, 1.7, 0.6]
};

section('axis spans');
for (const name of NAMES) {
  const s = survey(S[name](BIG, { seed: 4 }));
  const want = SPAN[name];
  ok(name + ' spans all axes it should',
    s.span[0] >= want[0] && s.span[1] >= want[1] && s.span[2] >= want[2],
    'span=' + s.span.map((v) => v.toFixed(3)).join('/') + ' min=' + want.join('/'));
  const m = survey(S[name](MID, { seed: 4 }));
  ok(name + ' still spans at n=' + MID,
    m.span[0] >= want[0] * 0.75 && m.span[1] >= want[1] * 0.6 && m.span[2] >= want[2] * 0.75,
    'span=' + m.span.map((v) => v.toFixed(3)).join('/'));
}

section('sphere — evenness and shell');
{
  const a = S.sphere(BIG, { seed: 21 });
  let rMin = Infinity, rMax = 0;
  for (let i = 0; i < a.length; i += 3) {
    const r = Math.hypot(a[i], a[i + 1], a[i + 2]);
    if (r < rMin) rMin = r;
    if (r > rMax) rMax = r;
  }
  ok('sphere shell=0 is a true shell', rMax - rMin < 1e-4, 'r in [' + rMin.toFixed(6) + ',' + rMax.toFixed(6) + ']');

  for (const n of [MID, BIG]) {
    const p = S.sphere(n, { seed: 21 });
    const expected = Math.sqrt((4 * Math.PI) / n);
    const nn = nearestNeighbours(p, expected, 600);
    ok('sphere n=' + n + ' nearest-neighbour spacing is even (cv < 0.08)', nn.cv < 0.08,
      'cv=' + nn.cv.toFixed(4) + ' mean=' + nn.mean.toFixed(5) + ' expected~' + expected.toFixed(5));
    ok('sphere n=' + n + ' spacing matches the lattice prediction',
      nn.mean > expected * 0.8 && nn.mean < expected * 1.2, 'mean/expected=' + (nn.mean / expected).toFixed(3));
  }

  /* a Poisson cloud of the same size must fail the same evenness test, or the
     test is not measuring anything */
  const rnd = lcg(99);
  const poisson = new Float32Array(BIG * 3);
  for (let i = 0; i < BIG; i++) {
    const z = rnd() * 2 - 1, t = rnd() * TAU, s = Math.sqrt(1 - z * z);
    poisson[3 * i] = s * Math.cos(t); poisson[3 * i + 1] = z; poisson[3 * i + 2] = s * Math.sin(t);
  }
  const pn = nearestNeighbours(poisson, Math.sqrt((4 * Math.PI) / BIG), 600);
  ok('control: a random sphere fails the evenness test', pn.cv > 0.3, 'cv=' + pn.cv.toFixed(3));

  const shelled = S.sphere(BIG, { seed: 21, shell: 0.5 });
  let sMin = Infinity, sMax = 0, inside = 0;
  for (let i = 0; i < shelled.length; i += 3) {
    const r = Math.hypot(shelled[i], shelled[i + 1], shelled[i + 2]);
    if (r < sMin) sMin = r;
    if (r > sMax) sMax = r;
    if (r < 0.75) inside++;
  }
  ok('sphere shell=0.5 sinks inward to ~0.5', sMin < 0.53 && sMin > 0.48, 'rMin=' + sMin.toFixed(4));
  ok('sphere shell=0.5 still reaches the surface', sMax > 0.999, 'rMax=' + sMax.toFixed(4));
  /* uniform in volume => the fraction inside r is (r^3 - 0.5^3)/(1 - 0.5^3) */
  const wantInside = (0.75 ** 3 - 0.5 ** 3) / (1 - 0.5 ** 3);
  ok('sphere shell=0.5 is uniform in volume, not in radius',
    Math.abs(inside / BIG - wantInside) < 0.02,
    'frac=' + (inside / BIG).toFixed(4) + ' expected=' + wantInside.toFixed(4));
}

section('galaxy — arms, core and disc');
{
  for (const arms of [2, 3, 5]) {
    const g = S.galaxy(BIG, { seed: 31, arms });
    const { h, total } = angularHistogram(g, 0.4, 0.47, 64);
    const p = countPeaks(h, 7, 1.15);
    ok('galaxy arms=' + arms + ' shows ' + arms + ' angular peaks', p.peaks === arms,
      'peaks=' + p.peaks + ' contrast=' + p.contrast.toFixed(1) + ' sample=' + total);
    ok('galaxy arms=' + arms + ' arms stand out from the inter-arm gaps', p.contrast > 3,
      'contrast=' + p.contrast.toFixed(2));
  }

  const g = S.galaxy(BIG, { seed: 31 });
  /* radial surface density: core must be far denser than the rim */
  let core = 0, rim = 0, yIn = 0, yOut = 0, nIn = 0, nOut = 0;
  for (let i = 0; i < g.length; i += 3) {
    const r = Math.hypot(g[i], g[i + 2]);
    if (r < 0.2) core++;
    if (r > 0.7 && r < 0.9) rim++;
    if (r > 0.05 && r < 0.2) { yIn += Math.abs(g[i + 1]); nIn++; }
    if (r > 0.7 && r < 0.9) { yOut += Math.abs(g[i + 1]); nOut++; }
  }
  const coreDensity = core / (Math.PI * 0.2 * 0.2);
  const rimDensity = rim / (Math.PI * (0.9 * 0.9 - 0.7 * 0.7));
  ok('galaxy core is much denser than the rim', coreDensity > rimDensity * 8,
    'core/rim=' + (coreDensity / rimDensity).toFixed(1));
  ok('galaxy disc thins outward', yIn / nIn > (yOut / nOut) * 2,
    'h(inner)=' + (yIn / nIn).toFixed(4) + ' h(outer)=' + (yOut / nOut).toFixed(4));
  ok('galaxy is a disc, not a ball', survey(g).span[1] < survey(g).span[0] * 0.25);

  /* the arms must wind: the peak angle at a small radius differs from a large one */
  const inner = angularHistogram(g, 0.3, 0.34, 64);
  const outer = angularHistogram(g, 0.7, 0.78, 64);
  const argmax = (hh) => { let bi = 0; for (let i = 1; i < hh.length; i++) if (hh[i] > hh[bi]) bi = i; return bi; };
  const shift = Math.abs(argmax(inner.h) - argmax(outer.h));
  ok('galaxy arms wind with radius (log spiral)', shift > 1 && shift < 63,
    'peak bin ' + argmax(inner.h) + ' -> ' + argmax(outer.h));
}

section('torusKnot — arc-length spacing and knot geometry');
{
  for (const [p, q] of [[2, 3], [3, 4], [2, 5]]) {
    const a = S.torusKnot(5000, { seed: 41, p, q, tube: 0 });
    /* points are emitted in arc-length order, so consecutive gaps are the
       spacing itself: a curve parameterised by raw t bunches and this explodes */
    const d = [];
    for (let i = 0; i < a.length - 3; i += 3) {
      d.push(Math.hypot(a[i + 3] - a[i], a[i + 4] - a[i + 1], a[i + 5] - a[i + 2]));
    }
    const mu = d.reduce((x, y) => x + y, 0) / d.length;
    const cv = Math.sqrt(d.reduce((x, y) => x + (y - mu) * (y - mu), 0) / d.length) / mu;
    ok('torusKnot (' + p + ',' + q + ') arc spacing is even (cv < 0.02)', cv < 0.02, 'cv=' + cv.toFixed(5));

    /* every point must lie on the torus the knot is drawn on */
    let rxyMax = 0;
    for (let i = 0; i < a.length; i += 3) rxyMax = Math.max(rxyMax, Math.hypot(a[i], a[i + 1]));
    const sc = rxyMax / 3;
    let worst = 0;
    for (let i = 0; i < a.length; i += 3) {
      const rxy = Math.hypot(a[i], a[i + 1]) / sc, z = a[i + 2] / sc;
      worst = Math.max(worst, Math.abs(Math.hypot(rxy - 2, z) - 1));
    }
    ok('torusKnot (' + p + ',' + q + ') lies on its torus', worst < 5e-3, 'residual=' + worst.toExponential(2));

    /* the curve must actually knot: it winds q times around the tube axis */
    let winds = 0, prev = null;
    for (let i = 0; i < a.length; i += 3) {
      const ang = Math.atan2(a[i + 2] / sc, Math.hypot(a[i], a[i + 1]) / sc - 2);
      if (prev !== null) {
        let d2 = ang - prev;
        while (d2 > Math.PI) d2 -= TAU;
        while (d2 < -Math.PI) d2 += TAU;
        winds += d2;
      }
      prev = ang;
    }
    const turns = Math.abs(Math.round(winds / TAU));
    ok('torusKnot (' + p + ',' + q + ') winds q=' + q + ' times around the tube', turns === q, 'turns=' + turns);
  }

  const t = S.torusKnot(BIG, { seed: 41 });
  let onAxis = 0;
  for (let i = 0; i < t.length; i += 3) if (Math.hypot(t[i], t[i + 1]) < 0.2) onAxis++;
  ok('torusKnot has a hole down the middle', onAxis === 0, 'points near the axis: ' + onAxis);
  const tubeRadii = [];
  {
    let rxyMax = 0;
    for (let i = 0; i < t.length; i += 3) rxyMax = Math.max(rxyMax, Math.hypot(t[i], t[i + 1]));
    const sc = (rxyMax - 0.12) / 3;
    for (let i = 0; i < t.length; i += 3000) {
      const rxy = Math.hypot(t[i], t[i + 1]) / sc, z = t[i + 2] / sc;
      tubeRadii.push(Math.abs(Math.hypot(rxy - 2, z) - 1));
    }
  }
  ok('torusKnot tube has thickness', Math.max(...tubeRadii) > 0.1, 'max offset=' + Math.max(...tubeRadii).toFixed(3));
}

section('wave — grid, jitter and interference');
{
  const a = S.wave(BIG, { seed: 51 });
  const s = survey(a);
  ok('wave is a sheet on xz', s.span[1] > 0.05 && s.span[1] < 0.4, 'y span=' + s.span[1].toFixed(3));

  /* a screen door would leave ~sqrt(n) distinct x values; stratified jitter
     leaves essentially n of them */
  const xs = new Set();
  for (let i = 0; i < a.length; i += 3) xs.add(a[i]);
  ok('wave is jittered, not a screen door', xs.size > BIG * 0.5,
    'distinct x=' + xs.size + ' of ' + BIG + ' (a plain grid would give ~' + Math.round(Math.sqrt(BIG)) + ')');

  /* coverage: every macro cell of the sheet holds its share */
  const G = 20, cells = new Float64Array(G * G);
  for (let i = 0; i < a.length; i += 3) {
    const cx = Math.min(G - 1, Math.floor(((a[i] - s.min[0]) / (s.span[0] + 1e-9)) * G));
    const cz = Math.min(G - 1, Math.floor(((a[i + 2] - s.min[2]) / (s.span[2] + 1e-9)) * G));
    cells[cz * G + cx]++;
  }
  let lo = Infinity, hi = 0;
  for (let i = 0; i < cells.length; i++) { if (cells[i] < lo) lo = cells[i]; if (cells[i] > hi) hi = cells[i]; }
  ok('wave covers the sheet evenly', lo > (BIG / (G * G)) * 0.7 && hi < (BIG / (G * G)) * 1.3,
    'cell counts ' + lo + '..' + hi + ' (mean ' + (BIG / (G * G)) + ')');

  /* interference: the height field must have real structure, both crests and
     troughs, and must not be a single ramp */
  let above = 0, below = 0, sum = 0;
  for (let i = 1; i < a.length; i += 3) { sum += a[i]; if (a[i] > 0.02) above++; if (a[i] < -0.02) below++; }
  ok('wave has crests and troughs', above > BIG * 0.15 && below > BIG * 0.15,
    'crest=' + above + ' trough=' + below);
  ok('wave is centred vertically', Math.abs(sum / BIG) < 0.02, 'mean y=' + (sum / BIG).toFixed(4));

  /* changing the frequency changes the field but not the footprint */
  const b = S.wave(BIG, { seed: 51, frequency: 40 });
  ok('wave frequency knob works', !bytesEqual(a, b) && Math.abs(survey(b).span[0] - s.span[0]) < 1e-5);
}

section('ring — flatness, gaps and banding');
{
  const a = S.ring(BIG, { seed: 61 });
  let ymax = 0;
  for (let i = 1; i < a.length; i += 3) ymax = Math.max(ymax, Math.abs(a[i]));
  ok('ring is flat in y', ymax < 0.05, '|y|max=' + ymax.toFixed(4));

  const BINS = 120, inner = 0.45, outer = 0.97;
  const h = new Float64Array(BINS);
  let held = 0;
  for (let i = 0; i < a.length; i += 3) {
    const r = Math.hypot(a[i], a[i + 2]);
    const x = (r - inner) / (outer - inner);
    if (x >= 0 && x < 1) { h[Math.floor(x * BINS)]++; held++; }
  }
  ok('ring is an annulus (nothing inside the hole)', held > BIG * 0.98, 'inside the band: ' + held);

  const sorted = Array.from(h).sort((p, q) => p - q);
  const median = sorted[BINS >> 1];
  const runs = [];
  let cur = null;
  for (let i = 3; i < BINS - 3; i++) {          /* skip the feathered edges */
    if (h[i] < median * 0.3) {
      if (!cur) cur = { from: i, to: i, min: h[i] };
      else { cur.to = i; cur.min = Math.min(cur.min, h[i]); }
    } else if (cur) { runs.push(cur); cur = null; }
  }
  if (cur) runs.push(cur);
  ok('ring has exactly two gaps', runs.length === 2,
    'gaps=' + JSON.stringify(runs.map((r) => [r.from, r.to])));
  ok('ring gaps are deep', runs.every((r) => r.min < median * 0.25),
    'gap minima=' + runs.map((r) => r.min).join(','));
  ok('ring gaps are separated', runs.length === 2 && runs[1].from - runs[0].to > 10);

  /* banding: outside the gaps the density still rises and falls */
  const band = [];
  for (let i = 3; i < BINS - 3; i++) {
    if (!runs.some((r) => i >= r.from - 2 && i <= r.to + 2)) band.push(h[i]);
  }
  const bm = band.reduce((p, q) => p + q, 0) / band.length;
  const bcv = Math.sqrt(band.reduce((p, q) => p + (q - bm) * (q - bm), 0) / band.length) / bm;
  ok('ring has radial density banding', bcv > 0.08 && bcv < 0.6, 'band cv=' + bcv.toFixed(3));

  /* angular coverage must be uniform — a ring with a bite out of it is broken */
  const ah = angularHistogram(a, inner, outer, 72);
  let alo = Infinity, ahi = 0;
  for (let i = 0; i < 72; i++) { if (ah.h[i] < alo) alo = ah.h[i]; if (ah.h[i] > ahi) ahi = ah.h[i]; }
  ok('ring is angularly uniform', ahi / alo < 1.15, 'max/min per 5deg=' + (ahi / alo).toFixed(3));
}

section('cube — faces and stratification');
{
  const half = 0.575;
  for (const n of [MID, BIG]) {
    const a = S.cube(n, { seed: 71 });
    const faces = new Array(6).fill(0);
    let worst = 0, offFace = 0;
    for (let i = 0; i < a.length; i += 3) {
      const v = [a[i], a[i + 1], a[i + 2]];
      let ax = 0;
      for (let k = 1; k < 3; k++) if (Math.abs(v[k]) > Math.abs(v[ax])) ax = k;
      faces[ax * 2 + (v[ax] > 0 ? 1 : 0)]++;
      worst = Math.max(worst, Math.abs(Math.abs(v[ax]) - half));
      for (let k = 0; k < 3; k++) if (Math.abs(v[k]) > half + 1e-6) offFace++;
    }
    ok('cube n=' + n + ' every point is on the surface', worst < 1e-5, 'max deviation=' + worst.toExponential(2));
    ok('cube n=' + n + ' no point escapes the box', offFace === 0);
    const lo = Math.min(...faces), hi = Math.max(...faces);
    ok('cube n=' + n + ' all six faces get an equal share', hi - lo <= 1, 'faces=' + faces.join(','));
  }
  const a = S.cube(BIG, { seed: 71 });
  /* stratification: one point per grid cell, jittered — so within a single face
     the in-plane coordinates are nearly all distinct */
  const vals = new Set();
  for (let i = 0; i < a.length; i += 3) if (Math.abs(a[i + 2] - half) < 1e-6) vals.add(a[i]);
  ok('cube faces are stratified, not a plain lattice', vals.size > (BIG / 6) * 0.5,
    'distinct x on +z face=' + vals.size + ' of ' + Math.round(BIG / 6));
  const s = survey(a);
  ok('cube reaches every face', Math.abs(s.min[0] + half) < 0.01 && Math.abs(s.max[1] - half) < 0.01);
}

section('heart — implicit surface, evenness and orientation');
{
  const a = S.heart(BIG, { seed: 81 });
  const K = S.HEART_SCALE;
  ok('HEART_SCALE is published', typeof K === 'number' && K > 0 && K < 10, 'K=' + K.toFixed(6));

  /* world -> equation space: world = (x_eq, z_eq, -y_eq) * K */
  let worst = 0, worstRaw = 0;
  for (let i = 0; i < a.length; i += 3) {
    const x = a[i] / K, y = -a[i + 2] / K, z = a[i + 1] / K;
    const f = heartF(x, y, z);
    worstRaw = Math.max(worstRaw, Math.abs(f));
    worst = Math.max(worst, Math.abs(f) / heartGradLen(x, y, z));
  }
  ok('heart points satisfy the implicit equation', worst < 2e-3,
    'max |F|/|grad F| = ' + worst.toExponential(2) + ' (raw |F| = ' + worstRaw.toExponential(2) + ')');

  /* evenness: density in the tip half and the lobe half must agree */
  const s = survey(a);
  const yLo = s.min[1], span = s.span[1];
  const spacing = 0.006;
  const tip = nearestNeighbours(a, spacing, 500, (x, y) => y < yLo + span * 0.25);
  const lobe = nearestNeighbours(a, spacing, 500, (x, y) => y > yLo + span * 0.7);
  const ratio = Math.max(tip.mean, lobe.mean) / Math.min(tip.mean, lobe.mean);
  ok('heart is sampled evenly over the surface', ratio < 1.25,
    'tip spacing=' + tip.mean.toFixed(5) + ' lobe spacing=' + lobe.mean.toFixed(5) + ' ratio=' + ratio.toFixed(3));
  const all = nearestNeighbours(a, spacing, 800);
  ok('heart has no clumps', all.cv < 0.3, 'nn cv=' + all.cv.toFixed(3));

  /* the area weighting must actually be doing the work: with it switched off
     the two regions drift apart */
  const raw = S.heart(BIG, { seed: 81, even: false });
  const rawTip = nearestNeighbours(raw, spacing, 500, (x, y) => y < yLo + span * 0.25);
  const rawLobe = nearestNeighbours(raw, spacing, 500, (x, y) => y > yLo + span * 0.7);
  const rawRatio = Math.max(rawTip.mean, rawLobe.mean) / Math.min(rawTip.mean, rawLobe.mean);
  ok('control: without area weighting the density is visibly uneven', rawRatio > ratio * 1.2,
    'unweighted ratio=' + rawRatio.toFixed(3) + ' vs weighted ' + ratio.toFixed(3));

  /* orientation: lobes up, tip down, thin axis facing the viewer */
  const widthAt = (frac, hw) => {
    let lo = Infinity, hi = -Infinity, n = 0;
    const yc = yLo + span * frac;
    for (let i = 0; i < a.length; i += 3) {
      if (Math.abs(a[i + 1] - yc) > span * hw) continue;
      if (a[i] < lo) lo = a[i];
      if (a[i] > hi) hi = a[i];
      n++;
    }
    return { w: hi - lo, n };
  };
  const tipW = widthAt(0.04, 0.02), lobeW = widthAt(0.85, 0.02);
  ok('heart tapers to a point at the bottom', tipW.w < lobeW.w * 0.35,
    'width at tip=' + tipW.w.toFixed(3) + ' at lobes=' + lobeW.w.toFixed(3));
  ok('heart is thin front-to-back', s.span[2] < s.span[0] * 0.8 && s.span[2] < s.span[1] * 0.8,
    'span x/y/z=' + s.span.map((v) => v.toFixed(3)).join('/'));
  ok('heart is left-right symmetric', Math.abs(s.min[0] + s.max[0]) < 1e-3,
    'x in [' + s.min[0].toFixed(4) + ',' + s.max[0].toFixed(4) + ']');

  /* the cleft: on the centre line the surface dips below the top of the lobes */
  let centreTop = -Infinity, lobeTop = -Infinity;
  for (let i = 0; i < a.length; i += 3) {
    if (Math.abs(a[i]) < 0.02 && Math.abs(a[i + 2]) < 0.05) centreTop = Math.max(centreTop, a[i + 1]);
    if (Math.abs(a[i]) > 0.15 && Math.abs(a[i]) < 0.35) lobeTop = Math.max(lobeTop, a[i + 1]);
  }
  ok('heart has a cleft between the lobes', centreTop < lobeTop - 0.05,
    'centre top=' + centreTop.toFixed(3) + ' lobe top=' + lobeTop.toFixed(3));
}

section('helix — strands, rungs and turns');
{
  const R = 0.33;
  const a = S.helix(BIG, { seed: 91 });
  const s = survey(a);
  let rmax = 0, nearAxis = 0;
  for (let i = 0; i < a.length; i += 3) {
    const r = Math.hypot(a[i], a[i + 2]);
    if (r > rmax) rmax = r;
    if (r < R * 0.5) nearAxis++;
  }
  ok('helix stays on its cylinder', rmax < R * 1.15, 'rmax=' + rmax.toFixed(4));
  ok('helix has rungs crossing the middle', nearAxis > BIG * 0.02, 'inner points=' + nearAxis);
  ok('helix is tall', s.span[1] > 1.7, 'y span=' + s.span[1].toFixed(3));

  /* a thin horizontal slab must cut the two strands, and only two */
  const y0 = (s.min[1] + s.max[1]) / 2, hw = s.span[1] * 0.004, B = 48;
  const hist = new Float64Array(B);
  let slab = 0;
  for (let i = 0; i < a.length; i += 3) {
    if (Math.abs(a[i + 1] - y0) > hw) continue;
    if (Math.hypot(a[i], a[i + 2]) < R * 0.85) continue;
    let th = Math.atan2(a[i + 2], a[i]);
    if (th < 0) th += TAU;
    hist[Math.floor((th / TAU) * B)]++;
    slab++;
  }
  let clusters = 0, run = false;
  for (let i = 0; i < B; i++) { const on = hist[i] > 0; if (on && !run) clusters++; run = on; }
  if (hist[0] > 0 && hist[B - 1] > 0 && clusters > 1) clusters--;   /* wrap-around */
  ok('helix is a double helix (two strands in a slab)', clusters === 2,
    'clusters=' + clusters + ' slab points=' + slab);

  /* turns knob: count how often a strand crosses the +x half plane */
  for (const turns of [2, 5]) {
    const b = S.helix(40000, { seed: 91, turns, rungRatio: 0 });
    let crossings = 0;
    const bs = survey(b);
    const steps = 400;
    let prevAng = null;
    for (let k = 0; k < steps; k++) {
      const yc = bs.min[1] + (bs.span[1] * (k + 0.5)) / steps;
      let best = null, bestD = Infinity;
      for (let i = 0; i < b.length; i += 3) {
        const d = Math.abs(b[i + 1] - yc);
        if (d < bestD) { bestD = d; best = i; }
      }
      const ang = Math.atan2(b[best + 2], b[best]);
      if (prevAng !== null) {
        let d2 = ang - prevAng;
        while (d2 > Math.PI) d2 -= TAU;
        while (d2 < -Math.PI) d2 += TAU;
        crossings += d2;
      }
      prevAng = ang;
    }
    const measured = Math.abs(crossings) / TAU;
    ok('helix turns=' + turns + ' really makes ' + turns + ' turns',
      Math.abs(measured - turns) < 0.35, 'measured=' + measured.toFixed(2));
  }

  const noRungs = S.helix(BIG, { seed: 91, rungRatio: 0 });
  let inner2 = 0;
  for (let i = 0; i < noRungs.length; i += 3) if (Math.hypot(noRungs[i], noRungs[i + 2]) < R * 0.5) inner2++;
  ok('helix rungRatio=0 removes the rungs', inner2 === 0, 'inner points=' + inner2);
}

/* ══ 5. scale ════════════════════════════════════════════════════════════ */

section('one million points');
{
  for (const name of NAMES) {
    const t0 = Date.now();
    const a = S[name](1000000, { seed: 7 });
    const ms = Date.now() - t0;
    const s = survey(a);
    ok(name + '(1000000)', a.length === 3000000 && s.finite && s.maxR <= HARD_RADIUS,
      'maxR=' + s.maxR.toFixed(4) + ' in ' + ms + 'ms');
  }
}

/* ── verdict ────────────────────────────────────────────────────────────── */

console.log('\n' + '-'.repeat(64));
console.log(failed === 0
  ? 'PASS  ' + passed + ' checks'
  : 'FAIL  ' + failed + ' of ' + (passed + failed) + ' checks failed');
process.exit(failed === 0 ? 0 : 1);
