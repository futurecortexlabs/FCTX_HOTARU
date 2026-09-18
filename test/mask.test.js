#!/usr/bin/env node
/* ─────────────────────────────────────────────────────────────────────────────
   test/mask.test.js — hotaru/mask.js
   Plain node script. Builds every mask synthetically, prints one line per case,
   exits non-zero on the first failure count > 0.
   ───────────────────────────────────────────────────────────────────────── */
"use strict";

require("../hotaru/mask.js"); /* IIFE: publishes globalThis.HotaruMask */
const { sampleMask, coverage } = globalThis.HotaruMask;

/* ── harness ────────────────────────────────────────────────────────────── */

let failures = 0;
let checks = 0;
let caseErrors = [];

function check(cond, msg) {
  checks++;
  if (!cond) caseErrors.push(msg);
}
function near(a, b, eps, msg) {
  check(Math.abs(a - b) <= eps, msg + " (got " + a + ", want " + b + " ±" + eps + ")");
}
function testCase(name, fn) {
  caseErrors = [];
  let note = "";
  try {
    note = fn() || "";
  } catch (e) {
    caseErrors.push("threw: " + (e && e.stack ? e.stack.split("\n")[0] : e));
  }
  if (caseErrors.length === 0) {
    console.log("  ok    " + name + (note ? "   " + note : ""));
  } else {
    failures += caseErrors.length;
    console.log("  FAIL  " + name);
    for (const e of caseErrors) console.log("          - " + e);
  }
}

/* ── synthetic masks ────────────────────────────────────────────────────── */

function blank(w, h) {
  return { a: new Uint8Array(w * h), w, h };
}
function put(m, x, y, v) {
  if (x >= 0 && y >= 0 && x < m.w && y < m.h) m.a[y * m.w + x] = v;
}

/* solid filled disc */
function disc(w, h, cx, cy, r, v = 255) {
  const m = blank(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const dx = x + 0.5 - cx,
        dy = y + 0.5 - cy;
      if (dx * dx + dy * dy <= r * r) m.a[y * w + x] = v;
    }
  return m;
}

/* thin annulus, ~2px wall */
function ring(w, h, cx, cy, rOuter, rInner) {
  const m = blank(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const dx = x + 0.5 - cx,
        dy = y + 0.5 - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 <= rOuter * rOuter && d2 >= rInner * rInner) m.a[y * w + x] = 255;
    }
  return m;
}

/* checkerboard of `cell`-sized squares */
function checker(w, h, cell) {
  const m = blank(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if ((((x / cell) | 0) + ((y / cell) | 0)) % 2 === 0) m.a[y * w + x] = 255;
  return m;
}

/* one-pixel-wide diagonal hairline */
function hairline(w, h) {
  const m = blank(w, h);
  for (let x = 0; x < w; x++) {
    const y = Math.floor((x * (h - 1)) / (w - 1));
    put(m, x, y, 255);
  }
  return m;
}

/* soft antialiased edge: a horizontal ramp column-by-column */
function softEdge(w, h) {
  const m = blank(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      m.a[y * w + x] = Math.max(0, Math.min(255, Math.round((x / (w - 1)) * 255)));
  return m;
}

/* asymmetric L: tall bar on the LEFT, foot along the BOTTOM */
function shapeL(w, h, thick) {
  const m = blank(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < thick; x++) m.a[y * w + x] = 255;
  for (let y = h - thick; y < h; y++) for (let x = 0; x < w; x++) m.a[y * w + x] = 255;
  return m;
}

/* ── shared assertions ──────────────────────────────────────────────────── */

/* Every point must sit within `tolPx` pixels of a lit pixel. We test in mask
   pixel space by inverting the world mapping. */
function makeInverse(m, opts) {
  const fit = opts && opts.fit !== undefined ? opts.fit : 1;
  const aspect = opts && opts.aspect !== undefined ? opts.aspect : m.w / m.h;
  const halfW = aspect >= 1 ? fit : fit * aspect;
  const halfH = aspect >= 1 ? fit / aspect : fit;
  return {
    halfW,
    halfH,
    px: (x) => ((x + halfW) / (2 * halfW)) * m.w,
    py: (y) => ((halfH - y) / (2 * halfH)) * m.h
  };
}

function assertInside(m, pts, opts, tolPx = 1, threshold = 16) {
  const inv = makeInverse(m, opts);
  let worst = 0,
    worstAt = -1;
  const n = pts.length / 3;
  for (let i = 0; i < n; i++) {
    const px = inv.px(pts[i * 3]),
      py = inv.py(pts[i * 3 + 1]);
    /* distance in pixels to the nearest lit pixel centre, searched in a small
       neighbourhood — anything further than tolPx+1 is already a failure. */
    const cx = Math.floor(px),
      cy = Math.floor(py);
    let best = Infinity;
    const R = Math.ceil(tolPx) + 1;
    for (let dy = -R; dy <= R; dy++)
      for (let dx = -R; dx <= R; dx++) {
        const gx = cx + dx,
          gy = cy + dy;
        if (gx < 0 || gy < 0 || gx >= m.w || gy >= m.h) continue;
        if (m.a[gy * m.w + gx] < threshold) continue;
        /* distance to that pixel's square, not its centre */
        const qx = Math.max(gx, Math.min(px, gx + 1));
        const qy = Math.max(gy, Math.min(py, gy + 1));
        const d = Math.hypot(px - qx, py - qy);
        if (d < best) best = d;
      }
    if (best > worst) {
      worst = best;
      worstAt = i;
    }
  }
  check(worst <= tolPx, "point " + worstAt + " lies " + worst.toFixed(3) + "px outside the lit region");
  return worst;
}

function assertFinite(pts) {
  for (let i = 0; i < pts.length; i++)
    if (!Number.isFinite(pts[i])) {
      check(false, "non-finite value at index " + i + ": " + pts[i]);
      return;
    }
  check(true, "");
}

function assertShape(pts, count) {
  check(pts instanceof Float32Array, "result is not a Float32Array");
  check(pts.length === count * 3, "length " + pts.length + " != " + count * 3);
}

/* ── cases ──────────────────────────────────────────────────────────────── */

console.log("\nhotaru/mask.js\n");

testCase("filled disc — shape, finiteness, containment", () => {
  const m = disc(128, 128, 64, 64, 50);
  const N = 50000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 7 });
  assertShape(pts, N);
  assertFinite(pts);
  const worst = assertInside(m, pts, {});
  return "worst overshoot " + worst.toFixed(4) + "px";
});

testCase("thin ring (2px wall) — points stay on the wall", () => {
  const m = ring(160, 160, 80, 80, 60, 58);
  const cov = coverage(m.a, m.w, m.h, 16);
  const N = 120000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 11 });
  assertShape(pts, N);
  assertFinite(pts);
  const worst = assertInside(m, pts, {});
  check(cov.litPixels > 500, "ring should be more than 500 px, got " + cov.litPixels);
  return "lit " + cov.litPixels + "px, worst overshoot " + worst.toFixed(4) + "px";
});

testCase("checkerboard — no point in a dark square", () => {
  const m = checker(96, 96, 8);
  const N = 60000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 3 });
  assertShape(pts, N);
  assertFinite(pts);
  /* stricter than the 1px rule: a checkerboard point must be inside a lit cell */
  const inv = makeInverse(m, {});
  let stray = 0;
  for (let i = 0; i < N; i++) {
    const gx = Math.floor(inv.px(pts[i * 3])),
      gy = Math.floor(inv.py(pts[i * 3 + 1]));
    if (gx < 0 || gy < 0 || gx >= m.w || gy >= m.h || m.a[gy * m.w + gx] < 16) stray++;
  }
  check(stray === 0, stray + " points landed off a lit cell");
  /* both colours of the board are sampled in proportion */
  return "stray " + stray;
});

testCase("1px diagonal hairline — every point on the line", () => {
  const m = hairline(200, 120);
  const N = 40000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 5 });
  assertShape(pts, N);
  assertFinite(pts);
  const inv = makeInverse(m, {});
  let stray = 0;
  for (let i = 0; i < N; i++) {
    const gx = Math.floor(inv.px(pts[i * 3])),
      gy = Math.floor(inv.py(pts[i * 3 + 1]));
    if (gx < 0 || gy < 0 || gx >= m.w || gy >= m.h || m.a[gy * m.w + gx] < 16) stray++;
  }
  check(stray === 0, stray + " points left the hairline");
  const cov = coverage(m.a, m.w, m.h, 16);
  return "lit " + cov.litPixels + "px, " + (N / cov.litPixels).toFixed(0) + " pts/px, stray " + stray;
});

testCase("soft antialiased edge — density tracks alpha", () => {
  const m = softEdge(64, 32);
  const N = 200000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 21, threshold: 16 });
  assertShape(pts, N);
  assertFinite(pts);
  const inv = makeInverse(m, {});
  /* count per column, compare against alpha-proportional expectation */
  const col = new Float64Array(m.w);
  for (let i = 0; i < N; i++) {
    const gx = Math.floor(inv.px(pts[i * 3]));
    if (gx >= 0 && gx < m.w) col[gx]++;
  }
  let totalW = 0;
  const wcol = new Float64Array(m.w);
  for (let x = 0; x < m.w; x++) {
    const a = m.a[x];
    if (a >= 16) {
      wcol[x] = a * m.h;
      totalW += wcol[x];
    }
  }
  /* Each pixel receives floor or ceil of its own quota, so a column of h
     pixels can only drift by at most h points from the ideal, whatever the
     count. Assert that hard bound, and report the relative error. */
  let maxRel = 0,
    maxAbs = 0;
  for (let x = 0; x < m.w; x++) {
    if (wcol[x] === 0) {
      check(col[x] === 0, "column " + x + " is below threshold but got " + col[x] + " points");
      continue;
    }
    const exp = (N * wcol[x]) / totalW;
    const dev = Math.abs(col[x] - exp);
    check(dev <= m.h + 1, "column " + x + " is off by " + dev.toFixed(1) + " points (bound " + (m.h + 1) + ")");
    maxAbs = Math.max(maxAbs, dev);
    if (exp < 200) continue;
    maxRel = Math.max(maxRel, dev / exp);
  }
  check(maxRel < 0.07, "column density deviates from alpha by " + (maxRel * 100).toFixed(2) + "%");
  /* the faintest lit column must get far fewer points than the solid one */
  const faint = col[4],
    solid = col[m.w - 1];
  check(faint * 4 < solid, "antialiased column not proportionally sparser (" + faint + " vs " + solid + ")");
  return (
    "max column drift " + maxAbs.toFixed(1) + " pts (bound " + (m.h + 1) + "), " +
    (maxRel * 100).toFixed(2) + "% rel, faint " + faint + " vs solid " + solid
  );
});

testCase("L-shaped mask — aspect preserved, y points up", () => {
  /* 200 wide, 100 tall: bar on the LEFT, foot along the BOTTOM of the image */
  const m = shapeL(200, 100, 20);
  const N = 40000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 2, fit: 1 });

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = pts[i * 3],
      y = pts[i * 3 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  /* aspect 2:1 → box is [-1,1] x [-0.5,0.5] */
  near(minX, -1, 0.01, "minX");
  near(maxX, 1, 0.01, "maxX");
  near(minY, -0.5, 0.01, "minY");
  near(maxY, 0.5, 0.01, "maxY");
  const extentRatio = (maxX - minX) / (maxY - minY);
  near(extentRatio, 2, 0.05, "extent aspect ratio");

  /* orientation: the foot is at the BOTTOM of the image, so the lower half of
     the world box must hold far more points on the right-hand side. */
  let lowerRight = 0,
    upperRight = 0;
  for (let i = 0; i < N; i++) {
    const x = pts[i * 3],
      y = pts[i * 3 + 1];
    if (x > 0.3) {
      if (y < 0) lowerRight++;
      else upperRight++;
    }
  }
  check(lowerRight > 100 * upperRight + 100, "y is upside down: lowerRight=" + lowerRight + " upperRight=" + upperRight);
  check(upperRight === 0, "points found where the L has no ink: " + upperRight);

  /* and the tall bar is on the LEFT in world space too */
  let left = 0;
  for (let i = 0; i < N; i++) if (pts[i * 3] < -0.85 && pts[i * 3 + 1] > 0.2) left++;
  check(left > 1000, "left bar missing in the upper-left quadrant (" + left + ")");
  return "box x[" + minX.toFixed(3) + "," + maxX.toFixed(3) + "] y[" + minY.toFixed(3) + "," + maxY.toFixed(3) + "], lowerRight/upperRight " + lowerRight + "/" + upperRight;
});

testCase("explicit aspect + fit opts respected", () => {
  /* full-bleed mask so the point cloud reaches the edges of the world box */
  const m = blank(100, 100);
  m.a.fill(255);
  const N = 20000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 4, fit: 2.5, aspect: 0.5, depth: 0.25 });
  let maxAx = 0,
    maxAy = 0,
    maxAz = 0;
  for (let i = 0; i < N; i++) {
    maxAx = Math.max(maxAx, Math.abs(pts[i * 3]));
    maxAy = Math.max(maxAy, Math.abs(pts[i * 3 + 1]));
    maxAz = Math.max(maxAz, Math.abs(pts[i * 3 + 2]));
  }
  /* aspect < 1 → the tall axis is the one clamped to fit */
  near(maxAy, 2.5, 0.01, "half-height");
  near(maxAx, 1.25, 0.01, "half-width");
  check(maxAz <= 0.25, "z exceeded depth: " + maxAz);
  near(maxAz, 0.25, 0.01, "z half-extent");
  /* depth defaults to 0.02 */
  const d = sampleMask(m.a, m.w, m.h, 5000, { seed: 4 });
  let mz = 0;
  for (let i = 0; i < 5000; i++) mz = Math.max(mz, Math.abs(d[i * 3 + 2]));
  near(mz, 0.02, 0.001, "default depth");
  return "halfW " + maxAx.toFixed(3) + ", halfH " + maxAy.toFixed(3) + ", |z| " + maxAz.toFixed(3);
});

testCase("determinism under a fixed seed", () => {
  const m = disc(64, 64, 32, 32, 25);
  const a = sampleMask(m.a, m.w, m.h, 30000, { seed: 1234 });
  const b = sampleMask(m.a, m.w, m.h, 30000, { seed: 1234 });
  let diff = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
  check(diff === 0, diff + " values differ between identical calls");
  const c = sampleMask(m.a, m.w, m.h, 30000, { seed: 1235 });
  let moved = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== c[i]) moved++;
  check(moved > a.length * 0.9, "a different seed produced a near-identical field (" + moved + ")");
  /* string seeds work too */
  const s1 = sampleMask(m.a, m.w, m.h, 500, { seed: "ほたる" });
  const s2 = sampleMask(m.a, m.w, m.h, 500, { seed: "ほたる" });
  for (let i = 0; i < s1.length; i++)
    if (s1[i] !== s2[i]) {
      check(false, "string seed is not deterministic");
      break;
    }
  return "identical 0 diffs, reseeded " + ((moved / a.length) * 100).toFixed(1) + "% moved";
});

testCase("library source is a plain, deterministic script", () => {
  const raw = require("fs").readFileSync(require("path").join(__dirname, "..", "hotaru", "mask.js"), "utf8");
  const code = raw.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
  check(!/Math\s*\.\s*random/.test(code), "hotaru/mask.js calls Math.random");
  check(!/\bwindow\b|\bdocument\b/.test(code), "library touches window/document");
  check(!/\brequire\s*\(|^\s*import\s|^\s*export\s/m.test(code), "library is not a plain script");
  check(/root\.HotaruMask\s*=/.test(code), "library does not publish HotaruMask");
  check(/globalThis/.test(code), "IIFE is not passed globalThis");
  return "no Math.random, no DOM, no modules";
});

testCase("degenerate input never throws", () => {
  const m = disc(32, 32, 16, 16, 10);
  const zero = new Uint8Array(32 * 32);

  const allZero = sampleMask(zero, 32, 32, 100, { seed: 1 });
  assertShape(allZero, 100);
  for (let i = 0; i < allZero.length; i++) if (allZero[i] !== 0) check(false, "empty mask must return the origin");

  const w0 = sampleMask(m.a, 0, 32, 10, { seed: 1 });
  assertShape(w0, 10);
  const h0 = sampleMask(m.a, 32, 0, 10, { seed: 1 });
  assertShape(h0, 10);
  for (let i = 0; i < 30; i++) check(w0[i] === 0 && h0[i] === 0, "zero-sized mask must return the origin");

  const c0 = sampleMask(m.a, 32, 32, 0, { seed: 1 });
  assertShape(c0, 0);
  const cNeg = sampleMask(m.a, 32, 32, -5, { seed: 1 });
  assertShape(cNeg, 0);

  const c1 = sampleMask(m.a, 32, 32, 1, { seed: 9 });
  assertShape(c1, 1);
  assertFinite(c1);
  assertInside(m, c1, {});

  /* malformed and missing arguments */
  assertShape(sampleMask(null, 32, 32, 8, {}), 8);
  assertShape(sampleMask(new Uint8Array(4), 32, 32, 8, {}), 8); /* too short */
  assertShape(sampleMask(m.a, 32, 32, 8), 8); /* no opts at all */
  assertShape(sampleMask(m.a, 32, 32, 8, { fit: NaN, aspect: 0, jitter: 99, depth: NaN, threshold: -3 }), 8);
  check(Number.isFinite(sampleMask(m.a, 32, 32, 8, { fit: NaN, jitter: 99 })[0]), "bad opts produced non-finite output");

  /* threshold above every alpha == empty mask */
  const over = sampleMask(m.a, 32, 32, 50, { seed: 1, threshold: 300 });
  for (let i = 0; i < over.length; i++) if (over[i] !== 0) check(false, "threshold 300 should empty the mask");

  return "all handled";
});

testCase("single lit pixel, 4096 points — spread, not stacked", () => {
  const m = blank(16, 16);
  put(m, 5, 9, 255);
  const N = 4096;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 8 });
  assertShape(pts, N);
  assertFinite(pts);
  assertInside(m, pts, {}, 0.001);

  /* every point must be inside pixel (5,9); the sub-pixel positions should
     fill a 16x16 sub-grid with no empty cell and low variance */
  const inv = makeInverse(m, {});
  const sub = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) {
    const px = inv.px(pts[i * 3]) - 5,
      py = inv.py(pts[i * 3 + 1]) - 9;
    check(px >= -1e-4 && px < 1.0001 && py >= -1e-4 && py < 1.0001, "point escaped the single lit pixel");
    sub[i * 2] = px;
    sub[i * 2 + 1] = py;
  }

  /* occupancy of a BxB sub-grid, as chi2/dof (1.0 == Poisson == white noise) */
  function grid(B, xy) {
    const c = new Int32Array(B * B);
    for (let i = 0; i < N; i++) {
      const cx = Math.min(B - 1, Math.max(0, Math.floor(xy[i * 2] * B)));
      const cy = Math.min(B - 1, Math.max(0, Math.floor(xy[i * 2 + 1] * B)));
      c[cy * B + cx]++;
    }
    const mean = N / (B * B);
    let v = 0,
      lo = Infinity,
      hi = 0;
    for (let i = 0; i < B * B; i++) {
      v += (c[i] - mean) ** 2;
      lo = Math.min(lo, c[i]);
      hi = Math.max(hi, c[i]);
    }
    return { chi2: v / (B * B) / mean, lo, hi, mean };
  }

  const g16 = grid(16, sub);
  check(g16.lo > 0, "a 16x16 sub-pixel cell was left empty (min " + g16.lo + ")");

  /* at the scale the default jitter operates on (quarter-pixel cells) the
     spread must be markedly flatter than white noise */
  const g4 = grid(4, sub);
  check(g4.chi2 < 0.25, "quarter-pixel spread is barely better than Poisson (chi2/dof " + g4.chi2.toFixed(4) + ")");

  /* jitter: 0 is the strictly stratified limit and must be flatter still */
  const strict = sampleMask(m.a, m.w, m.h, N, { seed: 8, jitter: 0 });
  const ssub = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) {
    ssub[i * 2] = inv.px(strict[i * 3]) - 5;
    ssub[i * 2 + 1] = inv.py(strict[i * 3 + 1]) - 9;
  }
  const s16 = grid(16, ssub);
  check(s16.chi2 < 0.1, "jitter:0 should be near-perfectly stratified, chi2/dof " + s16.chi2.toFixed(4));
  check(s16.lo > 0, "jitter:0 left a 16x16 cell empty");

  return (
    "16x16 min " + g16.lo + " max " + g16.hi + " (mean " + g16.mean.toFixed(0) + "), " +
    "chi2/dof 4x4 " + g4.chi2.toFixed(4) + ", jitter:0 16x16 " + s16.chi2.toFixed(4)
  );
});

testCase("count >> lit pixels — exact quota per pixel, no stacking", () => {
  const m = disc(48, 48, 24, 24, 18);
  const cov = coverage(m.a, m.w, m.h, 16);
  const N = 400000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 6 });
  const inv = makeInverse(m, {});
  const per = new Int32Array(m.w * m.h);
  for (let i = 0; i < N; i++) {
    const gx = Math.floor(inv.px(pts[i * 3])),
      gy = Math.floor(inv.py(pts[i * 3 + 1]));
    per[gy * m.w + gx]++;
  }
  /* uniform alpha → every lit pixel deserves N/lit points, floor or ceil */
  const ideal = N / cov.litPixels;
  let bad = 0,
    lo = Infinity,
    hi = 0;
  for (let i = 0; i < per.length; i++) {
    if (m.a[i] < 16) {
      if (per[i] !== 0) bad++;
      continue;
    }
    lo = Math.min(lo, per[i]);
    hi = Math.max(hi, per[i]);
    if (per[i] < Math.floor(ideal) - 1 || per[i] > Math.ceil(ideal) + 1) bad++;
  }
  check(bad === 0, bad + " pixels are off quota");
  check(hi - lo <= 2, "per-pixel counts span " + lo + ".." + hi + " (ideal " + ideal.toFixed(2) + ")");
  return cov.litPixels + " lit px, ideal " + ideal.toFixed(2) + "/px, actual " + lo + ".." + hi;
});

testCase("count << lit pixels — no holes", () => {
  const m = disc(256, 256, 128, 128, 120);
  const cov = coverage(m.a, m.w, m.h, 16);
  const N = 600; /* ~75x fewer points than lit pixels */
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 13 });
  assertShape(pts, N);
  assertInside(m, pts, {});
  /* bin into 12x12 world cells; every cell that is mostly ink must be occupied */
  const inv = makeInverse(m, {});
  const B = 12;
  const got = new Int32Array(B * B);
  const inkw = new Float64Array(B * B);
  for (let y = 0; y < m.h; y++)
    for (let x = 0; x < m.w; x++)
      if (m.a[y * m.w + x] >= 16) {
        const bx = Math.min(B - 1, Math.floor((x / m.w) * B));
        const by = Math.min(B - 1, Math.floor((y / m.h) * B));
        inkw[by * B + bx]++;
      }
  for (let i = 0; i < N; i++) {
    const bx = Math.min(B - 1, Math.floor((inv.px(pts[i * 3]) / m.w) * B));
    const by = Math.min(B - 1, Math.floor((inv.py(pts[i * 3 + 1]) / m.h) * B));
    got[by * B + bx]++;
  }
  const cellPx = (m.w / B) * (m.h / B);
  let holes = 0,
    solid = 0;
  for (let i = 0; i < B * B; i++) {
    if (inkw[i] > cellPx * 0.5) {
      solid++;
      if (got[i] === 0) holes++;
    }
  }
  check(holes === 0, holes + " of " + solid + " ink-filled cells received no point");
  return cov.litPixels + " lit px, " + N + " points, " + solid + " ink cells, " + holes + " holes";
});

testCase("coverage() — litPixels, weight, bbox", () => {
  const m = blank(40, 20);
  for (let y = 4; y < 12; y++) for (let x = 6; x < 30; x++) m.a[y * 40 + x] = 128;
  const cov = coverage(m.a, 40, 20, 16);
  check(cov.litPixels === 8 * 24, "litPixels " + cov.litPixels);
  near(cov.weight, (8 * 24 * 128) / 255, 1e-6, "weight");
  check(cov.bbox.x0 === 6 && cov.bbox.y0 === 4, "bbox origin " + JSON.stringify(cov.bbox));
  check(cov.bbox.x1 === 30 && cov.bbox.y1 === 12, "bbox end " + JSON.stringify(cov.bbox));
  check(cov.bbox.width === 24 && cov.bbox.height === 8, "bbox size " + JSON.stringify(cov.bbox));

  /* threshold excludes the faint band */
  for (let x = 0; x < 40; x++) m.a[18 * 40 + x] = 10;
  const c2 = coverage(m.a, 40, 20, 16);
  check(c2.litPixels === 8 * 24, "faint band should be below threshold");
  const c3 = coverage(m.a, 40, 20, 8);
  check(c3.litPixels === 8 * 24 + 40, "faint band should count at threshold 8");
  check(c3.bbox.y1 === 19, "bbox should grow to the faint band, got " + c3.bbox.y1);

  /* degenerate */
  const e1 = coverage(new Uint8Array(100), 10, 10, 16);
  check(e1.litPixels === 0 && e1.weight === 0 && e1.bbox.width === 0, "empty mask coverage");
  const e2 = coverage(null, 10, 10, 16);
  check(e2.litPixels === 0, "null alpha coverage");
  const e3 = coverage(m.a, 0, 0, 16);
  check(e3.litPixels === 0, "zero-size coverage");
  return "lit " + cov.litPixels + ", weight " + cov.weight.toFixed(2) + ", bbox " + cov.bbox.width + "x" + cov.bbox.height;
});

testCase("any prefix of the output is itself an even point set", () => {
  const m = disc(128, 128, 64, 64, 60);
  const N = 200000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 31 });
  const inv = makeInverse(m, {});
  const B = 16;
  /* take the first 5% of the array and check it still covers the whole disc */
  const P = Math.floor(N * 0.05);
  const hit = new Int32Array(B * B);
  for (let i = 0; i < P; i++) {
    const bx = Math.min(B - 1, Math.floor((inv.px(pts[i * 3]) / m.w) * B));
    const by = Math.min(B - 1, Math.floor((inv.py(pts[i * 3 + 1]) / m.h) * B));
    hit[by * B + bx]++;
  }
  const inkw = new Float64Array(B * B);
  let wtot = 0;
  for (let y = 0; y < m.h; y++)
    for (let x = 0; x < m.w; x++)
      if (m.a[y * m.w + x] >= 16) {
        const bx = Math.min(B - 1, Math.floor((x / m.w) * B));
        const by = Math.min(B - 1, Math.floor((y / m.h) * B));
        inkw[by * B + bx]++;
        wtot++;
      }
  let acc = 0,
    dof = 0;
  for (let i = 0; i < B * B; i++) {
    const e = (P * inkw[i]) / wtot;
    if (e < 25) continue;
    acc += ((hit[i] - e) * (hit[i] - e)) / e;
    dof++;
  }
  const chi = acc / dof;
  check(chi < 0.5, "the first 5% of the array is not evenly spread (chi2/dof " + chi.toFixed(3) + ")");
  /* and it must not be a top-to-bottom wipe: y must span the whole disc */
  let minY = Infinity,
    maxY = -Infinity;
  for (let i = 0; i < P; i++) {
    minY = Math.min(minY, pts[i * 3 + 1]);
    maxY = Math.max(maxY, pts[i * 3 + 1]);
  }
  check(maxY - minY > 0.9, "prefix covers only a band of the image (" + (maxY - minY).toFixed(3) + ")");
  return "first " + P + " pts: chi2/dof " + chi.toFixed(3) + ", y span " + (maxY - minY).toFixed(3);
});

testCase("accepts Uint8ClampedArray, subarrays and plain arrays", () => {
  const m = disc(64, 64, 32, 32, 26);
  const base = sampleMask(m.a, m.w, m.h, 5000, { seed: 17 });

  const clamped = Uint8ClampedArray.from(m.a);
  const c = sampleMask(clamped, m.w, m.h, 5000, { seed: 17 });
  let diff = 0;
  for (let i = 0; i < base.length; i++) if (base[i] !== c[i]) diff++;
  check(diff === 0, "Uint8ClampedArray gave a different result (" + diff + " values)");

  /* a view into a larger RGBA-sized buffer, as the caller will actually pass */
  const big = new Uint8Array(m.a.length + 128);
  big.set(m.a, 64);
  const view = big.subarray(64, 64 + m.a.length);
  const v = sampleMask(view, m.w, m.h, 5000, { seed: 17 });
  for (let i = 0; i < base.length; i++)
    if (base[i] !== v[i]) {
      check(false, "subarray view gave a different result");
      break;
    }

  const plain = Array.from(m.a);
  const p = sampleMask(plain, m.w, m.h, 5000, { seed: 17 });
  for (let i = 0; i < base.length; i++)
    if (base[i] !== p[i]) {
      check(false, "plain array gave a different result");
      break;
    }
  check(coverage(clamped, m.w, m.h, 16).litPixels === coverage(m.a, m.w, m.h, 16).litPixels, "coverage differs by array type");
  return "identical across 4 input types";
});

testCase("uniformity — stratified vs uniform-random pixel picking", () => {
  const m = disc(256, 256, 128, 128, 110);
  const N = 250000;
  const pts = sampleMask(m.a, m.w, m.h, N, { seed: 99 });

  /* naive baseline: pick a lit pixel uniformly at random, jitter inside it.
     (Local to the test — the library itself never calls Math.random.) */
  const litIdx = [];
  for (let i = 0; i < m.a.length; i++) if (m.a[i] >= 16) litIdx.push(i);
  const naive = new Float64Array(N * 2);
  let s = 123456789 >>> 0;
  const rnd = () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
  for (let i = 0; i < N; i++) {
    const p = litIdx[(rnd() * litIdx.length) | 0];
    naive[i * 2] = (p % m.w) + rnd();
    naive[i * 2 + 1] = ((p / m.w) | 0) + rnd();
  }

  /* bin both on a 32x32 grid over the mask, expectation from ink weight */
  const B = 32;
  const inv = makeInverse(m, {});
  const wbin = new Float64Array(B * B);
  let wtot = 0;
  for (let y = 0; y < m.h; y++)
    for (let x = 0; x < m.w; x++) {
      const a = m.a[y * m.w + x];
      if (a < 16) continue;
      const bx = Math.min(B - 1, Math.floor((x / m.w) * B));
      const by = Math.min(B - 1, Math.floor((y / m.h) * B));
      wbin[by * B + bx] += a;
      wtot += a;
    }

  const binOf = (px, py) =>
    Math.min(B - 1, Math.floor((py / m.h) * B)) * B + Math.min(B - 1, Math.floor((px / m.w) * B));

  const hs = new Int32Array(B * B);
  for (let i = 0; i < N; i++) hs[binOf(inv.px(pts[i * 3]), inv.py(pts[i * 3 + 1]))]++;
  const hn = new Int32Array(B * B);
  for (let i = 0; i < N; i++) hn[binOf(naive[i * 2], naive[i * 2 + 1])]++;

  /* chi-square per degree of freedom over well-populated bins */
  function chi2(hist) {
    let acc = 0,
      dof = 0;
    for (let i = 0; i < B * B; i++) {
      const e = (N * wbin[i]) / wtot;
      if (e < 100) continue;
      acc += ((hist[i] - e) * (hist[i] - e)) / e;
      dof++;
    }
    return { v: acc / dof, dof };
  }
  const S = chi2(hs),
    Nv = chi2(hn);

  console.log(
    "        occupancy chi2/dof over " + S.dof + " bins:  stratified " +
      S.v.toFixed(4) + "   uniform-random " + Nv.v.toFixed(4) +
      "   (" + (Nv.v / S.v).toFixed(1) + "x flatter)"
  );
  check(Nv.v > 0.5 && Nv.v < 2.0, "baseline should sit near 1.0, got " + Nv.v.toFixed(3));
  check(S.v < Nv.v / 10, "stratified sampling is not 10x flatter than random (" + S.v.toFixed(4) + " vs " + Nv.v.toFixed(4) + ")");
  return "stratified " + S.v.toFixed(4) + " vs random " + Nv.v.toFixed(4);
});

testCase("performance — 1,000,000 points from a 1024x512 mask", () => {
  const m = blank(1024, 512);
  /* a word-like mask: several soft-edged blobs across the canvas */
  for (let b = 0; b < 6; b++) {
    const cx = 100 + b * 160,
      cy = 256,
      r = 90;
    for (let y = 0; y < 512; y++)
      for (let x = 0; x < 1024; x++) {
        const dx = x - cx,
          dy = y - cy;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < r) {
          const a = Math.min(255, Math.round(255 * Math.min(1, (r - d) / 3)));
          if (a > m.a[y * 1024 + x]) m.a[y * 1024 + x] = a;
        }
      }
  }
  const cov = coverage(m.a, 1024, 512, 16);
  const N = 1000000;

  sampleMask(m.a, 1024, 512, 10000, { seed: 1 }); /* warm up the JIT */
  let best = Infinity;
  let pts = null;
  for (let r = 0; r < 3; r++) {
    const t0 = process.hrtime.bigint();
    pts = sampleMask(m.a, 1024, 512, N, { seed: 42 });
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    best = Math.min(best, ms);
  }
  assertShape(pts, N);
  check(Number.isFinite(pts[0]) && Number.isFinite(pts[N * 3 - 1]), "non-finite output");
  console.log(
    "        " + cov.litPixels.toLocaleString() + " lit px  ->  1,000,000 points in " +
      best.toFixed(1) + " ms  (budget 200 ms)"
  );
  check(best < 200, "took " + best.toFixed(1) + "ms, budget is 200ms");
  return "best of 3: " + best.toFixed(1) + " ms";
});

/* ── summary ────────────────────────────────────────────────────────────── */

console.log(
  "\n" + (failures === 0 ? "PASS" : "FAIL") + "  " + checks + " checks, " + failures + " failures\n"
);
process.exit(failures === 0 ? 0 : 1);
