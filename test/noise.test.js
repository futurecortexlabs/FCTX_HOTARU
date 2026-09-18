/* ─────────────────────────────────────────────────────────────────────────────
   Hotaru · test/noise.test.js
   ---------------------------------------------------------------------------
   Plain node script. Run with:  node test/noise.test.js

   Covers:
     A. module hygiene          (globals, no Math.random, no DOM)
     B. snoise3                 range / spread / continuity / determinism /
                                isotropy / bounded gradient
     C. the 7x7 gradient table  every one of the 289 permutation outputs maps
                                to a unit-ish gradient (this is the check that
                                catches the float64-vs-float32 floor() trap)
     D. curlNoise               divergence-free, continuous, bounded,
                                non-trivial, varies in all three components
     E. fbm                     range, octave clamping, and that more octaves
                                really do add high-frequency detail
     F. GLSL <-> JS structure   the shader source and the JS mirror are shown
                                to carry the same constants and the same
                                expression shapes
     G. GLSL lint               balanced brackets, declaration order, the
                                PRELUDE contains all three pieces
   ───────────────────────────────────────────────────────────────────────── */
"use strict";

const fs = require("fs");
const path = require("path");

const NOISE_PATH = path.join(__dirname, "..", "hotaru", "noise.js");
const NOISE_SRC = fs.readFileSync(NOISE_PATH, "utf8");

require(NOISE_PATH); // plain script: loaded for the global side effect
const N = globalThis.HotaruNoise;

/* ── tiny harness ──────────────────────────────────────────────────────── */

let passed = 0;
let failed = 0;
let group = "";

function describe(name) {
  group = name;
  console.log("\n" + name);
}

function ok(cond, label, detail) {
  if (cond) {
    passed++;
    console.log("  PASS  " + label + (detail ? "   [" + detail + "]" : ""));
  } else {
    failed++;
    console.log("  FAIL  " + label + (detail ? "   [" + detail + "]" : ""));
  }
}

function near(a, b, tol, label) {
  ok(Math.abs(a - b) <= tol, label, a.toPrecision(8) + " vs " + b + " +/- " + tol);
}

/* ── deterministic PRNG for the sampling (never Math.random) ───────────── */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampler(seed, span) {
  const r = mulberry32(seed);
  return function (out) {
    out[0] = (r() - 0.5) * span;
    out[1] = (r() - 0.5) * span;
    out[2] = (r() - 0.5) * span;
    return out;
  };
}

/* ── source stripping ──────────────────────────────────────────────────────
   Several checks below must look at noise.js as *code*, with comments and
   string literals removed. Both matter: the GLSL lives in string literals
   inside noise.js, and the JS mirror quotes the shader line by line in its
   comments — leaving either in would make the structural comparison in
   section F vacuous, and would make the "no Math.random" check in section A
   trip over the sentence in the file header that promises there is none. */

function jsCodeOnly(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === '"' || c === "'" || c === "`") {
      const quote = c;
      i++;
      while (i < n) {
        if (src[i] === "\\") { i += 2; continue; }
        if (src[i] === quote) { i++; break; }
        i++;
      }
      out += ' "" ';
      continue;
    }
    if (c === "/" && src[i + 1] === "/") {
      while (i < n && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && src[i + 1] === "*") {
      i += 2;
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

const NOISE_CODE = jsCodeOnly(NOISE_SRC);

const P = new Float64Array(3);
const A = new Float64Array(3);
const B = new Float64Array(3);
const CV = new Float64Array(3);
const DV = new Float64Array(3);
const EV = new Float64Array(3);
const FV = new Float64Array(3);
const GV = new Float64Array(3);

/* ═══════════════════════════════════════════════════════════════════════════
   A. module hygiene
   ═══════════════════════════════════════════════════════════════════════════ */

describe("A. module hygiene");

ok(typeof N === "object" && N !== null, "HotaruNoise global is published");
for (const k of [
  "GLSL_SIMPLEX3", "GLSL_FBM3", "GLSL_CURL3", "PRELUDE",
  "snoise3", "fbm3", "curl3", "potential3", "fillSnoise", "fillCurl"
]) {
  ok(N[k] !== undefined, "exports " + k);
}
ok(typeof N.snoise3 === "function" && N.snoise3.length === 3, "snoise3(x,y,z) arity 3");
ok(typeof N.fbm3 === "function" && N.fbm3.length === 4, "fbm3(x,y,z,octaves) arity 4");
ok(typeof N.curl3 === "function", "curl3 is a function");

ok(!/Math\s*\.\s*random/.test(NOISE_CODE), "library code calls no Math.random");
ok(!/\brequire\s*\(/.test(NOISE_CODE), "library code calls no require()");
ok(!/\b(import|export)\b/.test(NOISE_CODE), "library code has no import/export");
ok(!/\bwindow\b|\bdocument\b/.test(NOISE_CODE), "library code never touches window/document");
ok(/\(globalThis\)\s*;?\s*$/.test(NOISE_SRC.trim()), "library is an IIFE closed over globalThis");
ok((NOISE_CODE.match(/root\.\w+\s*=/g) || []).length === 1,
  "library publishes exactly one global");
ok(!/\bTODO\b|\bFIXME\b|\bXXX\b/.test(NOISE_SRC), "library source has no TODO / FIXME markers");

/* ═══════════════════════════════════════════════════════════════════════════
   B. snoise3
   ═══════════════════════════════════════════════════════════════════════════ */

describe("B. snoise3 — range, spread, continuity, determinism");

{
  const M = 120000;
  const next = sampler(0x5eed01, 400);
  let mn = Infinity, mx = -Infinity, sum = 0, sum2 = 0;
  let near1 = 0;
  for (let i = 0; i < M; i++) {
    next(P);
    const v = N.snoise3(P[0], P[1], P[2]);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
    sum2 += v * v;
    if (Math.abs(v) > 0.8) near1++;
  }
  const mean = sum / M;
  const std = Math.sqrt(sum2 / M - mean * mean);

  ok(mn >= -1.02 && mx <= 1.02, "stays inside [-1.02, 1.02] over " + M + " samples",
    "min " + mn.toFixed(5) + "  max " + mx.toFixed(5));
  ok(mn < -0.85 && mx > 0.85, "actually reaches the ends of the range",
    "min " + mn.toFixed(5) + "  max " + mx.toFixed(5));
  ok(std > 0.25 && std < 0.55, "is not collapsed near zero", "std " + std.toFixed(5));
  ok(Math.abs(mean) < 0.02, "is roughly zero-mean", "mean " + mean.toExponential(3));
  ok(near1 / M > 0.002, "produces strong peaks regularly",
    "|v|>0.8 in " + ((100 * near1) / M).toFixed(3) + "% of samples");
}

{
  // Continuity: a 1e-4 step must move the output by less than 1e-2.
  const M = 40000;
  const h = 1e-4;
  const next = sampler(0x5eed02, 200);
  let maxJump = 0;
  for (let i = 0; i < M; i++) {
    next(P);
    const v = N.snoise3(P[0], P[1], P[2]);
    const jx = Math.abs(N.snoise3(P[0] + h, P[1], P[2]) - v);
    const jy = Math.abs(N.snoise3(P[0], P[1] + h, P[2]) - v);
    const jz = Math.abs(N.snoise3(P[0], P[1], P[2] + h) - v);
    const jd = Math.abs(N.snoise3(P[0] + h, P[1] + h, P[2] + h) - v);
    const m = Math.max(jx, jy, jz, jd);
    if (m > maxJump) maxJump = m;
  }
  ok(maxJump < 1e-2, "continuous: a 1e-4 step changes the value by < 1e-2",
    "worst jump " + maxJump.toExponential(3));
}

{
  // Continuity, the hard way. Random sampling almost never lands on a simplex
  // cell boundary, so hunt the boundaries out: walk a segment, find the
  // steepest adjacent pair, bisect onto the crossing and measure what is left
  // when the bracket is narrower than a float. A smooth stretch bisects down
  // to zero; a genuine step does not. The 0.6 kernel radius of the reference
  // implementation leaves a small step here (0.5 would leave none, at the cost
  // of contrast) — this measures it rather than hoping it is not there.
  const SEGMENTS = 2500;
  const STEPS = 1200;
  const r = mulberry32(0x5eed05);
  let maxGap = 0, sumGap = 0;
  for (let s = 0; s < SEGMENTS; s++) {
    const ox = (r() - 0.5) * 200, oy = (r() - 0.5) * 200, oz = (r() - 0.5) * 200;
    let dx = r() - 0.5, dy = r() - 0.5, dz = r() - 0.5;
    const L = Math.hypot(dx, dy, dz);
    dx /= L; dy /= L; dz /= L;
    const f = (t) => N.snoise3(ox + dx * t, oy + dy * t, oz + dz * t);

    let bi = 1, bd = -1, prev = f(0);
    for (let i = 1; i <= STEPS; i++) {
      const v = f(i / STEPS);
      const d = Math.abs(v - prev);
      if (d > bd) { bd = d; bi = i; }
      prev = v;
    }
    let lo = (bi - 1) / STEPS, hi = bi / STEPS;
    for (let it = 0; it < 48; it++) {
      const m = (lo + hi) / 2;
      if (Math.abs(f(m) - f(lo)) > Math.abs(f(hi) - f(m))) hi = m; else lo = m;
    }
    const gap = Math.abs(f(hi) - f(lo));
    if (gap > maxGap) maxGap = gap;
    sumGap += gap;
  }
  ok(maxGap < 1e-2, "even on a hunted simplex cell boundary the step stays under 1e-2",
    "worst " + maxGap.toExponential(3) + "  mean " + (sumGap / SEGMENTS).toExponential(3) +
    " over " + SEGMENTS + " bisected crossings");
  ok(sumGap / SEGMENTS < 1e-3, "the typical boundary step is negligible",
    "mean " + (sumGap / SEGMENTS).toExponential(3));
}

{
  // Gradient magnitude bounded.
  const M = 30000;
  const e = 1e-3;
  const next = sampler(0x5eed03, 200);
  let maxG = 0, sumG = 0;
  for (let i = 0; i < M; i++) {
    next(P);
    const gx = (N.snoise3(P[0] + e, P[1], P[2]) - N.snoise3(P[0] - e, P[1], P[2])) / (2 * e);
    const gy = (N.snoise3(P[0], P[1] + e, P[2]) - N.snoise3(P[0], P[1] - e, P[2])) / (2 * e);
    const gz = (N.snoise3(P[0], P[1], P[2] + e) - N.snoise3(P[0], P[1], P[2] - e)) / (2 * e);
    const g = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (g > maxG) maxG = g;
    sumG += g;
  }
  ok(maxG < 8.0, "gradient magnitude is bounded", "max |grad| " + maxG.toFixed(4));
  ok(sumG / M > 0.5, "gradient is non-trivial", "mean |grad| " + (sumG / M).toFixed(4));
}

{
  // Determinism: same input, same bits — regardless of what was computed between.
  const next = sampler(0x5eed04, 100);
  const pts = [];
  const vals = [];
  for (let i = 0; i < 2000; i++) {
    next(P);
    pts.push([P[0], P[1], P[2]]);
    vals.push(N.snoise3(P[0], P[1], P[2]));
  }
  let same = true;
  for (let i = pts.length - 1; i >= 0; i--) {
    N.fbm3(i, i * 3, i * 7, 5); // deliberately perturb any hidden state
    N.curl3(i, -i, i * 2, A);
    if (N.snoise3(pts[i][0], pts[i][1], pts[i][2]) !== vals[i]) same = false;
  }
  ok(same, "deterministic: identical bits on re-evaluation, in any order");
  ok(N.snoise3(0.5, 0.5, 0.5) === N.snoise3(0.5, 0.5, 0.5), "deterministic for a fixed point");
  ok(Number.isFinite(N.snoise3(0, 0, 0)), "finite at the origin");
  ok(Number.isFinite(N.snoise3(-1e5, 1e5, 3e4)), "finite far from the origin");
}

describe("B2. snoise3 — isotropy (not separable, not axis-aligned)");

{
  // Correlate the field with a copy of itself shifted by `lag` along eight
  // directions: the three axes, the three face diagonals, the body diagonal
  // and one arbitrary direction. If the noise were separable or lattice
  // aligned, the axis directions would stand out.
  function corrAlong(dx, dy, dz, lag, seed) {
    const L = Math.hypot(dx, dy, dz);
    const ux = (dx / L) * lag, uy = (dy / L) * lag, uz = (dz / L) * lag;
    const next = sampler(seed, 300);
    const M = 24000;
    let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (let i = 0; i < M; i++) {
      next(P);
      const u = N.snoise3(P[0], P[1], P[2]);
      const v = N.snoise3(P[0] + ux, P[1] + uy, P[2] + uz);
      sa += u; sb += v; saa += u * u; sbb += v * v; sab += u * v;
    }
    const ma = sa / M, mb = sb / M;
    const cov = sab / M - ma * mb;
    const va = saa / M - ma * ma;
    const vb = sbb / M - mb * mb;
    return cov / Math.sqrt(va * vb);
  }

  const dirs = [
    ["+X", 1, 0, 0], ["+Y", 0, 1, 0], ["+Z", 0, 0, 1],
    ["XY", 1, 1, 0], ["XZ", 1, 0, 1], ["YZ", 0, 1, 1],
    ["XYZ", 1, 1, 1], ["oblique", 0.37, -0.81, 0.45]
  ];

  for (const lag of [0.25, 0.5, 1.0]) {
    const cs = dirs.map((d) => corrAlong(d[1], d[2], d[3], lag, 0x5eed10));
    const axis = cs.slice(0, 3);
    const diag = cs.slice(3);
    const spread = Math.max.apply(null, cs) - Math.min.apply(null, cs);
    const axisMean = axis.reduce((a, b) => a + b, 0) / axis.length;
    const diagMean = diag.reduce((a, b) => a + b, 0) / diag.length;

    ok(spread < 0.08,
      "lag " + lag + ": all eight directions correlate alike (no axis anomaly)",
      "spread " + spread.toFixed(4) + "  " +
      dirs.map((d, i) => d[0] + "=" + cs[i].toFixed(3)).join(" "));
    ok(Math.abs(axisMean - diagMean) < 0.06,
      "lag " + lag + ": axis-aligned and diagonal correlations agree",
      "axis " + axisMean.toFixed(4) + " vs diag " + diagMean.toFixed(4));
  }

  // The field must genuinely decorrelate — a separable / degenerate field
  // would stay correlated far out along the axes.
  const farX = corrAlong(1, 0, 0, 3.0, 0x5eed11);
  const farD = corrAlong(1, 1, 1, 3.0, 0x5eed11);
  ok(Math.abs(farX) < 0.08 && Math.abs(farD) < 0.08,
    "decorrelates by lag 3 in every direction",
    "X " + farX.toFixed(4) + "  XYZ " + farD.toFixed(4));

  // Separability check: if snoise(x,y,z) factored as f(x)g(y)h(z) it would be
  // constant in y and z whenever it is scanned along x at fixed y,z with the
  // same relative profile. Compare the x-profile at two different (y,z):
  // a separable field gives profiles that are exact scalar multiples.
  const n = 400;
  const p1 = new Float64Array(n);
  const p2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i / n) * 8 - 4;
    p1[i] = N.snoise3(x, 0.371, -2.113);
    p2[i] = N.snoise3(x, 5.917, 3.449);
  }
  // best scalar k minimising |p2 - k*p1|
  let num = 0, den = 0, e2 = 0, n2 = 0;
  for (let i = 0; i < n; i++) { num += p1[i] * p2[i]; den += p1[i] * p1[i]; }
  const k = num / den;
  for (let i = 0; i < n; i++) { const d = p2[i] - k * p1[i]; e2 += d * d; n2 += p2[i] * p2[i]; }
  ok(e2 / n2 > 0.5, "not separable: x-profiles at different (y,z) are not scalar multiples",
    "residual energy " + ((100 * e2) / n2).toFixed(1) + "%");
}

/* ═══════════════════════════════════════════════════════════════════════════
   C. the 7x7 gradient table  (guards the float32 floor() boundary)
   ═══════════════════════════════════════════════════════════════════════════ */

describe("C. gradient table derived from the shader's own arithmetic");

{
  // Re-derive the gradient lookup exactly as the shader does, for every value
  // the permutation can produce (0..288), and check that every gradient is
  // close to unit length after taylorInvSqrt. Evaluating the shader's literals
  // in float64 instead of float32 sends yu to 7 — one row past the 7x7 table —
  // and blows |g| up to ~4.7, which is what this guards.
  const floor = Math.floor, abs = Math.abs;
  const n_ = Math.fround(0.142857142857);
  const D = [0.0, 0.5, 1.0, 2.0];
  const nsx = n_ * D[3] - D[0];
  const nsy = n_ * D[1] - D[2];
  const nsz = n_ * D[2] - D[0];
  const tis = (r) => 1.79284291400159 - 0.85373472095314 * r;

  let minL = Infinity, maxL = -Infinity, badIndex = 0;
  const seen = new Set();
  for (let p = 0; p < 289; p++) {
    const j = p - 49.0 * floor(p * nsz * nsz);
    const xu = floor(j * nsz);
    const yu = floor(j - 7.0 * xu);
    if (j < 0 || j > 48 || xu < 0 || xu > 6 || yu < 0 || yu > 6) badIndex++;
    const X = xu * nsx + nsy;
    const Y = yu * nsx + nsy;
    const H = 1.0 - abs(X) - abs(Y);
    const sX = floor(X) * 2.0 + 1.0;
    const sY = floor(Y) * 2.0 + 1.0;
    const sh = H <= 0.0 ? -1.0 : 0.0;
    let gx = X + sX * sh, gy = Y + sY * sh, gz = H;
    const pre = gx * gx + gy * gy + gz * gz;
    const nr = tis(pre);
    gx *= nr; gy *= nr; gz *= nr;
    const L = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (L < minL) minL = L;
    if (L > maxL) maxL = L;
    seen.add(xu + ":" + yu);
  }
  ok(badIndex === 0, "all 289 permutation values index inside the 7x7 table",
    "out of range: " + badIndex);
  ok(seen.size === 49, "the table really has 49 distinct cells", "distinct " + seen.size);
  ok(minL > 0.85 && maxL < 1.06, "every normalised gradient is unit-ish",
    "|g| in [" + minL.toFixed(4) + ", " + maxL.toFixed(4) + "]");

  // The same derivation with the float64 rounding of the literal must fail,
  // which is the evidence that the float32 rounding in noise.js is load-bearing.
  const n64 = 0.142857142857;
  let overflow = 0;
  for (let p = 0; p < 289; p++) {
    const j = p - 49.0 * floor(p * n64 * n64);
    const xu = floor(j * n64);
    const yu = floor(j - 7.0 * xu);
    if (yu > 6 || xu > 6 || j > 48) overflow++;
  }
  ok(overflow > 0, "float64 rounding of the literal would overflow the table (guard is needed)",
    overflow + " of 289 entries");
  ok(/Math\.fround|f32\(/.test(NOISE_SRC), "noise.js pins those literals to float32");
}

/* ═══════════════════════════════════════════════════════════════════════════
   D. curlNoise
   ═══════════════════════════════════════════════════════════════════════════ */

describe("D. curl3 — divergence-free, continuous, bounded, non-trivial");

{
  const e = N.CURL_EPS;
  ok(typeof e === "number" && e > 0 && e < 0.1, "CURL_EPS is a small positive epsilon", "eps " + e);

  const M = 20000;
  const next = sampler(0x5eed20, 120);
  let maxDiv = 0, sumDiv = 0;
  let maxMag = 0, minMag = Infinity, sumMag = 0;
  const s = [0, 0, 0], s2 = [0, 0, 0];

  for (let i = 0; i < M; i++) {
    next(P);
    const x = P[0], y = P[1], z = P[2];

    N.curl3(x, y, z, CV);
    const mag = Math.hypot(CV[0], CV[1], CV[2]);
    if (mag > maxMag) maxMag = mag;
    if (mag < minMag) minMag = mag;
    sumMag += mag;
    for (let k = 0; k < 3; k++) { s[k] += CV[k]; s2[k] += CV[k] * CV[k]; }

    // Numerical divergence with the same central-difference stencil the curl
    // itself uses. div(curl) is then an exactly cancelling sum of commuting
    // difference operators, so this is zero up to rounding — not merely small.
    N.curl3(x - e, y, z, A); N.curl3(x + e, y, z, B);
    N.curl3(x, y - e, z, DV); N.curl3(x, y + e, z, EV);
    N.curl3(x, y, z - e, FV); N.curl3(x, y, z + e, GV);
    const div = ((B[0] - A[0]) + (EV[1] - DV[1]) + (GV[2] - FV[2])) / (2 * e);
    const ad = Math.abs(div);
    if (ad > maxDiv) maxDiv = ad;
    sumDiv += ad;
  }

  ok(maxDiv < 1e-3, "divergence is within 1e-3 of zero everywhere sampled",
    "max |div| " + maxDiv.toExponential(3) + "  mean " + (sumDiv / M).toExponential(3));
  ok(maxDiv < 1e-8, "divergence is in fact zero to floating-point rounding",
    "max |div| " + maxDiv.toExponential(3));

  ok(maxMag < 40 && Number.isFinite(maxMag), "magnitude is bounded",
    "|curl| in [" + minMag.toFixed(4) + ", " + maxMag.toFixed(4) + "]");
  ok(sumMag / M > 1.0, "magnitude is non-trivial", "mean |curl| " + (sumMag / M).toFixed(4));

  const sd = [0, 1, 2].map((k) => Math.sqrt(s2[k] / M - (s[k] / M) * (s[k] / M)));
  ok(sd.every((v) => v > 0.8), "varies in all three components",
    "std " + sd.map((v) => v.toFixed(4)).join(" / "));
  ok(Math.max.apply(null, sd) / Math.min.apply(null, sd) < 1.35,
    "the three components carry comparable energy (isotropic)",
    "ratio " + (Math.max.apply(null, sd) / Math.min.apply(null, sd)).toFixed(4));
  ok([0, 1, 2].every((k) => Math.abs(s[k] / M) < 0.12), "each component is roughly zero-mean",
    "mean " + [0, 1, 2].map((k) => (s[k] / M).toFixed(4)).join(" / "));
}

{
  // Continuity of the curl field. The curl is a difference quotient of the
  // potential, so its Lipschitz constant is the potential's divided by the
  // epsilon — roughly 30 here. Check that the jump really does shrink in
  // proportion to the step (which a discontinuous field would not do), and
  // separately bound the tail, where the 1/(2*eps) amplification of the
  // simplex cell step measured above shows up.
  const M = 30000;
  const ratios = [];
  let tailAt1e4 = 0;
  for (const h of [1e-3, 1e-4, 1e-5]) {
    const next = sampler(0x5eed21, 120);
    const jumps = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      next(P);
      N.curl3(P[0], P[1], P[2], A);
      N.curl3(P[0] + h, P[1] + h, P[2] + h, B);
      jumps[i] = Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]);
    }
    let sum = 0, mx = 0;
    for (let i = 0; i < M; i++) { sum += jumps[i]; if (jumps[i] > mx) mx = jumps[i]; }
    const mean = sum / M;
    ratios.push(mean / h);
    if (h === 1e-4) tailAt1e4 = mx;
    ok(mean / h < 60, "step " + h + ": mean jump scales with the step (Lipschitz)",
      "mean/h " + (mean / h).toFixed(2) + "  max " + mx.toExponential(3));
  }
  const rmax = Math.max.apply(null, ratios);
  const rmin = Math.min.apply(null, ratios);
  ok(rmax / rmin < 1.15,
    "the jump/step ratio is the same across three decades of step size",
    "ratios " + ratios.map((v) => v.toFixed(2)).join(" / "));
  ok(tailAt1e4 < 0.35,
    "worst-case jump at a 1e-4 step stays under 10% of the mean magnitude",
    "worst " + tailAt1e4.toExponential(3) + " vs mean |curl| ~3.5");
}

{
  // The epsilon stencil really does sample the continuous curl: compare it to a
  // 50x finer stencil on the same potential. If these agree, the sampled field
  // is the true (analytically divergence-free) curl to that accuracy.
  function curlAt(x, y, z, eps, out) {
    N.potential3(x - eps, y, z, A); N.potential3(x + eps, y, z, B);
    N.potential3(x, y - eps, z, CV); N.potential3(x, y + eps, z, DV);
    N.potential3(x, y, z - eps, EV); N.potential3(x, y, z + eps, FV);
    const inv = 1 / (2 * eps);
    out[0] = ((DV[2] - CV[2]) - (FV[1] - EV[1])) * inv;
    out[1] = ((FV[0] - EV[0]) - (B[2] - A[2])) * inv;
    out[2] = ((B[1] - A[1]) - (DV[0] - CV[0])) * inv;
    return out;
  }
  const M = 4000;
  const next = sampler(0x5eed22, 120);
  const coarse = new Float64Array(3);
  const fine = new Float64Array(3);
  let sumErr = 0, sumMag = 0;
  for (let i = 0; i < M; i++) {
    next(P);
    N.curl3(P[0], P[1], P[2], coarse);
    curlAt(P[0], P[1], P[2], N.CURL_EPS / 50, fine);
    sumErr += Math.hypot(coarse[0] - fine[0], coarse[1] - fine[1], coarse[2] - fine[2]);
    sumMag += Math.hypot(fine[0], fine[1], fine[2]);
  }
  const rel = sumErr / sumMag;
  ok(rel < 0.01, "the epsilon stencil tracks the continuous curl to better than 1%",
    "mean relative error " + (100 * rel).toFixed(3) + "%");
}

{
  // The potential's three components must be independent fields, otherwise the
  // curl degenerates. Correlate them pairwise.
  const M = 20000;
  const next = sampler(0x5eed23, 200);
  const s = [0, 0, 0], s2 = [0, 0, 0], sxy = [0, 0, 0];
  for (let i = 0; i < M; i++) {
    next(P);
    N.potential3(P[0], P[1], P[2], A);
    for (let k = 0; k < 3; k++) { s[k] += A[k]; s2[k] += A[k] * A[k]; }
    sxy[0] += A[0] * A[1]; sxy[1] += A[0] * A[2]; sxy[2] += A[1] * A[2];
  }
  const mean = s.map((v) => v / M);
  const sd = [0, 1, 2].map((k) => Math.sqrt(s2[k] / M - mean[k] * mean[k]));
  const pairs = [[0, 1], [0, 2], [1, 2]];
  const cc = pairs.map((p, i) => (sxy[i] / M - mean[p[0]] * mean[p[1]]) / (sd[p[0]] * sd[p[1]]));
  ok(cc.every((c) => Math.abs(c) < 0.05), "the three potential components are decorrelated",
    "r = " + cc.map((c) => c.toFixed(4)).join(" / "));
  ok(N.CURL_OFFSETS.length === 3, "three fixed offsets are published");
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      const a = N.CURL_OFFSETS[i], b = N.CURL_OFFSETS[j];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      ok(d > 50, "offsets " + i + " and " + j + " are far apart", "distance " + d.toFixed(2));
    }
  }
  ok(N.curl3(1, 2, 3) === N.curl3(4, 5, 6), "curl3 reuses one scratch buffer when no target is given");
  const own = new Float64Array(3);
  ok(N.curl3(1, 2, 3, own) === own, "curl3 writes into a supplied target");
  N.curl3(1.5, -2.25, 0.75, A);
  N.curl3(1.5, -2.25, 0.75, B);
  ok(A[0] === B[0] && A[1] === B[1] && A[2] === B[2], "curl3 is deterministic");
}

/* ═══════════════════════════════════════════════════════════════════════════
   E. fbm
   ═══════════════════════════════════════════════════════════════════════════ */

describe("E. fbm3 — range and octave detail");

{
  const M = 40000;
  for (const oct of [1, 2, 4, 6, 8]) {
    const next = sampler(0x5eed30, 80);
    let mn = Infinity, mx = -Infinity, s2 = 0;
    for (let i = 0; i < M; i++) {
      next(P);
      const v = N.fbm3(P[0], P[1], P[2], oct);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      s2 += v * v;
    }
    ok(mn >= -1.02 && mx <= 1.02, "octaves=" + oct + ": stays inside [-1.02, 1.02]",
      "min " + mn.toFixed(5) + "  max " + mx.toFixed(5));
    ok(Math.sqrt(s2 / M) > 0.1, "octaves=" + oct + ": is not collapsed near zero",
      "rms " + Math.sqrt(s2 / M).toFixed(5));
  }

  ok(N.fbm3(1.1, 2.2, 3.3, 0) === 0, "octaves=0 returns exactly 0");
  ok(N.fbm3(1.1, 2.2, 3.3, -4) === 0, "a negative octave count returns exactly 0");
  ok(N.fbm3(1.1, 2.2, 3.3, 99) === N.fbm3(1.1, 2.2, 3.3, N.FBM_MAX_OCTAVES),
    "the octave count is clamped to FBM_MAX_OCTAVES");
  ok(N.fbm3(0.3, -0.7, 1.9, 5) === N.fbm3(0.3, -0.7, 1.9, 5), "fbm3 is deterministic");
  near(N.fbm3(0.3, -0.7, 1.9, 1), N.snoise3(0.3, -0.7, 1.9), 1e-12,
    "one octave equals plain snoise (amplitude normalisation cancels)");
  ok(N.FBM_LACUNARITY === 2.0 && N.FBM_GAIN === 0.5, "lacunarity 2.0 and gain 0.5 are published");
}

{
  // High-frequency detail: the variance of a small finite difference must grow
  // with the octave count, because each octave halves the feature size.
  const h = 0.005;
  const M = 30000;
  const fdVar = [];
  for (let oct = 1; oct <= 8; oct++) {
    const next = sampler(0x5eed31, 40);
    let s = 0;
    for (let i = 0; i < M; i++) {
      next(P);
      const d = N.fbm3(P[0] + h, P[1], P[2], oct) - N.fbm3(P[0], P[1], P[2], oct);
      s += d * d;
    }
    fdVar.push(s / M);
  }
  console.log("        fd-variance by octave: " + fdVar.map((v) => v.toExponential(2)).join("  "));

  let monotone = true;
  for (let i = 2; i < fdVar.length; i++) if (fdVar[i] <= fdVar[i - 1]) monotone = false;
  ok(monotone, "fd-variance increases with every octave from 2 upward");
  ok(fdVar[7] > 1.6 * fdVar[1], "8 octaves carry far more small-scale energy than 2",
    "ratio " + (fdVar[7] / fdVar[1]).toFixed(3));
  ok(fdVar[7] > 1.1 * fdVar[0], "8 octaves carry more small-scale energy than 1",
    "ratio " + (fdVar[7] / fdVar[0]).toFixed(3));

  // ... while the large-scale structure is preserved: fbm(8) still correlates
  // strongly with fbm(2), i.e. the extra octaves are detail, not a new field.
  const next = sampler(0x5eed32, 40);
  let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 20000;
  for (let i = 0; i < n; i++) {
    next(P);
    const a = N.fbm3(P[0], P[1], P[2], 2);
    const b = N.fbm3(P[0], P[1], P[2], 8);
    sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
  }
  const r = (sab / n - (sa / n) * (sb / n)) /
    Math.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2));
  ok(r > 0.75, "8 octaves keep the 2-octave silhouette (detail is added, not replaced)",
    "r = " + r.toFixed(4));
}

describe("E2. bulk fill helpers");

{
  const count = 512;
  const pos = new Float32Array(count * 3);
  const rnd = mulberry32(0x5eed40);
  for (let i = 0; i < pos.length; i++) pos[i] = (rnd() - 0.5) * 20;

  const outS = new Float32Array(count);
  ok(N.fillSnoise(pos, outS, count, 0.5) === outS, "fillSnoise returns its target");
  let allSet = true;
  for (let i = 0; i < count; i++) {
    if (!Number.isFinite(outS[i]) || Math.abs(outS[i]) > 1.02) allSet = false;
  }
  ok(allSet, "fillSnoise writes finite in-range values into a Float32Array");
  ok(Math.abs(outS[3] - Math.fround(N.snoise3(pos[9] * 0.5, pos[10] * 0.5, pos[11] * 0.5))) < 1e-6,
    "fillSnoise agrees with snoise3 element-wise");

  const outC = new Float32Array(count * 3);
  N.fillCurl(pos, outC, count, 0.5);
  let curlOk = true;
  for (let i = 0; i < outC.length; i++) if (!Number.isFinite(outC[i])) curlOk = false;
  ok(curlOk, "fillCurl writes finite values into a Float32Array");
  N.curl3(pos[0] * 0.5, pos[1] * 0.5, pos[2] * 0.5, A);
  ok(Math.abs(outC[0] - Math.fround(A[0])) < 1e-5, "fillCurl agrees with curl3 element-wise");
}

/* ═══════════════════════════════════════════════════════════════════════════
   F. GLSL <-> JS structural equivalence
   ═══════════════════════════════════════════════════════════════════════════ */

describe("F. the JS mirror carries the same constants and shapes as the GLSL");

/** Extract `float snoise(vec3 v) { ... }` from the shader, braces matched. */
function extractGlslFunction(glsl, signature) {
  const start = glsl.indexOf(signature);
  if (start < 0) return null;
  let i = glsl.indexOf("{", start);
  if (i < 0) return null;
  let depth = 0;
  const from = i;
  for (; i < glsl.length; i++) {
    if (glsl[i] === "{") depth++;
    else if (glsl[i] === "}") {
      depth--;
      if (depth === 0) return glsl.slice(from, i + 1);
    }
  }
  return null;
}

/* The mirror hoists two shader literals into named constants so that
   Math.fround is applied once instead of on every call. The checker expands
   those two names back to the literal they stand for; both expansions are
   verified numerically right below, so the alias table cannot hide a change. */
const ALIASES = [
  ["INV_289", "(1.0/289.0)"],
  ["ONE_SEVENTH", "0.142857142857"]
];

function canon(src) {
  let s = src
    .replace(/\bMath\.(floor|abs|min|max|sqrt|fround|hypot)\b/g, "$1")
    .replace(/[\[\]]/g, (m) => (m === "[" ? "(" : ")"))
    .replace(/\s+/g, "");
  for (const [name, literal] of ALIASES) {
    s = s.split(name).join(literal);
  }
  return s;
}

const GLSL = N.GLSL_SIMPLEX3;
const GLSL_SNOISE = extractGlslFunction(GLSL, "float snoise(vec3 v)");
const GLSL_MOD289_3 = extractGlslFunction(GLSL, "vec3 mod289(vec3 x)");
const GLSL_MOD289_4 = extractGlslFunction(GLSL, "vec4 mod289(vec4 x)");
const GLSL_PERMUTE = extractGlslFunction(GLSL, "vec4 permute(vec4 x)");
const GLSL_TIS = extractGlslFunction(GLSL, "vec4 taylorInvSqrt(vec4 r)");

ok(GLSL_SNOISE !== null && GLSL_SNOISE.length > 1200, "extracted the GLSL snoise body",
  GLSL_SNOISE ? GLSL_SNOISE.length + " chars" : "not found");
ok(GLSL_MOD289_3 && GLSL_MOD289_4 && GLSL_PERMUTE && GLSL_TIS,
  "extracted mod289 (vec3 + vec4), permute and taylorInvSqrt");
ok(canon(GLSL_MOD289_3) === canon(GLSL_MOD289_4),
  "the two mod289 overloads have identical bodies");

const JS_CODE = NOISE_CODE;
ok(!/snoise\s*\(\s*vec3/.test(JS_CODE), "the GLSL text has been removed before comparing",
  "stripped " + (NOISE_SRC.length - JS_CODE.length) + " chars of strings and comments");

const JS_SNOISE = (function () {
  const m = JS_CODE.indexOf("function snoise3(");
  let i = JS_CODE.indexOf("{", m);
  let depth = 0;
  const from = i;
  for (; i < JS_CODE.length; i++) {
    if (JS_CODE[i] === "{") depth++;
    else if (JS_CODE[i] === "}") { depth--; if (depth === 0) return JS_CODE.slice(from, i + 1); }
  }
  return "";
})();
ok(JS_SNOISE.length > 1500, "extracted the JS snoise3 body", JS_SNOISE.length + " chars");

const cGLSL = canon(GLSL_SNOISE);
const cJS = canon(JS_SNOISE);
const cGLSLall = canon(GLSL);
const cJSall = canon(JS_CODE);

/* F1. the helper functions, expression for expression */
const SHAPES = [
  ["mod289 body", "returnx-floor(x*(1.0/289.0))*289.0;", canon(GLSL_MOD289_3), canon(JS_CODE)],
  ["permute body", "returnmod289(((x*34.0)+1.0)*x);", canon(GLSL_PERMUTE), canon(JS_CODE)],
  ["taylorInvSqrt body", "return1.79284291400159-0.85373472095314*r;", canon(GLSL_TIS), canon(JS_CODE)]
];
for (const [label, shape, inGlsl, inJs] of SHAPES) {
  ok(inGlsl.includes(shape) && inJs.includes(shape),
    label + " is character-for-character identical in GLSL and JS");
}

/* F2. the constants of snoise itself */
const CONSTANTS = [
  ["C = vec2(1/6, 1/3)", "(1.0/6.0,1.0/3.0)", cGLSL, cJSall],
  ["D = vec4(0, 0.5, 1, 2)", "(0.0,0.5,1.0,2.0)", cGLSL, cJSall],
  ["the 289 modulus", "289.0", cGLSLall, cJSall],
  ["the 1/289 reciprocal", "1.0/289.0", cGLSLall, cJSall],
  ["the 34 in permute", "34.0", cGLSLall, cJSall],
  ["the 49 = 7*7 wrap", "49.0*floor(", cGLSL, cJS],
  ["the 7 in mod(j,7)", "7.0*", cGLSL, cJS],
  ["1/7 as 0.142857142857", "0.142857142857", cGLSL, cJSall],
  ["taylorInvSqrt a", "1.79284291400159", cGLSLall, cJSall],
  ["taylorInvSqrt b", "0.85373472095314", cGLSLall, cJSall],
  ["the 0.6 kernel radius", "max(0.6-", cGLSL, cJS],
  ["the 42.0 output scale", "return42.0*", cGLSL, cJS]
];
for (const [label, token, inGlsl, inJs] of CONSTANTS) {
  ok(inGlsl.includes(token) && inJs.includes(token),
    "constant present in both: " + label,
    JSON.stringify(token));
}

/* F3. no stray extra copies of the two scale constants */
function count(hay, needle) {
  let n = 0, i = 0;
  for (;;) {
    const k = hay.indexOf(needle, i);
    if (k < 0) return n;
    n++;
    i = k + needle.length;
  }
}
ok(count(cGLSL, "42.0") === 1 && count(cJS, "42.0") === 1,
  "the 42.0 scale appears exactly once in each snoise body");
ok(count(cGLSL, "0.6") === 1, "0.6 appears once in the GLSL snoise body");
ok(count(cJS, "0.6") === 4, "0.6 appears once per unrolled lane (4) in the JS body");

/* F4. the vec4 lanes really are unrolled 4-wide */
ok(count(cGLSL, "permute(") === 3, "GLSL does 3 rounds of permute on a vec4",
  count(cGLSL, "permute(") + " calls");
ok(count(cJS, "permute(") === 12, "JS does the same 3 rounds unrolled over 4 lanes",
  count(cJS, "permute(") + " calls");
ok(count(cGLSL, "taylorInvSqrt(") === 1 && count(cJS, "taylorInvSqrt(") === 4,
  "taylorInvSqrt: 1 vec4 call in GLSL, 4 scalar calls in JS");
/* snoise takes six floors: one over a vec3 (the base corner) and five over a
   vec4 (the 7*7 wrap, x_, y_, s0, s1). Unrolled that is 1*3 + 5*4 = 23 scalar
   floors, and the JS must have exactly that many — no more, no fewer. */
ok(count(cGLSL, "floor(") === 6, "GLSL snoise takes 6 vector-wide floors",
  count(cGLSL, "floor(") + " calls");
ok(count(cJS, "floor(") === 1 * 3 + 5 * 4,
  "JS snoise3 unrolls them to exactly 1*3 + 5*4 = 23 scalar floors",
  count(cJS, "floor(") + " calls");

/* F5. the aliases really do stand for the literals they are expanded to */
{
  const invDecl = /INV_289\s*=\s*f32\(\s*1\.0\s*\/\s*289\.0\s*\)/.test(NOISE_SRC);
  const sevDecl = /ONE_SEVENTH\s*=\s*f32\(\s*0\.142857142857\s*\)/.test(NOISE_SRC);
  ok(invDecl, "INV_289 is declared as f32(1.0 / 289.0)");
  ok(sevDecl, "ONE_SEVENTH is declared as f32(0.142857142857)");
  ok(/var\s+f32\s*=\s*Math\.fround\s*;/.test(NOISE_SRC), "f32 is Math.fround, nothing else");
}

/* F6. the JS really computes the shader's ns vector */
{
  const n_ = Math.fround(0.142857142857);
  const D = [0.0, 0.5, 1.0, 2.0];
  near(n_ * D[3] - D[0], 2 * n_, 0, "ns.x == n_ * D.w - D.x == 2/7");
  near(n_ * D[1] - D[2], 0.5 * n_ - 1.0, 0, "ns.y == n_ * D.y - D.z");
  near(n_ * D[2] - D[0], n_, 0, "ns.z == n_ * D.z - D.x == 1/7");
  ok(/n_\s*\*\s*D\[3\]\s*-\s*D\[0\]/.test(NOISE_SRC), "JS spells ns.x as n_ * D[3] - D[0]");
  ok(/n_\s*\*\s*D\[1\]\s*-\s*D\[2\]/.test(NOISE_SRC), "JS spells ns.y as n_ * D[1] - D[2]");
  ok(/n_\s*\*\s*D\[2\]\s*-\s*D\[0\]/.test(NOISE_SRC), "JS spells ns.z as n_ * D[2] - D[0]");
}

/* ═══════════════════════════════════════════════════════════════════════════
   G. GLSL lint
   ═══════════════════════════════════════════════════════════════════════════ */

describe("G. GLSL source lint");

function balanced(src, open, close) {
  let d = 0;
  for (const ch of src) {
    if (ch === open) d++;
    else if (ch === close) { d--; if (d < 0) return false; }
  }
  return d === 0;
}

for (const [name, src] of [
  ["GLSL_SIMPLEX3", N.GLSL_SIMPLEX3],
  ["GLSL_FBM3", N.GLSL_FBM3],
  ["GLSL_CURL3", N.GLSL_CURL3],
  ["PRELUDE", N.PRELUDE]
]) {
  ok(typeof src === "string" && src.length > 100, name + " is a non-empty source string",
    src.length + " chars");
  ok(balanced(src, "{", "}"), name + ": braces balanced");
  ok(balanced(src, "(", ")"), name + ": parentheses balanced");
  ok(!/\bTODO\b|\bFIXME\b|\bXXX\b/.test(src), name + ": no TODO / FIXME markers");
  ok(!/#version/.test(src), name + ": no #version line (the host shader owns it)");
}

ok(/\bfloat\s+snoise\s*\(\s*vec3\s+v\s*\)/.test(N.GLSL_SIMPLEX3), "declares float snoise(vec3 v)");
ok(/\bfloat\s+fbm\s*\(\s*vec3\s+p\s*,\s*int\s+octaves\s*\)/.test(N.GLSL_FBM3),
  "declares float fbm(vec3 p, int octaves)");
ok(/\bvec3\s+curlNoise\s*\(\s*vec3\s+p\s*\)/.test(N.GLSL_CURL3), "declares vec3 curlNoise(vec3 p)");

ok(N.PRELUDE.indexOf(N.GLSL_SIMPLEX3) === 0, "PRELUDE starts with GLSL_SIMPLEX3");
ok(N.PRELUDE.includes(N.GLSL_FBM3) && N.PRELUDE.includes(N.GLSL_CURL3),
  "PRELUDE contains GLSL_FBM3 and GLSL_CURL3");
ok(N.PRELUDE.indexOf("float snoise(vec3 v)") < N.PRELUDE.indexOf("float fbm(vec3 p"),
  "snoise is defined before fbm uses it");
ok(N.PRELUDE.indexOf("float snoise(vec3 v)") < N.PRELUDE.indexOf("vec3 curlNoise(vec3 p)"),
  "snoise is defined before curlNoise uses it");
ok(N.PRELUDE.indexOf("vec3 hotaruPotential(vec3 p)") < N.PRELUDE.indexOf("vec3 curlNoise(vec3 p)"),
  "hotaruPotential is defined before curlNoise uses it");

// The GLSL fbm must spell out the lacunarity and gain the module advertises.
ok(/amp\s*\*=\s*0\.5\s*;/.test(N.GLSL_FBM3), "GLSL fbm applies gain 0.5");
ok(/q\s*=\s*q\s*\*\s*2\.0/.test(N.GLSL_FBM3), "GLSL fbm applies lacunarity 2.0");
ok(/sum\s*\/\s*norm/.test(N.GLSL_FBM3), "GLSL fbm normalises by the amplitude sum");

// The GLSL curl must use the same epsilon and the same three offsets as the JS.
ok(N.GLSL_CURL3.includes(String(N.CURL_EPS)), "GLSL curl uses the published CURL_EPS",
  String(N.CURL_EPS));
for (const off of N.CURL_OFFSETS) {
  for (const c of off) {
    ok(N.GLSL_CURL3.includes(String(c)), "GLSL curl carries offset component " + c);
  }
}
ok(/\(2\.0\s*\*\s*e\)/.test(N.GLSL_CURL3), "GLSL curl divides by 2*epsilon");
ok(/REQUIRES highp/.test(N.GLSL_CURL3),
  "GLSL curl states its highp requirement in the source (mediump eats the epsilon)");

{
  // Show that the requirement is real: at float16 precision the epsilon step
  // vanishes against the potential's own offsets and the curl collapses.
  const f16 = (x) => {
    // round-to-nearest at 10 explicit mantissa bits, the mediump guarantee
    if (x === 0 || !Number.isFinite(x)) return x;
    const e = Math.floor(Math.log2(Math.abs(x)));
    const q = Math.pow(2, e - 10);
    return Math.round(x / q) * q;
  };
  const off = N.CURL_OFFSETS[2][0]; // 91.77, the largest offset
  ok(f16(off + N.CURL_EPS) === f16(off),
    "at mediump precision, offset + CURL_EPS is indistinguishable from offset",
    f16(off) + " vs " + f16(off + N.CURL_EPS));
  ok(Math.fround(off + N.CURL_EPS) !== Math.fround(off),
    "at highp precision it is resolved cleanly",
    Math.fround(off + N.CURL_EPS) + " vs " + Math.fround(off));
  ok(Math.fround(1e4 + N.CURL_EPS) !== Math.fround(1e4),
    "highp still resolves the epsilon at |p| = 1e4");
}
ok(count(N.GLSL_CURL3, "hotaruPotential(p") === 6, "GLSL curl takes six potential samples",
  count(N.GLSL_CURL3, "hotaruPotential(p") + " samples");

// The GLSL fbm must match the JS fbm shift constants.
ok(/vec3\(17\.31,\s*9\.73,\s*23\.17\)/.test(N.GLSL_FBM3), "GLSL fbm octave shift is (17.31, 9.73, 23.17)");
ok(/qx\s*=\s*qx\s*\*\s*2\.0\s*\+\s*FBM_SHIFT_X/.test(NOISE_SRC), "JS fbm applies the same shift on x");
ok(/FBM_SHIFT_X\s*=\s*17\.31/.test(NOISE_SRC) &&
  /FBM_SHIFT_Y\s*=\s*9\.73/.test(NOISE_SRC) &&
  /FBM_SHIFT_Z\s*=\s*23\.17/.test(NOISE_SRC), "JS fbm shift constants equal the GLSL ones");

/* ── summary ───────────────────────────────────────────────────────────── */

console.log("\n" + "-".repeat(62));
console.log((failed === 0 ? "ALL GREEN" : "FAILURES") + "   " + passed + " passed, " + failed + " failed");
console.log("-".repeat(62));
process.exitCode = failed === 0 ? 0 : 1;
