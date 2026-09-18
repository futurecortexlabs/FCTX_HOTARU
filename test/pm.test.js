/* test/pm.test.js — correctness proof and iteration-count experiment for
 * hotaru/pm.js.
 *
 *   node test/pm.test.js            full run, ~3 minutes
 *   node test/pm.test.js --quick    skip the long integrations and shrink the
 *                                   headline experiment, ~40 seconds
 *
 * Plain node script: it require()s the library for its global side effect.
 * One line per case, non-zero exit on any failure.
 *
 * Sections 1-7 are the physics proofs the brief asks for. Section 8 is the
 * headline experiment — how many relaxation passes per frame the GPU needs —
 * and section 9 collects the constants that go into the shader.
 */
'use strict';

const path = require('path');

const SRC = path.join(__dirname, '..', 'hotaru', 'pm.js');
require(SRC);
const PM = globalThis.HotaruPM;

const QUICK = process.argv.indexOf('--quick') >= 0;

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
  console.log('\n== ' + title + ' ' + '='.repeat(Math.max(3, 62 - title.length)));
}

function note(s) { console.log('       ' + s); }

function pad(s, n) { s = String(s); return s.length >= n ? s : ' '.repeat(n - s.length) + s; }
function padr(s, n) { s = String(s); return s.length >= n ? s : s + ' '.repeat(n - s.length); }
function e2(x) { return Number(x).toExponential(2); }
function e1(x) { return Number(x).toExponential(1); }
function pct(x) { return (x * 100).toFixed(2) + '%'; }

const T0 = Date.now();
function elapsed() { return ((Date.now() - T0) / 1000).toFixed(1) + 's'; }

/* Findings that section 9 prints back as the deliverable. */
const FINDINGS = {};

/* ── fixtures ───────────────────────────────────────────────────────────── */

/*  A realistic density field: three clumps of different compactness, a bent
    filament between two of them, and a thin sea. Uniform noise would not do:
    it has almost no large-scale power, and the large scales are exactly what
    relaxation is bad at, so a noise field would flatter the solver enormously.

    Everything is generated inside a ball of radius RFIELD < L/2, and the
    advection is a RIGID rotation about a tilted axis. That combination is
    deliberate. The warm-start experiment asks what error a given per-frame
    budget can sustain, and the answer only means something if the problem
    being solved is statistically the same on frame 60 as on frame 5. A
    differential rotation is the obvious choice and it is wrong: it winds the
    clumps into ever thinner spirals, the field changes faster and faster,
    and the "steady-state" error climbs through the run — which is a property
    of the fixture, not of the solver. A rigid rotation moves every clump
    across many cells, excites exactly the low-order modes that are hard,
    never leaves the box, and preserves the field's shape exactly forever.
    A small fixed per-particle scatter is kept on top so the test is not
    measuring a pure symmetry; over the longest run it displaces a particle
    by about one cell.
*/
const RFIELD = 0.45;
function makeField(count, seed) {
  const rnd = PM.mulberry32(seed);
  const g = () => PM.gaussian(rnd);
  const pos = new Float64Array(3 * count);
  const vel = new Float64Array(3 * count);
  let p = 0;
  const put = (i, x, y, z) => {
    const r = Math.hypot(x, y, z);
    const s = r > RFIELD ? RFIELD / r : 1;
    pos[3 * i] = x * s; pos[3 * i + 1] = y * s; pos[3 * i + 2] = z * s;
  };
  const clumps = [
    [-0.20, 0.16, -0.05, 0.035, 0.26],
    [0.17, -0.18, 0.11, 0.050, 0.20],
    [0.05, 0.25, 0.20, 0.025, 0.12]
  ];
  for (const c of clumps) {
    const n = Math.round(count * c[4]);
    for (let i = 0; i < n && p < count; i++, p++) {
      put(p, c[0] + c[3] * g(), c[1] + c[3] * g(), c[2] + c[3] * g());
    }
  }
  const nf = Math.round(count * 0.26);
  for (let i = 0; i < nf && p < count; i++, p++) {
    const t = rnd(), s = 0.018;
    put(p,
      -0.27 + 0.52 * t + s * g(),
      -0.23 + 0.45 * t + s * g() + 0.11 * Math.sin(3 * t),
      0.18 - 0.38 * t + s * g());
  }
  while (p < count) {
    /* uniform inside the ball, so the rotation never carries anything out */
    const u = Math.cbrt(rnd()) * RFIELD;
    const ct = 2 * rnd() - 1, st = Math.sqrt(Math.max(0, 1 - ct * ct)), ph = 2 * Math.PI * rnd();
    put(p, u * st * Math.cos(ph), u * st * Math.sin(ph), u * ct);
    p++;
  }

  /* omega on a tilted axis, so the flow is not aligned with any grid plane */
  const on = Math.hypot(0.35, 0.62, 0.70);
  const ox = 0.35 / on, oy = 0.62 / on, oz = 0.70 / on;
  let speed = 0;
  for (let i = 0; i < count; i++) {
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    vel[3 * i] = oy * z - oz * y;
    vel[3 * i + 1] = oz * x - ox * z;
    vel[3 * i + 2] = ox * y - oy * x;
    speed += Math.hypot(vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
  }
  speed /= count;
  for (let i = 0; i < count; i++) {
    vel[3 * i] += 0.05 * speed * g();
    vel[3 * i + 1] += 0.05 * speed * g();
    vel[3 * i + 2] += 0.05 * speed * g();
  }
  return { pos, vel, count };
}

function rmsSpeed(vel, count) {
  let s = 0;
  for (let i = 0; i < 3 * count; i++) s += vel[i] * vel[i];
  return Math.sqrt(s / count);
}

/*  Plummer sphere, positions from the inverse of the cumulative mass profile
    and velocities by Aarseth's rejection sampling of the Plummer distribution
    function q^2 (1-q^2)^{7/2}. Truncated at RCUT so the sphere is compact
    inside the periodic box. The net momentum is zeroed so the cluster does
    not sail across the box and start interacting with its own images. */
function plummerSphere(np, opts) {
  const G = opts.G, M = opts.M, a = opts.a, rcut = opts.rcut;
  const rnd = PM.mulberry32(opts.seed);
  const pos = new Float64Array(3 * np);
  const vel = new Float64Array(3 * np);
  const xmax = Math.pow(1 + Math.pow(a / rcut, 2), -1.5);
  for (let i = 0; i < np; i++) {
    const X = rnd() * xmax;
    const r = a / Math.sqrt(Math.pow(X, -2 / 3) - 1);
    let ct = 2 * rnd() - 1, st = Math.sqrt(Math.max(0, 1 - ct * ct)), ph = 2 * Math.PI * rnd();
    pos[3 * i] = r * st * Math.cos(ph);
    pos[3 * i + 1] = r * st * Math.sin(ph);
    pos[3 * i + 2] = r * ct;
    let q, y;
    do { q = rnd(); y = rnd() * 0.1; } while (y > q * q * Math.pow(1 - q * q, 3.5));
    const vesc = Math.SQRT2 * Math.sqrt(G * M / a) * Math.pow(1 + r * r / (a * a), -0.25);
    const v = q * vesc;
    ct = 2 * rnd() - 1; st = Math.sqrt(Math.max(0, 1 - ct * ct)); ph = 2 * Math.PI * rnd();
    vel[3 * i] = v * st * Math.cos(ph);
    vel[3 * i + 1] = v * st * Math.sin(ph);
    vel[3 * i + 2] = v * ct;
  }
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < np; i++) { cx += vel[3 * i]; cy += vel[3 * i + 1]; cz += vel[3 * i + 2]; }
  cx /= np; cy /= np; cz /= np;
  for (let i = 0; i < np; i++) { vel[3 * i] -= cx; vel[3 * i + 1] -= cy; vel[3 * i + 2] -= cz; }
  return { pos, vel, count: np, mass: M / np };
}

/* Least-squares slope of y against x — used to separate a systematic energy
   drift from an oscillation. */
function slope(xs, ys) {
  const n = xs.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const d = n * sxx - sx * sx;
  return d === 0 ? 0 : (n * sxy - sx * sy) / d;
}

/* ══ 0. the two solvers invert the SAME operator ═════════════════════════ */

section('0. operator identities');
{
  const L = 1, G = 1, N = 32, N3 = N * N * N;
  const f = makeField(20000, 11);
  const rho = new Float64Array(N3), phiF = new Float64Array(N3);
  const m = 1 / f.count;
  PM.depositCIC(f.pos, f.count, N, L, rho, { mass: m });
  PM.solvePoissonFFT(rho, N, L, G, phiF);

  /*  The whole comparison in section 8 rests on this: the FFT must invert the
      7-point stencil exactly, not the continuum Laplacian. If the continuum
      k^2 were used instead, this residual would sit at O((kh)^2) ~ 1e-2 and
      never move, and the Jacobi iteration table would be measuring
      discretisation error rather than iteration error. */
  const resF = PM.poissonResidual(phiF, rho, N, L, G);
  ok('FFT solution satisfies the 7-point stencil to round-off',
    resF < 1e-12, 'relative residual ' + e2(resF));

  const phiC = new Float64Array(N3);
  PM.solvePoissonChebyshev(rho, N, L, G, phiC, 220, { rhoMean: 1 / (L * L * L) });
  const agree = PM.relativeL2(phiC, phiF);
  ok('relaxation converges to the FFT answer, not near it',
    agree < 1e-8, 'relative L2 after 220 Chebyshev passes ' + e2(agree));

  /* Jacobi must monotonically reduce the residual, which is the property the
     GPU can check without an FFT to compare against. */
  const phiJ = new Float64Array(N3);
  let prev = Infinity, monotone = true;
  for (let k = 0; k < 8; k++) {
    PM.solvePoissonJacobi(rho, N, L, G, phiJ, 8, { rhoMean: 1 / (L * L * L) });
    const r = PM.poissonResidual(phiJ, rho, N, L, G);
    if (r >= prev) monotone = false;
    prev = r;
  }
  ok('Jacobi residual falls monotonically', monotone, 'after 64 passes ' + e2(prev));

  /* rhoMean passed in analytically must equal the reduction. On the GPU this
     is what lets you skip a full reduction pass every frame. */
  const measured = PM.meanOf(rho, N3);
  const analytic = 1 / (L * L * L);
  ok('rho_mean = total mass / L^3, so the GPU needs no reduction',
    Math.abs(measured - analytic) / analytic < 1e-12,
    'measured ' + measured.toFixed(12) + ' vs ' + analytic);

  /*  Why the mean subtraction is not optional. Write the sweep the obvious
      way, phi = (sum6 - h^2 4 pi G rho)/6 with the raw rho, and every pass
      shifts the mean of phi by -h^2 4 pi G rho_mean / 6. The box has net
      mass, so that shift never cancels: the whole potential slides downward
      without limit, linearly in the pass count, forever. The gradient is
      unaffected in exact arithmetic — but on the GPU phi lives in float32,
      and once the common offset is large the useful field is being carried
      in the low bits of a big number. Subtracting rho_mean (equivalently,
      zeroing the k=0 mode, which is what the FFT does) removes the drive
      entirely and the mean is then conserved exactly.  */
  {
    const naive = new Float64Array(N3), fixed = new Float64Array(N3);
    const h = L / N;
    const perPass = -h * h * 4 * Math.PI * G * analytic / 6;
    const means = [];
    for (let b = 0; b < 4; b++) {
      PM.solvePoissonJacobi(rho, N, L, G, naive, 50, { rhoMean: 0 });
      PM.solvePoissonJacobi(rho, N, L, G, fixed, 50, { rhoMean: analytic });
      means.push([50 * (b + 1), PM.meanOf(naive, N3), PM.meanOf(fixed, N3)]);
    }
    console.log('       passes   <phi> without the subtraction   <phi> with it');
    for (const m of means) {
      console.log('       ' + pad(m[0], 6) + '   ' + pad(m[1].toExponential(4), 26) +
        '   ' + m[2].toExponential(4));
    }
    const last = means[means.length - 1];
    const predicted = perPass * last[0];
    ok('without the subtraction the mean of phi runs away linearly, as predicted',
      Math.abs(last[1] - predicted) / Math.abs(predicted) < 1e-9,
      'after ' + last[0] + ' passes ' + last[1].toExponential(4) +
      ' vs predicted ' + predicted.toExponential(4));
    ok('with the subtraction the mean of phi is conserved exactly',
      Math.abs(last[2]) < 1e-14, '|<phi>| = ' + Math.abs(last[2]).toExponential(2));
  }

  PM.releaseScratch(N);
}

/* ══ 1. mass conservation through deposition ═════════════════════════════ */

section('1. CIC deposition conserves mass');
{
  const L = 1.0, N = 16, N3 = N * N * N, h = L / N;
  const rho = new Float64Array(N3);

  /* random */
  {
    const rnd = PM.mulberry32(20260919);
    const count = 50000;
    const pos = new Float64Array(3 * count);
    for (let i = 0; i < 3 * count; i++) pos[i] = (rnd() - 0.5) * L;
    const total = PM.depositCIC(pos, count, N, L, rho, { mass: 1 / count });
    const got = PM.totalMass(rho, N, L);
    ok('random positions', Math.abs(got - total) / total < 1e-13,
      'total ' + total.toFixed(15) + ' grid ' + got.toFixed(15) + ' rel ' + e2(Math.abs(got - total) / (total || 1)));
  }

  /*  Pathological placements. Every one of these has a floor() or a modulo
      sitting exactly on a boundary, which is where a deposition routine
      silently loses or doubles a particle. */
  const half = L / 2;
  const cases = [
    ['exactly on a cell corner (origin)', [0, 0, 0]],
    ['exactly on an interior node', [-half + 5 * h, -half + 5 * h, -half + 5 * h]],
    ['on a cell face (x on a node, y,z mid-cell)', [-half + 5 * h, -half + 5.5 * h, -half + 5.5 * h]],
    ['on a cell edge', [-half + 5 * h, -half + 5 * h, -half + 5.5 * h]],
    ['the low box corner', [-half, -half, -half]],
    ['the high box edge (wraps to the low one)', [half, half, half]],
    ['one ulp below the high box edge', [half - Number.EPSILON, half, -half]],
    ['just outside the box (wraps)', [half + 0.3 * h, -half - 0.7 * h, half + 1.9 * h]],
    ['far outside the box (wraps many times)', [7.25 * L, -12.5 * L, 3.75 * L]],
    ['dead centre of a cell', [-half + 5.5 * h, -half + 9.5 * h, -half + 0.5 * h]]
  ];
  let worst = 0, worstName = '';
  for (const c of cases) {
    const pos = Float64Array.from(c[1]);
    const total = PM.depositCIC(pos, 1, N, L, rho, { mass: 1 });
    const got = PM.totalMass(rho, N, L);
    const rel = Math.abs(got - total) / total;
    if (rel >= worst) { worst = rel; worstName = c[0]; }
    /* non-negative everywhere too — a wrap bug usually shows as a negative
       weight long before it shows as lost mass */
    let neg = false;
    for (let i = 0; i < N3; i++) if (rho[i] < 0) { neg = true; break; }
    ok(c[0], rel < 1e-13 && !neg, 'rel ' + e2(rel) + (neg ? ' NEGATIVE CELLS' : ''));
  }
  note('worst pathological case: ' + worstName + ' at ' + e2(worst));

  /* all of them at once, with unequal masses */
  {
    const count = cases.length;
    const pos = new Float64Array(3 * count);
    const mass = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      pos[3 * i] = cases[i][1][0]; pos[3 * i + 1] = cases[i][1][1]; pos[3 * i + 2] = cases[i][1][2];
      mass[i] = 0.1 + i * 0.37;
    }
    const total = PM.depositCIC(pos, count, N, L, rho, { mass: mass });
    const got = PM.totalMass(rho, N, L);
    ok('all pathological cases together, unequal masses',
      Math.abs(got - total) / total < 1e-13, 'rel ' + e2(Math.abs(got - total) / total));
  }

  /* and at a size the sim actually uses */
  {
    const N2 = 64, rho2 = new Float64Array(N2 * N2 * N2);
    const rnd = PM.mulberry32(5150);
    const count = 200000;
    const pos = new Float64Array(3 * count);
    for (let i = 0; i < 3 * count; i++) pos[i] = (rnd() - 0.5) * L * 1.4; /* 40% outside, wrapping */
    const total = PM.depositCIC(pos, count, N2, L, rho2, { mass: 1 / count });
    const got = PM.totalMass(rho2, N2, L);
    ok('200k particles on a 64^3 grid, many outside the box',
      Math.abs(got - total) / total < 1e-13, 'rel ' + e2(Math.abs(got - total) / total));
  }
}

/* ══ 2. a single point mass against -G m / r ═════════════════════════════ */

section('2. point mass vs the analytic potential');
{
  const L = 1, G = 1, m = 1, N = 64, N3 = N * N * N, h = L / N;
  /* on a node, so the CIC cloud is a clean delta and the comparison is not
     confounded by the cloud's own shape */
  const pos = new Float64Array([0, 0, 0]);
  const rho = new Float64Array(N3), phi = new Float64Array(N3), ag = new Float64Array(3 * N3);
  PM.depositCIC(pos, 1, N, L, rho, { mass: m });
  PM.solvePoissonFFT(rho, N, L, G, phi);
  PM.gradient(phi, N, L, ag);

  const nb = N / 2;
  const sP = new Float64Array(nb), sA = new Float64Array(nb), sR = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const x = (i - N / 2) * h, y = (j - N / 2) * h, z = (k - N / 2) * h;
        const r = Math.hypot(x, y, z);
        if (r === 0) continue;
        const b = Math.round(r / h);
        if (b < 1 || b >= nb) continue;
        const idx = (k * N + j) * N + i, o = 3 * idx;
        sP[b] += phi[idx];
        sA[b] += -(ag[o] * x + ag[o + 1] * y + ag[o + 2] * z) / r;  /* inward = positive */
        sR[b] += r;
        cnt[b]++;
      }
    }
  }

  /*  The periodic box has no isolated-point potential to compare against: the
      k=0 mode was thrown away, so phi has zero mean while -Gm/r does not, and
      the infinite lattice of images adds a slowly varying offset on top. One
      additive constant absorbs both to leading order; it is fitted over the
      comparison window and printed so you can see it is the size the lattice
      sum predicts, not a fudge that grows to cover a bad solution. The force
      needs no such constant, which is why it is the primary number here. */
  const RLO = 3, RHI = Math.floor(N / 6);
  let acc = 0, nfit = 0;
  for (let b = RLO; b <= RHI; b++) {
    if (!cnt[b]) continue;
    acc += (sP[b] / cnt[b]) - (-G * m / (sR[b] / cnt[b]));
    nfit++;
  }
  const C = acc / nfit;

  /*  Sanity on the offset: the solver forces <phi> = 0 over the box, so if
      phi really is -Gm/r plus a constant, that constant has to be the box
      average of +Gm/r. Computing that average directly (minimum image,
      skipping the singular cell) says what C should be to within the image
      corrections, so the fit cannot be quietly absorbing a bad solution. */
  {
    let s = 0, n = 0;
    for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const x = (i - N / 2) * h, y = (j - N / 2) * h, z = (k - N / 2) * h;
      const r = Math.hypot(x, y, z);
      if (r === 0) continue;
      s += G * m / r; n++;
    }
    const expected = s / n;
    note('fitted offset C = ' + C.toFixed(4) + ', and <G m / r> over the box is ' +
      expected.toFixed(4) + ' — they must agree because the solver sets <phi> = 0');
    ok('the fitted offset is the one the zero-mean condition demands',
      Math.abs(C - expected) / expected < 0.25,
      'C ' + C.toFixed(4) + ' vs ' + expected.toFixed(4) +
      ' (' + pct(Math.abs(C - expected) / expected) + ' apart; the rest is the image lattice)');
  }
  console.log('       r/h     r        phi_pm        -Gm/r + C     err%      a_pm         Gm/r^2       err%');

  let maxPhiErr = 0, maxAccErr = 0;
  for (let b = 1; b < nb; b++) {
    if (!cnt[b]) continue;
    const r = sR[b] / cnt[b];
    const pp = sP[b] / cnt[b], pe = -G * m / r + C;
    const aa = sA[b] / cnt[b], ae = G * m / (r * r);
    const ep = Math.abs(pp - pe) / Math.abs(G * m / r);
    const ea = (aa - ae) / ae;
    const inWindow = b >= RLO && b <= RHI;
    if (inWindow) {
      if (ep > maxPhiErr) maxPhiErr = ep;
      if (Math.abs(ea) > maxAccErr) maxAccErr = Math.abs(ea);
    }
    if (b <= 20 || b % 4 === 0) {
      console.log('       ' + pad(b, 3) + '   ' + r.toFixed(5) + '  ' + pad(pp.toExponential(4), 12) +
        '  ' + pad(pe.toExponential(4), 12) + '  ' + pad((ep * 100).toFixed(2), 6) +
        '  ' + pad(aa.toExponential(4), 12) + ' ' + pad(ae.toExponential(4), 12) +
        '  ' + pad((ea * 100).toFixed(2), 7) + (inWindow ? '   <' : ''));
    }
  }
  note('window marked "<" is ' + RLO + ' <= r/h <= ' + RHI +
    '; past that the periodic images dominate, which is physics, not solver error');
  ok('potential within 1% of -Gm/r for 3 <= r/h <= ' + RHI,
    maxPhiErr < 0.01, 'worst ' + pct(maxPhiErr));
  ok('force within 3% of Gm/r^2 for 3 <= r/h <= ' + RHI,
    maxAccErr < 0.03, 'worst ' + pct(maxAccErr));

  PM.releaseScratch(N);
}

/* ══ 3. momentum conservation ════════════════════════════════════════════ */

section('3. momentum conservation over 2000 steps');
{
  const L = 1, G = 1, N = 32, h = L / N, m = 1;
  const d = 6 * h;
  const v = Math.sqrt(G * m / (2 * d));
  const T = 2 * Math.PI * (d / 2) / v;
  const steps = QUICK ? 400 : 2000;
  const dt = T / 64;

  /*  Deliberately not symmetric about the origin and with a net drift, so
      that a momentum leak would show as the pair sliding, not cancel out. */
  const st = {
    count: 2,
    pos: new Float64Array([-d / 2 + 0.031, 0.017, -0.024, d / 2 + 0.031, 0.017, -0.024]),
    vel: new Float64Array([0.13, -v + 0.05, 0.02, 0.13, v + 0.05, 0.02]),
    mass: m
  };
  const params = { N, L, G, mass: m, iterations: 0 };
  const P0 = [
    m * (st.vel[0] + st.vel[3]),
    m * (st.vel[1] + st.vel[4]),
    m * (st.vel[2] + st.vel[5])
  ];
  const scale = m * v;
  let worst = 0;
  for (let s = 0; s < steps; s++) {
    PM.step(st, dt, params);
    const dP = Math.hypot(
      m * (st.vel[0] + st.vel[3]) - P0[0],
      m * (st.vel[1] + st.vel[4]) - P0[1],
      m * (st.vel[2] + st.vel[5]) - P0[2]);
    if (dP / scale > worst) worst = dP / scale;
  }
  /*  Exactly zero in exact arithmetic: the total force is a quadratic form in
      rho with an odd kernel (see the note at the top of pm.js), so what is
      left is float64 round-off accumulated over the run. */
  ok(steps + ' steps, isolated pair, |dP|/(m v) stays at round-off',
    worst < 1e-11, 'worst ' + e2(worst) + ' over ' + steps + ' steps');

  /* and with many particles, where cancellation is not trivial */
  const f = makeField(20000, 77);
  const st2 = { count: f.count, pos: f.pos, vel: f.vel, mass: 1 / f.count };
  const p2 = { N, L, G, mass: st2.mass, iterations: 0 };
  const sumP = (s) => {
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < s.count; i++) { x += s.vel[3 * i]; y += s.vel[3 * i + 1]; z += s.vel[3 * i + 2]; }
    return [x * s.mass, y * s.mass, z * s.mass];
  };
  const q0 = sumP(st2);
  const sc2 = st2.mass * rmsSpeed(st2.vel, st2.count) * Math.sqrt(st2.count);
  for (let s = 0; s < (QUICK ? 10 : 40); s++) PM.step(st2, 1e-3, p2);
  const q1 = sumP(st2);
  const dq = Math.hypot(q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]) / sc2;
  ok('20000 particles, total momentum unchanged', dq < 1e-11, 'rel ' + e2(dq));

  PM.releaseScratch(N);
}

/* ══ 4. a two-body circular orbit ════════════════════════════════════════ */

section('4. two-body circular orbit: radius and energy');
{
  const L = 1, G = 1, N = 64, h = L / N, m = 1;
  const d = 8 * h;                        /* 8 cells — comfortably resolved */
  const v = Math.sqrt(G * m / (2 * d));   /* set up analytically: m v^2/(d/2) = G m^2/d^2 */
  const T = 2 * Math.PI * (d / 2) / v;
  const perOrbit = 128;
  const orbits = QUICK ? 1 : 4;
  const dt = T / perOrbit;
  const steps = perOrbit * orbits;

  note('d = 8h = ' + d.toFixed(5) + ', v = ' + v.toFixed(4) + ', T = ' + e2(T) +
    ', dt = T/' + perOrbit + ' (v dt/h = ' + (v * dt / h).toFixed(3) + '), ' + orbits + ' orbits');

  const st = {
    count: 2,
    pos: new Float64Array([-d / 2, 0, 0, d / 2, 0, 0]),
    vel: new Float64Array([0, -v, 0, 0, v, 0]),
    mass: m
  };
  const params = { N, L, G, mass: m, iterations: 0 };

  let dmin = Infinity, dmax = 0, E0 = null, Emin = Infinity, Emax = -Infinity;
  const ts = [], Es = [];
  for (let s = 0; s < steps; s++) {
    PM.step(st, dt, params);
    const dx = PM.minImage(st.pos[3] - st.pos[0], L);
    const dy = PM.minImage(st.pos[4] - st.pos[1], L);
    const dz = PM.minImage(st.pos[5] - st.pos[2], L);
    const sep = Math.hypot(dx, dy, dz);
    if (sep < dmin) dmin = sep;
    if (sep > dmax) dmax = sep;
    /*  Energy from the analytic two-body expression, not from the grid
        potential. sum(1/2) m phi(x_p) would also count each particle's own
        self-potential, which is finite on a grid and wobbles as the particle
        moves across a cell — a bookkeeping artefact, not a physical energy. */
    const K = 0.5 * m * (st.vel[0] ** 2 + st.vel[1] ** 2 + st.vel[2] ** 2)
            + 0.5 * m * (st.vel[3] ** 2 + st.vel[4] ** 2 + st.vel[5] ** 2);
    const E = K - G * m * m / sep;
    if (E0 === null) E0 = E;
    if (E < Emin) Emin = E;
    if (E > Emax) Emax = E;
    ts.push(s * dt); Es.push((E - E0) / Math.abs(E0));
  }

  const spread = (dmax - dmin) / d;
  const eSpread = (Emax - Emin) / Math.abs(E0);
  const trend = slope(ts, Es) * (steps * dt);      /* total systematic change */
  const eFinal = Es[Es.length - 1];

  console.log('       orbit   t/T      dE/E0');
  for (let o = 1; o <= orbits; o++) {
    const i = o * perOrbit - 1;
    console.log('       ' + pad(o, 5) + '   ' + (ts[i] / T).toFixed(2) + '   ' + e2(Es[i]));
  }
  note('separation ranged over [' + dmin.toFixed(5) + ', ' + dmax.toFixed(5) + '] against d = ' + d.toFixed(5));
  note('energy spread ' + pct(eSpread) + ', final ' + pct(eFinal) +
    ', least-squares trend over the whole run ' + pct(trend));

  ok('radius held to within 3% over ' + orbits + ' orbits', spread < 0.03, 'spread ' + pct(spread));
  ok('energy spread under 1%', eSpread < 0.01, pct(eSpread));
  /*  Symplectic integrators oscillate; they do not drift. If the trend is a
      good fraction of the oscillation, the leapfrog has been broken into an
      Euler step somewhere. */
  ok('energy oscillates rather than drifting monotonically',
    Math.abs(trend) < 0.5 * eSpread, '|trend| ' + pct(Math.abs(trend)) + ' vs spread ' + pct(eSpread));

  FINDINGS.orbitSpread = spread;
  FINDINGS.orbitEnergySpread = eSpread;
  PM.releaseScratch(N);
}

/* ══ 4b. how large a timestep survives ═══════════════════════════════════ */

section('4b. timestep stability ladder');
{
  const L = 1, G = 1, N = 32, h = L / N, m = 1;
  const d = 6 * h;
  const v = Math.sqrt(G * m / (2 * d));
  const T = 2 * Math.PI * (d / 2) / v;
  const orbits = 3;
  console.log('       steps/orbit   v dt/h   dt/T     sep spread   E spread   E final');
  const rows = [];
  for (const nstep of [8, 12, 16, 24, 32, 48, 64, 128]) {
    const dt = T / nstep, steps = nstep * orbits;
    const st = {
      count: 2,
      pos: new Float64Array([-d / 2, 0, 0, d / 2, 0, 0]),
      vel: new Float64Array([0, -v, 0, 0, v, 0]),
      mass: m
    };
    const params = { N, L, G, mass: m, iterations: 0 };
    let dmin = Infinity, dmax = 0, E0 = null, Emin = Infinity, Emax = -Infinity, Elast = 0, blew = false;
    for (let s = 0; s < steps; s++) {
      PM.step(st, dt, params);
      const dx = PM.minImage(st.pos[3] - st.pos[0], L);
      const dy = PM.minImage(st.pos[4] - st.pos[1], L);
      const dz = PM.minImage(st.pos[5] - st.pos[2], L);
      const sep = Math.hypot(dx, dy, dz);
      if (!isFinite(sep)) { blew = true; break; }
      if (sep < dmin) dmin = sep;
      if (sep > dmax) dmax = sep;
      const K = 0.5 * m * (st.vel[0] ** 2 + st.vel[1] ** 2 + st.vel[2] ** 2)
              + 0.5 * m * (st.vel[3] ** 2 + st.vel[4] ** 2 + st.vel[5] ** 2);
      const E = K - G * m * m / sep;
      if (E0 === null) E0 = E;
      if (E < Emin) Emin = E;
      if (E > Emax) Emax = E;
      Elast = E;
    }
    const courant = v * dt / h;
    if (blew) {
      console.log('       ' + pad(nstep, 11) + '   ' + pad(courant.toFixed(3), 6) + '   BLEW UP');
      rows.push([courant, Infinity]);
      continue;
    }
    const sp = (dmax - dmin) / d, es = (Emax - Emin) / Math.abs(E0);
    console.log('       ' + pad(nstep, 11) + '   ' + pad(courant.toFixed(3), 6) +
      '   ' + pad((dt / T).toFixed(4), 6) + '   ' + pad(pct(sp), 9) +
      '   ' + pad(pct(es), 8) + '   ' + pad(pct((Elast - E0) / Math.abs(E0)), 8));
    rows.push([courant, sp]);
  }
  /* the largest Courant number that still holds the radius to 5% */
  let best = 0;
  for (const r of rows) if (r[1] < 0.05 && r[0] > best) best = r[0];
  FINDINGS.courant = best;
  note('largest v dt / h that still holds the orbit radius to 5%: ' + best.toFixed(2));
  ok('the leapfrog is stable out to v dt/h >= 0.5', best >= 0.5, 'measured ' + best.toFixed(2));
  ok('and degrades gracefully rather than exploding at 4x that',
    rows[0][1] === Infinity ? false : true,
    'at v dt/h = ' + rows[0][0].toFixed(2) + ' the radius spread is ' +
      (rows[0][1] === Infinity ? 'NaN' : pct(rows[0][1])));
  PM.releaseScratch(N);
}

/* ══ 5. Newton's third law ═══════════════════════════════════════════════ */

section("5. Newton's third law");
{
  const L = 1, G = 1, N = 64, N3 = N * N * N, h = L / N, m = 1;
  const rho = new Float64Array(N3), phi = new Float64Array(N3);
  const ag = new Float64Array(3 * N3), ap = new Float64Array(6);
  const pos = new Float64Array(6);

  function forces(p) {
    pos.set(p);
    PM.depositCIC(pos, 2, N, L, rho, { mass: m });
    PM.solvePoissonFFT(rho, N, L, G, phi);
    PM.gradient(phi, N, L, ag);
    PM.sampleCIC(ag, N, L, pos, 2, ap);
    return ap;
  }

  /*  The node lattice is invariant under x -> -x modulo L (node i maps to
      node N-i), so a pair placed symmetrically about the origin must produce
      exactly opposite forces. "Exactly" here means to float64 round-off — the
      two particles' arithmetic runs through different memory, so the bits do
      not have to match, only the values. */
  const configs = [
    ['on nodes, along x', [-4 * h, 0, 0, 4 * h, 0, 0]],
    ['on nodes, along a face diagonal', [-3 * h, -3 * h, 0, 3 * h, 3 * h, 0]],
    ['on nodes, along the body diagonal', [-2 * h, -2 * h, -2 * h, 2 * h, 2 * h, 2 * h]],
    ['mid-cell, along x', [-4.5 * h, 0, 0, 4.5 * h, 0, 0]],
    ['generic sub-cell offsets', [-0.0731, 0.0412, -0.0195, 0.0731, -0.0412, 0.0195]],
    ['very close (1 cell apart)', [-0.5 * h, 0, 0, 0.5 * h, 0, 0]],
    ['nearly a third of the box apart', [-0.16, 0.03, -0.05, 0.16, -0.03, 0.05]]
  ];
  let worst = 0;
  for (const c of configs) {
    const a = forces(c[1]);
    const mag = Math.hypot(a[0], a[1], a[2]);
    const rel = Math.hypot(a[0] + a[3], a[1] + a[4], a[2] + a[5]) / mag;
    if (rel > worst) worst = rel;
    ok(c[0] + ': F1 = -F2', rel < 1e-9, '|F1+F2|/|F1| = ' + e2(rel) + ', |F1| = ' + e2(mag));
  }

  /*  Off-centre too: translation invariance means the identity cannot depend
      on where the pair sits, only on their separation. Here the pair is NOT
      symmetric about the origin, so this is the sum rule rather than a
      mirror symmetry — and it is the one that makes momentum conserve. */
  {
    const a = forces([0.2013, -0.1077, 0.0451, 0.2013 + 5 * h, -0.1077 + 2 * h, 0.0451 - 3 * h]);
    const mag = Math.hypot(a[0], a[1], a[2]);
    const rel = Math.hypot(a[0] + a[3], a[1] + a[4], a[2] + a[5]) / mag;
    if (rel > worst) worst = rel;
    ok('off-centre pair, no mirror symmetry to help: sum of forces is zero',
      rel < 1e-9, e2(rel));
  }

  /*  A ring of eight particles at the cube corners: by symmetry every force
      must point at the centre and all magnitudes must agree. */
  {
    const s = 5 * h, n = 8;
    const p = new Float64Array(3 * n);
    let q = 0;
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
      p[3 * q] = sx * s; p[3 * q + 1] = sy * s; p[3 * q + 2] = sz * s; q++;
    }
    const rho8 = new Float64Array(N3), phi8 = new Float64Array(N3), a8 = new Float64Array(3 * n);
    PM.depositCIC(p, n, N, L, rho8, { mass: m });
    PM.solvePoissonFFT(rho8, N, L, G, phi8);
    PM.gradient(phi8, N, L, ag);
    PM.sampleCIC(ag, N, L, p, n, a8);
    let mags = [], radial = 0;
    for (let i = 0; i < n; i++) {
      const ax = a8[3 * i], ay = a8[3 * i + 1], az = a8[3 * i + 2];
      const mg = Math.hypot(ax, ay, az);
      mags.push(mg);
      const ux = p[3 * i] / (s * Math.sqrt(3)), uy = p[3 * i + 1] / (s * Math.sqrt(3)), uz = p[3 * i + 2] / (s * Math.sqrt(3));
      const rad = -(ax * ux + ay * uy + az * uz);
      radial = Math.max(radial, Math.abs(mg - rad) / mg);   /* 0 if purely inward */
    }
    const spread = (Math.max.apply(null, mags) - Math.min.apply(null, mags)) / mags[0];
    ok('cube of 8: all magnitudes equal', spread < 1e-9, 'spread ' + e2(spread));
    ok('cube of 8: every force points exactly at the centre', radial < 1e-9, e2(radial));
  }

  note('worst antisymmetry over all configurations: ' + e2(worst) + ' (float64 round-off)');
  PM.releaseScratch(N);
}

/* ══ 6. a Plummer sphere in virial equilibrium ═══════════════════════════ */

section('6. Plummer sphere holds together');
{
  const L = 1, G = 1, M = 1, a = 0.08, N = 64, h = L / N;
  const np = 4096, rcut = 0.30;
  const steps = QUICK ? 200 : 1000;
  const dt = 2e-4;
  const eps = 1.2 * h;
  const tdyn = Math.sqrt(a * a * a / (G * M));

  const st = plummerSphere(np, { G, M, a, rcut, seed: 20260919 });
  const params = { N, L, G, mass: st.mass, iterations: 0 };

  note('a = ' + a + ' = ' + (a / h).toFixed(1) + ' cells, ' + np + ' particles, M = ' + M +
    ', t_dyn = ' + e2(tdyn) + ', dt = ' + e2(dt) + ' (' + (steps * dt / tdyn).toFixed(1) + ' t_dyn total)');

  /*  The potential energy is a direct softened pair sum, not the grid
      potential. In a periodic box the grid phi also contains the cluster's
      interaction with its own infinite lattice of images, a large constant
      offset that would make 2T/|W| read ~0.7 forever and mean nothing. The
      direct sum uses the same softening the grid imposes so the comparison is
      like for like. */
  function energies() {
    const m = st.mass;
    let T = 0;
    for (let i = 0; i < np; i++) {
      T += 0.5 * m * (st.vel[3 * i] ** 2 + st.vel[3 * i + 1] ** 2 + st.vel[3 * i + 2] ** 2);
    }
    let W = 0;
    const e2s = eps * eps;
    for (let i = 0; i < np; i++) {
      const x = st.pos[3 * i], y = st.pos[3 * i + 1], z = st.pos[3 * i + 2];
      for (let j = i + 1; j < np; j++) {
        const dx = x - st.pos[3 * j], dy = y - st.pos[3 * j + 1], dz = z - st.pos[3 * j + 2];
        W -= G * m * m / Math.sqrt(dx * dx + dy * dy + dz * dz + e2s);
      }
    }
    return { T, W };
  }
  const radii = new Float64Array(np);
  function lagrange(frac) {
    for (let i = 0; i < np; i++) radii[i] = Math.hypot(st.pos[3 * i], st.pos[3 * i + 1], st.pos[3 * i + 2]);
    const s = Array.prototype.slice.call(radii).sort((p, q) => p - q);
    return s[Math.min(np - 1, Math.floor(frac * np))];
  }

  console.log('       step   t/t_dyn   2T/|W|    r10       r50       r90       dE/E0');
  const virials = [], r50s = [], dEs = [];
  let E0 = null;
  const every = Math.max(1, Math.round(steps / 10));
  for (let s = 0; s <= steps; s++) {
    if (s % every === 0 || s === steps) {
      const e = energies();
      const E = e.T + e.W;
      if (E0 === null) E0 = E;
      const vr = 2 * e.T / Math.abs(e.W);
      const r50 = lagrange(0.5);
      virials.push(vr); r50s.push(r50); dEs.push((E - E0) / Math.abs(E0));
      console.log('       ' + pad(s, 4) + '   ' + pad((s * dt / tdyn).toFixed(2), 7) +
        '   ' + vr.toFixed(4) + '   ' + lagrange(0.1).toFixed(5) + '   ' + r50.toFixed(5) +
        '   ' + lagrange(0.9).toFixed(5) + '   ' + pad(pct((E - E0) / Math.abs(E0)), 8));
    }
    if (s < steps) PM.step(st, dt, params);
  }

  const vmin = Math.min.apply(null, virials), vmax = Math.max.apply(null, virials);
  const rmin = Math.min.apply(null, r50s), rmax = Math.max.apply(null, r50s);
  const dEmax = Math.max.apply(null, dEs.map(Math.abs));
  note('virial ratio stayed in [' + vmin.toFixed(3) + ', ' + vmax.toFixed(3) +
    '], half-mass radius in [' + rmin.toFixed(5) + ', ' + rmax.toFixed(5) + ']');
  note('the slow rise then fall of 2T/|W| is a breathing mode: the analytic Plummer ' +
    'velocities assume a Newtonian core, the grid softens it, so the sphere settles');

  ok('virial ratio stays near 1', vmin > 0.85 && vmax < 1.25,
    '[' + vmin.toFixed(3) + ', ' + vmax.toFixed(3) + ']');
  ok('half-mass radius neither collapses nor blows up',
    rmax / rmin < 1.25, 'r50 varied by ' + pct(rmax / rmin - 1));
  ok('total energy stays within 5%', dEmax < 0.05, 'worst ' + pct(dEmax));
  ok('all particles still inside the box and finite', (() => {
    for (let i = 0; i < 3 * np; i++) if (!Number.isFinite(st.pos[i]) || Math.abs(st.pos[i]) > L) return false;
    return true;
  })());

  FINDINGS.plummerVirial = [vmin, vmax];
  PM.releaseScratch(N);
}

/* ══ 7. self-force ═══════════════════════════════════════════════════════ */

section('7. self-force');
{
  const L = 1, G = 1, m = 1;
  /*  A lone particle must feel nothing. The proof is at the top of pm.js:
      the deposit-solve-gradient-interpolate chain is an odd kernel summed
      against an even autocorrelation, which cancels term by term for ANY
      sub-cell position. What is measured here is float64 round-off, which is
      why the numbers are ~1e-17 of the characteristic force and not ~1e-3. */
  const positions = [
    ['exactly on a node', [0, 0, 0]],
    ['dead centre of a cell', [1 / 128, 1 / 128, 1 / 128]],
    ['generic sub-cell position', [0.0131, -0.2072, 0.08813]],
    ['near the box edge', [0.5 - 1e-4, -0.5 + 1e-4, 0.4999]],
    ['on a cell face', [3 / 64, 0.0231, -0.0177]]
  ];
  for (const N of [32, 64]) {
    const N3 = N * N * N, h = L / N;
    const rho = new Float64Array(N3), phi = new Float64Array(N3);
    const ag = new Float64Array(3 * N3), ap = new Float64Array(3);
    const scale = G * m / (h * h);   /* the force a neighbour one cell away would give */
    for (const c of positions) {
      const pos = Float64Array.from(c[1]);
      PM.depositCIC(pos, 1, N, L, rho, { mass: m });
      PM.solvePoissonFFT(rho, N, L, G, phi);
      PM.gradient(phi, N, L, ag);
      PM.sampleCIC(ag, N, L, pos, 1, ap);
      const rel = Math.hypot(ap[0], ap[1], ap[2]) / scale;
      ok('N=' + N + ' FFT, ' + c[0], rel < 1e-12, '|a| / (G m / h^2) = ' + e2(rel));
    }
    /*  And with the relaxation solver, because K Jacobi passes from a zero
        start are a polynomial in the (even) stencil and inherit the same
        cancellation — an odd number of passes included, in case anyone
        suspects the ping-pong parity. */
    for (const K of [1, 7, 32]) {
      const pos = Float64Array.from([0.0131, -0.2072, 0.08813]);
      const pj = new Float64Array(N3);
      PM.depositCIC(pos, 1, N, L, rho, { mass: m });
      PM.solvePoissonJacobi(rho, N, L, G, pj, K, { rhoMean: m / (L * L * L) });
      PM.gradient(pj, N, L, ag);
      PM.sampleCIC(ag, N, L, pos, 1, ap);
      const rel = Math.hypot(ap[0], ap[1], ap[2]) / (G * m / (h * h));
      ok('N=' + N + ' Jacobi x' + K + ', generic position', rel < 1e-12, e2(rel));
    }
    PM.releaseScratch(N);
  }
}

/* ══ 7b. the softening the grid implies ══════════════════════════════════ */

section('7b. effective softening of the grid force');
{
  const L = 1, G = 1, N = 64, N3 = N * N * N, h = L / N, m = 1;
  const rho = new Float64Array(N3), phi = new Float64Array(N3);
  const ag = new Float64Array(3 * N3), ap = new Float64Array(6);
  const pos = new Float64Array(6);
  const rnd = PM.mulberry32(31337);
  const trials = QUICK ? 12 : 32;

  /*  The pair force is measured over random orientations AND random sub-cell
      placements of the pair's midpoint, because the PM force depends on both.
      The mean tells you the softening; the scatter tells you the grid
      anisotropy, and it is the scatter, not the mean, that decides how far
      apart two particles have to be before the force can be trusted. */
  function pair(r) {
    const vals = [];
    for (let t = 0; t < trials; t++) {
      let ux, uy, uz, s2;
      do {
        ux = 2 * rnd() - 1; uy = 2 * rnd() - 1; uz = 2 * rnd() - 1;
        s2 = ux * ux + uy * uy + uz * uz;
      } while (s2 > 1 || s2 < 1e-6);
      const s = Math.sqrt(s2); ux /= s; uy /= s; uz /= s;
      const cx = (rnd() - 0.5) * h, cy = (rnd() - 0.5) * h, cz = (rnd() - 0.5) * h;
      pos[0] = cx - 0.5 * r * ux; pos[1] = cy - 0.5 * r * uy; pos[2] = cz - 0.5 * r * uz;
      pos[3] = cx + 0.5 * r * ux; pos[4] = cy + 0.5 * r * uy; pos[5] = cz + 0.5 * r * uz;
      PM.depositCIC(pos, 2, N, L, rho, { mass: m });
      PM.solvePoissonFFT(rho, N, L, G, phi);
      PM.gradient(phi, N, L, ag);
      PM.sampleCIC(ag, N, L, pos, 2, ap);
      vals.push(ap[0] * ux + ap[1] * uy + ap[2] * uz);
    }
    const mean = vals.reduce((x, y) => x + y, 0) / vals.length;
    let vsum = 0;
    for (const v of vals) vsum += (v - mean) * (v - mean);
    return { mean, sd: Math.sqrt(vsum / vals.length) };
  }

  console.log('       r/h    F_pm / F_newton   scatter    implied Plummer eps/h');
  const ladder = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8];
  const ratios = [];
  for (const rc of ladder) {
    const r = rc * h;
    const p = pair(r);
    const ratio = p.mean / (G * m / (r * r));
    ratios.push([rc, ratio, p.sd / Math.abs(p.mean)]);
    /*  A Plummer-equivalent epsilon is only meaningful where the force is
        actually suppressed. Past ~1.5 cells the PM kernel recovers far more
        sharply than a Plummer sphere does, so quoting an epsilon there would
        be fitting a shape the force does not have. */
    const epsStr = (rc <= 1.5 && ratio > 0 && ratio < 1)
      ? (rc * Math.sqrt(Math.pow(ratio, -2 / 3) - 1)).toFixed(3)
      : '   -  ';
    console.log('       ' + pad(rc, 4) + '    ' + pad(ratio.toFixed(4), 7) +
      '         ' + pad(pct(p.sd / Math.abs(p.mean)), 7) + '     ' + epsStr);
  }

  /* half-force radius: where F_pm / F_newton crosses 0.5 */
  let halfR = null;
  for (let i = 1; i < ratios.length; i++) {
    if (ratios[i - 1][1] < 0.5 && ratios[i][1] >= 0.5) {
      const t = (0.5 - ratios[i - 1][1]) / (ratios[i][1] - ratios[i - 1][1]);
      halfR = ratios[i - 1][0] + t * (ratios[i][0] - ratios[i - 1][0]);
      break;
    }
  }
  /* smallest separation at which the mean force is within 2% and the
     placement scatter is under 5% */
  let trust2 = null, trust5 = null;
  for (const r of ratios) {
    if (trust2 === null && Math.abs(r[1] - 1) < 0.02) trust2 = r[0];
    if (trust5 === null && r[2] < 0.05 && Math.abs(r[1] - 1) < 0.03) trust5 = r[0];
  }
  note('half-force radius (the softening) ~ ' + (halfR === null ? '?' : halfR.toFixed(2)) + ' cells');
  note('mean force within 2% of Newton from r >= ' + trust2 + ' cells');
  note('placement scatter under 5% from r >= ' + trust5 + ' cells — this is the real resolution limit');

  ok('force is suppressed, not amplified, below one cell', ratios[0][1] < 0.3, 'F/F_N at r=h/2 is ' + ratios[0][1].toFixed(3));
  ok('force recovers to Newtonian by a few cells', Math.abs(ratios[ratios.length - 1][1] - 1) < 0.05,
    'F/F_N at r=8h is ' + ratios[ratios.length - 1][1].toFixed(4));
  ok('half-force radius is of order one cell', halfR !== null && halfR > 0.5 && halfR < 2.0,
    halfR === null ? 'not bracketed' : halfR.toFixed(2) + ' cells');

  FINDINGS.softening = halfR;
  FINDINGS.trust2 = trust2;
  FINDINGS.trust5 = trust5;
  PM.releaseScratch(N);
}

/* ══ 8. THE HEADLINE EXPERIMENT ══════════════════════════════════════════ */

section('8. how many relaxation passes per frame');
console.log(`
  Everything above establishes that the reference is right. This is what the
  reference was built to answer: with the potential warm-started from the
  previous frame, how many stencil passes does the GPU have to run to stay
  within a few percent of exact?

  Two relaxations are measured side by side. Both use the identical 7-point
  sweep, both are double-buffered, both warm-start from last frame's phi:

    JACOBI     phi_new = (sum of 6 neighbours - h^2 4 pi G (rho - rho_mean))/6
    CHEBYSHEV  the same sweep, but the next iterate is a Chebyshev combination
               of the sweep's output and the two previous iterates

  Error is the relative L2 difference from solvePoissonFFT, which inverts the
  same stencil exactly, so what is measured is iteration error and nothing
  else. Both the potential error and the error in the interpolated particle
  acceleration are reported, because they are not the same number.
`);

const HEAD = {};

/* ── 8a. cold start: the raw convergence rate ───────────────────────────── */
{
  const L = 1, G = 1;
  const Ns = QUICK ? [32, 64] : [32, 64, 128];
  const ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

  console.log('  --- 8a. cold start (phi = 0), single frame -----------------------------');
  console.log('      the asymptotic per-pass factors are analytic: Jacobi damps the');
  console.log('      longest wave by (2 + cos(2 pi/N))/3 per pass, Chebyshev by');
  console.log('      exp(-acosh(eta)) — a square root of the condition number better.');
  console.log('');
  for (const N of Ns) {
    const N3 = N * N * N;
    const f = makeField(120000, 4242);
    const rho = new Float64Array(N3), phiF = new Float64Array(N3);
    PM.depositCIC(f.pos, f.count, N, L, rho, { mass: 1 / f.count });
    PM.solvePoissonFFT(rho, N, L, G, phiF);
    const o = { rhoMean: 1 / (L * L * L) };

    console.log('      N=' + N + '   Jacobi rate/pass ' + PM.jacobiRate(N).toFixed(6) +
      '   Chebyshev rate/pass ' + PM.chebyshevRate(N).toFixed(4) +
      '   (' + Math.round(Math.log(0.01) / Math.log(PM.jacobiRate(N))) + ' vs ' +
      Math.round(Math.log(0.01) / Math.log(PM.chebyshevRate(N))) + ' passes for 1% from cold)');
    console.log('        K      Jacobi      Chebyshev');
    const j = new Float64Array(N3);
    let done = 0;
    const jrow = [], crow = [];
    for (const K of ladder) {
      PM.solvePoissonJacobi(rho, N, L, G, j, K - done, o);
      done = K;
      jrow.push(PM.relativeL2(j, phiF));
    }
    for (const K of ladder) {
      const c = new Float64Array(N3);            /* Chebyshev restarts, so no incremental trick */
      PM.solvePoissonChebyshev(rho, N, L, G, c, K, o);
      crow.push(PM.relativeL2(c, phiF));
    }
    for (let i = 0; i < ladder.length; i++) {
      console.log('      ' + pad(ladder[i], 5) + '     ' + pad(e2(jrow[i]), 8) + '    ' + pad(e2(crow[i]), 8));
    }
    console.log('');
    PM.releaseScratch(N);
  }
}

/* ── 8b. warm start: the number that matters ────────────────────────────── */
{
  const L = 1, G = 1, COUNT = 120000;

  /*  Protocol. Every variant sees exactly the same sequence of density
      fields, because the particles are advected by a fixed kinematic
      velocity field rather than by gravity — so a variant with a large error
      cannot alter the problem it is being scored on. Each variant carries its
      own phi from frame to frame and gets its budget of K passes on it. The
      error is read at the last frame, by which point the recycling has
      reached whatever accuracy that budget can sustain.
  */
  function runWarm(N, frames, spec, disp) {
    const N3 = N * N * N, h = L / N, m = 1 / COUNT;
    const f = makeField(COUNT, 4242);
    const dt = disp * h / rmsSpeed(f.vel, COUNT);
    const rho = new Float64Array(N3), phiF = new Float64Array(N3), err = new Float64Array(N3);
    const aerr = new Float64Array(N3);
    const agF = new Float64Array(3 * N3), agV = new Float64Array(3 * N3);
    const apF = new Float64Array(3 * COUNT), apV = new Float64Array(3 * COUNT);
    const vars = spec.map(() => new Float64Array(N3));
    const hist = spec.map(() => []);
    const o = { rhoMean: 1 / (L * L * L) };
    let final = null;
    /*  How much the exact potential itself changes from frame to frame. This
        is the quantity the whole budget is fighting: a solver that reduces
        the error by a factor f per frame settles at roughly delta/(1-f), so
        knowing delta tells you what f — and therefore what K — you need. */
    const phiPrev = new Float64Array(N3);
    let deltaSum = 0, deltaN = 0;

    for (let fr = 0; fr < frames; fr++) {
      PM.depositCIC(f.pos, COUNT, N, L, rho, { mass: m });
      PM.solvePoissonFFT(rho, N, L, G, phiF);
      if (fr > 0) { deltaSum += PM.relativeL2(phiPrev, phiF); deltaN++; }
      phiPrev.set(phiF);
      for (let i = 0; i < spec.length; i++) {
        spec[i].solve(rho, N, L, G, vars[i], spec[i].K, o);
        hist[i].push(PM.relativeL2(vars[i], phiF));
      }
      if (fr === frames - 1) {
        PM.gradient(phiF, N, L, agF);
        PM.sampleCIC(agF, N, L, f.pos, COUNT, apF);
        final = spec.map((s, i) => {
          PM.gradient(vars[i], N, L, agV);
          PM.sampleCIC(agV, N, L, f.pos, COUNT, apV);
          for (let q = 0; q < N3; q++) err[q] = vars[i][q] - phiF[q];
          /*  The brief asks whether the force error is a smooth bias or
              noise. Answer it about the FORCE, not about phi: take the
              x-component of the acceleration error on the grid and measure
              its wavelength. Tens of cells is a coherent bias the eye reads
              as slightly-wrong gravity; two to four cells is grid hash that
              makes particles jitter. */
          for (let q = 0; q < N3; q++) aerr[q] = agV[3 * q] - agF[3 * q];
          const Rp = PM.roughness(err, N);
          const Ra = PM.roughness(aerr, N);
          /*  Has the recycling settled? A warm-started solver starts from
              phi = 0 on frame 1 and the error falls for as long as the
              recycling keeps gaining more than the moving field loses. The
              time constant is the same slow mode the solver is bad at, so a
              too-short run reports an error that is still on its way down and
              flatters the budget. Comparing the last frame with 60% of the
              way through says whether the number can be trusted. */
          const eLast = hist[i][hist[i].length - 1];
          const eMid = hist[i][Math.floor((frames - 1) * 0.6)];
          return {
            fErr: PM.relativeL2Raw(apV, apF),
            lamPhi: PM.roughnessWavelength(Rp),
            lamAcc: PM.roughnessWavelength(Ra),
            settled: Math.abs(eLast - eMid) <= 0.15 * eMid
          };
        });
      }
      for (let q = 0; q < 3 * COUNT; q++) f.pos[q] = PM.wrapCoord(f.pos[q] + dt * f.vel[q], L);
    }
    return { hist, final, frames, delta: deltaN ? deltaSum / deltaN : 0 };
  }

  /*  The ladders below are coarse because a Jacobi point at N=128 costs real
      seconds. The pass count needed for a given accuracy is read off by
      interpolating geometrically between the two bracketing measurements —
      the steady-state error falls smoothly and near-geometrically in K, so
      this is an interpolation, not an extrapolation, wherever it returns a
      value inside the ladder. */
  function passesFor(points, target) {
    if (!points.length) return null;
    if (points[0][1] <= target) return points[0][0];
    for (let i = 1; i < points.length; i++) {
      const k0 = points[i - 1][0], e0 = points[i - 1][1];
      const k1 = points[i][0], e1 = points[i][1];
      if (e0 > target && e1 <= target) {
        const t = (Math.log(target) - Math.log(e0)) / (Math.log(e1) - Math.log(e0));
        return Math.ceil(k0 * Math.pow(k1 / k0, t));
      }
    }
    return null;
  }

  const DISP = 0.3;
  const TARGET = 0.03;
  console.log('  --- 8b. warm start, ' + DISP + ' cell of motion per frame ------------------------');
  console.log('      "phi err" is the relative L2 against the exact solve at the last');
  console.log('      frame; "force err" is the same for the acceleration actually');
  console.log('      interpolated back to the particles. lam_phi and lam_a are the');
  console.log('      wavelengths of the potential-error and force-error fields in');
  console.log('      cells: tens of cells means a smooth bias, a few means grid noise.');
  console.log('      A "!" marks a row still on its way down at the last frame: the run');
  console.log('      was too short for that budget to settle, so its error is an UPPER');
  console.log('      bound on the steady state and the pass count read off it is');
  console.log('      pessimistic. Every warm start begins from phi = 0 on frame 1 and');
  console.log('      improves from there, so an unsettled row never flatters a solver.');
  console.log('');

  /*  Frame counts are chosen so the interesting budgets settle. The recycling
      relaxes toward its steady state on the solver's own slow mode, so it
      takes about 1/(1 - rate) passes in total: ~160 at N=32, ~620 at N=64 and
      ~2500 at N=128 for Jacobi, which is why the small-K Jacobi rows at the
      larger grids carry a "!" no matter how long this is allowed to run. */
  const plan = QUICK
    ? [[32, 20, [8, 32, 128], [1, 2, 4, 8, 16, 32]],
       [64, 16, [32, 128], [1, 2, 4, 8, 16, 32]]]
    : [[32, 60, [8, 16, 32, 64, 128], [1, 2, 4, 6, 8, 12, 16, 24, 32]],
       [64, 48, [16, 32, 64, 128, 256], [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]],
       [128, 24, [64, 256], [2, 4, 8, 12, 16, 24, 32, 48]]];

  for (const pl of plan) {
    const N = pl[0], frames = pl[1], jK = pl[2], cK = pl[3];
    const spec = []
      .concat(jK.map(K => ({ name: 'Jacobi', K, solve: PM.solvePoissonJacobi })))
      .concat(cK.map(K => ({ name: 'Chebyshev', K, solve: PM.solvePoissonChebyshev })));

    const t = Date.now();
    const r = runWarm(N, frames, spec, DISP);
    console.log('      N=' + N + ', ' + frames + ' frames  (' + ((Date.now() - t) / 1000).toFixed(1) + 's)' +
      '   the exact phi itself moves ' + pct(r.delta) + ' per frame');
    console.log('        solver       K    frame 1   frame ' + Math.min(5, frames) +
      '   last     force err   lam_phi  lam_a');
    const jPts = [], cPts = [];
    for (let i = 0; i < spec.length; i++) {
      const H = r.hist[i], F = r.final[i];
      const last = H[frames - 1];
      console.log('      ' + padr(spec[i].name, 11) + ' ' + pad(spec[i].K, 4) + '   ' +
        pad(e1(H[0]), 7) + '   ' + pad(e1(H[Math.min(4, frames - 1)]), 7) + '   ' +
        pad(e1(last), 7) + '  ' + pad(e1(F.fErr), 8) + '   ' +
        pad(F.lamPhi.toFixed(0), 5) + '   ' + pad(F.lamAcc.toFixed(1), 5) +
        (F.settled ? '' : '  !'));
      (spec[i].name === 'Jacobi' ? jPts : cPts).push([spec[i].K, F.fErr, F.settled]);
    }
    const rec = {
      jacobi: passesFor(jPts, TARGET),
      cheb: passesFor(cPts, TARGET),
      jacobiMax: jPts[jPts.length - 1][0],
      chebMax: cPts[cPts.length - 1][0],
      chebSettled: cPts.every(p => p[2] || p[1] > TARGET)
    };
    HEAD[N] = rec;
    console.log('      -> ' + pct(TARGET) + ' force error needs   Jacobi ' +
      (rec.jacobi === null ? '> ' + rec.jacobiMax : '~' + rec.jacobi) +
      '   Chebyshev ' + (rec.cheb === null ? '> ' + rec.chebMax : '~' + rec.cheb) + '  passes/frame');
    console.log('');
    PM.releaseScratch(N);
  }

  /* ── 8c. sensitivity to how fast the field moves ─────────────────────── */
  if (!QUICK) {
    console.log('  --- 8c. does it depend on how fast the field moves? (N=64) ------------');
    console.log('        cells/frame   K=8       K=16      K=32');
    for (const disp of [0.1, 0.3, 1.0]) {
      const spec = [8, 16, 32].map(K => ({ name: 'Chebyshev', K, solve: PM.solvePoissonChebyshev }));
      const r = runWarm(64, 24, spec, disp);
      console.log('      ' + pad(disp.toFixed(1), 11) + '     ' +
        r.final.map(f => pad(e1(f.fErr), 8)).join('  '));
      PM.releaseScratch(64);
    }
    note('slower motion is an easier warm start, but the dependence is weak — the');
    note('budget is set by how fast the solver converges, not by how far things moved');
    console.log('');
  }

  /* ── 8d. what too few passes actually looks like ─────────────────────── */
  {
    console.log('  --- 8d. the failure mode when starved of passes (N=64, warm) ----------');
    const spec = [1, 2, 4, 8, 16, 32, 48].map(K =>
      ({ name: 'Chebyshev', K, solve: PM.solvePoissonChebyshev }));
    const r = runWarm(64, QUICK ? 10 : 20, spec, DISP);
    console.log('        passes   phi err   force err   lam_phi   lam_a (cells)');
    let starvedSmooth = true, richLam = null;
    for (let i = 0; i < spec.length; i++) {
      const F = r.final[i], last = r.hist[i][r.frames - 1];
      console.log('      ' + pad(spec[i].K, 8) + '   ' + pad(e1(last), 7) + '   ' +
        pad(e1(F.fErr), 8) + '    ' + pad(F.lamPhi.toFixed(0), 6) + '    ' + F.lamAcc.toFixed(1));
      if (spec[i].K <= 4 && F.lamAcc < 6) starvedSmooth = false;
      if (i === spec.length - 1) richLam = F.lamAcc;
    }
    ok('a starved solver fails smoothly, not noisily', starvedSmooth,
      'at 1-4 passes the force-error field is coherent over many cells');
    note('so the failure mode is a coherent, slowly varying bias in gravity, not');
    note('jitter. That is the dangerous kind: too few passes still looks like a');
    note('plausible field on screen while being quietly the wrong one, which is');
    note('exactly why this number had to be measured instead of tuned by eye.');
    note('at a full budget the residue shortens to ~' + (richLam === null ? '?' : richLam.toFixed(1)) +
      ' cells but is by then well under a percent, so it never shows as noise.');
    PM.releaseScratch(64);
  }
}

/* ══ 9. the numbers that go into the shader ══════════════════════════════ */

section('9. summary for the GPU implementation');
{
  const reco = [];
  for (const N of [32, 64, 128]) {
    const r = HEAD[N];
    if (!r) continue;
    const j = r.jacobi === null ? '> ' + r.jacobiMax : '~' + r.jacobi;
    const c = r.cheb === null ? '> ' + r.chebMax : '~' + r.cheb;
    const ratio = (r.jacobi === null ? r.jacobiMax : r.jacobi) / (r.cheb === null ? r.chebMax : r.cheb);
    reco.push('    N=' + pad(N, 3) + '   plain Jacobi ' + padr(j + ' passes', 16) +
      '   Chebyshev ' + padr(c + ' passes', 14) +
      '   (' + ratio.toFixed(0) + 'x fewer' + (r.jacobi === null ? ', at least' : '') + ')');
  }
  console.log('  passes per frame to hold the particle force error under 3%, warm-started');
  console.log('  from the previous frame with the field moving 0.3 cell per frame:');
  console.log(reco.join('\n'));
  console.log('');
  console.log('  Both rows run the identical 7-point sweep. Chebyshev differs only in');
  console.log('  what it writes: alpha*(c1*sweep - c2*phi_k) - beta*phi_{k-1}, three');
  console.log('  ping-pong buffers instead of two, two float uniforms per pass that the');
  console.log('  CPU computes from r_{k+1} = 1/(2 eta - r_k). See HotaruPM');
  console.log('  .solvePoissonChebyshev; HotaruPM.chebyshevBounds(N) gives eta.');
  console.log('');
  console.log('  Read the Jacobi column as a floor rather than a figure. Its rows at');
  console.log('  the larger grids were still improving when the run ended, so those');
  console.log('  pass counts are pessimistic — but only by the margin a longer run');
  console.log('  would recover, and the cold-start table in 8a bounds that: at N=128');
  console.log('  Jacobi needs ~11500 passes for 1% from cold against Chebyshev\'s ~162,');
  console.log('  and no amount of warm starting changes the ratio of the two rates.');
  console.log('');
  console.log('  grid:      the effective softening is the half-force radius, ' +
    (FINDINGS.softening === null || FINDINGS.softening === undefined ? '?' : FINDINGS.softening.toFixed(2)) + ' cells;');
  console.log('             the force is only trustworthy beyond ~' + FINDINGS.trust5 + ' cells (placement scatter).');
  console.log('  timestep:  stable to v dt / h = ' + (FINDINGS.courant || 0).toFixed(2) +
    '; use dt <= 0.3 h / v_max, and also');
  console.log('             dt <= 0.1 sqrt(h / a_max) so the deepest cell stays resolved.');
  console.log('');

  ok('Chebyshev needs fewer passes than Jacobi at every N',
    [32, 64, 128].every(N => {
      const r = HEAD[N];
      if (!r) return true;
      const j = r.jacobi === null ? Infinity : r.jacobi;
      const c = r.cheb === null ? Infinity : r.cheb;
      return c < j;
    }), 'see the table above');

  ok('Chebyshev reaches 3% force error inside a 48-pass budget at every N',
    [32, 64, 128].every(N => !HEAD[N] || HEAD[N].cheb !== null));
}

/* ── verdict ────────────────────────────────────────────────────────────── */

console.log('\n' + '-'.repeat(66));
console.log((failed === 0
  ? 'PASS  ' + passed + ' checks'
  : 'FAIL  ' + failed + ' of ' + (passed + failed) + ' checks failed') + '   (' + elapsed() + ')');
process.exit(failed === 0 ? 0 : 1);
