/* ─────────────────────────────────────────────────────────────────────────────
   Hotaru · pm  —  exact CPU reference Particle-Mesh gravity solver
   ---------------------------------------------------------------------------
   This file exists so the WebGL2 implementation can be checked against
   something exact instead of tuned by eye. Everything here is the same
   algorithm the GPU will run, written once in double precision with no
   texture packing, no atlas addressing and no float32 rounding in the way.

   THE GRID
     A periodic cubic lattice of N^3 nodes covering a box of side L centred on
     the origin. Cell size h = L/N. Node (i,j,k) sits at

         x = -L/2 + i*h,   y = -L/2 + j*h,   z = -L/2 + k*h

     with i,j,k in [0,N) and everything wrapping mod N. Flat index is

         idx = (k*N + j)*N + i          (x fastest — the cache-friendly axis)

     On the GPU this same lattice is a 2D atlas of N z-slices; the index
     arithmetic changes, the algorithm does not.

   THE CHAIN
     depositCIC        particles  -> rho        (mass density, trilinear)
     solvePoisson*     rho        -> phi        (FFT exact, or Jacobi)
     gradient          phi        -> acc grid   (central differences)
     sampleCIC         acc grid   -> particles  (trilinear, SAME kernel)
     step              one kick-drift-kick leapfrog using the above

   WHY THE SELF-FORCE IS EXACTLY ZERO
     Write the whole chain as one linear map from the particle's position to
     the acceleration it feels from its own mass:

         a(p) = SUM_g SUM_g'  W(g-p) * (-D G)(g-g') * W(g'-p)

     W is the CIC kernel, G the discrete Green's function of the 7-point
     stencil, D the central-difference operator. On a periodic cubic lattice
     G is EVEN (G(-d) = G(d)) because the stencil is symmetric, and D is ODD,
     so (-D G) is odd. Substituting d = g-g' turns the double sum into

         SUM_d (-D G)(d) * C(d),    C(d) = SUM_g W(g+d-p) W(g-p)

     and C, an autocorrelation of W with itself, is EVEN in d. An odd function
     summed against an even one over the whole torus is exactly zero — for any
     sub-cell position of the particle, not just for particles sitting on a
     node. That identity is the reason deposition and interpolation MUST use
     the same kernel: replace one W by anything else and C stops being even,
     C picks up an odd part, and every particle starts pushing itself across
     the box. The same argument makes the total momentum of the whole system
     exactly conserved, and it survives the Jacobi solver too, because K
     Jacobi sweeps from a zero start are a polynomial in the (even) stencil.

   THE MEAN-DENSITY SUBTRACTION (the Jeans swindle)
     A periodic box with net mass has no solution to grad^2 phi = 4 pi G rho:
     the k=0 mode of the equation reads 0 = 4 pi G rho_mean. The standard fix
     is to solve for rho - rho_mean instead, i.e. to put the box on a uniform
     neutralising background. In the FFT that is "set the k=0 mode to zero".
     The Jacobi solver must do the same thing or it does not solve the same
     problem: a Jacobi sweep shifts the mean of phi by -h^2*4*pi*G*rho_mean/6
     EVERY iteration, so with a net mass the potential slides downward without
     limit and float32 loses the field underneath it. Both solvers here
     subtract the mean. Note for the GPU: rho_mean is not something you have
     to reduce for, it is (total particle mass)/L^3, a constant you already
     know, because CIC conserves mass exactly.

   THE DISCRETE LAPLACIAN EIGENVALUE
     solvePoissonFFT deliberately does NOT use the continuum k^2. It uses

         k^2  ->  (2/h)^2 * (sin^2(pi i/N) + sin^2(pi j/N) + sin^2(pi k/N))

     which is minus the eigenvalue of the very 7-point stencil the Jacobi
     solver relaxes. The two solvers therefore invert the SAME operator and
     their difference is pure iteration error, which is the only thing the
     headline experiment is trying to measure. Use the continuum k^2 and you
     bake in an O((kh)^2) discretisation difference that never converges away
     and the iteration table becomes meaningless.

   Plain script. Publishes exactly one global: HotaruPM.
   No import/export, no require, no window, no document, no Math.random.
   ───────────────────────────────────────────────────────────────────────── */
(function (root) {
  "use strict";

  var FOUR_PI = 4 * Math.PI;
  var SIXTH = 1 / 6;

  /* ═══════════════════════════════════════════════════════════════════════
     0. small utilities
     ═══════════════════════════════════════════════════════════════════════ */

  /* mulberry32 — the only randomness in this file, and only when a caller
     asks for it. Library code never calls Math.random. */
  function mulberry32(seed) {
    var a = (seed >>> 0) || 0x9E3779B9;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Box-Muller on top of mulberry32, for test fixtures that want normals. */
  function gaussian(rnd) {
    var u = 1 - rnd();
    var v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function isPow2(n) { return n > 0 && (n & (n - 1)) === 0; }

  function cellSize(N, L) { return L / N; }

  /* Fold a coordinate back into [-L/2, +L/2). */
  function wrapCoord(x, L) {
    var half = L * 0.5;
    var y = x + half;
    y -= L * Math.floor(y / L);
    if (y >= L) y -= L;          /* guards the floor() rounding up at |x| huge */
    if (y < 0) y += L;
    return y - half;
  }

  /* Shortest separation across the periodic boundary. */
  function minImage(d, L) {
    var y = d + L * 0.5;
    y -= L * Math.floor(y / L);
    if (y >= L) y -= L;
    if (y < 0) y += L;
    return y - L * 0.5;
  }

  /* Compensated summation. Naive summation over 2M cells loses ~1e-10
     relative, which would swamp the mass-conservation assertion we actually
     want to make. */
  function kahanSum(a, n) {
    if (n === undefined) n = a.length;
    var s = 0, c = 0, i, y, t;
    for (i = 0; i < n; i++) {
      y = a[i] - c;
      t = s + y;
      c = (t - s) - y;
      s = t;
    }
    return s;
  }

  /* Total mass held by a density grid. */
  function totalMass(rho, N, L) {
    var h = L / N;
    return kahanSum(rho, N * N * N) * (h * h * h);
  }

  function meanOf(a, n) {
    if (n === undefined) n = a.length;
    return kahanSum(a, n) / n;
  }

  /* Subtract the mean in place; returns the mean that was removed. A constant
     offset is physically meaningless here (only grad phi is observable) so
     every comparison between two potentials removes it first. */
  function removeMean(a, n) {
    if (n === undefined) n = a.length;
    var m = meanOf(a, n);
    for (var i = 0; i < n; i++) a[i] -= m;
    return m;
  }

  /* ||a - b||_2 / ||b||_2, both means removed first (non-destructively). */
  function relativeL2(a, b, n) {
    if (n === undefined) n = b.length;
    var ma = meanOf(a, n), mb = meanOf(b, n);
    var num = 0, den = 0, i, d, v;
    for (i = 0; i < n; i++) {
      d = (a[i] - ma) - (b[i] - mb);
      v = b[i] - mb;
      num += d * d;
      den += v * v;
    }
    if (den === 0) return num === 0 ? 0 : Infinity;
    return Math.sqrt(num / den);
  }

  /* Plain ||a-b||/||b|| with no mean removal — for vector fields, where a
     constant offset is NOT meaningless. */
  function relativeL2Raw(a, b, n) {
    if (n === undefined) n = b.length;
    var num = 0, den = 0, i, d;
    for (i = 0; i < n; i++) {
      d = a[i] - b[i];
      num += d * d;
      den += b[i] * b[i];
    }
    if (den === 0) return num === 0 ? 0 : Infinity;
    return Math.sqrt(num / den);
  }

  /* ═══════════════════════════════════════════════════════════════════════
     1. roughness — is an error field a smooth bias or is it noise?

     S(e) = (sum of 6 neighbours - 6 e)/6  is (h^2/6) times the 7-point
     Laplacian. Its eigenvalue on a Fourier mode is

         -(2/3) * (sin^2(qx/2) + sin^2(qy/2) + sin^2(qz/2)),   q = k*h

     which runs from 0 (constant) to -2 (checkerboard). So

         R = ||S e|| / ||e||   in [0, 2]

     is a single number saying how wiggly e is, with no FFT needed — cheap
     enough to run on the GPU as a live diagnostic. Reading it as one
     isotropic mode, R = 2 sin^2(Q / (2*sqrt 3)) with Q = |k| h, giving an
     effective error wavelength of 2*pi/Q cells. A smooth bias comes out at
     tens of cells; grid noise comes out at 2-4 cells.
     ═══════════════════════════════════════════════════════════════════════ */
  function roughness(e, N) {
    var N2 = N * N, num = 0, den = 0;
    var k, j, i, k0, kp, km, j0, jp, jm, row, rzp, rzm, ryp, rym, ip, imn, v, s;
    for (k = 0; k < N; k++) {
      k0 = k * N2;
      kp = ((k + 1 === N) ? 0 : k + 1) * N2;
      km = ((k === 0) ? N - 1 : k - 1) * N2;
      for (j = 0; j < N; j++) {
        j0 = j * N;
        jp = ((j + 1 === N) ? 0 : j + 1) * N;
        jm = ((j === 0) ? N - 1 : j - 1) * N;
        row = k0 + j0; rzp = kp + j0; rzm = km + j0; ryp = k0 + jp; rym = k0 + jm;
        for (i = 0; i < N; i++) {
          ip = (i + 1 === N) ? 0 : i + 1;
          imn = (i === 0) ? N - 1 : i - 1;
          v = e[row + i];
          s = (e[row + ip] + e[row + imn] + e[ryp + i] + e[rym + i] +
               e[rzp + i] + e[rzm + i]) * SIXTH - v;
          num += s * s;
          den += v * v;
        }
      }
    }
    return den === 0 ? 0 : Math.sqrt(num / den);
  }

  /* Effective wavelength, in cells, implied by a roughness value. */
  function roughnessWavelength(R) {
    if (R <= 0) return Infinity;
    var s = Math.min(1, Math.sqrt(Math.min(R, 2) / 2));
    var Q = 2 * Math.sqrt(3) * Math.asin(s);
    return Q === 0 ? Infinity : (2 * Math.PI) / Q;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     2. CIC deposition — particles to mass density

     Trilinear ("cloud in cell"): each particle is a cube of side h centred on
     itself, and the overlap with each of the 8 surrounding nodes' cells gives
     that node's weight. The 8 weights are (1-fx or fx)(1-fy or fy)(1-fz or fz)
     and sum to exactly 1 in exact arithmetic, so total mass is conserved.
     rhoOut is filled with MASS DENSITY (mass per unit volume), which is what
     the Poisson equation wants; multiply by h^3 to get mass back.
     ═══════════════════════════════════════════════════════════════════════ */
  function depositCIC(pos, count, N, L, rhoOut, opts) {
    opts = opts || {};
    var massOpt = (opts.mass === undefined) ? 1 : opts.mass;
    var massArr = (typeof massOpt === "number") ? null : massOpt;
    var mScalar = massArr ? 0 : massOpt;

    var h = L / N, inv = 1 / h, half = L * 0.5;
    var N2 = N * N, N3 = N2 * N;

    rhoOut.fill(0);

    var total = 0, comp = 0;
    var p, m, b, gx, gy, gz, fx, fy, fz, i0, j0, k0, i1, j1, k1;
    var x0, x1, y0, y1, z0, z1, r0, r1, s0, s1, wy, y, t;

    for (p = 0; p < count; p++) {
      m = massArr ? massArr[p] : mScalar;
      b = 3 * p;

      gx = (pos[b] + half) * inv;
      gy = (pos[b + 1] + half) * inv;
      gz = (pos[b + 2] + half) * inv;

      i0 = Math.floor(gx); fx = gx - i0;
      j0 = Math.floor(gy); fy = gy - j0;
      k0 = Math.floor(gz); fz = gz - k0;

      i0 = ((i0 % N) + N) % N;
      j0 = ((j0 % N) + N) % N;
      k0 = ((k0 % N) + N) % N;
      i1 = (i0 + 1 === N) ? 0 : i0 + 1;
      j1 = (j0 + 1 === N) ? 0 : j0 + 1;
      k1 = (k0 + 1 === N) ? 0 : k0 + 1;

      x0 = 1 - fx; x1 = fx;
      y0 = 1 - fy; y1 = fy;
      z0 = 1 - fz; z1 = fz;

      /* four row bases: (j0,k0) (j1,k0) (j0,k1) (j1,k1) */
      r0 = (k0 * N + j0) * N; r1 = (k0 * N + j1) * N;
      s0 = (k1 * N + j0) * N; s1 = (k1 * N + j1) * N;

      wy = m * z0 * y0; rhoOut[r0 + i0] += wy * x0; rhoOut[r0 + i1] += wy * x1;
      wy = m * z0 * y1; rhoOut[r1 + i0] += wy * x0; rhoOut[r1 + i1] += wy * x1;
      wy = m * z1 * y0; rhoOut[s0 + i0] += wy * x0; rhoOut[s0 + i1] += wy * x1;
      wy = m * z1 * y1; rhoOut[s1 + i0] += wy * x0; rhoOut[s1 + i1] += wy * x1;

      y = m - comp; t = total + y; comp = (t - total) - y; total = t;
    }

    var ivol = 1 / (h * h * h);
    for (var q = 0; q < N3; q++) rhoOut[q] *= ivol;

    return total;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     3. radix-2 complex FFT, and the exact Poisson solve built on it
     ═══════════════════════════════════════════════════════════════════════ */

  var planCache = Object.create(null);

  function getPlan(n) {
    var p = planCache[n];
    if (p) return p;
    if (!isPow2(n)) throw new Error("HotaruPM: FFT length must be a power of two, got " + n);
    var bits = 0; while ((1 << bits) < n) bits++;
    var rev = new Uint32Array(n);
    var i, b, r, x;
    for (i = 0; i < n; i++) {
      r = 0; x = i;
      for (b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    var half = n >> 1;
    var cs = new Float64Array(half), sn = new Float64Array(half);
    for (i = 0; i < half; i++) {
      var ang = 2 * Math.PI * i / n;
      cs[i] = Math.cos(ang);
      sn[i] = Math.sin(ang);
    }
    /* sin^2(pi m / n), the per-axis piece of the discrete Laplacian eigenvalue */
    var ss = new Float64Array(n);
    for (i = 0; i < n; i++) {
      var s = Math.sin(Math.PI * i / n);
      ss[i] = s * s;
    }
    p = { n: n, bits: bits, rev: rev, cos: cs, sin: sn, sinsq: ss };
    planCache[n] = p;
    return p;
  }

  /*  In-place iterative Cooley-Tukey on re[off .. off+n) / im[off .. off+n).

      The `off` parameter is not decoration: the x-axis of the 3D transform is
      contiguous in memory, so it runs straight on the field with off = base
      and skips a gather/scatter that costs as much as the butterflies do at
      these sizes. The len=2 pass is peeled out because its twiddle is exactly
      1 and it is a sixth of all the work at n=64.
  */
  function fft1d(re, im, off, plan, inverse) {
    var n = plan.n, rev = plan.rev, cs = plan.cos, sn = plan.sin;
    var end = off + n;
    var i, j, t, len, half, step, k, base, wr, wi, ur, ui, ar, ai, vr, vi, a, bIdx;

    for (i = 0; i < n; i++) {
      j = rev[i];
      if (j > i) {
        a = off + i; bIdx = off + j;
        t = re[a]; re[a] = re[bIdx]; re[bIdx] = t;
        t = im[a]; im[a] = im[bIdx]; im[bIdx] = t;
      }
    }

    for (a = off; a < end; a += 2) {
      bIdx = a + 1;
      ur = re[a]; ui = im[a]; vr = re[bIdx]; vi = im[bIdx];
      re[a] = ur + vr; im[a] = ui + vi;
      re[bIdx] = ur - vr; im[bIdx] = ui - vi;
    }

    var sgn = inverse ? 1 : -1;
    for (len = 4; len <= n; len <<= 1) {
      half = len >> 1;
      step = n / len;
      for (base = off; base < end; base += len) {
        for (k = 0, t = 0; k < half; k++, t += step) {
          wr = cs[t];
          wi = sgn * sn[t];
          a = base + k; bIdx = a + half;
          ur = re[a]; ui = im[a];
          ar = re[bIdx]; ai = im[bIdx];
          vr = ar * wr - ai * wi;
          vi = ar * wi + ai * wr;
          re[a] = ur + vr; im[a] = ui + vi;
          re[bIdx] = ur - vr; im[bIdx] = ui - vi;
        }
      }
    }
  }

  /* 3D transform of an N^3 field held as two flat arrays. Lines are gathered
     into a contiguous scratch before transforming: the strided axes are
     otherwise cache-hostile and cost several times more than the butterflies.
     Not scaled — the inverse's 1/N^3 is applied by the caller. */
  function fft3d(re, im, N, inverse, bufRe, bufIm) {
    var plan = getPlan(N);
    var N2 = N * N;
    var a, b, c, base, off, s;

    /* axis 0 — stride 1, transformed where it lies */
    for (c = 0; c < N2; c++) fft1d(re, im, c * N, plan, inverse);

    /* axis 1 — stride N */
    for (c = 0; c < N; c++) {
      for (b = 0; b < N; b++) {
        base = c * N2 + b;
        for (a = 0, off = base; a < N; a++, off += N) { bufRe[a] = re[off]; bufIm[a] = im[off]; }
        fft1d(bufRe, bufIm, 0, plan, inverse);
        for (a = 0, off = base; a < N; a++, off += N) { re[off] = bufRe[a]; im[off] = bufIm[a]; }
      }
    }

    /* axis 2 — stride N^2 */
    s = N2;
    for (c = 0; c < N; c++) {
      for (b = 0; b < N; b++) {
        base = c * N + b;
        for (a = 0, off = base; a < N; a++, off += s) { bufRe[a] = re[off]; bufIm[a] = im[off]; }
        fft1d(bufRe, bufIm, 0, plan, inverse);
        for (a = 0, off = base; a < N; a++, off += s) { re[off] = bufRe[a]; im[off] = bufIm[a]; }
      }
    }
  }

  /* Scratch for the 3D FFT, one set per N, allocated once. */
  var fftWork = Object.create(null);
  function getFFTWork(N) {
    var w = fftWork[N];
    if (w) return w;
    var N3 = N * N * N;
    w = {
      re: new Float64Array(N3),
      im: new Float64Array(N3),
      bufRe: new Float64Array(N),
      bufIm: new Float64Array(N)
    };
    fftWork[N] = w;
    return w;
  }

  /* Drop cached FFT scratch — 128^3 holds 32 MB and a long test run does not
     want three of those alive at once. */
  function releaseScratch(N) {
    if (N === undefined) { fftWork = Object.create(null); jacobiWork = Object.create(null); return; }
    delete fftWork[N];
    delete jacobiWork[N + "/f64"];
    delete jacobiWork[N + "/f32"];
  }

  /*  solvePoissonFFT — the exact inverse of the 7-point stencil.

      phi_k = -4 pi G rho_k / k^2,   k^2 = (2/h)^2 * sum sin^2(pi m / N)

      with the k=0 mode set to zero (a periodic box has no defined mean
      potential). Because k^2 here is the DISCRETE eigenvalue, the phi that
      comes back satisfies the 7-point difference equation to round-off — the
      Jacobi solver is iterating toward exactly this field, not near it.
  */
  function solvePoissonFFT(rho, N, L, G, phiOut) {
    var N3 = N * N * N;
    var h = L / N;
    var w = getFFTWork(N);
    var re = w.re, im = w.im;
    var i;

    for (i = 0; i < N3; i++) { re[i] = rho[i]; im[i] = 0; }

    fft3d(re, im, N, false, w.bufRe, w.bufIm);

    var plan = getPlan(N);
    var ss = plan.sinsq;
    var invKfac = (h * h) / 4;              /* 1/k^2 = h^2 / (4 * S) */
    var coef = -FOUR_PI * G * invKfac;      /* phi_k = coef * rho_k / S   */
    var kz, jy, ix, sz, sy, S, f, base, idx;

    for (kz = 0; kz < N; kz++) {
      sz = ss[kz];
      for (jy = 0; jy < N; jy++) {
        sy = sz + ss[jy];
        base = (kz * N + jy) * N;
        for (ix = 0; ix < N; ix++) {
          idx = base + ix;
          S = sy + ss[ix];
          if (S === 0) { re[idx] = 0; im[idx] = 0; continue; }
          f = coef / S;
          re[idx] *= f; im[idx] *= f;
        }
      }
    }

    fft3d(re, im, N, true, w.bufRe, w.bufIm);

    var scale = 1 / N3;
    for (i = 0; i < N3; i++) phiOut[i] = re[i] * scale;

    return phiOut;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     4. Jacobi relaxation — what the GPU actually runs

         phi_new = (sum of 6 neighbours - h^2 * 4 pi G * (rho - rho_mean)) / 6

     Double-buffered (a Jacobi sweep must read only the old field — doing it
     in place is Gauss-Seidel, which is a different and non-symmetric
     operator, and on a GPU you get a race instead). Warm-started from
     whatever phiInOut already holds, which is the entire point: across
     frames the field moves by a fraction of a cell, so last frame's solution
     is already a very good guess and the per-frame budget can be small.

     The result is always left in phiInOut regardless of parity.
     ═══════════════════════════════════════════════════════════════════════ */

  /*  Scratch for the relaxation solvers. Keyed by N and by precision: pass a
      Float32Array as phiInOut and every intermediate is stored in float32
      too, which is what the GPU does with an RGBA32F ping-pong. It is not a
      bit-exact GPU emulation — a real fragment shader also computes in
      float32 registers, while this rounds on each store — but it captures
      the thing that actually decides whether a relaxation survives the move
      to the GPU, which is how much precision the iterate loses between
      passes. test/pm.test.js runs the whole warm-start experiment both ways. */
  var jacobiWork = Object.create(null);
  function getJacobiWork(N, Ctor) {
    if (!Ctor) Ctor = Float64Array;
    var key = N + (Ctor === Float32Array ? '/f32' : '/f64');
    var w = jacobiWork[key];
    if (w) return w;
    var N3 = N * N * N;
    w = {
      tmp: new Ctor(N3),   /* y = one Jacobi sweep of x           */
      src: new Ctor(N3),   /* h^2 * 4 pi G * (rho - rho_mean)     */
      prev: new Ctor(N3)   /* x_{k-1}, Chebyshev only             */
    };
    jacobiWork[key] = w;
    return w;
  }

  function workCtor(phi) {
    return (phi instanceof Float32Array) ? Float32Array : Float64Array;
  }

  /* srho[i] = h^2 * 4 pi G * (rho[i] - rho_mean) — every per-iteration
     constant folded into one array, so the sweep is six adds, one subtract
     and one multiply. Returns the mean that was used. */
  function prepareSource(rho, N, L, G, srho, rhoMeanOpt) {
    var N3 = N * N * N;
    var h = L / N;
    var rhoMean = (rhoMeanOpt === undefined) ? meanOf(rho, N3) : rhoMeanOpt;
    var c = h * h * FOUR_PI * G;
    for (var i = 0; i < N3; i++) srho[i] = c * (rho[i] - rhoMean);
    return rhoMean;
  }

  /*  One Jacobi sweep: dst = (1-omega)*src + omega*(sum6 - srho)/6.
      omega === 1 is the plain sweep the brief specifies. Reads only `src`
      and writes only `dst` — doing it in place would be Gauss-Seidel, a
      different (and on a GPU, racy) operator. */
  function jacobiSweep(src, dst, srho, N, omega) {
    var N2 = N * N;
    var k, j, k0, kp, km, j0, jp, jm, row, rzp, rzm, ryp, rym, ix, s;
    var plain = (omega === 1);
    var wj = omega * SIXTH, wk = 1 - omega;

    for (k = 0; k < N; k++) {
      k0 = k * N2;
      kp = ((k + 1 === N) ? 0 : k + 1) * N2;
      km = ((k === 0) ? N - 1 : k - 1) * N2;
      for (j = 0; j < N; j++) {
        j0 = j * N;
        jp = ((j + 1 === N) ? 0 : j + 1) * N;
        jm = ((j === 0) ? N - 1 : j - 1) * N;
        row = k0 + j0; rzp = kp + j0; rzm = km + j0; ryp = k0 + jp; rym = k0 + jm;

        if (plain) {
          /* x = 0 — wraps left */
          dst[row] = (src[row + 1] + src[row + N - 1] + src[ryp] + src[rym] +
                      src[rzp] + src[rzm] - srho[row]) * SIXTH;
          /* interior — no branches in the hot loop */
          for (ix = 1; ix < N - 1; ix++) {
            dst[row + ix] = (src[row + ix + 1] + src[row + ix - 1] +
                             src[ryp + ix] + src[rym + ix] +
                             src[rzp + ix] + src[rzm + ix] - srho[row + ix]) * SIXTH;
          }
          /* x = N-1 — wraps right */
          ix = N - 1;
          dst[row + ix] = (src[row] + src[row + ix - 1] + src[ryp + ix] + src[rym + ix] +
                           src[rzp + ix] + src[rzm + ix] - srho[row + ix]) * SIXTH;
        } else {
          s = src[row + 1] + src[row + N - 1] + src[ryp] + src[rym] + src[rzp] + src[rzm];
          dst[row] = wk * src[row] + wj * (s - srho[row]);
          for (ix = 1; ix < N - 1; ix++) {
            s = src[row + ix + 1] + src[row + ix - 1] + src[ryp + ix] + src[rym + ix] +
                src[rzp + ix] + src[rzm + ix];
            dst[row + ix] = wk * src[row + ix] + wj * (s - srho[row + ix]);
          }
          ix = N - 1;
          s = src[row] + src[row + ix - 1] + src[ryp + ix] + src[rym + ix] +
              src[rzp + ix] + src[rzm + ix];
          dst[row + ix] = wk * src[row + ix] + wj * (s - srho[row + ix]);
        }
      }
    }
  }

  function solvePoissonJacobi(rho, N, L, G, phiInOut, iterations, opts) {
    opts = opts || {};
    iterations = iterations | 0;
    var w = getJacobiWork(N, workCtor(phiInOut));
    prepareSource(rho, N, L, G, w.src, opts.rhoMean);
    if (iterations <= 0) return phiInOut;

    var omega = (opts.omega === undefined) ? 1 : opts.omega;
    var a = phiInOut, b = w.tmp, sw, it;

    for (it = 0; it < iterations; it++) {
      jacobiSweep(a, b, w.src, N, omega);
      sw = a; a = b; b = sw;
    }
    if (a !== phiInOut) phiInOut.set(a);
    return phiInOut;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     4b. Chebyshev-accelerated Jacobi  —  the same sweep, ~35x fewer of them

     Plain Jacobi is a terrible Poisson solver and the measurements in
     test/pm.test.js say so out loud: its slowest mode decays by a factor
     (2 + cos(2 pi / N)) / 3 per sweep, which at N = 64 is 0.9984, so cutting
     the error by 100x takes ~2900 sweeps. That is not a tuning problem, it
     is the spectrum of the operator. The lowest mode is also exactly the one
     carrying most of phi's power, because phi_k ~ rho_k / k^2.

     Chebyshev semi-iteration fixes it without changing the stencil. Instead
     of taking the Jacobi output as the next iterate, it forms a specific
     linear combination of the Jacobi output and the two previous iterates,
     chosen so that after K passes the error has been multiplied by the
     Chebyshev polynomial that is smallest on the spectrum. The error decays
     by roughly 1 - 2*pi*sqrt(2/3)/N per pass instead of 1 - 2*pi^2/(3*N^2) —
     a square-root improvement in the condition number, the same speedup SOR
     gets, but with a fully parallel double-buffered sweep and no red/black
     checkerboard to work out inside a z-slice atlas.

     Let G be the Jacobi iteration operator. Its eigenvalues on the
     non-constant subspace (the constant mode is the k=0 null space we
     removed with the mean subtraction) run over

         mu = (cos qx + cos qy + cos qz) / 3,   q = 2 pi n / N

     so a = -1 (the checkerboard) and b = (2 + cos(2 pi / N)) / 3 (the
     longest wave). With eta = (2 - a - b)/(b - a), c1 = 4/(b-a),
     c2 = 2(a+b)/(b-a), and mu_k = T_k(eta):

         y_k    = jacobiSweep(x_k)
         x_1    = (c1 y_0 - c2 x_0) / (2 eta)
         x_k+1  = [ mu_k (c1 y_k - c2 x_k) - mu_k-1 x_k-1 ] / mu_k+1

     T_k(eta) overflows quickly, so the coefficients are carried as the ratio
     r_k = mu_k-1 / mu_k, which obeys r_k+1 = 1 / (2 eta - r_k) and stays in
     (0,1) — the GPU only ever sees two float uniforms per pass.

     SHADER SHAPE: one extra texture (x_k-1) and two uniforms. Read the
     stencil from x_k as usual, then write
         alpha * (c1 * y - c2 * x_k) - beta * x_k-1
     with alpha = r_k+1, beta = r_k * r_k+1. Ping-pong three buffers instead
     of two. Restart the recurrence (k = 0) every frame; the warm start is
     the previous frame's phi, exactly as with plain Jacobi.
     ═══════════════════════════════════════════════════════════════════════ */
  function chebyshevBounds(N) {
    return { a: -1, b: (2 + Math.cos(2 * Math.PI / N)) / 3 };
  }

  /* Asymptotic error reduction per pass, for the record. */
  function chebyshevRate(N) {
    var bd = chebyshevBounds(N);
    var eta = (2 - bd.a - bd.b) / (bd.b - bd.a);
    return Math.exp(-Math.acosh(eta));
  }
  function jacobiRate(N) {
    return chebyshevBounds(N).b;
  }

  function solvePoissonChebyshev(rho, N, L, G, phiInOut, iterations, opts) {
    opts = opts || {};
    iterations = iterations | 0;
    var N3 = N * N * N;
    var w = getJacobiWork(N, workCtor(phiInOut));
    prepareSource(rho, N, L, G, w.src, opts.rhoMean);
    if (iterations <= 0) return phiInOut;

    var bd = chebyshevBounds(N);
    var a = (opts.a === undefined) ? bd.a : opts.a;
    var b = (opts.b === undefined) ? bd.b : opts.b;
    var eta = (2 - a - b) / (b - a);
    var c1 = 4 / (b - a);
    var c2 = 2 * (a + b) / (b - a);

    var x = phiInOut;        /* x_k   */
    var xp = w.prev;         /* x_k-1 */
    var y = w.tmp;           /* y_k   */
    var i, it, r, rNext, alpha, beta, tmpRef;

    /* k = 0 */
    jacobiSweep(x, y, w.src, N, 1);
    /* x_1 goes into xp's storage; x_0 must survive as the new x_{k-1}. */
    var inv2eta = 1 / (2 * eta);
    for (i = 0; i < N3; i++) xp[i] = (c1 * y[i] - c2 * x[i]) * inv2eta;
    tmpRef = x; x = xp; xp = tmpRef;   /* x = x_1, xp = x_0 */
    r = 1 / eta;                        /* r_1 = mu_0 / mu_1 */

    for (it = 1; it < iterations; it++) {
      jacobiSweep(x, y, w.src, N, 1);
      rNext = 1 / (2 * eta - r);
      alpha = rNext;
      beta = r * rNext;
      /* x_{k+1} overwrites x_{k-1}, which is dead after this line. */
      for (i = 0; i < N3; i++) {
        xp[i] = alpha * (c1 * y[i] - c2 * x[i]) - beta * xp[i];
      }
      tmpRef = x; x = xp; xp = tmpRef;
      r = rNext;
    }

    if (x !== phiInOut) phiInOut.set(x);
    return phiInOut;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     4c. red-black SOR — the other same-stencil accelerator, for comparison

     Converges at about the same rate per stencil pass as Chebyshev, and
     needs only one field instead of three. It costs a checkerboard test
     ((i+j+k) & 1) and two passes per sweep, each of which must still write
     every texel of the ping-pong target because WebGL2 cannot read and write
     the same texture. Included so the choice between the two is made on
     measured numbers rather than taste.
     ═══════════════════════════════════════════════════════════════════════ */
  function sorOmega(N) {
    /* 2 / (1 + sqrt(1 - rho_J^2)) with rho_J the Jacobi radius on the
       non-constant subspace. */
    var rj = chebyshevBounds(N).b;
    return 2 / (1 + Math.sqrt(Math.max(0, 1 - rj * rj)));
  }

  function solvePoissonSOR(rho, N, L, G, phiInOut, sweeps, opts) {
    opts = opts || {};
    sweeps = sweeps | 0;
    var w = getJacobiWork(N, workCtor(phiInOut));
    prepareSource(rho, N, L, G, w.src, opts.rhoMean);
    if (sweeps <= 0) return phiInOut;

    var omega = (opts.omega === undefined) ? sorOmega(N) : opts.omega;
    var srho = w.src, a = phiInOut;
    var N2 = N * N;
    var wj = omega * SIXTH, wk = 1 - omega;
    var it, colour, k, j, i, k0, kp, km, j0, jp, jm, row, rzp, rzm, ryp, rym, ip, imn, s;

    for (it = 0; it < sweeps; it++) {
      for (colour = 0; colour < 2; colour++) {
        for (k = 0; k < N; k++) {
          k0 = k * N2;
          kp = ((k + 1 === N) ? 0 : k + 1) * N2;
          km = ((k === 0) ? N - 1 : k - 1) * N2;
          for (j = 0; j < N; j++) {
            j0 = j * N;
            jp = ((j + 1 === N) ? 0 : j + 1) * N;
            jm = ((j === 0) ? N - 1 : j - 1) * N;
            row = k0 + j0; rzp = kp + j0; rzm = km + j0; ryp = k0 + jp; rym = k0 + jm;
            for (i = ((j + k + colour) & 1); i < N; i += 2) {
              ip = (i + 1 === N) ? 0 : i + 1;
              imn = (i === 0) ? N - 1 : i - 1;
              s = a[row + ip] + a[row + imn] + a[ryp + i] + a[rym + i] +
                  a[rzp + i] + a[rzm + i];
              a[row + i] = wk * a[row + i] + wj * (s - srho[row + i]);
            }
          }
        }
      }
    }
    return phiInOut;
  }

  /* Relative residual of the 7-point equation:
        || (sum6 - 6 phi)/h^2 - 4 pi G (rho - mean) ||  /  || 4 pi G (rho-mean) ||
     Zero-ish means "this phi solves the discrete Poisson equation". Run it on
     the FFT output to prove the eigenvalue is right; run it on the Jacobi
     output to watch convergence without needing the exact answer. */
  function poissonResidual(phi, rho, N, L, G, opts) {
    opts = opts || {};
    var N2 = N * N, N3 = N2 * N;
    var h = L / N, ih2 = 1 / (h * h);
    var rhoMean = (opts.rhoMean === undefined) ? meanOf(rho, N3) : opts.rhoMean;
    var num = 0, den = 0;
    var k, j, i, k0, kp, km, j0, jp, jm, row, rzp, rzm, ryp, rym, ip, imn, lap, s, r;
    for (k = 0; k < N; k++) {
      k0 = k * N2;
      kp = ((k + 1 === N) ? 0 : k + 1) * N2;
      km = ((k === 0) ? N - 1 : k - 1) * N2;
      for (j = 0; j < N; j++) {
        j0 = j * N;
        jp = ((j + 1 === N) ? 0 : j + 1) * N;
        jm = ((j === 0) ? N - 1 : j - 1) * N;
        row = k0 + j0; rzp = kp + j0; rzm = km + j0; ryp = k0 + jp; rym = k0 + jm;
        for (i = 0; i < N; i++) {
          ip = (i + 1 === N) ? 0 : i + 1;
          imn = (i === 0) ? N - 1 : i - 1;
          lap = (phi[row + ip] + phi[row + imn] + phi[ryp + i] + phi[rym + i] +
                 phi[rzp + i] + phi[rzm + i] - 6 * phi[row + i]) * ih2;
          s = FOUR_PI * G * (rho[row + i] - rhoMean);
          r = lap - s;
          num += r * r;
          den += s * s;
        }
      }
    }
    if (den === 0) return num === 0 ? 0 : Infinity;
    return Math.sqrt(num / den);
  }

  /* ═══════════════════════════════════════════════════════════════════════
     5. gradient — acceleration per cell, a = -grad phi

     Central differences, (phi[i+1] - phi[i-1]) / (2h). This operator is ODD,
     which is half of why the self-force vanishes; a one-sided or a 4th-order
     non-symmetric difference would break that. accOut is interleaved
     [ax,ay,az] per cell, which is both what a GPU RGB texture looks like and
     what sampleCIC wants to read.
     ═══════════════════════════════════════════════════════════════════════ */
  function gradient(phi, N, L, accOut) {
    var N2 = N * N;
    var h = L / N;
    var f = -1 / (2 * h);
    var k, j, i, k0, kp, km, j0, jp, jm, row, rzp, rzm, ryp, rym, ip, imn, o;
    for (k = 0; k < N; k++) {
      k0 = k * N2;
      kp = ((k + 1 === N) ? 0 : k + 1) * N2;
      km = ((k === 0) ? N - 1 : k - 1) * N2;
      for (j = 0; j < N; j++) {
        j0 = j * N;
        jp = ((j + 1 === N) ? 0 : j + 1) * N;
        jm = ((j === 0) ? N - 1 : j - 1) * N;
        row = k0 + j0; rzp = kp + j0; rzm = km + j0; ryp = k0 + jp; rym = k0 + jm;
        for (i = 0; i < N; i++) {
          ip = (i + 1 === N) ? 0 : i + 1;
          imn = (i === 0) ? N - 1 : i - 1;
          o = 3 * (row + i);
          accOut[o]     = f * (phi[row + ip] - phi[row + imn]);
          accOut[o + 1] = f * (phi[ryp + i]  - phi[rym + i]);
          accOut[o + 2] = f * (phi[rzp + i]  - phi[rzm + i]);
        }
      }
    }
    return accOut;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     6. CIC interpolation — grid back to particles

     Byte for byte the same weights as depositCIC. That is a correctness
     requirement, not a convenience: see the self-force note at the top.
     ═══════════════════════════════════════════════════════════════════════ */
  function sampleCIC(accGrid, N, L, pos, count, accOut) {
    var h = L / N, inv = 1 / h, half = L * 0.5;
    var p, b, gx, gy, gz, fx, fy, fz, i0, j0, k0, i1, j1, k1;
    var x0, x1, y0, y1, z0, z1, r0, r1, s0, s1, w, o, ax, ay, az;

    for (p = 0; p < count; p++) {
      b = 3 * p;
      gx = (pos[b] + half) * inv;
      gy = (pos[b + 1] + half) * inv;
      gz = (pos[b + 2] + half) * inv;

      i0 = Math.floor(gx); fx = gx - i0;
      j0 = Math.floor(gy); fy = gy - j0;
      k0 = Math.floor(gz); fz = gz - k0;

      i0 = ((i0 % N) + N) % N;
      j0 = ((j0 % N) + N) % N;
      k0 = ((k0 % N) + N) % N;
      i1 = (i0 + 1 === N) ? 0 : i0 + 1;
      j1 = (j0 + 1 === N) ? 0 : j0 + 1;
      k1 = (k0 + 1 === N) ? 0 : k0 + 1;

      x0 = 1 - fx; x1 = fx;
      y0 = 1 - fy; y1 = fy;
      z0 = 1 - fz; z1 = fz;

      r0 = (k0 * N + j0) * N; r1 = (k0 * N + j1) * N;
      s0 = (k1 * N + j0) * N; s1 = (k1 * N + j1) * N;

      ax = 0; ay = 0; az = 0;

      w = z0 * y0 * x0; o = 3 * (r0 + i0); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z0 * y0 * x1; o = 3 * (r0 + i1); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z0 * y1 * x0; o = 3 * (r1 + i0); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z0 * y1 * x1; o = 3 * (r1 + i1); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z1 * y0 * x0; o = 3 * (s0 + i0); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z1 * y0 * x1; o = 3 * (s0 + i1); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z1 * y1 * x0; o = 3 * (s1 + i0); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];
      w = z1 * y1 * x1; o = 3 * (s1 + i1); ax += w * accGrid[o]; ay += w * accGrid[o + 1]; az += w * accGrid[o + 2];

      accOut[b] = ax; accOut[b + 1] = ay; accOut[b + 2] = az;
    }
    return accOut;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     7. the whole chain, and the leapfrog
     ═══════════════════════════════════════════════════════════════════════ */

  /* Per-state scratch. Allocated once, reused for the life of the state — the
     "typed arrays, allocated once, filled in place" rule applies to the
     reference too, or the 1000-step tests spend their time in the GC. */
  function workspace(state, N, count) {
    var w = state._pm;
    if (w && w.N === N && w.count >= count) return w;
    var N3 = N * N * N;
    w = {
      N: N,
      count: count,
      rho: new Float64Array(N3),
      phi: new Float64Array(N3),
      accGrid: new Float64Array(3 * N3),
      acc: new Float64Array(3 * count),
      accValid: false,
      lastRhoMean: 0
    };
    state._pm = w;
    return w;
  }

  /* particles -> acceleration, the full chain. Leaves phi in w.phi so the
     next frame can warm-start from it. */
  function computeAccel(state, params, w) {
    var N = params.N, L = params.L, G = params.G;
    var count = state.count;
    var mass = (state.mass !== undefined) ? state.mass : (params.mass === undefined ? 1 : params.mass);

    var total = depositCIC(state.pos, count, N, L, w.rho, { mass: mass });
    var rhoMean = total / (L * L * L);
    w.lastRhoMean = rhoMean;

    var iters = params.iterations | 0;
    if (iters > 0) {
      solvePoissonJacobi(w.rho, N, L, G, w.phi, iters, { rhoMean: rhoMean });
    } else {
      solvePoissonFFT(w.rho, N, L, G, w.phi);
    }

    gradient(w.phi, N, L, w.accGrid);
    sampleCIC(w.accGrid, N, L, state.pos, count, w.acc);
    w.accValid = true;
    return w.acc;
  }

  /*  step — one kick-drift-kick leapfrog.

        v += (dt/2) a(x)
        x += dt v                 (wrapped back into the box)
        recompute a(x)
        v += (dt/2) a(x)

      One force evaluation per step: the second half-kick's acceleration is
      carried over to become the first half-kick's of the next step. KDK is
      symplectic and time-reversible, which is why the energy in a bound orbit
      oscillates instead of drifting — see test 4. Do NOT "simplify" this into
      an Euler update; the drift is what kills a gravity sim, not the error.

      params: { N, L, G, mass, iterations }  (iterations 0 or absent -> FFT)
      state:  { pos, vel, count, mass? }     (typed arrays, length >= 3*count)
      Returns the state. state._pm.phi holds the potential, warm-started into
      the next call automatically.
  */
  function step(state, dt, params) {
    var N = params.N, L = params.L;
    var count = state.count;
    var w = workspace(state, N, count);

    if (!w.accValid) computeAccel(state, params, w);

    var acc = w.acc, pos = state.pos, vel = state.vel;
    var hdt = 0.5 * dt;
    var n3 = 3 * count, i;

    for (i = 0; i < n3; i++) vel[i] += hdt * acc[i];

    if (params.wrap === false) {
      for (i = 0; i < n3; i++) pos[i] += dt * vel[i];
    } else {
      for (i = 0; i < n3; i++) pos[i] = wrapCoord(pos[i] + dt * vel[i], L);
    }

    computeAccel(state, params, w);

    for (i = 0; i < n3; i++) vel[i] += hdt * acc[i];

    return state;
  }

  /* Force evaluation without advancing anything — for diagnostics. */
  function accelerations(state, params) {
    var w = workspace(state, params.N, state.count);
    computeAccel(state, params, w);
    return w.acc;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     8. exports
     ═══════════════════════════════════════════════════════════════════════ */

  root.HotaruPM = {
    /* the chain */
    depositCIC: depositCIC,
    solvePoissonFFT: solvePoissonFFT,
    solvePoissonJacobi: solvePoissonJacobi,
    solvePoissonChebyshev: solvePoissonChebyshev,
    solvePoissonSOR: solvePoissonSOR,
    jacobiSweep: jacobiSweep,
    gradient: gradient,
    sampleCIC: sampleCIC,
    step: step,
    accelerations: accelerations,

    /* solver spectra — the numbers behind the iteration counts */
    jacobiRate: jacobiRate,
    chebyshevRate: chebyshevRate,
    chebyshevBounds: chebyshevBounds,
    sorOmega: sorOmega,

    /* grid helpers */
    cellSize: cellSize,
    wrapCoord: wrapCoord,
    minImage: minImage,
    totalMass: totalMass,

    /* diagnostics — all of these port to the GPU as validation passes */
    poissonResidual: poissonResidual,
    relativeL2: relativeL2,
    relativeL2Raw: relativeL2Raw,
    roughness: roughness,
    roughnessWavelength: roughnessWavelength,
    removeMean: removeMean,
    meanOf: meanOf,
    kahanSum: kahanSum,

    /* internals, exposed so tests can reach them */
    fft3d: fft3d,
    fft1d: fft1d,
    getPlan: getPlan,
    releaseScratch: releaseScratch,
    mulberry32: mulberry32,
    gaussian: gaussian
  };

})(globalThis);
