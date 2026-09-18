/* hotaru/shapes.js — procedural 3D shape generators for the Hotaru particle field.
 *
 * Every generator has the signature fn(count, opts) -> Float32Array(count * 3),
 * xyz triples in a right-handed space (+x right, +y up, +z toward the viewer),
 * centred on the origin and scaled to sit comfortably inside the unit sphere.
 * No generator ever returns a point further than 1.0 from the origin.
 *
 * Determinism is absolute: the only entropy is opts.seed, fed through mulberry32.
 * Math.random() appears nowhere. Nothing here touches the DOM, timers or I/O.
 *
 * Plain script: one IIFE publishing the global HotaruShapes.
 */
(function (root) {
  'use strict';

  /* ── constants ─────────────────────────────────────────────────────────── */

  var TAU = Math.PI * 2;
  /* 2pi / phi^2 — the golden angle. Successive multiples never repeat a
     direction, which is what keeps Fibonacci lattices free of visible seams. */
  var GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
  var GOLDEN_FRACT = 0.6180339887498949;
  var FIT_LIMIT = 1.0; /* every cloud is scaled down into this radius */

  /* ── seeding and pseudo-randomness ─────────────────────────────────────── */

  /* Accepts a number, a string or nothing and returns a well-mixed uint32, so
     that neighbouring seeds (1, 2, 3) produce completely unrelated streams. */
  function hashSeed(seed) {
    var h;
    if (typeof seed === 'string') {
      h = 0x811c9dc5;
      for (var i = 0; i < seed.length; i++) {
        h ^= seed.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
      }
    } else if (typeof seed === 'number' && isFinite(seed)) {
      h = (Math.floor(seed) | 0) >>> 0;
    } else {
      h = 0x9e3779b9;
    }
    h ^= h >>> 16;
    h = Math.imul(h, 0x21f0aaad);
    h ^= h >>> 15;
    h = Math.imul(h, 0x735a2d97);
    h ^= h >>> 15;
    return h >>> 0;
  }

  /* mulberry32: 32 bits of state, passes gjrand/BigCrush-lite, four lines long. */
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Box–Muller, one normal per call (the spare is discarded on purpose: keeping
     it would make a generator's output depend on how many gaussians came before). */
  function gauss(rng) {
    var u = rng();
    if (u < 1e-12) u = 1e-12;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * rng());
  }

  /* A gaussian with the tail cut off, so one unlucky draw cannot fling a
     particle out of the shape and force the whole cloud to be scaled down. */
  function gaussClamped(rng, limit) {
    var g = gauss(rng);
    if (g > limit) return limit;
    if (g < -limit) return -limit;
    return g;
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  /* Numeric knobs only: anything that is not a finite number falls back to the
     tuned default, so one bad option cannot fill a buffer with NaN. */
  function pick(opts, key, dflt) {
    if (!opts) return dflt;
    var v = opts[key];
    return typeof v === 'number' && isFinite(v) ? v : dflt;
  }

  /* Boolean knobs: accepts true/false and 1/0 alike. */
  function flag(opts, key, dflt) {
    if (!opts) return dflt;
    var v = opts[key];
    if (v === undefined || v === null) return dflt;
    return !!v && v !== 0;
  }

  function smoothstep(edge0, edge1, x) {
    var t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function alloc(count) {
    var n = count > 0 ? Math.floor(count) : 0;
    return new Float32Array(n * 3);
  }

  /* Uniform point inside a ball of the given radius — used to give curves
     (helix strands, knot tube) a soft glowing thickness instead of a hairline. */
  function ballOffset(rng, radius, out) {
    var z = rng() * 2 - 1;
    var t = rng() * TAU;
    var s = Math.sqrt(Math.max(0, 1 - z * z));
    var r = radius * Math.cbrt(rng());
    out[0] = r * s * Math.cos(t);
    out[1] = r * z;
    out[2] = r * s * Math.sin(t);
  }

  /* ── small linear algebra ──────────────────────────────────────────────── */

  /* Shoemake's uniform random quaternion. Used to give seeded clouds a random
     orientation, so two seeds never line up even when the lattice is fixed. */
  function randomQuat(rng) {
    var u1 = rng(), u2 = rng(), u3 = rng();
    var s1 = Math.sqrt(1 - u1), s2 = Math.sqrt(u1);
    return [s1 * Math.sin(TAU * u2), s1 * Math.cos(TAU * u2), s2 * Math.sin(TAU * u3), s2 * Math.cos(TAU * u3)];
  }

  /* v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v) */
  function rotateByQuat(q, x, y, z, out) {
    var qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    var tx = qy * z - qz * y + qw * x;
    var ty = qz * x - qx * z + qw * y;
    var tz = qx * y - qy * x + qw * z;
    out[0] = x + 2 * (qy * tz - qz * ty);
    out[1] = y + 2 * (qz * tx - qx * tz);
    out[2] = z + 2 * (qx * ty - qy * tx);
  }

  /* Rodrigues rotation of v about a unit axis. */
  function rotateAxis(vx, vy, vz, ax, ay, az, angle, out) {
    var c = Math.cos(angle), s = Math.sin(angle);
    var d = ax * vx + ay * vy + az * vz;
    out[0] = vx * c + (ay * vz - az * vy) * s + ax * d * (1 - c);
    out[1] = vy * c + (az * vx - ax * vz) * s + ay * d * (1 - c);
    out[2] = vz * c + (ax * vy - ay * vx) * s + az * d * (1 - c);
  }

  /* ── radial density sampling ───────────────────────────────────────────── */

  /* Tabulate a 1-D density over [0,1] and return its normalised CDF, so any
     hand-authored profile (exponential disc, ringed annulus) can be sampled
     exactly and cheaply. */
  function buildCdf(bins, density) {
    var c = new Float64Array(bins + 1);
    var acc = 0;
    for (var i = 0; i < bins; i++) {
      var d = density((i + 0.5) / bins);
      acc += d > 0 ? d : 0;
      c[i + 1] = acc;
    }
    if (!(acc > 0)) {
      for (var j = 0; j <= bins; j++) c[j] = j / bins;
      return c;
    }
    for (var k = 0; k <= bins; k++) c[k] /= acc;
    c[bins] = 1;
    return c;
  }

  /* Inverse transform with linear interpolation inside the hit bin, so the
     result is continuous rather than quantised to bin edges. */
  function sampleCdf(c, u) {
    var bins = c.length - 1;
    if (!(u >= 0)) u = 0;
    if (u > 0.999999999) u = 0.999999999;
    var lo = 0, hi = bins - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (c[mid + 1] <= u) lo = mid + 1;
      else hi = mid;
    }
    var a = c[lo], b = c[lo + 1];
    var f = b > a ? (u - a) / (b - a) : 0.5;
    return (lo + f) / bins;
  }

  /* ── grid helper ───────────────────────────────────────────────────────── */

  /* A near-square grid holding exactly `n` cells. When cols*rows overshoots we
     drop the surplus cells with a Bresenham cadence, so the missing cells are
     sprinkled through the sheet instead of leaving a bitten-off last row. */
  function visitGrid(n, visit) {
    var cols = Math.max(1, Math.round(Math.sqrt(n)));
    var rows = Math.max(1, Math.ceil(n / cols));
    var total = cols * rows;
    var emitted = 0;
    var prev = 0;
    for (var c = 0; c < total && emitted < n; c++) {
      var next = Math.floor(((c + 1) * n) / total);
      if (next > prev) {
        visit(emitted++, c % cols, Math.floor(c / cols), cols, rows);
      }
      prev = next;
    }
    /* Defensive: rounding can never leave this short, but never emit garbage. */
    while (emitted < n) {
      visit(emitted, emitted % cols, Math.floor(emitted / cols) % rows, cols, rows);
      emitted++;
    }
  }

  /* ── final safety fit ──────────────────────────────────────────────────── */

  /* Shrink (never grow) the cloud so nothing escapes the unit sphere. Generators
     are tuned to land inside on their own; this is the belt to that braces. */
  function fit(out, limit) {
    var n = out.length;
    var maxSq = 0;
    for (var i = 0; i < n; i += 3) {
      var d = out[i] * out[i] + out[i + 1] * out[i + 1] + out[i + 2] * out[i + 2];
      if (d > maxSq) maxSq = d;
    }
    var max = Math.sqrt(maxSq);
    if (max > limit && max > 0) {
      var s = limit / max;
      for (var j = 0; j < n; j++) out[j] *= s;
    }
    return out;
  }

  /* ══ sphere ════════════════════════════════════════════════════════════ */

  /* Fibonacci (golden-angle) lattice: equal-area bands in y, golden-angle steps
     in longitude. Spacing is as even as a deterministic sphere covering gets —
     no pole clustering, no seams. opts.shell lets points sink inward, sampled
     uniformly in volume across the shell so the density stays flat. */
  function sphere(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var shell = clamp(pick(opts, 'shell', 0), 0, 1);
    var radius = pick(opts, 'radius', 1);
    var rng = mulberry32(hashSeed(opts && opts.seed));
    var q = randomQuat(rng);

    var inner = 1 - shell;
    var inner3 = inner * inner * inner;
    var v = [0, 0, 0];

    for (var i = 0; i < n; i++) {
      var y = 1 - (2 * i + 1) / n;
      var s = Math.sqrt(Math.max(0, 1 - y * y));
      var th = GOLDEN_ANGLE * i;
      var r = radius;
      if (shell > 0) {
        /* stratified so the radial spread stays smooth at any count */
        var u = (i + rng()) / n;
        r = radius * Math.cbrt(inner3 + (1 - inner3) * u);
      }
      rotateByQuat(q, Math.cos(th) * s, y, Math.sin(th) * s, v);
      var o = i * 3;
      out[o] = v[0] * r;
      out[o + 1] = v[1] * r;
      out[o + 2] = v[2] * r;
    }
    return fit(out, FIT_LIMIT);
  }

  /* ══ galaxy ════════════════════════════════════════════════════════════ */

  /* A grand-design spiral: exponential disc (dense core, sparse rim), arms laid
     on a logarithmic spiral theta = ln(r/r0)/tan(pitch), angular scatter that
     widens outward, a disc that thins outward, a flattened central bulge, and a
     pinch of inter-arm stars so the arms have something to stand out against. */
  function galaxy(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var arms = Math.max(1, Math.round(pick(opts, 'arms', 3)));
    var pitch = pick(opts, 'pitch', 0.235);          /* radians; 13.5 deg */
    var spread = pick(opts, 'spread', 0.155);        /* arm half-width, radians */
    var bulgeFrac = clamp(pick(opts, 'bulge', 0.17), 0, 1);
    var bulgeRadius = pick(opts, 'bulgeRadius', 0.2);
    var thickness = pick(opts, 'thickness', 0.062);
    var scaleLength = pick(opts, 'scaleLength', 0.28); /* exponential disc h */
    var interArm = clamp(pick(opts, 'interArm', 0.11), 0, 1);
    var rMax = pick(opts, 'radius', 0.97);
    var spin = pick(opts, 'spin', 1) >= 0 ? 1 : -1;

    var rng = mulberry32(hashSeed(opts && opts.seed));
    var phase = rng() * TAU;

    /* N(r) dr ∝ r * exp(-r/h) dr — a real exponential disc — with the rim
       feathered off so the galaxy dissolves instead of ending at a wall. */
    var cdf = buildCdf(1024, function (x) {
      var r = x * rMax;
      var edge = 1 - smoothstep(0.78, 1, x) * 0.93;
      return r * Math.exp(-r / scaleLength) * edge;
    });

    var rMin = 0.055;                  /* spiral reference radius */
    var invTan = 1 / Math.tan(pitch);
    var armStep = TAU / arms;

    var nBulge = Math.round(n * bulgeFrac);
    var nDisc = n - nBulge;
    var bAcc = 0, bUsed = 0, dUsed = 0;

    for (var i = 0; i < n; i++) {
      var o = i * 3;
      var x, y, z;
      bAcc += bulgeFrac;
      if (bAcc >= 1 && bUsed < nBulge) {
        /* ── bulge: flattened, steeply concentrated spheroid ── */
        bAcc -= 1;
        var ub = (bUsed + rng()) / Math.max(1, nBulge);
        var rb = bulgeRadius * Math.pow(ub, 1.55);
        var cz = rng() * 2 - 1;
        var sa = Math.sqrt(Math.max(0, 1 - cz * cz));
        var ang = rng() * TAU;
        x = rb * sa * Math.cos(ang);
        z = rb * sa * Math.sin(ang);
        y = rb * cz * 0.6;
        bUsed++;
      } else {
        /* ── disc: logarithmic spiral arms ── */
        var ud = (dUsed + rng()) / Math.max(1, nDisc);
        var r = sampleCdf(cdf, ud) * rMax;
        dUsed++;

        var wind = Math.log(Math.max(r, rMin) / rMin) * invTan * spin;
        var th;
        if (rng() < interArm) {
          th = rng() * TAU;                       /* field star */
        } else {
          var arm = i % arms;
          var sigma = spread * (0.45 + 0.8 * (r / rMax));
          th = arm * armStep + wind + gaussClamped(rng, 2.6) * sigma;
        }
        th += phase;

        /* disc scale height falls off outward: thin, flaring core */
        var hz = thickness * Math.exp(-r / 0.42);
        x = r * Math.cos(th);
        z = r * Math.sin(th);
        y = gaussClamped(rng, 2.6) * hz;
      }
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = z;
    }
    return fit(out, FIT_LIMIT);
  }

  /* ══ torus knot ════════════════════════════════════════════════════════ */

  /* Points on the tube around a (p,q) torus knot. The curve is tabulated once,
     re-parameterised by arc length (raw t bunches badly where the curve bends),
     and carried by a rotation-minimising frame whose closing twist is undone so
     the tube has no seam. */
  function torusKnot(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var p = Math.max(1, Math.round(pick(opts, 'p', 2)));
    var q = Math.max(1, Math.round(pick(opts, 'q', 3)));
    var tube = Math.max(0, pick(opts, 'tube', 0.12));
    var fill = clamp(pick(opts, 'fill', 0.3), 0, 1); /* radial thickness of the shell */
    var rng = mulberry32(hashSeed(opts && opts.seed));

    var SAMPLES = 2048;
    var px = new Float64Array(SAMPLES + 1);
    var py = new Float64Array(SAMPLES + 1);
    var pz = new Float64Array(SAMPLES + 1);
    var tx = new Float64Array(SAMPLES + 1);
    var ty = new Float64Array(SAMPLES + 1);
    var tz = new Float64Array(SAMPLES + 1);
    var nx = new Float64Array(SAMPLES + 1);
    var ny = new Float64Array(SAMPLES + 1);
    var nz = new Float64Array(SAMPLES + 1);
    var arc = new Float64Array(SAMPLES + 1);

    var i, t, rr, maxR = 0;
    for (i = 0; i <= SAMPLES; i++) {
      t = (i / SAMPLES) * TAU;
      rr = Math.cos(q * t) + 2;
      px[i] = rr * Math.cos(p * t);
      py[i] = rr * Math.sin(p * t);
      pz[i] = -Math.sin(q * t);
      /* analytic tangent keeps the frame clean at the seam */
      var dr = -q * Math.sin(q * t);
      var dx = dr * Math.cos(p * t) - rr * p * Math.sin(p * t);
      var dy = dr * Math.sin(p * t) + rr * p * Math.cos(p * t);
      var dz = -q * Math.cos(q * t);
      var dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      tx[i] = dx / dl; ty[i] = dy / dl; tz[i] = dz / dl;
      var rad = Math.sqrt(px[i] * px[i] + py[i] * py[i] + pz[i] * pz[i]);
      if (rad > maxR) maxR = rad;
    }
    for (i = 1; i <= SAMPLES; i++) {
      var ax = px[i] - px[i - 1], ay = py[i] - py[i - 1], az = pz[i] - pz[i - 1];
      arc[i] = arc[i - 1] + Math.sqrt(ax * ax + ay * ay + az * az);
    }
    var length = arc[SAMPLES];

    /* parallel transport: start with any normal, rotate it by the minimal
       rotation that carries each tangent to the next */
    var seedVec = Math.abs(tz[0]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    var n0x = ty[0] * seedVec[2] - tz[0] * seedVec[1];
    var n0y = tz[0] * seedVec[0] - tx[0] * seedVec[2];
    var n0z = tx[0] * seedVec[1] - ty[0] * seedVec[0];
    var n0l = Math.sqrt(n0x * n0x + n0y * n0y + n0z * n0z) || 1;
    nx[0] = n0x / n0l; ny[0] = n0y / n0l; nz[0] = n0z / n0l;
    var tmp = [0, 0, 0];
    for (i = 1; i <= SAMPLES; i++) {
      var cx = ty[i - 1] * tz[i] - tz[i - 1] * ty[i];
      var cy = tz[i - 1] * tx[i] - tx[i - 1] * tz[i];
      var cz = tx[i - 1] * ty[i] - ty[i - 1] * tx[i];
      var sn = Math.sqrt(cx * cx + cy * cy + cz * cz);
      if (sn < 1e-12) {
        nx[i] = nx[i - 1]; ny[i] = ny[i - 1]; nz[i] = nz[i - 1];
      } else {
        var cs = tx[i - 1] * tx[i] + ty[i - 1] * ty[i] + tz[i - 1] * tz[i];
        rotateAxis(nx[i - 1], ny[i - 1], nz[i - 1], cx / sn, cy / sn, cz / sn, Math.atan2(sn, cs), tmp);
        nx[i] = tmp[0]; ny[i] = tmp[1]; nz[i] = tmp[2];
      }
    }
    /* undo the holonomy accumulated over the loop, spread along arc length */
    var hx = ny[0] * nz[SAMPLES] - nz[0] * ny[SAMPLES];
    var hy = nz[0] * nx[SAMPLES] - nx[0] * nz[SAMPLES];
    var hz2 = nx[0] * ny[SAMPLES] - ny[0] * nx[SAMPLES];
    var sinH = hx * tx[0] + hy * ty[0] + hz2 * tz[0];
    var cosH = nx[0] * nx[SAMPLES] + ny[0] * ny[SAMPLES] + nz[0] * nz[SAMPLES];
    var holo = Math.atan2(sinH, cosH);
    for (i = 0; i <= SAMPLES; i++) {
      var k = length > 0 ? arc[i] / length : 0;
      rotateAxis(nx[i], ny[i], nz[i], tx[i], ty[i], tz[i], -holo * k, tmp);
      nx[i] = tmp[0]; ny[i] = tmp[1]; nz[i] = tmp[2];
    }

    /* Exact fit: the widest point of the tube is
         max_phi |s*P + tube*(cos phi N + sin phi B)|
               = sqrt(s^2|P|^2 + tube^2 + 2 s tube sqrt((P.N)^2 + (P.B)^2))
       which is monotonic in s, so a few bisections put the knot's true hull
       exactly on the target radius whatever p, q and tube are. */
    var perp = new Float64Array(SAMPLES + 1);
    var plen = new Float64Array(SAMPLES + 1);
    for (i = 0; i <= SAMPLES; i++) {
      var bx0 = ty[i] * nz[i] - tz[i] * ny[i];
      var by0 = tz[i] * nx[i] - tx[i] * nz[i];
      var bz0 = tx[i] * ny[i] - ty[i] * nx[i];
      var dn = px[i] * nx[i] + py[i] * ny[i] + pz[i] * nz[i];
      var db = px[i] * bx0 + py[i] * by0 + pz[i] * bz0;
      perp[i] = Math.sqrt(dn * dn + db * db);
      plen[i] = Math.sqrt(px[i] * px[i] + py[i] * py[i] + pz[i] * pz[i]);
    }
    var hull = function (s) {
      var m = 0;
      for (var a = 0; a <= SAMPLES; a++) {
        var v2 = s * s * plen[a] * plen[a] + tube * tube + 2 * s * tube * perp[a];
        if (v2 > m) m = v2;
      }
      return Math.sqrt(m);
    };
    var target = pick(opts, 'radius', 0.995);
    var sLo = 0, sHi = target / Math.max(1e-6, maxR) + 1;
    for (i = 0; i < 24; i++) {
      var sm = (sLo + sHi) * 0.5;
      if (hull(sm) > target) sHi = sm; else sLo = sm;
    }
    var scale = (sLo + sHi) * 0.5;
    var phase = rng() * TAU;
    var sPhase = rng();

    for (var j = 0; j < n; j++) {
      /* perfectly even arc-length spacing, offset by a seeded phase */
      var s = (((j + sPhase) / n) % 1) * length;
      /* locate s in the arc table */
      var lo = 0, hi = SAMPLES;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (arc[mid + 1] <= s) lo = mid + 1;
        else hi = mid;
      }
      var seg = arc[lo + 1] - arc[lo];
      var f = seg > 1e-15 ? (s - arc[lo]) / seg : 0;
      var i0 = lo, i1 = lo + 1;

      var cxp = px[i0] + (px[i1] - px[i0]) * f;
      var cyp = py[i0] + (py[i1] - py[i0]) * f;
      var czp = pz[i0] + (pz[i1] - pz[i0]) * f;
      var fnx = nx[i0] + (nx[i1] - nx[i0]) * f;
      var fny = ny[i0] + (ny[i1] - ny[i0]) * f;
      var fnz = nz[i0] + (nz[i1] - nz[i0]) * f;
      var ftx = tx[i0] + (tx[i1] - tx[i0]) * f;
      var fty = ty[i0] + (ty[i1] - ty[i0]) * f;
      var ftz = tz[i0] + (tz[i1] - tz[i0]) * f;
      var tl = Math.sqrt(ftx * ftx + fty * fty + ftz * ftz) || 1;
      ftx /= tl; fty /= tl; ftz /= tl;
      /* re-orthogonalise the interpolated normal against the tangent */
      var dot = fnx * ftx + fny * fty + fnz * ftz;
      fnx -= ftx * dot; fny -= fty * dot; fnz -= ftz * dot;
      var nl = Math.sqrt(fnx * fnx + fny * fny + fnz * fnz) || 1;
      fnx /= nl; fny /= nl; fnz /= nl;
      var fbx = fty * fnz - ftz * fny;
      var fby = ftz * fnx - ftx * fnz;
      var fbz = ftx * fny - fty * fnx;

      /* golden-angle roll: the tube fills evenly without stripes */
      var phi = GOLDEN_ANGLE * j + phase;
      var rt = tube > 0 ? tube * Math.sqrt(1 - fill * rng() * (2 - fill)) : 0;
      var ca = Math.cos(phi) * rt, sa2 = Math.sin(phi) * rt;

      var o = j * 3;
      out[o] = (cxp + fnx * ca + fbx * sa2) * scale;
      out[o + 1] = (cyp + fny * ca + fby * sa2) * scale;
      out[o + 2] = (czp + fnz * ca + fbz * sa2) * scale;
    }
    return fit(out, FIT_LIMIT);
  }

  /* ══ wave ══════════════════════════════════════════════════════════════ */

  /* A near-square sheet on the xz plane lifted by two circular waves from two
     sources — the two-slit interference figure, which has far more structure
     than crossed plane waves. Each grid cell gets one stratified sample so the
     sheet reads as a continuous surface rather than a fly screen. */
  function wave(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var half = pick(opts, 'extent', 0.7);
    var amp = pick(opts, 'amplitude', 0.09);
    var k = pick(opts, 'frequency', 26);
    var sep = pick(opts, 'separation', 0.5);
    var decay = pick(opts, 'decay', 1.6);
    var jitter = clamp(pick(opts, 'jitter', 0.85), 0, 1);
    var rng = mulberry32(hashSeed(opts && opts.seed));

    var ph1 = rng() * TAU, ph2 = rng() * TAU;
    var s1x = -sep, s1z = -sep * 0.35;
    var s2x = sep, s2z = sep * 0.35;

    visitGrid(n, function (idx, col, rowI, cols, rows) {
      var jx = 0.5 + (rng() - 0.5) * jitter;
      var jz = 0.5 + (rng() - 0.5) * jitter;
      var x = ((col + jx) / cols) * 2 * half - half;
      var z = ((rowI + jz) / rows) * 2 * half - half;

      var d1 = Math.sqrt((x - s1x) * (x - s1x) + (z - s1z) * (z - s1z));
      var d2 = Math.sqrt((x - s2x) * (x - s2x) + (z - s2z) * (z - s2z));
      var a1 = 1 / (1 + decay * d1);
      var a2 = 1 / (1 + decay * d2);
      var y = amp * (a1 * Math.sin(k * d1 - ph1) + a2 * Math.sin(k * d2 - ph2));

      var o = idx * 3;
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = z;
    });
    return fit(out, FIT_LIMIT);
  }

  /* ══ ring ══════════════════════════════════════════════════════════════ */

  /* A Saturn annulus: area-correct radial sampling through a hand-authored
     density profile with two Cassini-style gaps and a stack of finer bands.
     Angles walk by the golden fraction, which spreads the points like a Vogel
     spiral — dense, even, and free of the clumps plain random angles give. */
  function ring(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var inner = pick(opts, 'inner', 0.45);
    var outer = pick(opts, 'outer', 0.97);
    var thickness = pick(opts, 'thickness', 0.008);
    var bandDepth = clamp(pick(opts, 'bands', 0.3), 0, 0.8);
    var rng = mulberry32(hashSeed(opts && opts.seed));

    /* {centre (0..1 across the annulus), width, depth} */
    var gaps = opts && opts.gaps ? opts.gaps : [
      { at: 0.6, width: 0.045, depth: 0.97 },   /* Cassini division */
      { at: 0.89, width: 0.014, depth: 0.85 }   /* Encke gap */
    ];

    var span = outer - inner;
    var cdf = buildCdf(2048, function (x) {
      var r = inner + x * span;
      var d = r;                                   /* area weighting */
      d *= 1 + bandDepth * Math.sin(TAU * 6.3 * x + 0.7)
             + bandDepth * 0.55 * Math.sin(TAU * 13.9 * x + 2.1);
      for (var g = 0; g < gaps.length; g++) {
        var t = (x - gaps[g].at) / gaps[g].width;
        d *= 1 - gaps[g].depth * Math.exp(-t * t);
      }
      d *= smoothstep(0, 0.02, x) * (1 - smoothstep(0.97, 1, x) * 0.9);
      return d;
    });

    var phase = rng();
    for (var i = 0; i < n; i++) {
      var u = (i + rng()) / n;
      var r = inner + sampleCdf(cdf, u) * span;
      var th = TAU * ((i * GOLDEN_FRACT + phase) % 1);
      var o = i * 3;
      out[o] = r * Math.cos(th);
      out[o + 1] = gaussClamped(rng, 2.5) * thickness;
      out[o + 2] = r * Math.sin(th);
    }
    return fit(out, FIT_LIMIT);
  }

  /* ══ cube ══════════════════════════════════════════════════════════════ */

  /* Six faces, equal area so equal share of the points, stratified inside each
     face. Half-size 0.575 puts the corners at radius 0.996 — as large as a cube
     can be inside the unit sphere. */
  function cube(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var half = pick(opts, 'size', 0.575);
    var jitter = clamp(pick(opts, 'jitter', 0.9), 0, 1);
    var rng = mulberry32(hashSeed(opts && opts.seed));

    var base = Math.floor(n / 6), extra = n % 6;
    /* which faces get the leftover points rotates with the seed */
    var offset = Math.floor(rng() * 6);
    var cursor = 0;
    for (var f = 0; f < 6; f++) {
      var face = (f + offset) % 6;
      var cnt = base + (f < extra ? 1 : 0);
      if (cnt === 0) continue;
      var axis = face >> 1;
      var sign = (face & 1) ? 1 : -1;
      var a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
      var start = cursor;
      visitGrid(cnt, function (idx, col, rowI, cols, rows) {
        var ju = 0.5 + (rng() - 0.5) * jitter;
        var jv = 0.5 + (rng() - 0.5) * jitter;
        var u = ((col + ju) / cols) * 2 - 1;
        var v = ((rowI + jv) / rows) * 2 - 1;
        var o = (start + idx) * 3;
        out[o + axis] = sign * half;
        out[o + a0] = u * half;
        out[o + a1] = v * half;
      });
      cursor += cnt;
    }
    return fit(out, FIT_LIMIT);
  }

  /* ══ heart ═════════════════════════════════════════════════════════════ */

  /* Taubin's heart: (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3 = 0.
   *
   * Along a ray p = t*d the implicit collapses to a two-coefficient polynomial
   *     F(t) = (A t^2 - 1)^3 - B t^5,  A = dx^2 + 9/4 dy^2 + dz^2,
   *                                    B = (dx^2 + 9/80 dy^2) dz^3
   * so the surface radius in any direction costs a handful of multiplies. The
   * surface is star-shaped about the origin, so one root per direction is all
   * there is.
   *
   * Even directions do NOT give even points — a star-shaped surface stretches
   * solid angle by r^2 / cos(angle between the ray and the normal). So we weigh
   * a Fibonacci lattice of directions by exactly that factor, resample it, and
   * jitter each survivor inside its own lattice cell.
   *
   * Equation space has z up and y as the thin axis; the output is rotated -90
   * degrees about x, giving lobes up (+y), the tip down, and the heart facing a
   * viewer who looks down -z.
   */
  var HEART_TMAX = 3;

  function heartRoot(dx, dy, dz) {
    var A = dx * dx + 2.25 * dy * dy + dz * dz;
    var B = (dx * dx + 0.1125 * dy * dy) * dz * dz * dz;
    var STEPS = 40;
    var prevT = 0, prevF = -1;         /* F(0) = -1 */
    var loT = -1, hiT = -1;
    for (var i = 1; i <= STEPS; i++) {
      var t = (i / STEPS) * HEART_TMAX;
      var g = A * t * t - 1;
      var fv = g * g * g - B * t * t * t * t * t;
      if ((fv >= 0) !== (prevF >= 0)) { loT = prevT; hiT = t; }
      prevT = t; prevF = fv;
    }
    if (hiT < 0) return 0;
    for (var k = 0; k < 26; k++) {
      var m = (loT + hiT) * 0.5;
      var gm = A * m * m - 1;
      var fm = gm * gm * gm - B * m * m * m * m * m;
      if (fm >= 0) hiT = m; else loT = m;
    }
    return (loT + hiT) * 0.5;
  }

  /* Gradient of the implicit at an equation-space point (used for the area
     weight). Returned through `o` as [gx, gy, gz]. */
  function heartGrad(x, y, z, o) {
    var g = x * x + 2.25 * y * y + z * z - 1;
    var g2 = 3 * g * g;
    var z3 = z * z * z, z2 = z * z;
    o[0] = g2 * 2 * x - 2 * x * z3;
    o[1] = g2 * 4.5 * y - 0.225 * y * z3;
    o[2] = g2 * 2 * z - 3 * z2 * (x * x + 0.1125 * y * y);
  }

  /* One fixed scan at load time fixes the scale for every count and seed, so a
     10-point heart and a million-point heart are exactly the same size. */
  var HEART_RAW_MAX = (function () {
    var N = 4096, m = 0;
    for (var i = 0; i < N; i++) {
      var y = 1 - (2 * i + 1) / N;
      var s = Math.sqrt(Math.max(0, 1 - y * y));
      var th = GOLDEN_ANGLE * i;
      var r = heartRoot(Math.cos(th) * s, y, Math.sin(th) * s);
      if (r > m) m = r;
    }
    return m;
  })();
  var HEART_SCALE = 0.97 / HEART_RAW_MAX;

  function heart(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var scale = pick(opts, 'scale', HEART_SCALE);
    var even = flag(opts, 'even', true);
    var rng = mulberry32(hashSeed(opts && opts.seed));
    var q = randomQuat(rng);
    var spinOnly = flag(opts, 'randomRotation', false);
    var lat = rng() * TAU;              /* lattice phase so seeds differ */

    var dirs = new Float64Array(n * 3);
    var rad = new Float64Array(n);
    var wsum = 0;
    var weights = even ? new Float64Array(n) : null;
    var grad = [0, 0, 0];

    for (var i = 0; i < n; i++) {
      var y = 1 - (2 * i + 1) / n;
      var s = Math.sqrt(Math.max(0, 1 - y * y));
      var th = GOLDEN_ANGLE * i + lat;
      var dx = Math.cos(th) * s, dy = y, dz = Math.sin(th) * s;
      dirs[i * 3] = dx; dirs[i * 3 + 1] = dy; dirs[i * 3 + 2] = dz;
      var r = heartRoot(dx, dy, dz);
      rad[i] = r;
      if (even) {
        heartGrad(r * dx, r * dy, r * dz, grad);
        var gl = Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
        var cosA = gl > 1e-12 ? Math.abs((grad[0] * dx + grad[1] * dy + grad[2] * dz) / gl) : 1;
        var w = (r * r) / Math.max(cosA, 0.12);
        weights[i] = w;
        wsum += w;
      }
    }

    /* angular radius of one lattice cell: 2pi(1 - cos a) = 4pi/n */
    var cell = n > 1 ? Math.acos(clamp(1 - 2 / n, -1, 1)) : Math.PI;
    var v = [0, 0, 0];
    var tmpv = [0, 0, 0];

    /* Systematic (low-variance) resampling proportional to surface area. The
       picks come out sorted, so a run of equal picks is exactly a cell that has
       to carry several points. */
    var picks = new Int32Array(n);
    if (even) {
      var step = wsum / n;
      var walk = rng() * step;
      var src = 0, acc = weights[0];
      for (var a = 0; a < n; a++) {
        var target = walk + a * step;
        while (acc < target && src < n - 1) { src++; acc += weights[src]; }
        picks[a] = src;
      }
    } else {
      for (var b = 0; b < n; b++) picks[b] = b;
    }

    for (var j = 0; j < n;) {
      var idx = picks[j];
      var k = 1;
      while (j + k < n && picks[j + k] === idx) k++;

      var bx = dirs[idx * 3], by = dirs[idx * 3 + 1], bz = dirs[idx * 3 + 2];
      /* a perpendicular to tilt away from */
      var ux = Math.abs(bz) < 0.9 ? 0 : 1, uy = 0, uz = Math.abs(bz) < 0.9 ? 1 : 0;
      var ax = by * uz - bz * uy, ay = bz * ux - bx * uz, az = bx * uy - by * ux;
      var al = Math.sqrt(ax * ax + ay * ay + az * az) || 1;
      ax /= al; ay /= al; az /= al;
      var phase = rng() * TAU;

      for (var m = 0; m < k; m++) {
        var ex, ey, ez, rr;
        if (k === 1) {
          /* the common case: sit exactly on the lattice, radius already solved */
          ex = bx; ey = by; ez = bz; rr = rad[idx];
        } else {
          /* A cell asked to carry k points subdivides into a k-point golden
             spiral inside itself. Random jitter would let two of them land on
             top of each other and burn a hot spot into the render. */
          /* 0.85 of the cell radius: tuned so the sub-points sit as far apart as
             their neighbours in the untouched cells around them */
          var tilt = cell * 0.85 * Math.sqrt((m + 0.5) / k);
          rotateAxis(bx, by, bz, ax, ay, az, tilt, tmpv);
          rotateAxis(tmpv[0], tmpv[1], tmpv[2], bx, by, bz, GOLDEN_ANGLE * m + phase, v);
          var vl = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1;
          ex = v[0] / vl; ey = v[1] / vl; ez = v[2] / vl;
          rr = heartRoot(ex, ey, ez);
        }

        /* equation space (z up, y thin) -> world (y up, z toward the viewer) */
        var wx = rr * ex, wy = rr * ez, wz = -rr * ey;
        if (spinOnly) {
          rotateByQuat(q, wx, wy, wz, v);
          wx = v[0]; wy = v[1]; wz = v[2];
        }
        var o = (j + m) * 3;
        out[o] = wx * scale;
        out[o + 1] = wy * scale;
        out[o + 2] = wz * scale;
      }
      j += k;
    }
    return fit(out, 1.1);
  }

  /* ══ helix ═════════════════════════════════════════════════════════════ */

  /* A DNA double helix: two strands around the y axis, offset by 133 degrees so
     the major and minor grooves read correctly, plus base-pair rungs bridging
     them. Strand points are evenly spaced in t (a helix has constant speed, so
     that is already even arc length) and given a soft tube thickness. */
  function helix(count, opts) {
    var out = alloc(count);
    var n = out.length / 3;
    if (n === 0) return out;

    var turns = pick(opts, 'turns', 3.5);
    var radius = pick(opts, 'radius', 0.33);
    var height = pick(opts, 'height', 1.78);
    var strandPhase = pick(opts, 'strandPhase', 2.32);
    var rungRatio = clamp(pick(opts, 'rungRatio', 0.3), 0, 0.9);
    var rungsPerTurn = pick(opts, 'rungsPerTurn', 9);
    var strandTube = pick(opts, 'strandTube', 0.022);
    var rungTube = pick(opts, 'rungTube', 0.013);
    var rng = mulberry32(hashSeed(opts && opts.seed));
    var phase = rng() * TAU;

    var nRung = Math.round(n * rungRatio);
    var nStrand = n - nRung;
    var perStrand = [Math.ceil(nStrand / 2), Math.floor(nStrand / 2)];
    var used = [0, 0];
    var rungs = Math.max(1, Math.round(turns * rungsPerTurn));
    var T = turns * TAU;
    var off = [0, 0, 0];

    var rAcc = 0, rUsed = 0;
    for (var i = 0; i < n; i++) {
      var o = i * 3;
      var x, y, z, u, t, ang;
      rAcc += rungRatio;
      if (rAcc >= 1 && rUsed < nRung) {
        /* ── rung: a bar between the two strands ── */
        rAcc -= 1;
        var ri = rUsed % rungs;
        u = (ri + 0.5) / rungs;
        t = u * T;
        var yc = (u - 0.5) * height;
        var a0 = t + phase, a1 = t + phase + strandPhase;
        var x0 = radius * Math.cos(a0), z0 = radius * Math.sin(a0);
        var x1 = radius * Math.cos(a1), z1 = radius * Math.sin(a1);
        /* stay off the strands themselves so the rung reads as a bridge */
        var f = 0.07 + 0.86 * rng();
        x = x0 + (x1 - x0) * f;
        z = z0 + (z1 - z0) * f;
        y = yc;
        ballOffset(rng, rungTube, off);
        rUsed++;
      } else {
        /* ── strand ── */
        var strand = used[0] <= used[1] ? 0 : 1;
        if (perStrand[strand] === 0) strand = 1 - strand;
        var k = used[strand]++;
        u = (k + 0.5) / Math.max(1, perStrand[strand]);
        t = u * T;
        ang = t + phase + (strand ? strandPhase : 0);
        x = radius * Math.cos(ang);
        z = radius * Math.sin(ang);
        y = (u - 0.5) * height;
        ballOffset(rng, strandTube, off);
      }
      out[o] = x + off[0];
      out[o + 1] = y + off[1];
      out[o + 2] = z + off[2];
    }
    return fit(out, FIT_LIMIT);
  }

  /* ── registry ──────────────────────────────────────────────────────────── */

  var ORDER = ['sphere', 'galaxy', 'torusKnot', 'wave', 'ring', 'cube', 'heart', 'helix'];

  var LABELS = {
    sphere: '球',            /* 球 */
    galaxy: '銀河',      /* 銀河 */
    torusKnot: '結び目', /* 結び目 */
    wave: '波',              /* 波 */
    ring: '環',              /* 環 */
    cube: '立方体',  /* 立方体 */
    heart: 'ハート', /* ハート */
    helix: '螺旋'        /* 螺旋 */
  };

  function list() {
    return ORDER.slice();
  }

  root.HotaruShapes = {
    sphere: sphere,
    galaxy: galaxy,
    torusKnot: torusKnot,
    wave: wave,
    ring: ring,
    cube: cube,
    heart: heart,
    helix: helix,
    list: list,
    labels: LABELS,
    /* exported so callers (and the tests) can map world coordinates back into
       the heart's equation space: eq = (wx/S, -wz/S, wy/S) */
    HEART_SCALE: HEART_SCALE
  };
})(globalThis);
