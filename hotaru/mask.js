/* ─────────────────────────────────────────────────────────────────────────────
   Hotaru · mask.js — glyph raster  ➜  even point cloud
   ─────────────────────────────────────────────────────────────────────────────

   The page rasterises typed text (Latin, kana, kanji, emoji) onto a 2D canvas
   and hands us the alpha channel. This module turns that byte field into
   particle targets that read as a clean, soft-edged letterform rather than as
   a clumpy spray or a screen door.

   HOW IT SAMPLES
   --------------
   1. Every pixel with alpha >= threshold becomes a "lit" pixel carrying the
      weight alpha/255. Antialiased edge pixels therefore receive proportionally
      fewer points than solid interior pixels, so the edge fades instead of
      ending in a hard staircase.

   2. A cumulative weight table is built over the lit pixels. Points are drawn
      from it by *systematic* (stratified) sampling: sample k consumes the
      stratum [(k+d)/N] of the total weight, d being one seeded offset in [0,1).
      This is the low-discrepancy limit of CDF sampling — pixel i receives
      exactly floor or ceil of count*w_i/W points, never a Poisson clump and
      never a bald patch. The table is walked with a monotone cursor, so the
      whole draw is O(count + litPixels) with no binary search.

   3. Within the chosen pixel the offset comes from the R2 low-discrepancy
      sequence (the 2D plastic-constant lattice), advanced once per sample.
      Consecutive samples fall in the same or neighbouring pixels, and
      consecutive R2 points are maximally far apart, so the m points that land
      in one pixel form an m-point blue-noise-ish set rather than a stack, and
      neighbouring pixels never share an offset (no moiré against the pixel
      grid). `jitter` then mixes white noise into that offset *modulo one*,
      which keeps the distribution exactly uniform over the pixel and keeps
      every point inside its own pixel: 0 = strictly stratified, 1 = a full
      pixel of noise, 0.5 = default. The amplitude is divided by sqrt(points
      in this pixel), so the noise is always the same fraction of the local
      inter-point distance — enough to look organic rather than machined, never
      enough to clump. A pixel holding one point still gets the full jitter.

   4. Output order is scrambled by a modular bijection so that any prefix of the
      returned array is itself a complete, evenly covering point set (the
      renderer can draw fewer particles, or fade them in by index, without the
      image appearing as a top-to-bottom wipe).

   COORDINATES
   -----------
   The full w x h mask maps to a box centred on the origin whose width:height
   ratio is `aspect` (default w/h) and which fits inside [-fit, fit] on its
   longer axis. y points UP, so row 0 of the image is at the top. z is a small
   seeded value in [-depth, depth] to give the field a little thickness.

   DEGENERATE INPUT — never throws
   -------------------------------
     count <= 0 ................. Float32Array(0)
     w <= 0, h <= 0, no alpha,
       alpha shorter than w*h,
       all-zero / all-below-
       threshold mask ........... Float32Array(count*3) of zeros, i.e. every
                                  point sits at the origin
     count === 1 ................ one point, inside the lit region
     one lit pixel, huge count .. all points spread evenly inside that pixel

   Plain script. Publishes the single global HotaruMask. No DOM, no Math.random.
   ───────────────────────────────────────────────────────────────────────── */

(function (root) {
  "use strict";

  /* ── seeded PRNG ────────────────────────────────────────────────────────── */

  /* mulberry32 — 32 bits of state, one multiply-xorshift round, uniform [0,1). */
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Accepts any seed — number, string, undefined — and returns a uint32. */
  function hashSeed(seed) {
    if (seed === undefined || seed === null) return 0x9e3779b9;
    if (typeof seed === "number" && isFinite(seed)) {
      var x = Math.floor(seed) >>> 0;
      x ^= 0x9e3779b9;
      x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
      x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
      return (x ^ (x >>> 16)) >>> 0;
    }
    var s = String(seed);
    var hsh = 0x811c9dc5;
    for (var i = 0; i < s.length; i++) {
      hsh ^= s.charCodeAt(i);
      hsh = Math.imul(hsh, 0x01000193) >>> 0;
    }
    return hsh >>> 0;
  }

  /* ── small helpers ──────────────────────────────────────────────────────── */

  function num(v, dflt) {
    return typeof v === "number" && isFinite(v) ? v : dflt;
  }

  function isByteArray(a) {
    return !!a && typeof a.length === "number" && typeof a[0] !== "object";
  }

  function gcd(a, b) {
    while (b) {
      var t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

  /* An odd-ish stride coprime to n, near n*phi, used as a cheap bijection on
     [0,n) so that output index and spatial position are decorrelated. */
  function coprimeStride(n) {
    if (n < 3) return 1;
    var a = Math.floor(n * 0.6180339887498949) | 0;
    if (a < 2) a = 2;
    for (var i = 0; i < 64; i++) {
      var c = a + i;
      if (c >= n) c = ((a - i) % n + n) % n;
      if (c > 1 && gcd(c, n) === 1) return c;
    }
    return 1;
  }

  /* R2 sequence — the 2D generalisation of the golden ratio. Successive points
     are about as far apart as a 2D sequence can make them. */
  var R2_A1 = 0.7548776662466927; /* 1/g   */
  var R2_A2 = 0.5698402909980532; /* 1/g^2 */

  /* ── coverage ───────────────────────────────────────────────────────────── */

  /**
   * Measure a mask before sampling it.
   *
   * @param  {Uint8Array|Uint8ClampedArray} alpha  w*h bytes, one per pixel
   * @param  {number} w
   * @param  {number} h
   * @param  {number} [threshold=16]  pixels below this alpha are ignored;
   *                                  clamped to [1,256], so 256 or more means
   *                                  "nothing is lit"
   * @return {{litPixels:number, weight:number,
   *           bbox:{x0:number,y0:number,x1:number,y1:number,
   *                 width:number,height:number}}}
   *         `weight` is the sum of alpha/255 over lit pixels (the "ink area").
   *         `bbox` is in pixel coordinates, x0/y0 inclusive and x1/y1 exclusive,
   *         so width = x1-x0. An empty mask reports every field as 0.
   */
  function coverage(alpha, w, h, threshold) {
    var empty = {
      litPixels: 0,
      weight: 0,
      bbox: { x0: 0, y0: 0, x1: 0, y1: 0, width: 0, height: 0 }
    };
    w = Math.floor(num(w, 0));
    h = Math.floor(num(h, 0));
    if (w <= 0 || h <= 0 || !isByteArray(alpha) || alpha.length < w * h) return empty;

    var thr = Math.floor(num(threshold, 16));
    if (thr < 1) thr = 1;
    if (thr > 256) thr = 256; /* 256 and up: nothing is lit */

    var lit = 0,
      weight = 0,
      x0 = w,
      y0 = h,
      x1 = -1,
      y1 = -1;

    for (var y = 0; y < h; y++) {
      var row = y * w,
        rowHit = false;
      for (var x = 0; x < w; x++) {
        var a = alpha[row + x];
        if (a >= thr) {
          lit++;
          weight += a;
          rowHit = true;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
        }
      }
      if (rowHit) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }

    if (lit === 0) return empty;
    return {
      litPixels: lit,
      weight: weight / 255,
      bbox: {
        x0: x0,
        y0: y0,
        x1: x1 + 1,
        y1: y1 + 1,
        width: x1 + 1 - x0,
        height: y1 + 1 - y0
      }
    };
  }

  /* ── sampleMask ─────────────────────────────────────────────────────────── */

  /**
   * Turn a rasterised glyph mask into `count` particle targets.
   *
   * @param  {Uint8Array|Uint8ClampedArray} alpha  w*h bytes, one per pixel
   * @param  {number} w
   * @param  {number} h
   * @param  {number} count   number of points to produce (may far exceed or
   *                          fall far short of the number of lit pixels)
   * @param  {object} [opts]
   * @param  {number|string} [opts.seed]     any number or string; hashed to a
   *                                         uint32. Same seed, same field.
   * @param  {number} [opts.threshold=16]    alpha cut-off for "lit", clamped to
   *                                         [1,256]; 256 means nothing is lit
   * @param  {number} [opts.fit=1.0]         half-extent of the longer axis
   * @param  {number} [opts.aspect=w/h]      width:height of the mask in world units
   * @param  {number} [opts.jitter=0.5]      white noise mixed into the sub-pixel
   *                                         offset, scaled by the local point
   *                                         spacing; 0 strictly stratified,
   *                                         1 a full pixel of noise
   * @param  {number} [opts.depth=0.02]      z is seeded uniform in [-depth, depth]
   * @return {Float32Array} length count*3, laid out x,y,z,x,y,z,…
   */
  function sampleMask(alpha, w, h, count, opts) {
    opts = opts || {};

    count = Math.floor(num(count, 0));
    if (!(count > 0)) return new Float32Array(0);

    var out = new Float32Array(count * 3); /* zero-filled: the origin fallback */

    w = Math.floor(num(w, 0));
    h = Math.floor(num(h, 0));
    if (w <= 0 || h <= 0 || !isByteArray(alpha) || alpha.length < w * h) return out;

    var thr = Math.floor(num(opts.threshold, 16));
    if (thr < 1) thr = 1;
    if (thr > 256) thr = 256; /* 256 and up: nothing is lit */

    var fit = Math.abs(num(opts.fit, 1));
    var aspect = num(opts.aspect, w / h);
    if (!(aspect > 0)) aspect = w / h;
    var jitter = num(opts.jitter, 0.5);
    if (jitter < 0) jitter = 0;
    if (jitter > 1) jitter = 1;
    var depth = Math.abs(num(opts.depth, 0.02));

    /* ---- pass 1: how many pixels are lit? --------------------------------- */
    var n = w * h,
      lit = 0,
      i;
    for (i = 0; i < n; i++) if (alpha[i] >= thr) lit++;
    if (lit === 0) return out; /* nothing to draw: everything stays at origin */

    /* ---- world-space box -------------------------------------------------- */
    var halfW, halfH;
    if (aspect >= 1) {
      halfW = fit;
      halfH = fit / aspect;
    } else {
      halfW = fit * aspect;
      halfH = fit;
    }
    /* one pixel step in world units */
    var sx = (2 * halfW) / w;
    var sy = (2 * halfH) / h;

    /* ---- pass 2: cumulative weights + precomputed pixel corners ----------- */
    /* bx/by hold the world position of each lit pixel's top-left corner, so the
       inner loop is a single multiply-add per axis. */
    var cum = new Float64Array(lit);
    var bx = new Float32Array(lit);
    var by = new Float32Array(lit);
    var acc = 0,
      j = 0;
    for (var y = 0; y < h; y++) {
      var row = y * w;
      var wy = halfH - y * sy; /* y UP: image row 0 sits at the top */
      for (var x = 0; x < w; x++) {
        var a = alpha[row + x];
        if (a >= thr) {
          acc += a;
          cum[j] = acc;
          bx[j] = -halfW + x * sx;
          by[j] = wy;
          j++;
        }
      }
    }
    var total = acc;

    /* ---- sampling --------------------------------------------------------- */
    var rnd = mulberry32(hashSeed(opts.seed));
    var delta = rnd(); /* the one random offset of the systematic scan */

    /* decorrelate R2 phase from the seed too */
    var r2x = rnd(),
      r2y = rnd();

    var step = total / count;
    var stride = coprimeStride(count);
    var dst = (Math.floor(rnd() * count) % count) | 0; /* scrambled write cursor */

    var cursor = 0;
    var last = lit - 1;
    var jz = depth * 2;

    /* Jitter amplitude tracks the LOCAL point spacing rather than the pixel.
       A pixel that receives m points has an inter-point spacing of about
       1/sqrt(m) of a pixel, so the perturbation is scaled to that: it always
       breaks up lattice regularity by the same fraction of the neighbour
       distance, and never grows large enough to undo the even coverage.
       Recomputed only when the cursor steps to a new pixel. */
    var seen = -1;
    var amp = jitter;

    for (var k = 0; k < count; k++) {
      /* systematic walk of the cumulative table */
      var target = (k + delta) * step;
      while (cursor < last && cum[cursor] < target) cursor++;

      if (cursor !== seen) {
        seen = cursor;
        var wi = cursor > 0 ? cum[cursor] - cum[cursor - 1] : cum[0];
        var m = wi / step; /* points this pixel will receive */
        amp = m > 1 ? jitter / Math.sqrt(m) : jitter;
      }

      /* R2 sub-pixel offset, then jittered modulo one so it stays in-pixel */
      r2x += R2_A1;
      if (r2x >= 1) r2x -= 1;
      r2y += R2_A2;
      if (r2y >= 1) r2y -= 1;

      var u = r2x + amp * (rnd() - 0.5);
      if (u >= 1) u -= 1;
      else if (u < 0) u += 1;
      var v = r2y + amp * (rnd() - 0.5);
      if (v >= 1) v -= 1;
      else if (v < 0) v += 1;

      var o = dst * 3;
      out[o] = bx[cursor] + u * sx;
      out[o + 1] = by[cursor] - v * sy;
      out[o + 2] = (rnd() - 0.5) * jz;

      dst += stride;
      if (dst >= count) dst -= count;
    }

    return out;
  }

  root.HotaruMask = {
    sampleMask: sampleMask,
    coverage: coverage
  };
})(globalThis);
