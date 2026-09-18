/* ─────────────────────────────────────────────────────────────────────────────
   Hotaru · noise
   ---------------------------------------------------------------------------
   GLSL ES 3.00 source for 3D simplex noise, fBm and curl noise, plus a
   JavaScript mirror of each so the shaders can be validated on the CPU.

   The simplex implementation is the Ashima Arts / Stefan Gustavson
   "webgl-noise" noise3D.glsl (MIT / public domain), transcribed verbatim.
   The JavaScript below is a line-for-line scalar expansion of that same
   source: every constant is written with the same literal it has in the
   shader (1.0/6.0, 289.0, 34.0, 49.0, 0.142857142857, 1.79284291400159,
   0.85373472095314, 0.6, 42.0) so that test/noise.test.js can diff the two
   textually as well as numerically.

   Plain script. Publishes exactly one global: HotaruNoise.
   No import/export, no require, no window, no document, no Math.random.
   ───────────────────────────────────────────────────────────────────────── */
(function (root) {
  "use strict";

  /* ═══════════════════════════════════════════════════════════════════════
     1. GLSL — simplex 3D (Ashima Arts / Stefan Gustavson, verbatim)

     One deliberate property of this implementation is worth knowing before
     you build on it: the kernel radius is 0.6 (max(0.6 - r*r, 0.0)), not the
     0.5 that would make the kernel vanish exactly at the neighbouring corner.
     0.6 gives noticeably more contrast — which is why webgl-noise ships it —
     at the cost of a small step at simplex cell boundaries, where a corner
     enters or leaves the sum while its weight is still non-zero. Measured
     over thousands of bisected boundary crossings that step is at most about
     6e-3 out of a [-1, 1] range (mean 3e-4), on a measure-zero surface.
     It is invisible in the field and harmless to the curl's divergence, which
     is an exact discrete identity rather than a property of smoothness.
     Kept as-is: this is the reference implementation, not a variant of it.
     ═══════════════════════════════════════════════════════════════════════ */

  var GLSL_SIMPLEX3 = [
    "// Description : Array and textureless GLSL 2D/3D/4D simplex",
    "//               noise functions.",
    "//      Author : Ian McEwan, Ashima Arts.",
    "//  Maintainer : stegu",
    "//     Lastmod : 20110822 (ijm)",
    "//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.",
    "//               Distributed under the MIT License. See LICENSE file.",
    "//               https://github.com/ashima/webgl-noise",
    "//               https://github.com/stegu/webgl-noise",
    "",
    "vec3 mod289(vec3 x) {",
    "  return x - floor(x * (1.0 / 289.0)) * 289.0;",
    "}",
    "",
    "vec4 mod289(vec4 x) {",
    "  return x - floor(x * (1.0 / 289.0)) * 289.0;",
    "}",
    "",
    "vec4 permute(vec4 x) {",
    "     return mod289(((x*34.0)+1.0)*x);",
    "}",
    "",
    "vec4 taylorInvSqrt(vec4 r) {",
    "  return 1.79284291400159 - 0.85373472095314 * r;",
    "}",
    "",
    "float snoise(vec3 v) {",
    "  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;",
    "  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);",
    "",
    "// First corner",
    "  vec3 i  = floor(v + dot(v, C.yyy) );",
    "  vec3 x0 =   v - i + dot(i, C.xxx) ;",
    "",
    "// Other corners",
    "  vec3 g = step(x0.yzx, x0.xyz);",
    "  vec3 l = 1.0 - g;",
    "  vec3 i1 = min( g.xyz, l.zxy );",
    "  vec3 i2 = max( g.xyz, l.zxy );",
    "",
    "  //   x0 = x0 - 0.0 + 0.0 * C.xxx;",
    "  //   x1 = x0 - i1  + 1.0 * C.xxx;",
    "  //   x2 = x0 - i2  + 2.0 * C.xxx;",
    "  //   x3 = x0 - 1.0 + 3.0 * C.xxx;",
    "  vec3 x1 = x0 - i1 + C.xxx;",
    "  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y",
    "  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y",
    "",
    "// Permutations",
    "  i = mod289(i);",
    "  vec4 p = permute( permute( permute(",
    "             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))",
    "           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))",
    "           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));",
    "",
    "// Gradients: 7x7 points over a square, mapped onto an octahedron.",
    "// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)",
    "  float n_ = 0.142857142857; // 1.0/7.0",
    "  vec3  ns = n_ * D.wyz - D.xzx;",
    "",
    "  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)",
    "",
    "  vec4 x_ = floor(j * ns.z);",
    "  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)",
    "",
    "  vec4 x = x_ *ns.x + ns.yyyy;",
    "  vec4 y = y_ *ns.x + ns.yyyy;",
    "  vec4 h = 1.0 - abs(x) - abs(y);",
    "",
    "  vec4 b0 = vec4( x.xy, y.xy );",
    "  vec4 b1 = vec4( x.zw, y.zw );",
    "",
    "  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;",
    "  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;",
    "  vec4 s0 = floor(b0)*2.0 + 1.0;",
    "  vec4 s1 = floor(b1)*2.0 + 1.0;",
    "  vec4 sh = -step(h, vec4(0.0));",
    "",
    "  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;",
    "  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;",
    "",
    "  vec3 p0 = vec3(a0.xy,h.x);",
    "  vec3 p1 = vec3(a0.zw,h.y);",
    "  vec3 p2 = vec3(a1.xy,h.z);",
    "  vec3 p3 = vec3(a1.zw,h.w);",
    "",
    "//Normalise gradients",
    "  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));",
    "  p0 *= norm.x;",
    "  p1 *= norm.y;",
    "  p2 *= norm.z;",
    "  p3 *= norm.w;",
    "",
    "// Mix final noise value",
    "  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);",
    "  m = m * m;",
    "  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),",
    "                                dot(p2,x2), dot(p3,x3) ) );",
    "}",
    ""
  ].join("\n");

  /* ═══════════════════════════════════════════════════════════════════════
     2. GLSL — fBm (lacunarity 2.0, gain 0.5, amplitude-normalised)
     ═══════════════════════════════════════════════════════════════════════ */

  // Each octave is shifted by a fixed irrational-looking vector so the octaves
  // do not all share the same lattice phase at the origin (which would show up
  // as a static "hot spot" in the middle of the field).
  var FBM_SHIFT_X = 17.31;
  var FBM_SHIFT_Y = 9.73;
  var FBM_SHIFT_Z = 23.17;
  var FBM_MAX_OCTAVES = 8;

  var GLSL_FBM3 = [
    "// Fractal Brownian motion over snoise().",
    "// lacunarity 2.0, gain 0.5, divided by the sum of the amplitudes so the",
    "// result stays inside roughly [-1, 1] for any octave count.",
    "// Requires: snoise(vec3)",
    "float fbm(vec3 p, int octaves) {",
    "  float sum  = 0.0;",
    "  float amp  = 0.5;",
    "  float norm = 0.0;",
    "  vec3  q    = p;",
    "  for (int i = 0; i < 8; i++) {",
    "    if (i >= octaves) break;",
    "    sum  += amp * snoise(q);",
    "    norm += amp;",
    "    amp  *= 0.5;                                       // gain",
    "    q     = q * 2.0 + vec3(17.31, 9.73, 23.17);        // lacunarity + shift",
    "  }",
    "  return norm > 0.0 ? sum / norm : 0.0;",
    "}",
    ""
  ].join("\n");

  /* ═══════════════════════════════════════════════════════════════════════
     3. GLSL — curl noise (divergence-free by construction)
     ═══════════════════════════════════════════════════════════════════════ */

  // Three fixed offsets, far enough apart that the three potential components
  // are effectively independent scalar fields (simplex noise decorrelates over
  // about one lattice unit; these are ~100 apart in every pair), yet small
  // enough that mediump/highp float32 still resolves the epsilon step.
  var CURL_O1 = [37.13, 11.71, 83.29];
  var CURL_O2 = [-19.43, 67.87, -41.51];
  var CURL_O3 = [91.77, -53.19, 23.63];
  var CURL_EPS = 0.01;

  var GLSL_CURL3 = [
    "// Curl of a vector potential built from three decorrelated snoise fields.",
    "// Because curl(psi) is taken with central differences and div() of a",
    "// central-difference curl is an exactly cancelling sum of commuting",
    "// difference operators, the sampled field is divergence-free to rounding.",
    "// Requires: snoise(vec3), and precision highp float.",
    "//",
    "// REQUIRES highp. The central difference adds 0.01 to coordinates that are",
    "// already offset by up to ~92, so the step must survive the addition. highp",
    "// (float32) resolves it down to about 1e-5 of a unit and stays exact for",
    "// |p| up to ~1e4; mediump (float16 on mobile, ~1e-3 relative) would swallow",
    "// the step entirely and return a zero or garbage field.",
    "const float HOTARU_CURL_EPS = 0.01;",
    "",
    "vec3 hotaruPotential(vec3 p) {",
    "  return vec3(",
    "    snoise(p + vec3( 37.13,  11.71,  83.29)),",
    "    snoise(p + vec3(-19.43,  67.87, -41.51)),",
    "    snoise(p + vec3( 91.77, -53.19,  23.63))",
    "  );",
    "}",
    "",
    "vec3 curlNoise(vec3 p) {",
    "  float e = HOTARU_CURL_EPS;",
    "  vec3 dx = vec3(e, 0.0, 0.0);",
    "  vec3 dy = vec3(0.0, e, 0.0);",
    "  vec3 dz = vec3(0.0, 0.0, e);",
    "",
    "  vec3 xm = hotaruPotential(p - dx);",
    "  vec3 xp = hotaruPotential(p + dx);",
    "  vec3 ym = hotaruPotential(p - dy);",
    "  vec3 yp = hotaruPotential(p + dy);",
    "  vec3 zm = hotaruPotential(p - dz);",
    "  vec3 zp = hotaruPotential(p + dz);",
    "",
    "  //  curl = ( dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy )",
    "  float cx = (yp.z - ym.z) - (zp.y - zm.y);",
    "  float cy = (zp.x - zm.x) - (xp.z - xm.z);",
    "  float cz = (xp.y - xm.y) - (yp.x - ym.x);",
    "",
    "  return vec3(cx, cy, cz) / (2.0 * e);",
    "}",
    ""
  ].join("\n");

  var PRELUDE = GLSL_SIMPLEX3 + "\n" + GLSL_FBM3 + "\n" + GLSL_CURL3;

  /* ═══════════════════════════════════════════════════════════════════════
     4. JavaScript mirror
     ═══════════════════════════════════════════════════════════════════════ */

  var floor = Math.floor;
  var abs = Math.abs;
  var min = Math.min;
  var max = Math.max;
  var f32 = Math.fround;

  // const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  var C = [1.0 / 6.0, 1.0 / 3.0];
  // const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  var D = [0.0, 0.5, 1.0, 2.0];

  /* Two of the shader's literals sit exactly on a floor() boundary, so the
     branch they take depends on how the literal was rounded. A GLSL compiler
     stores them as float32, and BOTH round UP:

       float32(1.0/289.0)      = 0.0034602077212184668  >  1/289
       float32(0.142857142857) = 0.1428571492433548     >  1/7

     which is precisely what makes floor(j * ns.z) land on 1 (not 0) when
     j == 7, and floor(p * ns.z * ns.z) land on 1 (not 0) when p == 49.
     Evaluating the same literals in float64 rounds them DOWN and produces
     yu == 7 — an entry off the end of the 7x7 gradient table, with a
     gradient roughly twice the intended length. So the mirror adopts the
     float32 value of these two literals, exactly as the GPU does.
     Every other constant is representable identically in both formats. */
  var INV_289 = f32(1.0 / 289.0);
  var ONE_SEVENTH = f32(0.142857142857);

  // vec3/vec4 mod289(x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  function mod289(x) {
    return x - floor(x * INV_289) * 289.0;
  }

  // vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  function permute(x) {
    return mod289(((x * 34.0) + 1.0) * x);
  }

  // vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  function taylorInvSqrt(r) {
    return 1.79284291400159 - 0.85373472095314 * r;
  }

  /**
   * float snoise(vec3 v) — scalar expansion of the Ashima shader above.
   * Returns a value in roughly [-1, 1].
   */
  function snoise3(vx, vy, vz) {
    // vec3 i  = floor(v + dot(v, C.yyy));
    var dvc = (vx + vy + vz) * C[1];
    var ix = floor(vx + dvc);
    var iy = floor(vy + dvc);
    var iz = floor(vz + dvc);
    // vec3 x0 = v - i + dot(i, C.xxx);
    var dic = (ix + iy + iz) * C[0];
    var x0x = vx - ix + dic;
    var x0y = vy - iy + dic;
    var x0z = vz - iz + dic;

    // vec3 g = step(x0.yzx, x0.xyz);  vec3 l = 1.0 - g;
    var gx = x0x >= x0y ? 1.0 : 0.0;
    var gy = x0y >= x0z ? 1.0 : 0.0;
    var gz = x0z >= x0x ? 1.0 : 0.0;
    var lx = 1.0 - gx;
    var ly = 1.0 - gy;
    var lz = 1.0 - gz;

    // vec3 i1 = min(g.xyz, l.zxy);  vec3 i2 = max(g.xyz, l.zxy);
    var i1x = min(gx, lz), i1y = min(gy, lx), i1z = min(gz, ly);
    var i2x = max(gx, lz), i2y = max(gy, lx), i2z = max(gz, ly);

    // vec3 x1 = x0 - i1 + C.xxx;
    var x1x = x0x - i1x + C[0];
    var x1y = x0y - i1y + C[0];
    var x1z = x0z - i1z + C[0];
    // vec3 x2 = x0 - i2 + C.yyy;
    var x2x = x0x - i2x + C[1];
    var x2y = x0y - i2y + C[1];
    var x2z = x0z - i2z + C[1];
    // vec3 x3 = x0 - D.yyy;
    var x3x = x0x - D[1];
    var x3y = x0y - D[1];
    var x3z = x0z - D[1];

    // i = mod289(i);
    ix = mod289(ix);
    iy = mod289(iy);
    iz = mod289(iz);

    // vec4 p = permute(permute(permute(
    //            i.z + vec4(0.0, i1.z, i2.z, 1.0))
    //          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
    //          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    var q0 = permute(iz + 0.0);
    var q1 = permute(iz + i1z);
    var q2 = permute(iz + i2z);
    var q3 = permute(iz + 1.0);
    q0 = permute(q0 + iy + 0.0);
    q1 = permute(q1 + iy + i1y);
    q2 = permute(q2 + iy + i2y);
    q3 = permute(q3 + iy + 1.0);
    q0 = permute(q0 + ix + 0.0);
    q1 = permute(q1 + ix + i1x);
    q2 = permute(q2 + ix + i2x);
    q3 = permute(q3 + ix + 1.0);

    // float n_ = 0.142857142857;  vec3 ns = n_ * D.wyz - D.xzx;
    var n_ = ONE_SEVENTH;
    var nsx = n_ * D[3] - D[0];
    var nsy = n_ * D[1] - D[2];
    var nsz = n_ * D[2] - D[0];

    // vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    var j0 = q0 - 49.0 * floor(q0 * nsz * nsz);
    var j1 = q1 - 49.0 * floor(q1 * nsz * nsz);
    var j2 = q2 - 49.0 * floor(q2 * nsz * nsz);
    var j3 = q3 - 49.0 * floor(q3 * nsz * nsz);

    // vec4 x_ = floor(j * ns.z);  vec4 y_ = floor(j - 7.0 * x_);
    var xu0 = floor(j0 * nsz), xu1 = floor(j1 * nsz);
    var xu2 = floor(j2 * nsz), xu3 = floor(j3 * nsz);
    var yu0 = floor(j0 - 7.0 * xu0), yu1 = floor(j1 - 7.0 * xu1);
    var yu2 = floor(j2 - 7.0 * xu2), yu3 = floor(j3 - 7.0 * xu3);

    // vec4 x = x_ * ns.x + ns.yyyy;  vec4 y = y_ * ns.x + ns.yyyy;
    var X0 = xu0 * nsx + nsy, X1 = xu1 * nsx + nsy;
    var X2 = xu2 * nsx + nsy, X3 = xu3 * nsx + nsy;
    var Y0 = yu0 * nsx + nsy, Y1 = yu1 * nsx + nsy;
    var Y2 = yu2 * nsx + nsy, Y3 = yu3 * nsx + nsy;

    // vec4 h = 1.0 - abs(x) - abs(y);
    var H0 = 1.0 - abs(X0) - abs(Y0);
    var H1 = 1.0 - abs(X1) - abs(Y1);
    var H2 = 1.0 - abs(X2) - abs(Y2);
    var H3 = 1.0 - abs(X3) - abs(Y3);

    // vec4 b0 = vec4(x.xy, y.xy);  vec4 b1 = vec4(x.zw, y.zw);
    // vec4 s0 = floor(b0)*2.0 + 1.0;  vec4 s1 = floor(b1)*2.0 + 1.0;
    var s0x = floor(X0) * 2.0 + 1.0;
    var s0y = floor(X1) * 2.0 + 1.0;
    var s0z = floor(Y0) * 2.0 + 1.0;
    var s0w = floor(Y1) * 2.0 + 1.0;
    var s1x = floor(X2) * 2.0 + 1.0;
    var s1y = floor(X3) * 2.0 + 1.0;
    var s1z = floor(Y2) * 2.0 + 1.0;
    var s1w = floor(Y3) * 2.0 + 1.0;

    // vec4 sh = -step(h, vec4(0.0));
    var sh0 = H0 <= 0.0 ? -1.0 : 0.0;
    var sh1 = H1 <= 0.0 ? -1.0 : 0.0;
    var sh2 = H2 <= 0.0 ? -1.0 : 0.0;
    var sh3 = H3 <= 0.0 ? -1.0 : 0.0;

    // vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    // vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    // vec3 p0 = vec3(a0.xy, h.x); ... vec3 p3 = vec3(a1.zw, h.w);
    var g0x = X0 + s0x * sh0, g0y = Y0 + s0z * sh0, g0z = H0;
    var g1x = X1 + s0y * sh1, g1y = Y1 + s0w * sh1, g1z = H1;
    var g2x = X2 + s1x * sh2, g2y = Y2 + s1z * sh2, g2z = H2;
    var g3x = X3 + s1y * sh3, g3y = Y3 + s1w * sh3, g3z = H3;

    // vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    var nrm0 = taylorInvSqrt(g0x * g0x + g0y * g0y + g0z * g0z);
    var nrm1 = taylorInvSqrt(g1x * g1x + g1y * g1y + g1z * g1z);
    var nrm2 = taylorInvSqrt(g2x * g2x + g2y * g2y + g2z * g2z);
    var nrm3 = taylorInvSqrt(g3x * g3x + g3y * g3y + g3z * g3z);
    g0x *= nrm0; g0y *= nrm0; g0z *= nrm0;
    g1x *= nrm1; g1y *= nrm1; g1z *= nrm1;
    g2x *= nrm2; g2y *= nrm2; g2z *= nrm2;
    g3x *= nrm3; g3y *= nrm3; g3z *= nrm3;

    // vec4 m = max(0.6 - vec4(dot(x0,x0), ...), 0.0);  m = m * m;
    var m0 = max(0.6 - (x0x * x0x + x0y * x0y + x0z * x0z), 0.0);
    var m1 = max(0.6 - (x1x * x1x + x1y * x1y + x1z * x1z), 0.0);
    var m2 = max(0.6 - (x2x * x2x + x2y * x2y + x2z * x2z), 0.0);
    var m3 = max(0.6 - (x3x * x3x + x3y * x3y + x3z * x3z), 0.0);
    m0 = m0 * m0;
    m1 = m1 * m1;
    m2 = m2 * m2;
    m3 = m3 * m3;

    // return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    return 42.0 * (
      m0 * m0 * (g0x * x0x + g0y * x0y + g0z * x0z) +
      m1 * m1 * (g1x * x1x + g1y * x1y + g1z * x1z) +
      m2 * m2 * (g2x * x2x + g2y * x2y + g2z * x2z) +
      m3 * m3 * (g3x * x3x + g3y * x3y + g3z * x3z)
    );
  }

  /**
   * float fbm(vec3 p, int octaves) — mirror of GLSL_FBM3.
   * lacunarity 2.0, gain 0.5, normalised by the amplitude sum.
   */
  function fbm3(px, py, pz, octaves) {
    var n = octaves | 0;
    if (n < 0) n = 0;
    if (n > FBM_MAX_OCTAVES) n = FBM_MAX_OCTAVES;
    var sum = 0.0;
    var amp = 0.5;
    var nrm = 0.0;
    var qx = px, qy = py, qz = pz;
    for (var i = 0; i < n; i++) {
      sum += amp * snoise3(qx, qy, qz);
      nrm += amp;
      amp *= 0.5;
      qx = qx * 2.0 + FBM_SHIFT_X;
      qy = qy * 2.0 + FBM_SHIFT_Y;
      qz = qz * 2.0 + FBM_SHIFT_Z;
    }
    return nrm > 0.0 ? sum / nrm : 0.0;
  }

  /** vec3 hotaruPotential(vec3 p) — mirror; writes into out[0..2]. */
  function potential3(px, py, pz, out) {
    out[0] = snoise3(px + CURL_O1[0], py + CURL_O1[1], pz + CURL_O1[2]);
    out[1] = snoise3(px + CURL_O2[0], py + CURL_O2[1], pz + CURL_O2[2]);
    out[2] = snoise3(px + CURL_O3[0], py + CURL_O3[1], pz + CURL_O3[2]);
    return out;
  }

  // Scratch buffers — allocated once, reused by every curl3() call.
  var _xm = new Float64Array(3), _xp = new Float64Array(3);
  var _ym = new Float64Array(3), _yp = new Float64Array(3);
  var _zm = new Float64Array(3), _zp = new Float64Array(3);
  var _curl = new Float64Array(3);

  /**
   * vec3 curlNoise(vec3 p) — mirror of GLSL_CURL3.
   * @param {Float64Array|Float32Array|Array} [out] optional length-3 target.
   * @returns {Float64Array|Array} the curl vector (a shared scratch buffer if
   *          `out` is omitted — copy it before the next call).
   */
  function curl3(px, py, pz, out) {
    var e = CURL_EPS;
    var xm = potential3(px - e, py, pz, _xm);
    var xp = potential3(px + e, py, pz, _xp);
    var ym = potential3(px, py - e, pz, _ym);
    var yp = potential3(px, py + e, pz, _yp);
    var zm = potential3(px, py, pz - e, _zm);
    var zp = potential3(px, py, pz + e, _zp);

    var inv = 1.0 / (2.0 * e);
    var cx = ((yp[2] - ym[2]) - (zp[1] - zm[1])) * inv;
    var cy = ((zp[0] - zm[0]) - (xp[2] - xm[2])) * inv;
    var cz = ((xp[1] - xm[1]) - (yp[0] - ym[0])) * inv;

    var dst = out || _curl;
    dst[0] = cx;
    dst[1] = cy;
    dst[2] = cz;
    return dst;
  }

  /* ═══════════════════════════════════════════════════════════════════════
     5. Bulk helpers — allocate once, fill in place
     ═══════════════════════════════════════════════════════════════════════ */

  /**
   * Fill `dst[i] = snoise3(pos[3i], pos[3i+1], pos[3i+2]) * freq`-sampled noise.
   * @param {Float32Array} pos  positions, length 3*count
   * @param {Float32Array} dst  output, length count
   */
  function fillSnoise(pos, dst, count, freq) {
    var f = freq === undefined ? 1.0 : freq;
    for (var i = 0, j = 0; i < count; i++, j += 3) {
      dst[i] = snoise3(pos[j] * f, pos[j + 1] * f, pos[j + 2] * f);
    }
    return dst;
  }

  /**
   * Fill `dst[3i..3i+2]` with curlNoise(pos * freq).
   * @param {Float32Array} pos  positions, length 3*count
   * @param {Float32Array} dst  output, length 3*count
   */
  function fillCurl(pos, dst, count, freq) {
    var f = freq === undefined ? 1.0 : freq;
    var tmp = _curl;
    for (var i = 0, j = 0; i < count; i++, j += 3) {
      curl3(pos[j] * f, pos[j + 1] * f, pos[j + 2] * f, tmp);
      dst[j] = tmp[0];
      dst[j + 1] = tmp[1];
      dst[j + 2] = tmp[2];
    }
    return dst;
  }

  /* ═══════════════════════════════════════════════════════════════════════ */

  root.HotaruNoise = {
    GLSL_SIMPLEX3: GLSL_SIMPLEX3,
    GLSL_FBM3: GLSL_FBM3,
    GLSL_CURL3: GLSL_CURL3,
    PRELUDE: PRELUDE,

    snoise3: snoise3,
    fbm3: fbm3,
    curl3: curl3,
    potential3: potential3,

    fillSnoise: fillSnoise,
    fillCurl: fillCurl,

    CURL_EPS: CURL_EPS,
    CURL_OFFSETS: [CURL_O1, CURL_O2, CURL_O3],
    FBM_MAX_OCTAVES: FBM_MAX_OCTAVES,
    FBM_LACUNARITY: 2.0,
    FBM_GAIN: 0.5
  };
})(globalThis);
