/* Hotaru — GPGPU particle field.
   Position and velocity live in floating point textures. Every frame one
   fragment shader integrates all particles at once (multiple render targets,
   ping-ponged), then a single gl.POINTS draw reads them back by gl_VertexID.
   Scene goes to an offscreen buffer, gets a bright-pass and a separable blur,
   and is composited with tone mapping, vignette and grain. */
(function (root) {
  'use strict';

  /* ── tiny mat4 ─────────────────────────────────────────────────────── */

  function m4() { return new Float32Array(16); }

  function perspective(out, fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1;
    out[12] = 0; out[13] = 0; out[14] = 2 * far * near * nf; out[15] = 0;
    return out;
  }

  function lookAt(out, eye, center, up) {
    var zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    var zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
    var xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    var xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
    var yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  function mul(out, a, b) {
    for (var c = 0; c < 4; c++) {
      var b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4]     = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  }

  /* ── deterministic rng ─────────────────────────────────────────────── */

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ── shader sources ────────────────────────────────────────────────── */

  // Used only if hotaru/noise.js was not loaded. Cheap value noise, so the
  // page still moves rather than failing to compile.
  var FALLBACK_NOISE = [
    'vec3 hash33(vec3 p){',
    '  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)));',
    '  return fract(sin(p) * 43758.5453123) * 2.0 - 1.0;',
    '}',
    'float snoise(vec3 p){',
    '  vec3 i = floor(p), f = fract(p);',
    '  vec3 u = f * f * (3.0 - 2.0 * f);',
    '  float n = 0.0;',
    '  for (int dx = 0; dx < 2; dx++) for (int dy = 0; dy < 2; dy++) for (int dz = 0; dz < 2; dz++) {',
    '    vec3 o = vec3(float(dx), float(dy), float(dz));',
    '    float w = mix(1.0 - u.x, u.x, o.x) * mix(1.0 - u.y, u.y, o.y) * mix(1.0 - u.z, u.z, o.z);',
    '    n += w * dot(hash33(i + o), f - o);',
    '  }',
    '  return clamp(n * 2.0, -1.0, 1.0);',
    '}',
    'vec3 curlNoise(vec3 p){',
    '  const float e = 0.12;',
    '  vec3 ax = vec3(137.3, 0.0, 0.0), ay = vec3(0.0, 271.9, 0.0), az = vec3(0.0, 0.0, 419.1);',
    '  float x1 = snoise(p + vec3(0.0, e, 0.0) + az), x2 = snoise(p - vec3(0.0, e, 0.0) + az);',
    '  float x3 = snoise(p + vec3(0.0, 0.0, e) + ay), x4 = snoise(p - vec3(0.0, 0.0, e) + ay);',
    '  float y1 = snoise(p + vec3(0.0, 0.0, e) + ax), y2 = snoise(p - vec3(0.0, 0.0, e) + ax);',
    '  float y3 = snoise(p + vec3(e, 0.0, 0.0) + az), y4 = snoise(p - vec3(e, 0.0, 0.0) + az);',
    '  float z1 = snoise(p + vec3(e, 0.0, 0.0) + ay), z2 = snoise(p - vec3(e, 0.0, 0.0) + ay);',
    '  float z3 = snoise(p + vec3(0.0, e, 0.0) + ax), z4 = snoise(p - vec3(0.0, e, 0.0) + ax);',
    '  return normalize(vec3((x1 - x2) - (x3 - x4), (y1 - y2) - (y3 - y4), (z1 - z2) - (z3 - z4)) / (2.0 * e) + 1e-6);',
    '}'
  ].join('\n');

  function noisePrelude() {
    var N = root.HotaruNoise;
    if (N && typeof N.PRELUDE === 'string' && N.PRELUDE.indexOf('curlNoise') > -1) return N.PRELUDE;
    return FALLBACK_NOISE;
  }

  var VS_FULL =
'#version 300 es\n' +
'precision highp float;\n' +
'out vec2 vUv;\n' +
'void main() {\n' +
'  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n' +
'  vUv = p;\n' +
'  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n' +
'}\n';

  /* ── gravity: shared GLSL ───────────────────────────────────────────
     The 3D grid lives in a 2D atlas of z-slices, so every cell lookup has to
     go through HotaruAtlas's mapping; a plain texture() would blend across a
     slice seam and pull the whole field sideways. */
  function atlasPrelude(atlas) {
    if (!atlas) return '';
    // cellToTexel already wraps the cell index into [0, N), so a neighbour
    // offset that walks off the grid lands on the periodic image by itself.
    return atlas.glsl + '\n' +
'float fetchCell(sampler2D tex, ivec3 c) { return texelFetch(tex, cellToTexel(c), 0).r; }\n';
  }

  function simFS(atlas) {
    return (
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uPos;\n' +
'uniform sampler2D uVel;\n' +
'uniform sampler2D uTgt;\n' +
'uniform float uDt, uTime, uSpring, uDamp, uNoiseAmp, uNoiseScale, uFlowSpeed;\n' +
'uniform float uPointerActive, uPointerRadius, uPointerPush, uPointerSwirl, uAspect;\n' +
'uniform vec2  uPointer;\n' +
'uniform vec3  uCamRight, uCamUp;\n' +
'uniform mat4  uViewProj;\n' +
'uniform float uHome;\n' +
'uniform sampler2D uPhi;\n' +
'uniform float uGravity, uBoxL, uCellH, uForceClamp;\n' +
'uniform float uKickSigma, uKickSpin;\n' +
'layout(location = 0) out vec4 oPos;\n' +
'layout(location = 1) out vec4 oVel;\n' +
noisePrelude() + '\n' +
atlasPrelude(atlas) + '\n' +
'void main() {\n' +
'  ivec2 uv = ivec2(gl_FragCoord.xy);\n' +
'  vec4 P = texelFetch(uPos, uv, 0);\n' +
'  vec4 V = texelFetch(uVel, uv, 0);\n' +
'  vec4 T = texelFetch(uTgt, uv, 0);\n' +
'  vec3 p = P.xyz;\n' +
'  vec3 v = V.xyz;\n' +
'  float seed = P.w;\n' +
'  float jitter = 0.72 + 0.56 * seed;\n' +
'\n' +
'  vec3 acc = (T.xyz - p) * (uSpring * jitter);\n' +
'\n' +
'  vec3 np = p * uNoiseScale + vec3(0.0, uTime * uFlowSpeed, uTime * uFlowSpeed * 0.35);\n' +
'  acc += curlNoise(np) * (uNoiseAmp * jitter);\n' +
'\n' +
'  float r = length(p);\n' +
'  if (r > 2.4) acc -= p * ((r - 2.4) * 2.0);\n' +
'\n' +
'  if (uPointerActive > 0.0) {\n' +
'    vec4 clip = uViewProj * vec4(p, 1.0);\n' +
'    vec2 ndc = clip.xy / max(abs(clip.w), 1e-4);\n' +
'    vec2 d = (ndc - uPointer) * vec2(uAspect, 1.0);\n' +
'    float dist = length(d);\n' +
'    if (dist < uPointerRadius && clip.w > 0.0) {\n' +
'      float f = 1.0 - dist / uPointerRadius;\n' +
'      f = f * f * uPointerActive;\n' +
'      vec2 dir = dist > 1e-5 ? d / dist : vec2(0.7071, 0.7071);\n' +
'      acc += (uCamRight * dir.x + uCamUp * dir.y) * (f * uPointerPush * jitter);\n' +
'      acc += (uCamRight * -dir.y + uCamUp * dir.x) * (f * uPointerSwirl);\n' +
'    }\n' +
'  }\n' +
'\n' +
'  acc += (T.xyz - p) * uHome;\n' +
'\n' +
(atlas
  // gradGrid differentiates with respect to GRID coordinates, so the result is
  // scaled by N/L to become an acceleration per world unit.
? '  if (uGravity > 0.0) {\n' +
  '    vec3 g = worldToGrid(p, uBoxL);\n' +
  '    vec3 a = -gradGrid(uPhi, g, 1.0) * (HA_NF / uBoxL) * uGravity;\n' +
  '    float m = length(a);\n' +
  '    if (m > uForceClamp) a *= uForceClamp / m;\n' +
  '    acc += a;\n' +
  '  }\n'
: '') +
'\n' +
'  if (uKickSigma > 0.0 || uKickSpin != 0.0) {\n' +
'    v += curlNoise(p * 2.7 + vec3(seed * 37.0)) * uKickSigma;\n' +
'    v += cross(vec3(0.0, 1.0, 0.0), p) * uKickSpin;\n' +
'  }\n' +
'\n' +
'  v += acc * uDt;\n' +
'  v *= pow(uDamp, uDt * 60.0);\n' +
'  float sp = length(v);\n' +
'  if (sp > 8.0) v *= 8.0 / sp;\n' +
'  p += v * uDt;\n' +
'\n' +
'  oPos = vec4(p, seed);\n' +
'  oVel = vec4(v, mix(V.w, clamp((sp - 0.22) * 1.25, 0.0, 1.0), 1.0 - pow(0.86, uDt * 60.0)));\n' +
'}\n');
  }

  var VS_POINTS =
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uPos;\n' +
'uniform sampler2D uVel;\n' +
'uniform mat4 uViewProj;\n' +
'uniform ivec2 uTexSize;\n' +
'uniform float uPointScale, uSizeMin, uSizeMax, uBrightness;\n' +
'out vec3 vColor;\n' +
'out float vAlpha;\n' +
'void main() {\n' +
'  int id = gl_VertexID;\n' +
'  ivec2 uv = ivec2(id % uTexSize.x, id / uTexSize.x);\n' +
'  vec4 P = texelFetch(uPos, uv, 0);\n' +
'  vec4 V = texelFetch(uVel, uv, 0);\n' +
'  vec4 clip = uViewProj * vec4(P.xyz, 1.0);\n' +
'  gl_Position = clip;\n' +
'  float w = max(clip.w, 0.15);\n' +
'  float near = clamp(1.0 - (w - 2.2) / 3.4, 0.0, 1.0);\n' +
'  gl_PointSize = clamp(uPointScale * (0.62 + 0.76 * P.w) / w, uSizeMin, uSizeMax);\n' +
'\n' +
'  vec3 deep = vec3(0.96, 0.34, 0.05);\n' +
'  vec3 warm = vec3(1.00, 0.64, 0.19);\n' +
'  vec3 hot  = vec3(1.00, 0.93, 0.74);\n' +
'  vec3 cool = vec3(0.24, 0.58, 0.82);\n' +
'  float heat = clamp(V.w, 0.0, 1.0);\n' +
'  vec3 c = mix(deep, warm, smoothstep(0.0, 0.45, near + P.w * 0.22));\n' +
'  c = mix(c, cool, (1.0 - near) * 0.38 * (0.35 + 0.65 * P.w));\n' +
'  c = mix(c, hot, heat * heat * 0.58);\n' +
'  float twinkle = 0.78 + 0.22 * sin(P.w * 43.7 + float(id % 97) * 0.13);\n' +
'  vColor = c * uBrightness * twinkle;\n' +
'  vAlpha = (0.30 + 0.70 * near) * (0.55 + 0.45 * P.w);\n' +
'}\n';

  var FS_POINTS =
'#version 300 es\n' +
'precision highp float;\n' +
'in vec3 vColor;\n' +
'in float vAlpha;\n' +
'out vec4 frag;\n' +
'void main() {\n' +
'  vec2 d = gl_PointCoord - 0.5;\n' +
'  float r2 = dot(d, d);\n' +
'  if (r2 > 0.25) discard;\n' +
'  float core = exp(-r2 * 26.0);\n' +
'  float halo = exp(-r2 * 5.2) * 0.22;\n' +
'  float a = (core + halo) * vAlpha;\n' +
'  frag = vec4(vColor * a, a);\n' +
'}\n';

  /* Mass deposition: every particle is drawn as a single point into the cell it
     occupies, with additive blending, so the atlas ends up holding a particle
     count per cell. Nearest-grid-point rather than cloud-in-cell — a point
     sprite can only write one texel, and the grid already smooths the field. */
  function depositVS(atlas) {
    return (
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uPos;\n' +
'uniform ivec2 uTexSize;\n' +
'uniform float uBoxL;\n' +
atlas.glsl + '\n' +
'void main() {\n' +
'  int id = gl_VertexID;\n' +
'  ivec2 uv = ivec2(id % uTexSize.x, id / uTexSize.x);\n' +
'  vec3 p = texelFetch(uPos, uv, 0).xyz;\n' +
'  vec3 g = worldToGrid(p, uBoxL);\n' +
'  ivec2 t = cellToTexel(ivec3(floor(g + 0.5)));\n' +
'  vec2 ndc = (vec2(t) + 0.5) / vec2(float(HA_W), float(HA_H)) * 2.0 - 1.0;\n' +
'  gl_Position = vec4(ndc, 0.0, 1.0);\n' +
'  gl_PointSize = 1.0;\n' +
'}\n');
  }

  var FS_DEPOSIT =
'#version 300 es\n' +
'precision highp float;\n' +
'out vec4 frag;\n' +
'void main() { frag = vec4(1.0, 0.0, 0.0, 0.0); }\n';

  /* One 7-point Poisson sweep, combined with the two previous iterates by the
     Chebyshev recurrence. The sweep is identical to plain Jacobi; only what is
     written differs. A CPU reference (hotaru/pm.js, test/pm.test.js) measured
     the cost of both: to hold the particle force error under 3% at N=64,
     warm-started, plain Jacobi needs ~83 passes and this needs ~10. The mean
     density is subtracted because a periodic box has no solution otherwise. */
  function relaxFS(atlas) {
    return ['#version 300 es',
      'precision highp float;',
      'uniform sampler2D uPhi;',
      'uniform sampler2D uPhiPrev;',
      'uniform sampler2D uRho;',
      'uniform float uPoissonK, uMeanCount, uC1, uC2, uAlpha, uBeta, uFirst;',
      atlas.glsl,
      'float fetchCell(sampler2D tex, ivec3 c) { return texelFetch(tex, cellToTexel(c), 0).r; }',
      'out vec4 frag;',
      'void main() {',
      '  ivec2 t = ivec2(gl_FragCoord.xy);',
      '  ivec3 c = ivec3(texelToCell(t));',
      '  float s = fetchCell(uPhi, c + ivec3(1, 0, 0)) + fetchCell(uPhi, c - ivec3(1, 0, 0))',
      '          + fetchCell(uPhi, c + ivec3(0, 1, 0)) + fetchCell(uPhi, c - ivec3(0, 1, 0))',
      '          + fetchCell(uPhi, c + ivec3(0, 0, 1)) + fetchCell(uPhi, c - ivec3(0, 0, 1));',
      '  float n = fetchCell(uRho, c) - uMeanCount;',
      '  float y = (s - uPoissonK * n) / 6.0;',
      '  float xk = fetchCell(uPhi, c);',
      '  float comb = uC1 * y - uC2 * xk;',
      '  float next = uFirst > 0.5',
      '    ? comb * uAlpha',
      '    : uAlpha * comb - uBeta * fetchCell(uPhiPrev, c);',
      '  frag = vec4(next, 0.0, 0.0, 1.0);',
      '}'].join(String.fromCharCode(10)) + String.fromCharCode(10);
  }
  var FS_BRIGHT =
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uSrc;\n' +
'uniform float uThreshold, uIntensity;\n' +
'in vec2 vUv;\n' +
'out vec4 frag;\n' +
'void main() {\n' +
'  vec3 c = texture(uSrc, vUv).rgb;\n' +
'  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));\n' +
'  float k = max(l - uThreshold, 0.0) / max(l, 1e-4);\n' +
'  frag = vec4(c * k * uIntensity, 1.0);\n' +
'}\n';

  var FS_BLUR =
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uSrc;\n' +
'uniform vec2 uStep;\n' +
'in vec2 vUv;\n' +
'out vec4 frag;\n' +
'const float W[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);\n' +
'void main() {\n' +
'  vec3 c = texture(uSrc, vUv).rgb * W[0];\n' +
'  for (int i = 1; i < 5; i++) {\n' +
'    vec2 o = uStep * float(i);\n' +
'    c += texture(uSrc, vUv + o).rgb * W[i];\n' +
'    c += texture(uSrc, vUv - o).rgb * W[i];\n' +
'  }\n' +
'  frag = vec4(c, 1.0);\n' +
'}\n';

  /* Auto-exposure. The scene's mipmap chain reduces it to one texel, which is
     its average brightness; that value is smoothed into a 1x1 buffer over time
     and used as a gain at composite. A simulation whose density changes by two
     orders of magnitude cannot be exposed by hand. */
  var FS_LUM =
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uScene;\n' +
'uniform sampler2D uPrev;\n' +
'uniform float uRate, uLod;\n' +
'out vec4 frag;\n' +
'void main() {\n' +
'  vec3 c = textureLod(uScene, vec2(0.5), uLod).rgb;\n' +
'  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));\n' +
'  float p = texelFetch(uPrev, ivec2(0, 0), 0).r;\n' +
'  frag = vec4(p <= 0.0 ? l : mix(p, l, uRate), 0.0, 0.0, 1.0);\n' +
'}\n';

  var FS_COMPOSITE =
'#version 300 es\n' +
'precision highp float;\n' +
'uniform sampler2D uScene;\n' +
'uniform sampler2D uBloom;\n' +
'uniform sampler2D uLum;\n' +
'uniform float uAuto, uAutoTarget, uAutoMin, uAutoMax;\n' +
'uniform float uBloomAmount, uExposure, uTime, uGrain;\n' +
'in vec2 vUv;\n' +
'out vec4 frag;\n' +
'vec3 aces(vec3 x) {\n' +
'  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);\n' +
'}\n' +
'void main() {\n' +
'  vec3 scene = texture(uScene, vUv).rgb;\n' +
'  vec3 bloom = texture(uBloom, vUv).rgb;\n' +
'  vec3 c = scene + bloom * uBloomAmount;\n' +
'  float l = max(texelFetch(uLum, ivec2(0, 0), 0).r, 1e-5);\n' +
'  float gain = mix(1.0, clamp(uAutoTarget / l, uAutoMin, uAutoMax), uAuto);\n' +
'  c = aces(c * uExposure * gain);\n' +
'  vec2 q = vUv - 0.5;\n' +
'  c += vec3(0.020, 0.028, 0.052) * (1.0 - smoothstep(0.0, 0.85, length(q)));\n' +
'  c *= 1.0 - 0.55 * dot(q, q);\n' +
'  float g = fract(sin(dot(gl_FragCoord.xy + uTime * 60.0, vec2(12.9898, 78.233))) * 43758.5453);\n' +
'  c += (g - 0.5) * uGrain;\n' +
'  frag = vec4(max(c, 0.0), 1.0);\n' +
'}\n';

  /* ── gl helpers ────────────────────────────────────────────────────── */

  function compile(gl, type, src, label) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      var log = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(label + ' shader failed: ' + log);
    }
    return s;
  }

  function program(gl, vsSrc, fsSrc, label) {
    var vs = compile(gl, gl.VERTEX_SHADER, vsSrc, label + ' vertex');
    var fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc, label + ' fragment');
    var p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.linkProgram(p);
    gl.deleteShader(vs); gl.deleteShader(fs);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      var log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(label + ' link failed: ' + log);
    }
    var u = {};
    var n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (var i = 0; i < n; i++) {
      var info = gl.getActiveUniform(p, i);
      u[info.name.replace('[0]', '')] = gl.getUniformLocation(p, info.name);
    }
    return { p: p, u: u };
  }

  function makeTex(gl, w, h, internal, format, type, data) {
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data || null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return t;
  }

  function linear(gl, t) {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return t;
  }

  function fbo(gl, attachments) {
    var f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    var bufs = [];
    for (var i = 0; i < attachments.length; i++) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, attachments[i], 0);
      bufs.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(bufs);
    var ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (!ok) throw new Error('framebuffer incomplete');
    return f;
  }

  /* ── engine ────────────────────────────────────────────────────────── */

  function Engine(canvas, opts) {
    opts = opts || {};
    this.canvas = canvas;
    this.onError = opts.onError || function () {};

    var attrs = { alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: !!opts.preserveBuffer,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false };
    var gl = canvas.getContext('webgl2', attrs);
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;

    this.floatRender = !!gl.getExtension('EXT_color_buffer_float');
    this.halfRender = this.floatRender || !!gl.getExtension('EXT_color_buffer_half_float');
    if (!this.floatRender && !this.halfRender) throw new Error('This GPU cannot render to floating point textures.');
    gl.getExtension('OES_texture_float_linear');

    this.simFmt = this.floatRender
      ? { internal: gl.RGBA32F, type: gl.FLOAT, Arr: Float32Array }
      : { internal: gl.RGBA16F, type: gl.HALF_FLOAT, Arr: Uint16Array };
    this.sceneFmt = this.floatRender ? gl.RGBA16F : gl.RGBA16F;

    var psr = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    this.pointMax = Math.max(4, Math.min(psr ? psr[1] : 8, 14));
    this.vertexUnits = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);
    if (this.vertexUnits < 2) throw new Error('This GPU cannot read textures in the vertex stage.');

    /* The gravity grid has to be chosen before the shaders are built, because
       its dimensions are compiled into them. */
    this.atlas = null;
    this.gravityError = null;
    this.floatBlend = !!gl.getExtension('EXT_float_blend');
    if (root.HotaruAtlas && opts.gravity !== false) {
      try {
        var lay = root.HotaruAtlas.layout(opts.gridN || ((opts.count || 262144) >= 262144 ? 64 : 32));
        this.atlas = { N: lay.N, w: lay.width, h: lay.height, glsl: root.HotaruAtlas.GLSL(lay) };
      } catch (e) { this.gravityError = 'grid layout: ' + e.message; }
    } else if (!root.HotaruAtlas) {
      this.gravityError = 'atlas module not loaded';
    }

    this.progSim = program(gl, VS_FULL, simFS(this.atlas), 'simulate');
    this.progPoints = program(gl, VS_POINTS, FS_POINTS, 'points');
    this.progBright = program(gl, VS_FULL, FS_BRIGHT, 'bright');
    this.progBlur = program(gl, VS_FULL, FS_BLUR, 'blur');
    this.progComp = program(gl, VS_FULL, FS_COMPOSITE, 'composite');
    this.progLum = program(gl, VS_FULL, FS_LUM, 'luminance');

    if (this.atlas) {
      try {
        this.progDeposit = program(gl, depositVS(this.atlas), FS_DEPOSIT, 'deposit');
        this.progRelax = program(gl, VS_FULL, relaxFS(this.atlas), 'relax');
        this._makeGrid();
      } catch (e) { this.atlas = null; this.gravityError = e.message; }
    }

    this.vao = gl.createVertexArray();

    this.view = m4(); this.proj = m4(); this.viewProj = m4();
    this.camRight = new Float32Array(3);
    this.camUp = new Float32Array(3);

    this.time = 0;
    this.spin = 0;
    this.pointer = { x: 0, y: 0, active: 0, target: 0 };
    this.params = {
      spring: 2.6, damp: 0.90, noiseAmp: 0.55, noiseScale: 1.35, flowSpeed: 0.09,
      pointerRadius: 0.42, pointerPush: 26.0, pointerSwirl: 16.0,
      // pointScale is "pixels at view-space depth 1" — the shader divides by w,
      // and the engine scales it by particle count so total coverage stays put.
      brightness: 0.15, bloomAmount: 0.55, bloomThreshold: 0.45, exposure: 1.25,
      grain: 0.018, pointScale: 8.5, home: 0.0, spinSpeed: 0.055, dist: 3.35, tilt: 0.17,
      auto: 1, autoTarget: 0.055, autoMin: 0.10, autoMax: 4.0, autoRate: 0.035,
      // Self-gravity. GM is G times the total mass of the cloud, so the physics
      // does not change when the particle count drops to a lower tier.
      gravity: 0, GM: 1.6, boxL: 5.0, relax: 24, forceClamp: 24.0,
      gravDamp: 0.9995
    };
    this.formPulse = 0;
    this._kick = null;

    this.count = 0;
    this.tw = 0; this.th = 0;
    this.buffers = null;
    this.targets = null;

    this.width = 0; this.height = 0;
    this.dpr = 1;
    this.sceneTex = null;
    this.frames = 0;
    this.fps = 60;
    this._fpsAcc = 0; this._fpsN = 0;
    this.lost = false;

    var self = this;
    this._onLost = function (e) { e.preventDefault(); self.lost = true; };
    this._onRestored = function () { self.lost = false; self.onError(new Error('graphics context was restored — reload to continue')); };
    canvas.addEventListener('webglcontextlost', this._onLost, false);
    canvas.addEventListener('webglcontextrestored', this._onRestored, false);

    this.setCount(opts.count || 262144);
    this.resize();
  }

  Engine.prototype._makeGrid = function () {
    var gl = this.gl, a = this.atlas;
    // Mass is accumulated with additive blending. Blending into a 32-bit float
    // target needs EXT_float_blend; without it a 16-bit target still blends,
    // but stops accumulating once a cell holds more than ~2048 particles.
    var massInternal = this.floatBlend ? gl.R32F : gl.R16F;
    var massType = this.floatBlend ? gl.FLOAT : gl.HALF_FLOAT;
    this.rhoTex = makeTex(gl, a.w, a.h, massInternal, gl.RED, massType, null);
    this.rhoFB = fbo(gl, [this.rhoTex]);
    // Chebyshev needs x_{k-1}, x_k and a target, and WebGL cannot read and
    // write one texture, so three buffers rotate instead of two.
    this.phiTex = [];
    this.phiFB = [];
    for (var pi = 0; pi < 3; pi++) {
      this.phiTex.push(makeTex(gl, a.w, a.h, gl.R32F, gl.RED, gl.FLOAT, null));
      this.phiFB.push(fbo(gl, [this.phiTex[pi]]));
    }
    this.phiCur = 0;
    // Jacobi spectral radius on the non-constant subspace of the 7-point
    // periodic stencil: max over modes of (cos+cos+cos)/3.
    var cb = { a: -1, b: (2 + Math.cos(2 * Math.PI / a.N)) / 3 };
    var eta = (2 - cb.a - cb.b) / (cb.b - cb.a);
    this.cheb = { eta: eta, c1: 4 / (cb.b - cb.a), c2: 2 * (cb.a + cb.b) / (cb.b - cb.a) };
    this.clearPotential();
  };

  Engine.prototype.clearPotential = function () {
    var gl = this.gl;
    if (!this.atlas) return;
    for (var i = 0; i < 3; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.phiFB[i]);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  /* Deposit mass, then relax the potential. The relaxation is warm-started from
     the previous frame, which is what makes a handful of Jacobi sweeps enough:
     the field moves only a little between frames. */
  Engine.prototype._gravityStep = function () {
    var gl = this.gl, a = this.atlas, P = this.params;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rhoFB);
    gl.viewport(0, 0, a.w, a.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    var D = this.progDeposit;
    gl.useProgram(D.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.pos[this.cur]);
    gl.uniform1i(D.u.uPos, 0);
    gl.uniform2i(D.u.uTexSize, this.tw, this.th);
    gl.uniform1f(D.u.uBoxL, P.boxL);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.disable(gl.BLEND);

    var h = P.boxL / a.N;
    var R = this.progRelax, C = this.cheb;
    gl.useProgram(R.p);
    // phi uses cell particle counts, so G*m folds into one constant:
    // 4*pi*G*(M/count)/h, which keeps the physics fixed across particle tiers.
    gl.uniform1f(R.u.uPoissonK, 4 * Math.PI * P.GM / (this.count * h));
    gl.uniform1f(R.u.uMeanCount, this.count / (a.N * a.N * a.N));
    gl.uniform1f(R.u.uC1, C.c1);
    gl.uniform1f(R.u.uC2, C.c2);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.rhoTex);
    gl.uniform1i(R.u.uRho, 2);

    var iters = Math.max(2, P.relax | 0);
    var prev = this.phiCur, cur = this.phiCur, r = 1 / C.eta;
    for (var k = 0; k < iters; k++) {
      // k = 0 has no x_{k-1} yet: it starts from the warm-started field alone.
      var first = k === 0;
      var next = first ? (cur + 1) % 3 : (3 - prev - cur);
      var alpha, beta, rNext;
      if (first) { alpha = 1 / (2 * C.eta); beta = 0; }
      else { rNext = 1 / (2 * C.eta - r); alpha = rNext; beta = r * rNext; r = rNext; }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.phiFB[next]);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.phiTex[cur]);
      gl.uniform1i(R.u.uPhi, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.phiTex[prev]);
      gl.uniform1i(R.u.uPhiPrev, 1);
      gl.uniform1f(R.u.uFirst, first ? 1 : 0);
      gl.uniform1f(R.u.uAlpha, alpha);
      gl.uniform1f(R.u.uBeta, beta);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      prev = cur; cur = next;
    }
    this.phiCur = cur;
  };

  /* Release: gravity on, and one frame of a velocity kick. A little random
     dispersion makes the cloud fragment instead of collapsing to one blob; a
     little rotation is what turns the remnant into a disc. */
  Engine.prototype.setGravity = function (on, kick) {
    if (!this.atlas) return false;
    this.params.gravity = on ? 1 : 0;
    if (on) {
      this.clearPotential();
      this._kick = kick || { sigma: 0.22, spin: 0.30 };
    } else {
      this._kick = null;
    }
    return true;
  };

  Engine.prototype.hasGravity = function () { return !!this.atlas; };

  /* Diagnostic: read a block of the position texture back and report how far
     the cloud extends. Collapse and expansion are indistinguishable by eye once
     the field fills the frame, so the tuning is done against these numbers. */
  Engine.prototype.sampleRadius = function (side) {
    var gl = this.gl;
    var w = Math.min(side || 64, this.tw), h = Math.min(side || 64, this.th);
    var buf = new Float32Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.buffers.fb[this.cur]);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    var n = w * h, sum = 0, sum2 = 0, mx = 0, bad = 0;
    for (var i = 0; i < n; i++) {
      var x = buf[i * 4], y = buf[i * 4 + 1], z = buf[i * 4 + 2];
      if (!isFinite(x) || !isFinite(y) || !isFinite(z)) { bad++; continue; }
      var r = Math.sqrt(x * x + y * y + z * z);
      sum += r; sum2 += r * r;
      if (r > mx) mx = r;
    }
    var k = Math.max(1, n - bad);
    return { mean: sum / k, rms: Math.sqrt(sum2 / k), max: mx, n: k, nonFinite: bad };
  };

  Engine.prototype.setCount = function (n) {
    var gl = this.gl;
    var tw = 1024;
    if (n < 65536) tw = 256;
    var th = Math.max(1, Math.ceil(n / tw));
    var count = tw * th;

    var prevTargets = this.targets;

    if (this.buffers) this._freeBuffers();

    var pos = new Float32Array(count * 4);
    var vel = new Float32Array(count * 4);
    var tgt = new Float32Array(count * 4);
    var rnd = mulberry32(0x1F2E3D);
    for (var i = 0; i < count; i++) {
      var u = rnd() * 2 - 1, phi = rnd() * Math.PI * 2, rr = Math.cbrt(rnd()) * 1.55;
      var s = Math.sqrt(Math.max(0, 1 - u * u));
      pos[i * 4]     = rr * s * Math.cos(phi);
      pos[i * 4 + 1] = rr * u;
      pos[i * 4 + 2] = rr * s * Math.sin(phi);
      pos[i * 4 + 3] = rnd();
      tgt[i * 4]     = pos[i * 4];
      tgt[i * 4 + 1] = pos[i * 4 + 1];
      tgt[i * 4 + 2] = pos[i * 4 + 2];
      tgt[i * 4 + 3] = rnd();
    }

    var F = this.simFmt;
    var mk = function (data) {
      return makeTex(gl, tw, th, F.internal, gl.RGBA, F.type, F.type === gl.FLOAT ? data : toHalf(data));
    };

    this.buffers = {
      pos: [mk(pos), mk(pos)],
      vel: [mk(vel), mk(vel)],
      tgt: mk(tgt),
      fb: []
    };
    this.buffers.fb = [
      fbo(gl, [this.buffers.pos[0], this.buffers.vel[0]]),
      fbo(gl, [this.buffers.pos[1], this.buffers.vel[1]])
    ];
    this.cur = 0;
    this.count = count;
    this.tw = tw; this.th = th;

    if (prevTargets) this.setTargets(prevTargets);
    return count;
  };

  Engine.prototype._freeBuffers = function () {
    var gl = this.gl, b = this.buffers;
    if (!b) return;
    gl.deleteTexture(b.pos[0]); gl.deleteTexture(b.pos[1]);
    gl.deleteTexture(b.vel[0]); gl.deleteTexture(b.vel[1]);
    gl.deleteTexture(b.tgt);
    gl.deleteFramebuffer(b.fb[0]); gl.deleteFramebuffer(b.fb[1]);
    this.buffers = null;
  };

  /* xyz -> the w channel keeps a stable per-particle phase */
  Engine.prototype.setTargets = function (xyz) {
    var gl = this.gl, n = this.count;
    this.targets = xyz;
    var data = new Float32Array(n * 4);
    var src = xyz.length / 3;
    var rnd = mulberry32(0x5A17C3);
    for (var i = 0; i < n; i++) {
      var j = (i % src) * 3;
      var spread = src < n ? 0.012 : 0.0;
      data[i * 4]     = xyz[j]     + (spread ? (rnd() - 0.5) * spread : 0);
      data[i * 4 + 1] = xyz[j + 1] + (spread ? (rnd() - 0.5) * spread : 0);
      data[i * 4 + 2] = xyz[j + 2] + (spread ? (rnd() - 0.5) * spread : 0);
      data[i * 4 + 3] = rnd();
    }
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.tgt);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.tw, this.th, gl.RGBA,
      this.simFmt.type, this.simFmt.type === gl.FLOAT ? data : toHalf(data));
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.formPulse = 1;
  };

  Engine.prototype.resize = function () {
    var gl = this.gl, c = this.canvas;
    var rect = c.getBoundingClientRect();
    var cap = Math.min(window.devicePixelRatio || 1, this.count > 500000 ? 1.75 : 2);
    var w = Math.max(1, Math.round(rect.width * cap));
    var h = Math.max(1, Math.round(rect.height * cap));
    if (w === this.width && h === this.height) return;
    this.width = w; this.height = h; this.dpr = cap;
    c.width = w; c.height = h;

    if (this.scene) {
      gl.deleteTexture(this.sceneTex); gl.deleteFramebuffer(this.scene);
      gl.deleteTexture(this.bloomTexA); gl.deleteFramebuffer(this.bloomA);
      gl.deleteTexture(this.bloomTexB); gl.deleteFramebuffer(this.bloomB);
    }
    var half = gl.HALF_FLOAT;
    this.sceneTex = linear(gl, makeTex(gl, w, h, gl.RGBA16F, gl.RGBA, half, null));
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.sceneLod = Math.ceil(Math.log2(Math.max(w, h)));
    this.scene = fbo(gl, [this.sceneTex]);
    if (!this.lumTex) {
      this.lumTex = [makeTex(gl, 1, 1, gl.R16F, gl.RED, gl.HALF_FLOAT, null),
                     makeTex(gl, 1, 1, gl.R16F, gl.RED, gl.HALF_FLOAT, null)];
      this.lumFB = [fbo(gl, [this.lumTex[0]]), fbo(gl, [this.lumTex[1]])];
      this.lumCur = 0;
      for (var li = 0; li < 2; li++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumFB[li]);
        gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    var bw = Math.max(1, w >> 2), bh = Math.max(1, h >> 2);
    this.bw = bw; this.bh = bh;
    this.bloomTexA = linear(gl, makeTex(gl, bw, bh, gl.RGBA16F, gl.RGBA, half, null));
    this.bloomA = fbo(gl, [this.bloomTexA]);
    this.bloomTexB = linear(gl, makeTex(gl, bw, bh, gl.RGBA16F, gl.RGBA, half, null));
    this.bloomB = fbo(gl, [this.bloomTexB]);
  };

  Engine.prototype.setPointer = function (nx, ny, strength) {
    this.pointer.x = nx; this.pointer.y = ny;
    this.pointer.target = strength;
  };

  Engine.prototype.frame = function (dtMs) {
    if (this.lost) return;
    var gl = this.gl, P = this.params;
    var dt = Math.min(Math.max(dtMs, 1), 48) / 1000;
    this.time += dt;
    this.spin += dt * P.spinSpeed;
    this.frames++;

    this._fpsAcc += dtMs; this._fpsN++;
    if (this._fpsAcc > 500) { this.fps = 1000 / (this._fpsAcc / this._fpsN); this._fpsAcc = 0; this._fpsN = 0; }

    this.pointer.active += (this.pointer.target - this.pointer.active) * Math.min(1, dt * 9);
    this.formPulse *= Math.pow(0.12, dt);

    var d = P.dist, a = this.spin, t = P.tilt;
    var eye = [Math.sin(a) * d * Math.cos(t), Math.sin(t) * d, Math.cos(a) * d * Math.cos(t)];
    lookAt(this.view, eye, [0, 0, 0], [0, 1, 0]);
    perspective(this.proj, 0.85, this.width / this.height, 0.1, 40);
    mul(this.viewProj, this.proj, this.view);
    this.camRight[0] = this.view[0]; this.camRight[1] = this.view[4]; this.camRight[2] = this.view[8];
    this.camUp[0] = this.view[1]; this.camUp[1] = this.view[5]; this.camUp[2] = this.view[9];

    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    /* 0. self-gravity: deposit mass, relax the potential */
    var gravityOn = this.atlas && P.gravity > 0;
    if (gravityOn) this._gravityStep();

    /* 1. integrate */
    var src = this.cur, dst = 1 - this.cur;
    var S = this.progSim;
    gl.useProgram(S.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.buffers.fb[dst]);
    gl.viewport(0, 0, this.tw, this.th);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.buffers.pos[src]); gl.uniform1i(S.u.uPos, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.buffers.vel[src]); gl.uniform1i(S.u.uVel, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.buffers.tgt); gl.uniform1i(S.u.uTgt, 2);
    gl.uniform1f(S.u.uDt, dt);
    gl.uniform1f(S.u.uTime, this.time);
    gl.uniform1f(S.u.uSpring, P.spring * (1 + this.formPulse * 1.5));
    gl.uniform1f(S.u.uDamp, P.damp);
    gl.uniform1f(S.u.uNoiseAmp, P.noiseAmp * (1 + this.formPulse * 2.2));
    gl.uniform1f(S.u.uNoiseScale, P.noiseScale);
    gl.uniform1f(S.u.uFlowSpeed, P.flowSpeed);
    gl.uniform1f(S.u.uHome, P.home);
    gl.uniform1f(S.u.uPointerActive, this.pointer.active);
    gl.uniform1f(S.u.uPointerRadius, P.pointerRadius);
    gl.uniform1f(S.u.uPointerPush, P.pointerPush);
    gl.uniform1f(S.u.uPointerSwirl, P.pointerSwirl);
    gl.uniform1f(S.u.uAspect, this.width / this.height);
    gl.uniform2f(S.u.uPointer, this.pointer.x, this.pointer.y);
    gl.uniform3fv(S.u.uCamRight, this.camRight);
    gl.uniform3fv(S.u.uCamUp, this.camUp);
    gl.uniformMatrix4fv(S.u.uViewProj, false, this.viewProj);
    if (this.atlas) {
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.phiTex[this.phiCur]);
      gl.uniform1i(S.u.uPhi, 3);
      gl.uniform1f(S.u.uGravity, gravityOn ? 1 : 0);
      gl.uniform1f(S.u.uBoxL, P.boxL);
      gl.uniform1f(S.u.uForceClamp, P.forceClamp);
      gl.uniform1f(S.u.uKickSigma, this._kick ? this._kick.sigma : 0);
      gl.uniform1f(S.u.uKickSpin, this._kick ? this._kick.spin : 0);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this._kick = null;
    this.cur = dst;

    /* 2. points -> scene */
    var R = this.progPoints;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.scene);
    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(R.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.buffers.pos[this.cur]); gl.uniform1i(R.u.uPos, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.buffers.vel[this.cur]); gl.uniform1i(R.u.uVel, 1);
    gl.uniformMatrix4fv(R.u.uViewProj, false, this.viewProj);
    gl.uniform2i(R.u.uTexSize, this.tw, this.th);
    // Fewer particles are drawn larger so the field keeps roughly the same
    // screen coverage — and therefore the same exposure — at every tier.
    var sizeFactor = Math.pow(1048576 / Math.max(this.count, 1024), 0.40);
    gl.uniform1f(R.u.uPointScale, P.pointScale * sizeFactor * (this.height / 900));
    gl.uniform1f(R.u.uSizeMin, Math.min(1.0, this.pointMax));
    gl.uniform1f(R.u.uSizeMax, this.pointMax);
    gl.uniform1f(R.u.uBrightness, P.brightness);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.disable(gl.BLEND);

    /* 2b. average brightness -> 1x1, smoothed over time */
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
    var L = this.progLum, lsrc = this.lumCur, ldst = 1 - this.lumCur;
    gl.useProgram(L.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumFB[ldst]);
    gl.viewport(0, 0, 1, 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.sceneTex); gl.uniform1i(L.u.uScene, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.lumTex[lsrc]); gl.uniform1i(L.u.uPrev, 1);
    gl.uniform1f(L.u.uRate, P.autoRate);
    gl.uniform1f(L.u.uLod, this.sceneLod);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.lumCur = ldst;

    /* 3. bloom */
    var B = this.progBright;
    gl.useProgram(B.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA);
    gl.viewport(0, 0, this.bw, this.bh);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.sceneTex); gl.uniform1i(B.u.uSrc, 0);
    gl.uniform1f(B.u.uThreshold, P.bloomThreshold);
    gl.uniform1f(B.u.uIntensity, 1.0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    var BL = this.progBlur;
    gl.useProgram(BL.p);
    for (var pass = 0; pass < 2; pass++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomB);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.bloomTexA); gl.uniform1i(BL.u.uSrc, 0);
      gl.uniform2f(BL.u.uStep, 1.3 / this.bw, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.bloomTexB); gl.uniform1i(BL.u.uSrc, 0);
      gl.uniform2f(BL.u.uStep, 0, 1.3 / this.bh);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    /* 4. composite */
    var C = this.progComp;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(C.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.sceneTex); gl.uniform1i(C.u.uScene, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.bloomTexA); gl.uniform1i(C.u.uBloom, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.lumTex[this.lumCur]); gl.uniform1i(C.u.uLum, 2);
    gl.uniform1f(C.u.uAuto, P.auto);
    gl.uniform1f(C.u.uAutoTarget, P.autoTarget);
    gl.uniform1f(C.u.uAutoMin, P.autoMin);
    gl.uniform1f(C.u.uAutoMax, P.autoMax);
    gl.uniform1f(C.u.uBloomAmount, P.bloomAmount);
    gl.uniform1f(C.u.uExposure, P.exposure);
    gl.uniform1f(C.u.uTime, this.time);
    gl.uniform1f(C.u.uGrain, P.grain);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindVertexArray(null);
  };

  Engine.prototype.dispose = function () {
    this.canvas.removeEventListener('webglcontextlost', this._onLost);
    this.canvas.removeEventListener('webglcontextrestored', this._onRestored);
    this._freeBuffers();
  };

  /* float32 -> float16 for GPUs without EXT_color_buffer_float */
  var f32 = new Float32Array(1), i32 = new Int32Array(f32.buffer);
  function halfOf(v) {
    f32[0] = v;
    var x = i32[0];
    var bits = (x >> 16) & 0x8000;
    var m = (x >> 12) & 0x07FF;
    var e = (x >> 23) & 0xFF;
    if (e < 103) return bits;
    if (e > 142) return bits | 0x7C00;
    if (e < 113) {
      m |= 0x0800;
      return bits | ((m >> (114 - e)) + ((m >> (113 - e)) & 1));
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
  }
  function toHalf(arr) {
    var out = new Uint16Array(arr.length);
    for (var i = 0; i < arr.length; i++) out[i] = halfOf(arr[i]);
    return out;
  }

  root.HotaruEngine = Engine;
})(globalThis);
