/* ─────────────────────────────────────────────────────────────────────────────
   Hotaru · atlas.js — an N^3 grid living inside one 2D texture
   ─────────────────────────────────────────────────────────────────────────────

   The Particle-Mesh gravity solver needs a 3D scalar/vector field: mass density
   goes in, potential comes out, its gradient goes back to the particles. WebGL2
   has 3D textures, but it cannot render to a slice of one without either a
   layered-rendering extension or one draw call per slice, and the Jacobi
   relaxation wants to write the whole volume in a single full-screen pass. So
   the volume is stored the way every WebGL PM solver stores it: as an atlas of
   z-slices tiled across one ordinary 2D texture.

   Every shader in the pipeline — deposit, Jacobi, gradient, gather — addresses
   the volume through the six functions below. If any two of them disagree by
   half a texel the whole field acquires a constant drift and the particle cloud
   slides off in some diagonal direction, which looks exactly like "gravity",
   only wrong. That is why this module exists twice: once as GLSL for the GPU
   and once as JavaScript for the CPU, written statement for statement so that
   test/atlas.test.js can execute both and demand they agree.

   THE TILING
   ----------
   N slices of N x N texels are laid out row-major in a tilesX by tilesY grid,
   slice z occupying tile (z mod tilesX, z div tilesX):

        +--------+--------+--------+--------+     tilesX = 4, tilesY = 4, N = 16
        |  z=12  |  z=13  |  z=14  |  z=15  |     texture 64 x 64
        +--------+--------+--------+--------+
        |  z=8   |  z=9   |  z=10  |  z=11  |     tile (tx,ty) starts at texel
        +--------+--------+--------+--------+     (tx*N, ty*N); v grows upward,
        |  z=4   |  z=5   |  z=6   |  z=7   |     matching the order texImage2D
        +--------+--------+--------+--------+     uploads rows, so row 0 of the
        |  z=0   |  z=1   |  z=2   |  z=3   |     Float32Array is v = 0.
        +--------+--------+--------+--------+

   tilesX and tilesY are both powers of two whose product is exactly N. Nothing
   else was worth considering:

     * exact coverage — tilesX*tilesY == N means width*height == N^3, so every
       texel of the atlas belongs to exactly one cell. The Jacobi pass can run
       over the whole framebuffer with no bounds test and no wasted bandwidth,
       and there is no dead region whose stale contents could leak into a
       trilinear fetch.
     * exact arithmetic — a power-of-two N turns every div and mod in the
       address math into a shift and a mask, which are exact for all int inputs
       and free on the hardware. A "nicer" near-square tiling like 6 x 6 for
       N = 32 would be closer to square, but it wastes four tiles, breaks the
       bijection, and puts an integer division in the innermost loop of the
       relaxation.
     * near-square anyway — tilesX = 2^ceil(k/2), tilesY = 2^floor(k/2) for
       N = 2^k. The atlas is square when k is even and 2:1 when k is odd, which
       is the squarest a power-of-two factorisation of 2^k can be.

           N     tiles      texture      texels
           16    4 x 4      64 x 64        4096
           32    8 x 4     256 x 128      32768
           64    8 x 8     512 x 512     262144
          128   16 x 8    2048 x 1024   2097152

     N = 128 is the ceiling: 2048 is the largest width WebGL2 guarantees
     (MAX_TEXTURE_SIZE >= 2048), and 2 M texels x RGBA32F is already 32 MB per
     buffer, of which the solver needs several.

   COORDINATES
   -----------
   Three systems, and the half-texel lives in the step between the last two:

     world   p in [-L/2, +L/2]^3, the simulation box, origin at its centre.
     grid    continuous g, with g == i exactly at the CENTRE of cell i. So the
             box maps to g in [-0.5, N-0.5], the faces sit half a cell outside
             the outermost cell centres, and sampleGrid at integer g returns
             that cell's value untouched. The alternative convention (g == i at
             the cell's lower corner) reads more naturally but makes deposit and
             gather disagree by half a cell, which is the bug this file is
             written to prevent.
     texel   integer (x, y) into the atlas, plus the [0,1]^2 UV of its centre.

   PERIODICITY
   -----------
   The grid is a torus. That is not a convenience: Jacobi relaxation of the
   discrete Poisson equation needs a boundary condition, and periodic is the
   only one that is both cheap (a mask) and consistent with the FFT solution
   everybody validates against. Every function here wraps, for every input,
   however far outside the box — a particle that escapes does not read garbage,
   it reads the other side.

   WHY sampleGrid CANNOT BE texture()
   ----------------------------------
   Hardware bilinear filtering blends the four texels around a UV. Inside one
   tile that is the right thing for x and y. Between tiles it is nonsense: the
   texel above the top row of slice z is the bottom row of slice z + tilesX,
   not slice z at all, and the texel to the right of the last column of slice z
   is the first column of slice z + 1. So the z blend, and the x/y blends at
   tile borders, have to be done by hand: eight texelFetch calls with the wrap
   applied to cell indices, then a nested mix. That is what sampleGrid does, and
   the test that catches a wrong tiling is the one that samples a field varying
   only in z and demands a straight line.

   PUBLISHED
   ---------
     HotaruAtlas.SIZES                   [16, 32, 64, 128]
     HotaruAtlas.layout(N)               -> {N, tilesX, tilesY, width, height}
     HotaruAtlas.constants(layout)       -> every baked number, by name
     HotaruAtlas.GLSL(layout)            -> GLSL ES 3.00 chunk, constants inlined
     HotaruAtlas.mirror(layout, opts)    -> the same six functions in JS

   The GLSL is a chunk, not a shader: no #version, no precision qualifier, no
   main. Paste it after the host shader's own #version line.

   mirror(layout, {fp32: true}) rounds every float operation through
   Math.fround, reproducing a highp GPU's arithmetic instead of JavaScript's
   doubles. The tests run both modes.
   ───────────────────────────────────────────────────────────────────────── */
(function (root) {
  "use strict";

  var SIZES = [16, 32, 64, 128];

  /* ── layout ──────────────────────────────────────────────────────────── */

  function isPow2(n) {
    return n > 0 && (n & (n - 1)) === 0;
  }

  function log2i(n) {
    var k = 0;
    while ((1 << k) < n) k++;
    return k;
  }

  function layout(N) {
    if (typeof N !== 'number' || !isFinite(N) || Math.floor(N) !== N) {
      throw new TypeError('HotaruAtlas.layout: N must be an integer, got ' + String(N));
    }
    if (!isPow2(N)) {
      throw new RangeError('HotaruAtlas.layout: N must be a power of two, got ' + N);
    }
    if (N < 16 || N > 128) {
      throw new RangeError('HotaruAtlas.layout: N must be one of ' + SIZES.join(', ') + ', got ' + N);
    }
    var k = log2i(N);
    var sx = Math.ceil(k / 2);
    var sy = k - sx;
    var tilesX = 1 << sx;
    var tilesY = 1 << sy;
    return {
      N: N,
      tilesX: tilesX,
      tilesY: tilesY,
      width: tilesX * N,
      height: tilesY * N
    };
  }

  /* A layout is re-derived and checked rather than trusted: a hand-built object
     with a plausible-looking but wrong tilesX would silently desynchronise the
     GLSL from the JS, which is precisely the failure this module prevents. */
  function check(lay) {
    if (!lay || typeof lay !== 'object') {
      throw new TypeError('HotaruAtlas: expected a layout object from HotaruAtlas.layout()');
    }
    var ref = layout(lay.N);
    var keys = ['N', 'tilesX', 'tilesY', 'width', 'height'];
    for (var i = 0; i < keys.length; i++) {
      if (lay[keys[i]] !== ref[keys[i]]) {
        throw new RangeError('HotaruAtlas: layout.' + keys[i] + ' is ' + lay[keys[i]] +
          ', expected ' + ref[keys[i]] + ' for N = ' + ref.N);
      }
    }
    return ref;
  }

  /* Every number the GLSL bakes in, so the tests can compare like for like.
     All of them are exact in binary32: N, the tile counts and the texture
     dimensions are powers of two, and so are all four reciprocals. */
  function constants(lay) {
    var L = check(lay);
    return {
      N: L.N,
      N_MASK: L.N - 1,
      N_SHIFT: log2i(L.N),
      TILES_X: L.tilesX,
      TILES_Y: L.tilesY,
      TILES_X_MASK: L.tilesX - 1,
      TILES_X_SHIFT: log2i(L.tilesX),
      W: L.width,
      H: L.height,
      NF: L.N,
      INV_N: 1 / L.N,
      TILES_XF: L.tilesX,
      INV_TILES_X: 1 / L.tilesX,
      INV_W: 1 / L.width,
      INV_H: 1 / L.height
    };
  }

  /* ── GLSL ────────────────────────────────────────────────────────────── */

  /* Print a float so that parseFloat returns the identical double. Every
     constant here is a power of two or its reciprocal, so the plain decimal
     expansion is exact and short; the fallback keeps the contract for anything
     that ever stops being one. */
  function glslFloat(x) {
    var s = (x === Math.floor(x) && Math.abs(x) < 1e21) ? x.toFixed(1) : String(x);
    if (parseFloat(s) !== x) s = x.toPrecision(17);
    if (s.indexOf('.') < 0 && s.indexOf('e') < 0 && s.indexOf('E') < 0) s += '.0';
    return s;
  }

  function GLSL(lay) {
    var C = constants(lay);
    var d = [];

    d.push('/* ---------------------------------------------------------------------------');
    d.push('   HotaruAtlas -- ' + C.N + '^3 grid as ' + C.TILES_X + ' x ' + C.TILES_Y +
      ' z-slice tiles in a ' + C.W + ' x ' + C.H + ' texture.');
    d.push('   Generated by hotaru/atlas.js; mirrored statement for statement by');
    d.push('   HotaruAtlas.mirror() and checked against it by test/atlas.test.js.');
    d.push('   GLSL ES 3.00 chunk: no #version, no precision qualifier, no main.');
    d.push('   --------------------------------------------------------------------------- */');
    d.push('');
    d.push('const int   HA_N             = ' + C.N + ';');
    d.push('const int   HA_N_MASK        = ' + C.N_MASK + ';');
    d.push('const int   HA_N_SHIFT       = ' + C.N_SHIFT + ';');
    d.push('const int   HA_TILES_X       = ' + C.TILES_X + ';');
    d.push('const int   HA_TILES_Y       = ' + C.TILES_Y + ';');
    d.push('const int   HA_TILES_X_MASK  = ' + C.TILES_X_MASK + ';');
    d.push('const int   HA_TILES_X_SHIFT = ' + C.TILES_X_SHIFT + ';');
    d.push('const int   HA_W             = ' + C.W + ';');
    d.push('const int   HA_H             = ' + C.H + ';');
    d.push('const float HA_NF            = ' + glslFloat(C.NF) + ';');
    d.push('const float HA_INV_N         = ' + glslFloat(C.INV_N) + ';');
    d.push('const float HA_TILES_XF      = ' + glslFloat(C.TILES_XF) + ';');
    d.push('const float HA_INV_TILES_X   = ' + glslFloat(C.INV_TILES_X) + ';');
    d.push('const float HA_INV_W         = ' + glslFloat(C.INV_W) + ';');
    d.push('const float HA_INV_H         = ' + glslFloat(C.INV_H) + ';');
    d.push('');

    /* cellToTexel -------------------------------------------------------- */
    d.push('/* Cell -> the one texel that stores it. Total: the cell index is wrapped');
    d.push('   into [0,N) on every axis first, so a neighbour offset that walks off the');
    d.push('   grid lands on the periodic image instead of another slice. */');
    d.push('ivec2 cellToTexel(ivec3 cell) {');
    d.push('  int cx = cell.x & HA_N_MASK;');
    d.push('  int cy = cell.y & HA_N_MASK;');
    d.push('  int cz = cell.z & HA_N_MASK;');
    d.push('  int tx = cz & HA_TILES_X_MASK;');
    d.push('  int ty = cz >> HA_TILES_X_SHIFT;');
    d.push('  return ivec2(tx * HA_N + cx, ty * HA_N + cy);');
    d.push('}');
    d.push('');

    /* texelToCell -------------------------------------------------------- */
    d.push('/* The inverse, for a pass that iterates over the atlas and has to know which');
    d.push('   cell it is standing on. Returns floats because every consumer feeds the');
    d.push('   result straight into the float side of the maths. */');
    d.push('vec3 texelToCell(ivec2 texel) {');
    d.push('  int cx = texel.x & HA_N_MASK;');
    d.push('  int cy = texel.y & HA_N_MASK;');
    d.push('  int tx = texel.x >> HA_N_SHIFT;');
    d.push('  int ty = texel.y >> HA_N_SHIFT;');
    d.push('  int cz = ty * HA_TILES_X + tx;');
    d.push('  return vec3(float(cx), float(cy), float(cz));');
    d.push('}');
    d.push('');

    /* cellToUV ----------------------------------------------------------- */
    d.push('/* Cell centre in [0,1]^2. cell.z is wrapped and floored to pick the tile;');
    d.push('   cell.x and cell.y may be fractional, in which case the result is the UV of');
    d.push('   that point WITHIN the slice -- only meaningful for a texture() lookup that');
    d.push('   stays inside one tile, which is why the solver uses sampleGrid instead. */');
    d.push('vec2 cellToUV(vec3 cell) {');
    d.push('  vec3 c  = cell - HA_NF * floor(cell * HA_INV_N);');
    d.push('  float z = floor(c.z);');
    d.push('  float ty = floor(z * HA_INV_TILES_X);');
    d.push('  float tx = z - HA_TILES_XF * ty;');
    d.push('  float u = (tx * HA_NF + c.x + 0.5) * HA_INV_W;');
    d.push('  float v = (ty * HA_NF + c.y + 0.5) * HA_INV_H;');
    d.push('  return vec2(u, v);');
    d.push('}');
    d.push('');

    /* worldToGrid / gridToWorld ------------------------------------------ */
    d.push('/* World -> continuous grid coordinates, box of side L centred on the origin.');
    d.push('   p = -L/2 -> -0.5, p = 0 -> N/2 - 0.5, p = +L/2 -> N - 0.5: integer grid');
    d.push('   coordinates sit at cell CENTRES, so the box faces are half a cell outside');
    d.push('   the outermost centres and deposit and gather use one convention. */');
    d.push('vec3 worldToGrid(vec3 p, float L) {');
    d.push('  return (p / L + 0.5) * HA_NF - 0.5;');
    d.push('}');
    d.push('');
    d.push('/* The exact inverse. */');
    d.push('vec3 gridToWorld(vec3 g, float L) {');
    d.push('  return ((g + 0.5) * HA_INV_N - 0.5) * L;');
    d.push('}');
    d.push('');

    /* sampleGrid --------------------------------------------------------- */
    d.push('/* Periodic trilinear sample. Eight texelFetches and a nested mix, because');
    d.push('   hardware filtering would blend across tile borders, where the neighbour in');
    d.push('   the texture is a different slice. gridPos is wrapped into [0,N) as a float');
    d.push('   before flooring, so arbitrarily distant positions cost nothing extra and');
    d.push('   the integer side never leaves [0,N). At integer gridPos every weight is 0');
    d.push('   or 1 and the cell value comes back bit-exact. */');
    d.push('vec4 sampleGrid(sampler2D tex, vec3 gridPos) {');
    d.push('  vec3 gw = gridPos - HA_NF * floor(gridPos * HA_INV_N);');
    d.push('  vec3 b  = floor(gw);');
    d.push('  vec3 f  = gw - b;');
    d.push('  ivec3 i0 = ivec3(b) & HA_N_MASK;');
    d.push('  ivec3 i1 = (i0 + 1) & HA_N_MASK;');
    d.push('  vec4 c000 = texelFetch(tex, cellToTexel(ivec3(i0.x, i0.y, i0.z)), 0);');
    d.push('  vec4 c100 = texelFetch(tex, cellToTexel(ivec3(i1.x, i0.y, i0.z)), 0);');
    d.push('  vec4 c010 = texelFetch(tex, cellToTexel(ivec3(i0.x, i1.y, i0.z)), 0);');
    d.push('  vec4 c110 = texelFetch(tex, cellToTexel(ivec3(i1.x, i1.y, i0.z)), 0);');
    d.push('  vec4 c001 = texelFetch(tex, cellToTexel(ivec3(i0.x, i0.y, i1.z)), 0);');
    d.push('  vec4 c101 = texelFetch(tex, cellToTexel(ivec3(i1.x, i0.y, i1.z)), 0);');
    d.push('  vec4 c011 = texelFetch(tex, cellToTexel(ivec3(i0.x, i1.y, i1.z)), 0);');
    d.push('  vec4 c111 = texelFetch(tex, cellToTexel(ivec3(i1.x, i1.y, i1.z)), 0);');
    d.push('  vec4 x00 = mix(c000, c100, f.x);');
    d.push('  vec4 x10 = mix(c010, c110, f.x);');
    d.push('  vec4 x01 = mix(c001, c101, f.x);');
    d.push('  vec4 x11 = mix(c011, c111, f.x);');
    d.push('  vec4 y0  = mix(x00, x10, f.y);');
    d.push('  vec4 y1  = mix(x01, x11, f.y);');
    d.push('  return mix(y0, y1, f.z);');
    d.push('}');
    d.push('');

    /* gradGrid ----------------------------------------------------------- */
    d.push('/* Central-difference gradient of the .x channel, in GRID units: multiply by');
    d.push('   N/L for a world-space gradient. h is a grid-cell offset, and h = 1.0 is the');
    d.push('   value to use -- with h = 1 the trilinear interpolation error of a quadratic');
    d.push('   field is identical at both stencil points and cancels exactly, so the');
    d.push('   gradient is exact on anything up to second order. Smaller h does not');
    d.push('   improve it; it divides a fixed interpolation error by a smaller number. */');
    d.push('vec3 gradGrid(sampler2D tex, vec3 gridPos, float h) {');
    d.push('  float inv = 0.5 / h;');
    d.push('  vec3 ex = vec3(h, 0.0, 0.0);');
    d.push('  vec3 ey = vec3(0.0, h, 0.0);');
    d.push('  vec3 ez = vec3(0.0, 0.0, h);');
    d.push('  float xp = sampleGrid(tex, gridPos + ex).x;');
    d.push('  float xm = sampleGrid(tex, gridPos - ex).x;');
    d.push('  float yp = sampleGrid(tex, gridPos + ey).x;');
    d.push('  float ym = sampleGrid(tex, gridPos - ey).x;');
    d.push('  float zp = sampleGrid(tex, gridPos + ez).x;');
    d.push('  float zm = sampleGrid(tex, gridPos - ez).x;');
    d.push('  return vec3(xp - xm, yp - ym, zp - zm) * inv;');
    d.push('}');
    d.push('');

    return d.join('\n');
  }

  /* ── JS mirror ───────────────────────────────────────────────────────── */

  function identity(x) { return x; }

  /* Every statement below is the statement above it in GLSL(), in order, with
     the same groupings, so that fp32 rounding lands on the same nodes. R is
     Math.fround in fp32 mode and a no-op otherwise. Int arithmetic is exact in
     both languages (JS bitwise ops are 32-bit signed, as is GLSL highp int), so
     it is never rounded. */
  function mirror(lay, opts) {
    var C = constants(lay);
    var fp32 = !!(opts && opts.fp32);
    var R = fp32 ? Math.fround : identity;

    var N = C.N, N_MASK = C.N_MASK, N_SHIFT = C.N_SHIFT;
    var TILES_X = C.TILES_X, TILES_X_MASK = C.TILES_X_MASK, TILES_X_SHIFT = C.TILES_X_SHIFT;
    var W = C.W, H = C.H;
    var NF = C.NF, INV_N = C.INV_N;
    var TILES_XF = C.TILES_XF, INV_TILES_X = C.INV_TILES_X;
    var INV_W = C.INV_W, INV_H = C.INV_H;

    /* Scratch, allocated once. sampleGrid's outputs are consumed before the
       next call in every path here, so one buffer each is enough. */
    var sTexel = new Int32Array(2);
    var sUV = new Float64Array(2);
    var sCell = new Float64Array(3);
    var sVec = new Float64Array(3);
    var sSample = new Float64Array(4);
    var sGrad = new Float64Array(3);
    var sFetch = new Float64Array(4);
    var i0 = new Int32Array(3);
    var i1 = new Int32Array(3);
    var gw = new Float64Array(3);
    var bb = new Float64Array(3);
    var ff = new Float64Array(3);
    var c = [
      new Float64Array(4), new Float64Array(4), new Float64Array(4), new Float64Array(4),
      new Float64Array(4), new Float64Array(4), new Float64Array(4), new Float64Array(4)
    ];
    var x00 = new Float64Array(4), x10 = new Float64Array(4);
    var x01 = new Float64Array(4), x11 = new Float64Array(4);
    var y0 = new Float64Array(4), y1 = new Float64Array(4);

    /* ---- storage ---- */

    function createAtlas() {
      return new Float32Array(W * H * 4);
    }

    /* Texel (x,y) -> element index. Row 0 is v = 0, matching the order
       texImage2D consumes a Float32Array. */
    function texelIndex(tx, ty) {
      return (ty * W + tx) * 4;
    }

    function texelFetch(tex, texel, out) {
      var tx = texel[0], ty = texel[1];
      if (!(tx >= 0 && tx < W && ty >= 0 && ty < H)) {
        throw new RangeError('texelFetch out of range: (' + tx + ',' + ty + ') in ' + W + 'x' + H);
      }
      var o = out || sFetch;
      var i = texelIndex(tx, ty);
      o[0] = tex[i]; o[1] = tex[i + 1]; o[2] = tex[i + 2]; o[3] = tex[i + 3];
      return o;
    }

    function store(tex, cx, cy, cz, v0, v1, v2, v3) {
      sCell[0] = cx; sCell[1] = cy; sCell[2] = cz;
      var t = cellToTexel(sCell, sTexel);
      var i = texelIndex(t[0], t[1]);
      tex[i] = v0; tex[i + 1] = v1; tex[i + 2] = v2; tex[i + 3] = v3;
    }

    function load(tex, cx, cy, cz, out) {
      sCell[0] = cx; sCell[1] = cy; sCell[2] = cz;
      return texelFetch(tex, cellToTexel(sCell, sTexel), out);
    }

    /* ---- the six ---- */

    function cellToTexel(cell, out) {
      var o = out || sTexel;
      var cx = (cell[0] | 0) & N_MASK;
      var cy = (cell[1] | 0) & N_MASK;
      var cz = (cell[2] | 0) & N_MASK;
      var tx = cz & TILES_X_MASK;
      var ty = cz >> TILES_X_SHIFT;
      o[0] = tx * N + cx;
      o[1] = ty * N + cy;
      return o;
    }

    function texelToCell(texel, out) {
      var o = out || sCell;
      var cx = texel[0] & N_MASK;
      var cy = texel[1] & N_MASK;
      var tx = texel[0] >> N_SHIFT;
      var ty = texel[1] >> N_SHIFT;
      var cz = ty * TILES_X + tx;
      o[0] = cx; o[1] = cy; o[2] = cz;
      return o;
    }

    function cellToUV(cell, out) {
      var o = out || sUV;
      var c0 = R(cell[0] - R(NF * Math.floor(R(cell[0] * INV_N))));
      var c1 = R(cell[1] - R(NF * Math.floor(R(cell[1] * INV_N))));
      var c2 = R(cell[2] - R(NF * Math.floor(R(cell[2] * INV_N))));
      var z = Math.floor(c2);
      var ty = Math.floor(R(z * INV_TILES_X));
      var tx = R(z - R(TILES_XF * ty));
      o[0] = R(R(R(R(tx * NF) + c0) + 0.5) * INV_W);
      o[1] = R(R(R(R(ty * NF) + c1) + 0.5) * INV_H);
      return o;
    }

    function worldToGrid(p, L, out) {
      var o = out || sVec;
      o[0] = R(R(R(R(p[0] / L) + 0.5) * NF) - 0.5);
      o[1] = R(R(R(R(p[1] / L) + 0.5) * NF) - 0.5);
      o[2] = R(R(R(R(p[2] / L) + 0.5) * NF) - 0.5);
      return o;
    }

    function gridToWorld(g, L, out) {
      var o = out || sVec;
      o[0] = R(R(R(R(g[0] + 0.5) * INV_N) - 0.5) * L);
      o[1] = R(R(R(R(g[1] + 0.5) * INV_N) - 0.5) * L);
      o[2] = R(R(R(R(g[2] + 0.5) * INV_N) - 0.5) * L);
      return o;
    }

    /* GLSL mix(x, y, a) is defined as x*(1-a) + y*a. Reproduced literally so
       that fp32 mode rounds where the hardware rounds — modulo an FMA the
       driver may fuse, which no CPU mirror can predict. */
    function mix4(x, y, a, o) {
      var ia = R(1 - a);
      o[0] = R(R(x[0] * ia) + R(y[0] * a));
      o[1] = R(R(x[1] * ia) + R(y[1] * a));
      o[2] = R(R(x[2] * ia) + R(y[2] * a));
      o[3] = R(R(x[3] * ia) + R(y[3] * a));
      return o;
    }

    function sampleGrid(tex, gridPos, out) {
      var o = out || sSample;
      var k;
      for (k = 0; k < 3; k++) {
        gw[k] = R(gridPos[k] - R(NF * Math.floor(R(gridPos[k] * INV_N))));
        bb[k] = Math.floor(gw[k]);
        ff[k] = R(gw[k] - bb[k]);
        i0[k] = (bb[k] | 0) & N_MASK;
        i1[k] = (i0[k] + 1) & N_MASK;
      }
      /* Same eight corners, same order as the GLSL. */
      fetchCell(tex, i0[0], i0[1], i0[2], c[0]);
      fetchCell(tex, i1[0], i0[1], i0[2], c[1]);
      fetchCell(tex, i0[0], i1[1], i0[2], c[2]);
      fetchCell(tex, i1[0], i1[1], i0[2], c[3]);
      fetchCell(tex, i0[0], i0[1], i1[2], c[4]);
      fetchCell(tex, i1[0], i0[1], i1[2], c[5]);
      fetchCell(tex, i0[0], i1[1], i1[2], c[6]);
      fetchCell(tex, i1[0], i1[1], i1[2], c[7]);
      mix4(c[0], c[1], ff[0], x00);
      mix4(c[2], c[3], ff[0], x10);
      mix4(c[4], c[5], ff[0], x01);
      mix4(c[6], c[7], ff[0], x11);
      mix4(x00, x10, ff[1], y0);
      mix4(x01, x11, ff[1], y1);
      return mix4(y0, y1, ff[2], o);
    }

    function fetchCell(tex, cx, cy, cz, out) {
      var tx = (cz & TILES_X_MASK) * N + (cx & N_MASK);
      var ty = (cz >> TILES_X_SHIFT) * N + (cy & N_MASK);
      if (!(tx >= 0 && tx < W && ty >= 0 && ty < H)) {
        throw new RangeError('sampleGrid fetch out of range: (' + tx + ',' + ty + ')');
      }
      var i = (ty * W + tx) * 4;
      out[0] = tex[i]; out[1] = tex[i + 1]; out[2] = tex[i + 2]; out[3] = tex[i + 3];
      return out;
    }

    var gp = new Float64Array(3);

    function gradGrid(tex, gridPos, h, out) {
      var o = out || sGrad;
      var inv = R(0.5 / h);
      var k, d;
      var res = [0, 0, 0];
      for (k = 0; k < 3; k++) {
        for (d = 0; d < 3; d++) gp[d] = gridPos[d];
        gp[k] = R(gridPos[k] + h);
        var plus = sampleGrid(tex, gp, sSample)[0];
        gp[k] = R(gridPos[k] - h);
        var minus = sampleGrid(tex, gp, sSample)[0];
        res[k] = R(plus - minus);
      }
      o[0] = R(res[0] * inv);
      o[1] = R(res[1] * inv);
      o[2] = R(res[2] * inv);
      return o;
    }

    return {
      layout: { N: C.N, tilesX: C.TILES_X, tilesY: C.TILES_Y, width: W, height: H },
      constants: C,
      fp32: fp32,
      createAtlas: createAtlas,
      texelIndex: texelIndex,
      texelFetch: texelFetch,
      store: store,
      load: load,
      cellToUV: cellToUV,
      cellToTexel: cellToTexel,
      texelToCell: texelToCell,
      worldToGrid: worldToGrid,
      gridToWorld: gridToWorld,
      sampleGrid: sampleGrid,
      gradGrid: gradGrid
    };
  }

  root.HotaruAtlas = {
    SIZES: SIZES,
    layout: layout,
    constants: constants,
    GLSL: GLSL,
    mirror: mirror
  };
})(globalThis);
