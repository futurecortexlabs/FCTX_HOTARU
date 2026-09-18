/* test/atlas.test.js — contract tests for hotaru/atlas.js.
 *
 *   node test/atlas.test.js
 *
 * Plain node script: it require()s the library for its global side effect.
 * One line per case, non-zero exit on any failure.
 *
 * WHAT MAKES THIS DIFFERENT FROM A UNIT TEST
 * ------------------------------------------
 * The thing being verified is a GLSL string. Comparing a JS mirror against
 * analytic truth proves the mirror is right and says nothing at all about the
 * shader — and a shader that is half a texel out is exactly the bug that
 * survives every eyeball test, because the field still looks like gravity.
 *
 * So this file carries a small GLSL ES 3.00 interpreter (section 1). It
 * tokenises and parses the generated chunk, then EXECUTES it: typed values
 * (int vs float kinds, vector widths), GLSL's truncating integer division,
 * componentwise bitwise ops, constructors, swizzles, mix() expanded to its
 * specified x*(1-a) + y*a, and texelFetch bound to the same Float32Array the
 * mirror reads. Mixing an int with a float is a hard error, as it is in GLSL,
 * and a texelFetch outside the texture throws instead of returning zero.
 *
 * That makes three independent implementations agreeing:
 *
 *    the GLSL chunk (executed)   vs   the JS mirror   vs   analytic truth
 *
 * and it is run twice, once with JavaScript doubles and once with every float
 * operation forced through Math.fround, which is what a highp GPU does.
 */
'use strict';

const path = require('path');

/* HOTARU_ATLAS overrides the module under test. It exists so the suite can be
   pointed at a deliberately broken copy: a test that never fails proves
   nothing, and every assertion here was checked by mutating atlas.js until it
   did. Unset, it is the real file. */
const SRC = process.env.HOTARU_ATLAS || path.join(__dirname, '..', 'hotaru', 'atlas.js');
require(SRC);
const A = globalThis.HotaruAtlas;

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

/* Deterministic sampler — a failing run reproduces exactly. */
function lcg(seed) {
  let s = seed >>> 0 || 1;
  return function () {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d;
  }
  return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
   1. A GLSL ES 3.00 interpreter, restricted to the subset atlas.js emits:
      const declarations, function definitions, and straight-line bodies of
      `type name = expr;` and `return expr;`.
   ═══════════════════════════════════════════════════════════════════════════ */

const TYPES = {
  float: ['f', 1], int: ['i', 1],
  vec2: ['f', 2], vec3: ['f', 3], vec4: ['f', 4],
  ivec2: ['i', 2], ivec3: ['i', 3], ivec4: ['i', 4]
};
const IS_TYPE = Object.assign({ void: true, sampler2D: true }, TYPES);

const SWIZZLE = {
  x: 0, y: 1, z: 2, w: 3,
  r: 0, g: 1, b: 2, a: 3,
  s: 0, t: 1, p: 2, q: 3
};

/* Two-character operators must be tried before one-character ones. */
const PUNCT = ['<<', '>>', '(', ')', '{', '}', '[', ']', ',', ';', '.',
  '+', '-', '*', '/', '%', '&', '|', '^', '='];

function tokenize(src) {
  const out = [];
  let i = 0;
  const isAlpha = (c) => (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c === '_';
  const isDigit = (c) => c >= '0' && c <= '9';
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if (c === '/' && src[i + 1] === '*') {
      const e = src.indexOf('*/', i + 2);
      i = e < 0 ? src.length : e + 2;
      continue;
    }
    if (c === '/' && src[i + 1] === '/') {
      const e = src.indexOf('\n', i);
      i = e < 0 ? src.length : e + 1;
      continue;
    }
    if (c === '#') {
      const e = src.indexOf('\n', i);
      i = e < 0 ? src.length : e + 1;
      continue;
    }
    if (isAlpha(c)) {
      let j = i;
      while (j < src.length && (isAlpha(src[j]) || isDigit(src[j]))) j++;
      out.push({ t: 'id', v: src.slice(i, j) });
      i = j;
      continue;
    }
    if (isDigit(c) || (c === '.' && isDigit(src[i + 1] || ''))) {
      let j = i, isF = false;
      while (j < src.length && isDigit(src[j])) j++;
      if (src[j] === '.') { isF = true; j++; while (j < src.length && isDigit(src[j])) j++; }
      if (src[j] === 'e' || src[j] === 'E') {
        isF = true; j++;
        if (src[j] === '+' || src[j] === '-') j++;
        while (j < src.length && isDigit(src[j])) j++;
      }
      const text = src.slice(i, j);
      if (src[j] === 'f' || src[j] === 'F') { isF = true; j++; }
      out.push({ t: 'num', v: parseFloat(text), f: isF, text: text });
      i = j;
      continue;
    }
    let m = null;
    for (const p of PUNCT) if (src.startsWith(p, i)) { m = p; break; }
    if (!m) throw new Error('GLSL: unexpected character ' + JSON.stringify(c) + ' at ' + i);
    out.push({ t: 'op', v: m });
    i += m.length;
  }
  out.push({ t: 'eof', v: '<eof>' });
  return out;
}

/* Precedence, loosest last. GLSL ES 3.00 section 5.1. */
const PREC = [['|'], ['^'], ['&'], ['<<', '>>'], ['+', '-'], ['*', '/', '%']];

function parseGLSL(src) {
  const tk = tokenize(src);
  let p = 0;
  const peek = () => tk[p];
  const next = () => tk[p++];
  function expect(v) {
    const t = tk[p];
    if (t.v !== v) throw new Error('GLSL: expected ' + v + ' but found ' + t.v);
    p++;
    return t;
  }
  function isOp(v) { return tk[p].t === 'op' && tk[p].v === v; }

  function parsePrimary() {
    const t = next();
    if (t.t === 'num') return { n: 'num', v: t.v, f: t.f };
    if (t.t === 'op' && t.v === '(') {
      const e = parseBin(0);
      expect(')');
      return e;
    }
    if (t.t === 'op' && t.v === '-') return { n: 'neg', e: parseUnary() };
    if (t.t === 'op' && t.v === '+') return parseUnary();
    if (t.t === 'id') {
      if (isOp('(')) {
        p++;
        const args = [];
        if (!isOp(')')) {
          for (;;) {
            args.push(parseBin(0));
            if (isOp(',')) { p++; continue; }
            break;
          }
        }
        expect(')');
        return { n: 'call', name: t.v, args: args };
      }
      return { n: 'id', v: t.v };
    }
    throw new Error('GLSL: unexpected token ' + t.v);
  }

  function parsePostfix() {
    let e = parsePrimary();
    while (isOp('.')) {
      p++;
      const s = next();
      if (s.t !== 'id') throw new Error('GLSL: bad swizzle');
      e = { n: 'swz', e: e, s: s.v };
    }
    return e;
  }

  function parseUnary() {
    if (isOp('-')) { p++; return { n: 'neg', e: parseUnary() }; }
    if (isOp('+')) { p++; return parseUnary(); }
    return parsePostfix();
  }

  function parseBin(level) {
    if (level >= PREC.length) return parseUnary();
    let left = parseBin(level + 1);
    for (;;) {
      const t = peek();
      if (t.t === 'op' && PREC[level].indexOf(t.v) >= 0) {
        p++;
        const right = parseBin(level + 1);
        left = { n: 'bin', op: t.v, a: left, b: right };
      } else break;
    }
    return left;
  }

  const consts = [];
  const funcs = [];

  while (peek().t !== 'eof') {
    const t = peek();
    if (t.t === 'id' && t.v === 'const') {
      p++;
      const type = next().v;
      if (!TYPES[type]) throw new Error('GLSL: bad const type ' + type);
      const name = next().v;
      expect('=');
      const e = parseBin(0);
      expect(';');
      consts.push({ type: type, name: name, e: e });
      continue;
    }
    if (t.t === 'id' && IS_TYPE[t.v]) {
      const ret = next().v;
      const name = next().v;
      expect('(');
      const params = [];
      if (!isOp(')')) {
        for (;;) {
          const pt = next().v;
          const pn = next().v;
          if (!IS_TYPE[pt]) throw new Error('GLSL: bad param type ' + pt);
          params.push({ type: pt, name: pn });
          if (isOp(',')) { p++; continue; }
          break;
        }
      }
      expect(')');
      expect('{');
      const body = [];
      while (!isOp('}')) {
        if (peek().t === 'id' && peek().v === 'return') {
          p++;
          const e = parseBin(0);
          expect(';');
          body.push({ s: 'ret', e: e });
          continue;
        }
        const dt = next().v;
        if (!TYPES[dt]) throw new Error('GLSL: statement must be a declaration or return, found ' + dt);
        const dn = next().v;
        expect('=');
        const de = parseBin(0);
        expect(';');
        body.push({ s: 'decl', type: dt, name: dn, e: de });
      }
      expect('}');
      funcs.push({ ret: ret, name: name, params: params, body: body });
      continue;
    }
    throw new Error('GLSL: unexpected top-level token ' + t.v);
  }
  return { consts: consts, funcs: funcs };
}

function V(k, n, v) { return { k: k, n: n, v: v }; }

function makeVM(src, opts) {
  const ast = parseGLSL(src);
  const fp32 = !!(opts && opts.fp32);
  const R = fp32 ? Math.fround : (x) => x;
  const fetch = opts && opts.texelFetch;

  const funcs = Object.create(null);
  for (const f of ast.funcs) funcs[f.name] = f;

  const globals = Object.create(null);

  function construct(type, args) {
    const spec = TYPES[type];
    const k = spec[0], n = spec[1];
    const comps = [];
    for (const a of args) for (let i = 0; i < a.n; i++) comps.push(a.v[i]);
    let out;
    if (args.length === 1 && args[0].n === 1 && n > 1) {
      out = new Array(n).fill(comps[0]);
    } else if (comps.length >= n) {
      out = comps.slice(0, n);
    } else {
      throw new Error('GLSL: ' + type + '() needs ' + n + ' components, got ' + comps.length);
    }
    for (let i = 0; i < n; i++) out[i] = k === 'i' ? (Math.trunc(out[i]) | 0) : R(out[i]);
    return V(k, n, out);
  }

  function comp(a, i) { return a.v[a.n === 1 ? 0 : i]; }

  function cw(name, a, b, fn, kind) {
    const n = Math.max(a.n, b ? b.n : 1);
    if (b && a.n !== b.n && a.n !== 1 && b.n !== 1) {
      throw new Error('GLSL: width mismatch in ' + name);
    }
    const out = new Array(n);
    const k = kind || a.k;
    for (let i = 0; i < n; i++) {
      const r = fn(comp(a, i), b ? comp(b, i) : 0);
      out[i] = k === 'i' ? (r | 0) : R(r);
    }
    return V(k, n, out);
  }

  const INT_ONLY = { '&': 1, '|': 1, '^': 1, '<<': 1, '>>': 1, '%': 1 };

  function binop(op, a, b) {
    if (a.k !== b.k) {
      throw new Error('GLSL: illegal implicit conversion, ' + a.k + ' ' + op + ' ' + b.k);
    }
    if (INT_ONLY[op] && a.k !== 'i') throw new Error('GLSL: ' + op + ' needs integers');
    const isInt = a.k === 'i';
    return cw(op, a, b, function (x, y) {
      switch (op) {
        case '+': return x + y;
        case '-': return x - y;
        case '*': return isInt ? Math.imul(x, y) : x * y;
        case '/': return isInt ? Math.trunc(x / y) : x / y;
        case '%': return x - Math.trunc(x / y) * y;
        case '&': return x & y;
        case '|': return x | y;
        case '^': return x ^ y;
        case '<<': return x << y;
        case '>>': return x >> y;
        default: throw new Error('GLSL: unknown operator ' + op);
      }
    });
  }

  function builtin(name, args) {
    switch (name) {
      case 'floor':
        return cw('floor', args[0], null, (x) => Math.floor(x), 'f');
      case 'abs':
        return cw('abs', args[0], null, (x) => Math.abs(x));
      case 'fract':
        return cw('fract', args[0], null, (x) => R(x - Math.floor(x)), 'f');
      case 'mod':
        return cw('mod', args[0], args[1], (x, y) => R(x - R(y * Math.floor(R(x / y)))), 'f');
      case 'min':
        return cw('min', args[0], args[1], (x, y) => Math.min(x, y));
      case 'max':
        return cw('max', args[0], args[1], (x, y) => Math.max(x, y));
      case 'mix': {
        /* Spec: mix(x, y, a) = x*(1-a) + y*a. Expanded so fp32 rounds where
           the hardware rounds (barring an FMA the driver may fuse). */
        const x = args[0], y = args[1], a = args[2];
        const n = Math.max(x.n, y.n);
        const out = new Array(n);
        for (let i = 0; i < n; i++) {
          const av = comp(a, i);
          out[i] = R(R(comp(x, i) * R(1 - av)) + R(comp(y, i) * av));
        }
        return V('f', n, out);
      }
      case 'texelFetch': {
        const tex = args[0], t = args[1];
        if (tex.k !== 's') throw new Error('GLSL: texelFetch needs a sampler');
        if (t.k !== 'i' || t.n !== 2) throw new Error('GLSL: texelFetch needs an ivec2');
        const v = fetch(tex.tex, t.v[0], t.v[1]);
        return V('f', 4, [R(v[0]), R(v[1]), R(v[2]), R(v[3])]);
      }
      default:
        throw new Error('GLSL: unknown function ' + name);
    }
  }

  function evalExpr(e, env) {
    switch (e.n) {
      case 'num':
        return e.f ? V('f', 1, [R(e.v)]) : V('i', 1, [e.v | 0]);
      case 'id': {
        if (e.v in env) return env[e.v];
        if (e.v in globals) return globals[e.v];
        throw new Error('GLSL: undefined identifier ' + e.v);
      }
      case 'neg': {
        const a = evalExpr(e.e, env);
        return cw('neg', a, null, (x) => -x);
      }
      case 'swz': {
        const a = evalExpr(e.e, env);
        const s = e.s;
        const out = new Array(s.length);
        for (let i = 0; i < s.length; i++) {
          const idx = SWIZZLE[s[i]];
          if (idx === undefined || idx >= a.n) throw new Error('GLSL: bad swizzle .' + s);
          out[i] = a.v[idx];
        }
        return V(a.k, s.length, out);
      }
      case 'bin':
        return binop(e.op, evalExpr(e.a, env), evalExpr(e.b, env));
      case 'call': {
        const args = e.args.map((x) => evalExpr(x, env));
        if (TYPES[e.name]) return construct(e.name, args);
        if (funcs[e.name]) return callFn(e.name, args);
        return builtin(e.name, args);
      }
      default:
        throw new Error('GLSL: bad node ' + e.n);
    }
  }

  function coerceDecl(type, val, what) {
    const spec = TYPES[type];
    if (!spec) throw new Error('GLSL: bad type ' + type);
    if (spec[0] !== val.k || spec[1] !== val.n) {
      throw new Error('GLSL: ' + what + ' declared ' + type + ' but expression is ' +
        (val.k === 'i' ? 'i' : 'f') + 'vec' + val.n);
    }
    return val;
  }

  function callFn(name, args) {
    const f = funcs[name];
    if (!f) throw new Error('GLSL: no function ' + name);
    if (args.length !== f.params.length) throw new Error('GLSL: ' + name + ' arity');
    const env = Object.create(null);
    for (let i = 0; i < args.length; i++) {
      const pt = f.params[i].type;
      if (pt === 'sampler2D') {
        if (args[i].k !== 's') throw new Error('GLSL: ' + name + ' wants a sampler');
        env[f.params[i].name] = args[i];
      } else {
        env[f.params[i].name] = coerceDecl(pt, args[i], name + ' parameter ' + f.params[i].name);
      }
    }
    for (const st of f.body) {
      if (st.s === 'decl') {
        env[st.name] = coerceDecl(st.type, evalExpr(st.e, env), name + ':' + st.name);
      } else {
        return coerceDecl(f.ret, evalExpr(st.e, env), name + ' return');
      }
    }
    throw new Error('GLSL: ' + name + ' fell off the end');
  }

  for (const c of ast.consts) {
    globals[c.name] = coerceDecl(c.type, evalExpr(c.e, Object.create(null)), 'const ' + c.name);
  }

  return {
    ast: ast,
    consts: globals,
    sampler: (tex) => ({ k: 's', n: 1, v: [0], tex: tex }),
    f: (x) => V('f', 1, [R(x)]),
    i: (x) => V('i', 1, [x | 0]),
    vec3: (a, b, c) => V('f', 3, [R(a), R(b), R(c)]),
    ivec3: (a, b, c) => V('i', 3, [a | 0, b | 0, c | 0]),
    ivec2: (a, b) => V('i', 2, [a | 0, b | 0]),
    call: callFn,
    R: R
  };
}

/* ═══════════════════════════════════════════════════════════════════════════
   2. Independent reference implementations — deliberately written with
      different arithmetic from the library (% and Math.floor instead of masks
      and shifts, a weighted sum of eight corners instead of a nested mix), so
      that agreement means something.
   ═══════════════════════════════════════════════════════════════════════════ */

function refIndex(lay, x, y, z) {
  const tx = (z % lay.tilesX) * lay.N + x;
  const ty = Math.floor(z / lay.tilesX) * lay.N + y;
  return (ty * lay.width + tx) * 4;
}

function refWrap(i, N) {
  const m = i % N;
  return m < 0 ? m + N : m;
}

function refTrilinear(tex, lay, gx, gy, gz, channel) {
  const N = lay.N;
  const bx = Math.floor(gx), by = Math.floor(gy), bz = Math.floor(gz);
  const fx = gx - bx, fy = gy - by, fz = gz - bz;
  let acc = 0;
  for (let dz = 0; dz < 2; dz++) {
    const wz = dz ? fz : 1 - fz;
    for (let dy = 0; dy < 2; dy++) {
      const wy = dy ? fy : 1 - fy;
      for (let dx = 0; dx < 2; dx++) {
        const wx = dx ? fx : 1 - fx;
        const i = refIndex(lay,
          refWrap(bx + dx, N), refWrap(by + dy, N), refWrap(bz + dz, N));
        acc += wx * wy * wz * tex[i + channel];
      }
    }
  }
  return acc;
}

function refWorldToGrid(p, L, N) { return (p / L + 0.5) * N - 0.5; }
function refGridToWorld(g, L, N) { return ((g + 0.5) / N - 0.5) * L; }

/* ═══════════════════════════════════════════════════════════════════════════
   3. layout()
   ═══════════════════════════════════════════════════════════════════════════ */

section('layout');

{
  const expect = {
    16: { tilesX: 4, tilesY: 4, width: 64, height: 64 },
    32: { tilesX: 8, tilesY: 4, width: 256, height: 128 },
    64: { tilesX: 8, tilesY: 8, width: 512, height: 512 },
    128: { tilesX: 16, tilesY: 8, width: 2048, height: 1024 }
  };
  for (const N of A.SIZES) {
    const L = A.layout(N);
    const e = expect[N];
    ok('layout(' + N + ')',
      L.N === N && L.tilesX === e.tilesX && L.tilesY === e.tilesY &&
      L.width === e.width && L.height === e.height,
      L.tilesX + 'x' + L.tilesY + ' tiles, ' + L.width + 'x' + L.height + ' texels');
    ok('layout(' + N + ') tiles cover the volume exactly',
      L.tilesX * L.tilesY === N && L.width * L.height === N * N * N,
      'width*height = ' + (L.width * L.height) + ', N^3 = ' + (N * N * N));
    ok('layout(' + N + ') is near-square',
      Math.max(L.width, L.height) / Math.min(L.width, L.height) <= 2,
      'aspect ' + (L.width / L.height));
    ok('layout(' + N + ') fits the WebGL2 minimum MAX_TEXTURE_SIZE',
      L.width <= 2048 && L.height <= 2048);
  }

  const bad = [8, 256, 24, 48, 100, 0, -16, 1, 3, 17, 33, 2048];
  let allRejected = true;
  const survivors = [];
  for (const N of bad) {
    let threw = false;
    try { A.layout(N); } catch (e) { threw = e instanceof RangeError || e instanceof TypeError; }
    if (!threw) { allRejected = false; survivors.push(N); }
  }
  ok('layout rejects out-of-range and non-power-of-two N', allRejected,
    survivors.length ? 'accepted ' + survivors.join(',') : bad.length + ' rejected');

  const junk = [16.5, NaN, Infinity, '32', null, undefined, {}, [], '  '];
  let allJunk = true;
  for (const N of junk) {
    let threw = false;
    try { A.layout(N); } catch (e) { threw = true; }
    if (!threw) allJunk = false;
  }
  ok('layout rejects non-integer and non-numeric N', allJunk);

  let caught = false;
  try { A.GLSL({ N: 32, tilesX: 4, tilesY: 8, width: 128, height: 256 }); }
  catch (e) { caught = e instanceof RangeError; }
  ok('a hand-built layout that disagrees with layout() is rejected', caught);
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. The GLSL chunk: shape, and the constants it bakes in
   ═══════════════════════════════════════════════════════════════════════════ */

section('GLSL chunk structure and baked constants');

const REQUIRED = [
  ['vec2', 'cellToUV', ['vec3']],
  ['ivec2', 'cellToTexel', ['ivec3']],
  ['vec3', 'texelToCell', ['ivec2']],
  ['vec3', 'worldToGrid', ['vec3', 'float']],
  ['vec3', 'sampleGrid', ['sampler2D', 'vec3']],
  ['vec3', 'gradGrid', ['sampler2D', 'vec3', 'float']]
];
/* sampleGrid returns vec4; fixed up here rather than in the table above. */
REQUIRED[4][0] = 'vec4';

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const src = A.GLSL(lay);
  const C = A.constants(lay);
  const tag = 'N=' + N;

  /* The chunk gets pasted after a host shader's own #version line, so it must
     not carry one — nor a precision qualifier, nor a main(). Checked on the
     code, not on the comments, which are allowed to say the words. */
  const code = src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
  ok(tag + ' GLSL is a chunk, not a shader',
    code.indexOf('#') < 0 && /\bprecision\b/.test(code) === false &&
    /\bmain\s*\(/.test(code) === false,
    code.indexOf('#') >= 0 ? 'preprocessor directive present' : '');

  /* ESSL's source character set is ASCII, and a stray multi-byte character in
     a comment is enough for some drivers to reject the whole shader this chunk
     is pasted into. */
  const nonAscii = [];
  for (const ch of src) {
    const cp = ch.codePointAt(0);
    if (cp > 126 || (cp < 32 && ch !== '\n')) nonAscii.push(JSON.stringify(ch));
  }
  ok(tag + ' GLSL is plain ASCII', nonAscii.length === 0,
    nonAscii.length ? 'found ' + Array.from(new Set(nonAscii)).join(' ') : '');

  let braces = 0, balanced = true;
  for (const ch of src) {
    if (ch === '{') braces++;
    else if (ch === '}') { braces--; if (braces < 0) balanced = false; }
  }
  ok(tag + ' braces balance', balanced && braces === 0);

  const ast = parseGLSL(src);
  const byName = Object.create(null);
  for (const f of ast.funcs) byName[f.name] = f;

  let sigsOk = true;
  const sigDetail = [];
  for (const [ret, name, params] of REQUIRED) {
    const f = byName[name];
    if (!f) { sigsOk = false; sigDetail.push(name + ' missing'); continue; }
    if (f.ret !== ret) { sigsOk = false; sigDetail.push(name + ' returns ' + f.ret); }
    const got = f.params.map((x) => x.type).join(',');
    if (got !== params.join(',')) { sigsOk = false; sigDetail.push(name + '(' + got + ')'); }
  }
  ok(tag + ' all six functions declared with the required signatures', sigsOk,
    sigDetail.join('; '));

  /* Constants: pull every `const int/float NAME = V;` straight out of the text
     and demand it equal the number the JS mirror closes over. */
  const found = Object.create(null);
  const re = /const\s+(int|float)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/g;
  let m;
  while ((m = re.exec(src)) !== null) {
    const name = m[2].replace(/^HA_/, '');
    const text = m[3].trim();
    const val = parseFloat(text);
    found[name] = { val: val, type: m[1], text: text };
  }

  const names = Object.keys(C);
  let constsOk = true;
  const constDetail = [];
  for (const k of names) {
    const f = found[k];
    if (!f) { constsOk = false; constDetail.push(k + ' absent from GLSL'); continue; }
    if (f.val !== C[k]) {
      constsOk = false;
      constDetail.push(k + ' GLSL ' + f.text + ' vs JS ' + C[k]);
    }
    const wantInt = Number.isInteger(C[k]) && k !== 'NF' && k !== 'TILES_XF';
    if (wantInt && f.type !== 'int') { constsOk = false; constDetail.push(k + ' typed ' + f.type); }
    if (!wantInt && f.type !== 'float') { constsOk = false; constDetail.push(k + ' typed ' + f.type); }
  }
  ok(tag + ' every baked constant equals the JS mirror\'s', constsOk,
    constDetail.length ? constDetail.join('; ') : names.length + ' constants matched');

  let extras = Object.keys(found).filter((k) => !(k in C));
  ok(tag + ' GLSL bakes in no constant the mirror does not know about',
    extras.length === 0, extras.join(','));

  /* Float literals must round-trip exactly, or the shader and the mirror are
     already different numbers before either runs. */
  let exact = true;
  const exactBad = [];
  for (const k of names) {
    const f = found[k];
    if (f && f.type === 'float' && Math.fround(f.val) !== Math.fround(C[k])) {
      exact = false;
      exactBad.push(k);
    }
    if (f && f.type === 'float' && f.val !== C[k]) { exact = false; exactBad.push(k + '(f64)'); }
  }
  ok(tag + ' float literals round-trip exactly in binary32 and binary64', exact,
    exactBad.join(','));
}

/* ═══════════════════════════════════════════════════════════════════════════
   5. cellToTexel is a bijection; texelToCell inverts it
   ═══════════════════════════════════════════════════════════════════════════ */

section('cellToTexel bijection and texelToCell inverse');

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const M = A.mirror(lay);
  const tag = 'N=' + N;
  const total = N * N * N;

  /* Exhaustive for every size — N=128 is two million cells, which costs less
     than a second and beats sampling. */
  const seen = new Uint8Array(lay.width * lay.height);
  const cell = new Float64Array(3);
  const texel = new Int32Array(2);
  const back = new Float64Array(3);
  let collisions = 0, outOfRange = 0, roundTripFails = 0, refMismatch = 0;

  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        cell[0] = x; cell[1] = y; cell[2] = z;
        M.cellToTexel(cell, texel);
        const tx = texel[0], ty = texel[1];
        if (!(tx >= 0 && tx < lay.width && ty >= 0 && ty < lay.height)) { outOfRange++; continue; }
        const flat = ty * lay.width + tx;
        if (seen[flat]) collisions++;
        seen[flat] = 1;
        if ((flat * 4) !== refIndex(lay, x, y, z)) refMismatch++;
        M.texelToCell(texel, back);
        if (back[0] !== x || back[1] !== y || back[2] !== z) roundTripFails++;
      }
    }
  }
  let unused = 0;
  for (let i = 0; i < seen.length; i++) if (!seen[i]) unused++;

  ok(tag + ' every cell maps in range', outOfRange === 0, outOfRange + ' escapes');
  ok(tag + ' cellToTexel is injective', collisions === 0, collisions + ' collisions');
  ok(tag + ' cellToTexel is onto every texel', unused === 0,
    total + ' cells, ' + (seen.length - unused) + ' of ' + seen.length + ' texels used');
  ok(tag + ' texelToCell(cellToTexel(c)) === c for all ' + total + ' cells',
    roundTripFails === 0, roundTripFails + ' failures');
  ok(tag + ' addressing agrees with the independent (% and floor) reference',
    refMismatch === 0, refMismatch + ' mismatches');

  /* And the other direction: every texel in range maps back to a legal cell
     that returns to it. */
  let texelFails = 0;
  for (let ty = 0; ty < lay.height; ty++) {
    for (let tx = 0; tx < lay.width; tx++) {
      texel[0] = tx; texel[1] = ty;
      M.texelToCell(texel, back);
      if (!(back[0] >= 0 && back[0] < N && back[1] >= 0 && back[1] < N &&
            back[2] >= 0 && back[2] < N)) { texelFails++; continue; }
      M.cellToTexel(back, texel);
      if (texel[0] !== tx || texel[1] !== ty) texelFails++;
    }
  }
  ok(tag + ' cellToTexel(texelToCell(t)) === t for all ' + (lay.width * lay.height) + ' texels',
    texelFails === 0, texelFails + ' failures');

  /* Periodic totality: a neighbour offset that walks off the grid lands on the
     periodic image, not on the next slice along. */
  let wrapFails = 0;
  const rnd = lcg(0x51ed + N);
  for (let s = 0; s < 4000; s++) {
    const x = (rnd() * N) | 0, y = (rnd() * N) | 0, z = (rnd() * N) | 0;
    const k = [-3, -2, -1, 1, 2, 3][(rnd() * 6) | 0];
    const axis = (rnd() * 3) | 0;
    const off = [x, y, z];
    off[axis] += k * N;
    cell[0] = off[0]; cell[1] = off[1]; cell[2] = off[2];
    M.cellToTexel(cell, texel);
    const a = texel[0], b = texel[1];
    cell[0] = x; cell[1] = y; cell[2] = z;
    M.cellToTexel(cell, texel);
    if (a !== texel[0] || b !== texel[1]) wrapFails++;
  }
  ok(tag + ' cellToTexel is periodic in every axis', wrapFails === 0, wrapFails + ' failures');
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. cellToUV
   ═══════════════════════════════════════════════════════════════════════════ */

section('cellToUV lands on texel centres');

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const M = A.mirror(lay);
  const tag = 'N=' + N;
  const cell = new Float64Array(3);
  const texel = new Int32Array(2);
  const uv = new Float64Array(2);
  const rnd = lcg(0xc0ffee + N);
  let worst = 0, outside = 0;
  const trials = N <= 32 ? N * N * N : 200000;
  for (let s = 0; s < trials; s++) {
    let x, y, z;
    if (N <= 32) {
      x = s % N; y = ((s / N) | 0) % N; z = (s / (N * N)) | 0;
    } else {
      x = (rnd() * N) | 0; y = (rnd() * N) | 0; z = (rnd() * N) | 0;
    }
    cell[0] = x; cell[1] = y; cell[2] = z;
    M.cellToUV(cell, uv);
    M.cellToTexel(cell, texel);
    const eu = (texel[0] + 0.5) / lay.width;
    const ev = (texel[1] + 0.5) / lay.height;
    worst = Math.max(worst, Math.abs(uv[0] - eu), Math.abs(uv[1] - ev));
    if (!(uv[0] > 0 && uv[0] < 1 && uv[1] > 0 && uv[1] < 1)) outside++;
  }
  ok(tag + ' cellToUV(c) is the centre of cellToTexel(c)', worst === 0,
    'max |delta| = ' + worst);
  ok(tag + ' cellToUV stays strictly inside [0,1]^2', outside === 0, outside + ' escapes');
}

/* ═══════════════════════════════════════════════════════════════════════════
   7. worldToGrid / gridToWorld
   ═══════════════════════════════════════════════════════════════════════════ */

section('worldToGrid corners, centre and inverse');

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const M = A.mirror(lay);
  const tag = 'N=' + N;
  const p = new Float64Array(3);
  const g = new Float64Array(3);
  const w = new Float64Array(3);

  for (const L of [1, 2, 4, 7.5, 0.25]) {
    /* The eight corners land on -0.5 or N-0.5 exactly; the centre on N/2-0.5. */
    let cornerOk = true;
    for (let c = 0; c < 8; c++) {
      p[0] = (c & 1 ? 0.5 : -0.5) * L;
      p[1] = (c & 2 ? 0.5 : -0.5) * L;
      p[2] = (c & 4 ? 0.5 : -0.5) * L;
      M.worldToGrid(p, L, g);
      for (let k = 0; k < 3; k++) {
        const want = (c & (1 << k)) ? N - 0.5 : -0.5;
        if (g[k] !== want) cornerOk = false;
      }
    }
    p[0] = 0; p[1] = 0; p[2] = 0;
    M.worldToGrid(p, L, g);
    const centreOk = g[0] === N / 2 - 0.5 && g[1] === N / 2 - 0.5 && g[2] === N / 2 - 0.5;
    ok(tag + ' L=' + L + ' box corners map to -0.5 and ' + (N - 0.5) + ' exactly', cornerOk);
    ok(tag + ' L=' + L + ' box centre maps to ' + (N / 2 - 0.5) + ' exactly', centreOk,
      '[' + g.join(', ') + ']');

    /* Cell centres sit where the tiling says they do, and gridToWorld is the
       exact inverse. */
    const rnd = lcg(0xbeef + N + L * 1000);
    let worstRound = 0, worstRef = 0, worstCentre = 0;
    for (let s = 0; s < 20000; s++) {
      p[0] = (rnd() - 0.5) * L * 3;
      p[1] = (rnd() - 0.5) * L * 3;
      p[2] = (rnd() - 0.5) * L * 3;
      M.worldToGrid(p, L, g);
      for (let k = 0; k < 3; k++) {
        worstRef = Math.max(worstRef, Math.abs(g[k] - refWorldToGrid(p[k], L, N)));
      }
      M.gridToWorld(g, L, w);
      const scale = Math.max(1, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
      worstRound = Math.max(worstRound, maxAbsDiff(w, p) / scale);
    }
    for (let i = 0; i < N; i++) {
      g[0] = i; g[1] = i; g[2] = i;
      M.gridToWorld(g, L, w);
      const want = (i + 0.5) * (L / N) - L / 2;
      worstCentre = Math.max(worstCentre, Math.abs(w[0] - want));
    }
    ok(tag + ' L=' + L + ' worldToGrid matches the independent reference',
      worstRef <= 1e-12, 'max |delta| = ' + worstRef);
    ok(tag + ' L=' + L + ' gridToWorld inverts worldToGrid', worstRound <= 1e-14,
      'max relative |delta| = ' + worstRound.toExponential(2));
    ok(tag + ' L=' + L + ' integer grid coordinates sit at cell centres',
      worstCentre <= 1e-14, 'max |delta| = ' + worstCentre.toExponential(2));
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   8. sampleGrid
   ═══════════════════════════════════════════════════════════════════════════ */

section('sampleGrid: cell centres, linearity, tiling, seam');

/* Fill channel c of the atlas from a function of the integer cell. */
function fill(M, tex, fns) {
  const N = M.layout.N;
  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        M.store(tex, x, y, z,
          fns[0](x, y, z), fns[1](x, y, z), fns[2](x, y, z), fns[3](x, y, z));
      }
    }
  }
}

/* Coefficients chosen as exact binary fractions so that the stored values are
   exact in float32 and the only error left to measure is the mapping's. */
const AX = 0.125, BX = 0.25;
const AY = 0.0625, BY = -0.5;
const AZ = 0.25, BZ = 1.5;

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const M = A.mirror(lay);
  const tag = 'N=' + N;
  const tex = M.createAtlas();
  fill(M, tex, [
    (x) => AX * x + BX,
    (x, y) => AY * y + BY,
    (x, y, z) => AZ * z + BZ,
    () => 0.75
  ]);

  const g = new Float64Array(3);
  const out = new Float64Array(4);
  const rnd = lcg(0x5eed + N);

  /* (a) at a cell centre the stored value comes back untouched. */
  let centreWorst = 0;
  const centreTrials = N <= 32 ? N * N * N : 100000;
  for (let s = 0; s < centreTrials; s++) {
    let x, y, z;
    if (N <= 32) { x = s % N; y = ((s / N) | 0) % N; z = (s / (N * N)) | 0; }
    else { x = (rnd() * N) | 0; y = (rnd() * N) | 0; z = (rnd() * N) | 0; }
    g[0] = x; g[1] = y; g[2] = z;
    M.sampleGrid(tex, g, out);
    centreWorst = Math.max(centreWorst,
      Math.abs(out[0] - (AX * x + BX)),
      Math.abs(out[1] - (AY * y + BY)),
      Math.abs(out[2] - (AZ * z + BZ)),
      Math.abs(out[3] - 0.75));
  }
  ok(tag + ' sampleGrid at a cell centre returns the cell exactly',
    centreWorst === 0, 'max |delta| = ' + centreWorst);

  /* (b) a field linear in x, sampled with y and z ANYWHERE — far outside the
     box, across every z-slice boundary, across the periodic seam in y and z.
     Channel 0 does not vary with y or z, so wrapping those axes is harmless
     and the x line must survive intact. The same for y and for z. This is the
     test that fails if the tiling is wrong: sampling channel 2 interpolates
     between two different tiles on every call. */
  const axes = [
    { name: 'x', ch: 0, a: AX, b: BX },
    { name: 'y', ch: 1, a: AY, b: BY },
    { name: 'z', ch: 2, a: AZ, b: BZ }
  ];
  for (const ax of axes) {
    let worst = 0, worstAt = null;
    for (let s = 0; s < 120000; s++) {
      /* the varying axis stays in the non-wrapping interior [0, N-1]; the
         other two roam over +/- 4 box widths */
      for (let k = 0; k < 3; k++) g[k] = (rnd() - 0.5) * 8 * N;
      g[ax.ch] = rnd() * (N - 1);
      M.sampleGrid(tex, g, out);
      const want = ax.a * g[ax.ch] + ax.b;
      const d = Math.abs(out[ax.ch] - want);
      if (d > worst) { worst = d; worstAt = Array.from(g); }
      /* the constant channel must stay constant no matter how far out we go */
      worst = Math.max(worst, Math.abs(out[3] - 0.75));
    }
    ok(tag + ' field linear in ' + ax.name +
      ' is reproduced across slice boundaries and the seam in the other axes',
      worst <= 1e-5, 'max |delta| = ' + worst.toExponential(2) +
      (worst > 1e-5 ? ' at ' + JSON.stringify(worstAt) : ''));
  }

  /* (c) the whole 4-vector against the independent weighted-sum reference, at
     positions ranging over eight box widths in every direction. */
  let refWorst = 0;
  for (let s = 0; s < 120000; s++) {
    for (let k = 0; k < 3; k++) g[k] = (rnd() - 0.5) * 8 * N;
    M.sampleGrid(tex, g, out);
    for (let c = 0; c < 4; c++) {
      refWorst = Math.max(refWorst, Math.abs(out[c] - refTrilinear(tex, lay, g[0], g[1], g[2], c)));
    }
  }
  ok(tag + ' sampleGrid matches the independent periodic-trilinear reference',
    refWorst <= 1e-5, 'max |delta| = ' + refWorst.toExponential(2));

  /* (d) the seam in closed form. For gx in [-0.5, 0) the stencil straddles the
     wrap, and the correct answer is the sawtooth blend between cell N-1 and
     cell 0 — not the linear function. Asserting the closed form pins down
     which cells the wrap chose, in a way the generic reference cannot. */
  let seamWorst = 0;
  for (let s = 0; s < 20000; s++) {
    const t = rnd();                      /* fraction into the seam cell */
    /* [-1,0) and [N-1,N) are the same point of the torus reached from either
       side: the first goes through the negative branch of the float wrap, the
       second does not, and both must land on the same blend. */
    g[0] = (rnd() < 0.5 ? -1 : N - 1) + t;
    g[1] = (rnd() * N) | 0;
    g[2] = (rnd() * N) | 0;
    M.sampleGrid(tex, g, out);
    const f0 = AX * (N - 1) + BX;         /* the last cell */
    const f1 = AX * 0 + BX;               /* wrapping round to the first */
    const want = f0 * (1 - t) + f1 * t;
    seamWorst = Math.max(seamWorst, Math.abs(out[0] - want));
  }
  ok(tag + ' the x seam blends cell N-1 into cell 0 in closed form',
    seamWorst <= 1e-5, 'max |delta| = ' + seamWorst.toExponential(2));

  /* (e) z seam: the slice that wraps is also the tile that wraps, from the top
     right tile back to the bottom left one. */
  let zSeamWorst = 0;
  for (let s = 0; s < 20000; s++) {
    const t = rnd();
    g[0] = (rnd() * N) | 0;
    g[1] = (rnd() * N) | 0;
    g[2] = (rnd() < 0.5 ? -1 : N - 1) + t;
    M.sampleGrid(tex, g, out);
    const want = (AZ * (N - 1) + BZ) * (1 - t) + (AZ * 0 + BZ) * t;
    zSeamWorst = Math.max(zSeamWorst, Math.abs(out[2] - want));
  }
  ok(tag + ' the z seam blends the last tile into the first in closed form',
    zSeamWorst <= 1e-5, 'max |delta| = ' + zSeamWorst.toExponential(2));

  /* (f) exact periodicity: shifting by whole multiples of N changes nothing. */
  let periodWorst = 0;
  const g2 = new Float64Array(3);
  const out2 = new Float64Array(4);
  for (let s = 0; s < 40000; s++) {
    for (let k = 0; k < 3; k++) g[k] = (rnd() - 0.5) * 2 * N;
    for (let k = 0; k < 3; k++) {
      const kk = ((rnd() * 11) | 0) - 5;
      g2[k] = g[k] + kk * N;
    }
    M.sampleGrid(tex, g, out);
    M.sampleGrid(tex, g2, out2);
    periodWorst = Math.max(periodWorst, maxAbsDiff(out, out2));
  }
  ok(tag + ' sampleGrid is exactly periodic for shifts of +/- 5 boxes',
    periodWorst <= 1e-5, 'max |delta| = ' + periodWorst.toExponential(2));

  /* (g) far outside the box, reached through worldToGrid, in all 26 directions
     plus some long throws. */
  const L = 3.5;
  const p = new Float64Array(3);
  const pg = new Float64Array(3);
  let farWorst = 0, threw = null;
  for (let dz = -2; dz <= 2; dz++) {
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        for (let s = 0; s < 200; s++) {
          p[0] = (rnd() - 0.5) * L + dx * L;
          p[1] = (rnd() - 0.5) * L + dy * L;
          p[2] = (rnd() - 0.5) * L + dz * L;
          try {
            M.worldToGrid(p, L, pg);
            M.sampleGrid(tex, pg, out);
          } catch (e) { threw = e; break; }
          for (let c = 0; c < 4; c++) {
            farWorst = Math.max(farWorst,
              Math.abs(out[c] - refTrilinear(tex, lay, pg[0], pg[1], pg[2], c)));
          }
        }
      }
    }
  }
  ok(tag + ' positions up to 2 box widths out in all 125 directions wrap correctly',
    threw === null && farWorst <= 1e-5,
    threw ? String(threw.message) : 'max |delta| = ' + farWorst.toExponential(2));
}

/* ═══════════════════════════════════════════════════════════════════════════
   9. gradGrid
   ═══════════════════════════════════════════════════════════════════════════ */

section('gradGrid: linear exact, quadratic exact at h=1, predicted error otherwise');

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const M = A.mirror(lay);
  const tag = 'N=' + N;
  const g = new Float64Array(3);
  const out = new Float64Array(3);
  const rnd = lcg(0x9e37 + N);

  /* (a) linear field: the gradient is a constant, and central differences are
     exact for it at any h. */
  {
    const tex = M.createAtlas();
    const a = 0.125, b = -0.0625, c = 0.25, d = 0.5;
    fill(M, tex, [
      (x, y, z) => a * x + b * y + c * z + d,
      () => 0, () => 0, () => 0
    ]);
    for (const h of [1, 0.5, 0.25, 0.125]) {
      let worst = 0;
      for (let s = 0; s < 30000; s++) {
        for (let k = 0; k < 3; k++) g[k] = h + rnd() * (N - 1 - 2 * h);
        M.gradGrid(tex, g, h, out);
        worst = Math.max(worst,
          Math.abs(out[0] - a), Math.abs(out[1] - b), Math.abs(out[2] - c));
      }
      ok(tag + ' h=' + h + ' gradGrid on a linear field is the exact constant gradient',
        worst <= 1e-5, 'max |delta| = ' + worst.toExponential(2));
    }
  }

  /* (b) quadratic field. With h = 1 the trilinear interpolation error of a
     quadratic is the same at both stencil points (it depends only on the
     fractional part, which repeats with period 1), so it cancels and the
     gradient is exact even at non-integer positions. Bilinear interpolation
     reproduces the cross terms exactly, so they contribute nothing either. */
  {
    const tex = M.createAtlas();
    const s2 = 1 / (N * N);
    const qa = 0.5 * s2, qb = 0.25 * s2, qc = 0.75 * s2;
    const qxy = 0.3 * s2, qyz = 0.2 * s2, qxz = 0.1 * s2;
    const q0 = 0.42;
    const q = (x, y, z) => qa * x * x + qb * y * y + qc * z * z +
      qxy * x * y + qyz * y * z + qxz * x * z + q0;
    const dq = (x, y, z) => [
      2 * qa * x + qxy * y + qxz * z,
      2 * qb * y + qxy * x + qyz * z,
      2 * qc * z + qyz * y + qxz * x
    ];
    fill(M, tex, [q, () => 0, () => 0, () => 0]);

    let worst = 0, scale = 0;
    for (let s = 0; s < 30000; s++) {
      for (let k = 0; k < 3; k++) g[k] = 1 + rnd() * (N - 3);
      M.gradGrid(tex, g, 1, out);
      const want = dq(g[0], g[1], g[2]);
      for (let k = 0; k < 3; k++) {
        worst = Math.max(worst, Math.abs(out[k] - want[k]));
        scale = Math.max(scale, Math.abs(want[k]));
      }
    }
    ok(tag + ' h=1 gradGrid on a quadratic field is exact (error cancels)',
      worst / scale <= 1e-4, 'max relative |delta| = ' + (worst / scale).toExponential(2));

    /* (c) h != 1: the error no longer cancels, and it is not noise — it is the
       interpolation error of the pure square terms differenced across the
       stencil. Asserting the closed form proves the scheme is understood
       rather than merely tolerated. */
    const phi = (u) => { const t = u - Math.floor(u); return t * (1 - t); };
    for (const h of [0.5, 0.25]) {
      let predWorst = 0, rawWorst = 0, rawScale = 0;
      const coef = [qa, qb, qc];
      for (let s = 0; s < 20000; s++) {
        for (let k = 0; k < 3; k++) g[k] = 1 + rnd() * (N - 3);
        M.gradGrid(tex, g, h, out);
        const want = dq(g[0], g[1], g[2]);
        for (let k = 0; k < 3; k++) {
          const pred = coef[k] * (phi(g[k] + h) - phi(g[k] - h)) / (2 * h);
          predWorst = Math.max(predWorst, Math.abs(out[k] - want[k] - pred));
          rawWorst = Math.max(rawWorst, Math.abs(out[k] - want[k]));
          rawScale = Math.max(rawScale, Math.abs(want[k]));
        }
      }
      ok(tag + ' h=' + h + ' gradGrid error equals the predicted interpolation term',
        predWorst / Math.max(rawScale, 1e-30) <= 1e-4,
        'residual/scale = ' + (predWorst / rawScale).toExponential(2) +
        ', raw error/scale = ' + (rawWorst / rawScale).toExponential(2));
    }
  }
}

/* (d) second order in the grid spacing: a cubic world field on a refining grid.
   Halving the cell size must quarter the error. */
{
  const L = 2;
  const F = (x, y, z) => x * x * x + 0.7 * y * y * y - 0.4 * z * z * z + x * y * z;
  const dF = (x, y, z) => [
    3 * x * x + y * z,
    2.1 * y * y + x * z,
    -1.2 * z * z + x * y
  ];
  const probes = [];
  {
    const rnd = lcg(0x1234);
    for (let s = 0; s < 3000; s++) {
      probes.push([(rnd() - 0.5) * 0.6 * L, (rnd() - 0.5) * 0.6 * L, (rnd() - 0.5) * 0.6 * L]);
    }
  }
  const errs = [];
  for (const N of A.SIZES) {
    const M = A.mirror(A.layout(N));
    const tex = M.createAtlas();
    const gw = new Float64Array(3);
    const cell = new Float64Array(3);
    fill(M, tex, [
      (x, y, z) => {
        cell[0] = x; cell[1] = y; cell[2] = z;
        M.gridToWorld(cell, L, gw);
        return F(gw[0], gw[1], gw[2]);
      },
      () => 0, () => 0, () => 0
    ]);
    const g = new Float64Array(3);
    const out = new Float64Array(3);
    const p = new Float64Array(3);
    let worst = 0;
    for (const pr of probes) {
      p[0] = pr[0]; p[1] = pr[1]; p[2] = pr[2];
      M.worldToGrid(p, L, g);
      M.gradGrid(tex, g, 1, out);
      const want = dF(p[0], p[1], p[2]);
      for (let k = 0; k < 3; k++) {
        worst = Math.max(worst, Math.abs(out[k] * (N / L) - want[k]));
      }
    }
    errs.push({ N: N, e: worst });
  }
  for (let i = 1; i < errs.length; i++) {
    const r = errs[i - 1].e / errs[i].e;
    ok('gradGrid is second order: N ' + errs[i - 1].N + ' -> ' + errs[i].N +
      ' quarters the error', r >= 3.0 && r <= 5.5,
      'ratio ' + r.toFixed(3) + '  (' + errs[i - 1].e.toExponential(2) +
      ' -> ' + errs[i].e.toExponential(2) + ')');
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   10. The generated GLSL, executed, against the JS mirror
   ═══════════════════════════════════════════════════════════════════════════ */

section('the GLSL chunk executed against the JS mirror');

for (const N of A.SIZES) {
  const lay = A.layout(N);
  const src = A.GLSL(lay);
  const tag = 'N=' + N;

  for (const fp32 of [false, true]) {
    const mode = fp32 ? 'fp32' : 'f64';
    const M = A.mirror(lay, { fp32: fp32 });
    const tex = M.createAtlas();
    const rnd = lcg(0xa71a5 + N + (fp32 ? 1 : 0));
    /* Values with plenty of mantissa so that any difference in the order of
       operations shows up immediately. */
    for (let i = 0; i < tex.length; i++) tex[i] = (rnd() - 0.5) * 4;

    const vm = makeVM(src, {
      fp32: fp32,
      texelFetch: (t, x, y) => {
        if (!(x >= 0 && x < lay.width && y >= 0 && y < lay.height)) {
          throw new RangeError('GLSL texelFetch out of range (' + x + ',' + y + ')');
        }
        const i = (y * lay.width + x) * 4;
        return [t[i], t[i + 1], t[i + 2], t[i + 3]];
      }
    });
    const sam = vm.sampler(tex);
    const R = vm.R;

    const cell = new Float64Array(3);
    const texel = new Int32Array(2);
    const o2 = new Float64Array(2);
    const o3 = new Float64Array(3);
    const o4 = new Float64Array(4);
    const gp = new Float64Array(3);

    let bad = null;
    let checks = 0;

    function cmp(what, a, b, tol) {
      checks++;
      if (bad) return;
      for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (!(d <= (tol || 0))) {
          bad = what + ': GLSL [' + Array.from(a).join(', ') + '] vs JS [' +
            Array.from(b).join(', ') + ']';
          return;
        }
      }
    }

    const trials = N <= 32 ? 6000 : 3000;
    for (let s = 0; s < trials && !bad; s++) {
      const cx = ((rnd() * 4 - 2) * N) | 0;
      const cy = ((rnd() * 4 - 2) * N) | 0;
      const cz = ((rnd() * 4 - 2) * N) | 0;

      /* cellToTexel */
      const gt = vm.call('cellToTexel', [vm.ivec3(cx, cy, cz)]);
      cell[0] = cx; cell[1] = cy; cell[2] = cz;
      M.cellToTexel(cell, texel);
      cmp('cellToTexel(' + [cx, cy, cz] + ')', gt.v, texel, 0);

      /* texelToCell */
      const tx = (rnd() * lay.width) | 0;
      const ty = (rnd() * lay.height) | 0;
      const gc = vm.call('texelToCell', [vm.ivec2(tx, ty)]);
      texel[0] = tx; texel[1] = ty;
      M.texelToCell(texel, o3);
      cmp('texelToCell(' + [tx, ty] + ')', gc.v, o3, 0);

      /* cellToUV, on fractional cells too */
      const fx = R((rnd() * 4 - 2) * N);
      const fy = R((rnd() * 4 - 2) * N);
      const fz = R((rnd() * 4 - 2) * N);
      const guv = vm.call('cellToUV', [vm.vec3(fx, fy, fz)]);
      cell[0] = fx; cell[1] = fy; cell[2] = fz;
      M.cellToUV(cell, o2);
      cmp('cellToUV(' + [fx, fy, fz] + ')', guv.v, o2, 0);

      /* worldToGrid / gridToWorld */
      const L = R([1, 2, 3.5, 0.25][(rnd() * 4) | 0]);
      const px = R((rnd() - 0.5) * 4 * L);
      const py = R((rnd() - 0.5) * 4 * L);
      const pz = R((rnd() - 0.5) * 4 * L);
      const gw = vm.call('worldToGrid', [vm.vec3(px, py, pz), vm.f(L)]);
      gp[0] = px; gp[1] = py; gp[2] = pz;
      M.worldToGrid(gp, L, o3);
      cmp('worldToGrid', gw.v, o3, 0);

      const gb = vm.call('gridToWorld', [vm.vec3(gw.v[0], gw.v[1], gw.v[2]), vm.f(L)]);
      o3[0] = gw.v[0]; o3[1] = gw.v[1]; o3[2] = gw.v[2];
      M.gridToWorld(o3, L, o3);
      cmp('gridToWorld', gb.v, o3, 0);

      /* sampleGrid */
      const sx = R((rnd() * 6 - 3) * N);
      const sy = R((rnd() * 6 - 3) * N);
      const sz = R((rnd() * 6 - 3) * N);
      const gs = vm.call('sampleGrid', [sam, vm.vec3(sx, sy, sz)]);
      gp[0] = sx; gp[1] = sy; gp[2] = sz;
      M.sampleGrid(tex, gp, o4);
      cmp('sampleGrid(' + [sx, sy, sz] + ')', gs.v, o4, 0);

      /* gradGrid */
      const h = R([1, 0.5, 0.25][(rnd() * 3) | 0]);
      const gg = vm.call('gradGrid', [sam, vm.vec3(sx, sy, sz), vm.f(h)]);
      M.gradGrid(tex, gp, h, o3);
      cmp('gradGrid(' + [sx, sy, sz] + ', h=' + h + ')', gg.v, o3, 0);
    }

    ok(tag + ' ' + mode + ': executed GLSL agrees bit for bit with the JS mirror',
      bad === null, bad || (checks + ' comparisons'));
  }
}

/* One more thing the interpreter is good for: proving the chunk is type-clean.
   Mixing an int with a float, or reading an undeclared name, throws above — so
   a clean run over every function is itself an assertion. */
{
  let clean = true, why = '';
  for (const N of A.SIZES) {
    const lay = A.layout(N);
    try {
      const vm = makeVM(A.GLSL(lay), {
        fp32: false,
        texelFetch: () => [0, 0, 0, 0]
      });
      vm.call('sampleGrid', [vm.sampler(null), vm.vec3(0.5, 0.5, 0.5)]);
      vm.call('gradGrid', [vm.sampler(null), vm.vec3(0.5, 0.5, 0.5), vm.f(1)]);
      vm.call('cellToUV', [vm.vec3(1, 2, 3)]);
      vm.call('cellToTexel', [vm.ivec3(1, 2, 3)]);
      vm.call('texelToCell', [vm.ivec2(1, 2)]);
      vm.call('worldToGrid', [vm.vec3(0, 0, 0), vm.f(2)]);
    } catch (e) {
      clean = false;
      why = 'N=' + N + ': ' + e.message;
      break;
    }
  }
  ok('the chunk is type-clean: no int/float mixing, no undeclared names, ' +
    'every declaration matches its expression', clean, why);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

console.log('\n' + '-'.repeat(66));
console.log(failed === 0
  ? 'atlas: ' + passed + ' passed'
  : 'atlas: ' + passed + ' passed, ' + failed + ' FAILED');
process.exit(failed === 0 ? 0 : 1);
