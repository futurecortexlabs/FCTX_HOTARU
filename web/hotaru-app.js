/* Hotaru — page glue. Owns the canvas lifecycle, turns typed text into a point
   cloud, switches scenes, forwards the pointer, and keeps the frame rate
   honest by dropping the particle count when the device cannot keep up. */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var SHAPES = [
    { key: 'free',      label: '蛍' },
    { key: 'sphere',    label: '球' },
    { key: 'galaxy',    label: '銀河' },
    { key: 'torusKnot', label: '結び目' },
    { key: 'wave',      label: '波' },
    { key: 'ring',      label: '環' },
    { key: 'heart',     label: 'ハート' },
    { key: 'helix',     label: '螺旋' }
  ];

  /* brightness is per-particle and additive, so it tracks how tightly the scene
     packs its points: a flat glyph concentrates every particle into a fraction
     of the screen and needs far less of it than a cloud filling the frame. */
  var PRESET = {
    free: {
      spring: 1.00, damp: 0.952, noiseAmp: 0.62, noiseScale: 1.05, flowSpeed: 0.070,
      spinSpeed: 0.042, pointScale: 7.6, brightness: 0.175, bloomAmount: 0.60, bloomThreshold: 0.44,
      exposure: 1.26, dist: 3.90, tilt: 0.16, pointerRadius: 0.44, pointerPush: 24, pointerSwirl: 17
    },
    /* spring and damp are a mass-spring pair: damp is applied as pow(damp, dt*60),
       so the continuous damping is c = -60*ln(damp). Critical damping is
       c = 2*sqrt(spring), and these are tuned to sit just at it — overdamped and
       the shape stays a smear for seconds, underdamped and it wobbles. */
    shape: {
      spring: 10.0, damp: 0.900, noiseAmp: 0.15, noiseScale: 1.95, flowSpeed: 0.135,
      spinSpeed: 0.075, pointScale: 6.8, brightness: 0.075, bloomAmount: 0.62, bloomThreshold: 0.52,
      exposure: 1.20, dist: 2.50, tilt: 0.18, pointerRadius: 0.40, pointerPush: 28, pointerSwirl: 18
    },
    text: {
      spring: 18.0, damp: 0.872, noiseAmp: 0.055, noiseScale: 2.7, flowSpeed: 0.17,
      spinSpeed: 0.0, pointScale: 4.8, brightness: 0.100, bloomAmount: 0.58, bloomThreshold: 0.34,
      exposure: 1.14, dist: 2.60, tilt: 0.0, pointerRadius: 0.36, pointerPush: 30, pointerSwirl: 20
    },
    /* Released. Nothing holds the shape any more: the spring is off, damping is
       almost absent so the dynamics are the simulation's rather than a brake's,
       and the pointer becomes an attractor instead of a repulsor. */
    /* GM is set from the free-fall time: a sphere of radius 1 measured a
       collapse in 8s at GM = 0.056, against 6.2s predicted, so the constant is
       right and the pacing is deliberate. noiseAmp MUST stay at zero here: with
       damping this light, a continuous random force is a heat source and the
       cloud evaporates instead of falling. The spin is about two thirds of the
       circular velocity, which flattens the remnant into a disc. */
    gravity: {
      spring: 0, damp: 0.9985, noiseAmp: 0.0, noiseScale: 2.2, flowSpeed: 0.05,
      spinSpeed: 0.050, pointScale: 6.0, brightness: 0.075, bloomAmount: 0.72, bloomThreshold: 0.52,
      exposure: 1.22, dist: 3.60, tilt: 0.22, pointerRadius: 0.50, pointerPush: -14, pointerSwirl: 6,
      gravity: 1, GM: 0.056, forceClamp: 6.0, relax: 14
    }
  };
  for (var _k in PRESET) if (PRESET[_k].gravity === undefined) PRESET[_k].gravity = 0;

  /* Each shape packs its points differently — a thin knot lights a few thousand
     pixels, a galaxy bulge lights a few hundred — so exposure and the viewing
     angle are set per shape rather than shared. */
  var SHAPE_TWEAK = {
    sphere:    { brightness: 0.070, tilt: 0.20 },
    galaxy:    { brightness: 0.042, tilt: 0.46, bloomThreshold: 0.52, dist: 2.35, noiseAmp: 0.075 },
    torusKnot: { brightness: 0.105, tilt: 0.26, noiseAmp: 0.10 },
    wave:      { brightness: 0.085, tilt: 0.56, dist: 2.30, noiseAmp: 0.09 },
    ring:      { brightness: 0.072, tilt: 0.52, dist: 2.35, noiseAmp: 0.08 },
    heart:     { brightness: 0.070, tilt: 0.10, dist: 2.45, noiseAmp: 0.10 },
    helix:     { brightness: 0.110, tilt: 0.12, dist: 2.40, noiseAmp: 0.09 }
  };

  var FOV = 0.85;

  var TIERS = [1048576, 524288, 262144, 131072, 65536];

  var engine = null;
  var mode = 'free';
  var scene = 'free';
  var word = '';
  var tier = 0;
  var probeAt = 0;
  var drops = 0;
  var last = 0;
  var running = true;
  var burstUntil = 0;
  var finePointer = false;

  /* ── deterministic rng ─────────────────────────────────────────────── */

  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* A drifting ball of embers: the resting state, and the shape the name means. */
  function cloud(n, seed) {
    var out = new Float32Array(n * 3);
    var r = rng(seed || 7);
    for (var i = 0; i < n; i++) {
      var u = r() * 2 - 1, phi = r() * Math.PI * 2;
      var rad = 0.35 + Math.pow(r(), 0.62) * 0.92;
      var s = Math.sqrt(Math.max(0, 1 - u * u));
      out[i * 3]     = rad * s * Math.cos(phi);
      out[i * 3 + 1] = rad * u * 0.78;
      out[i * 3 + 2] = rad * s * Math.sin(phi);
    }
    return out;
  }

  /* ── text -> point cloud ───────────────────────────────────────────── */

  var FONT = '700 %spx "Zen Kaku Gothic New", "Hiragino Sans", "Noto Sans JP", ' +
    'system-ui, -apple-system, sans-serif';

  function rasterize(text) {
    var size = 200;
    var probe = document.createElement('canvas').getContext('2d');
    probe.font = FONT.replace('%s', size);
    var m = probe.measureText(text);
    var wRaw = Math.max(1, m.width);
    var ascent = m.actualBoundingBoxAscent || size * 0.82;
    var descent = m.actualBoundingBoxDescent || size * 0.22;
    var hRaw = Math.max(1, ascent + descent);

    // Keep the mask under a sane pixel budget however wide the word is.
    var MAXW = 1600, MAXH = 520;
    var k = Math.min(MAXW / (wRaw + size * 0.3), MAXH / (hRaw + size * 0.3), 1);
    size = Math.max(48, Math.floor(size * k));

    var ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
    ctx.font = FONT.replace('%s', size);
    m = ctx.measureText(text);
    ascent = m.actualBoundingBoxAscent || size * 0.82;
    descent = m.actualBoundingBoxDescent || size * 0.22;
    var pad = Math.round(size * 0.14);
    var w = Math.max(2, Math.ceil(m.width) + pad * 2);
    var h = Math.max(2, Math.ceil(ascent + descent) + pad * 2);

    ctx.canvas.width = w;
    ctx.canvas.height = h;
    ctx.font = FONT.replace('%s', size);
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#fff';
    ctx.fillText(text, pad, pad + ascent);

    var px = ctx.getImageData(0, 0, w, h).data;
    var alpha = new Uint8Array(w * h);
    for (var i = 0, n = w * h; i < n; i++) alpha[i] = px[i * 4 + 3];
    return { alpha: alpha, w: w, h: h };
  }

  /* The sampler maps the glyph box into [-fit, fit] on its longer axis, so the
     right fit depends on the viewport: the same word has to be much smaller on
     a tall phone than on a wide laptop to stay on screen. */
  function fitForMask(mw, mh) {
    var c = $('stage');
    var aspect = Math.max(c.clientWidth, 1) / Math.max(c.clientHeight, 1);
    var halfH = PRESET.text.dist * Math.tan(FOV / 2);
    var halfW = halfH * aspect;
    var rx = mw >= mh ? 1 : mw / mh;
    var ry = mw >= mh ? mh / mw : 1;
    return Math.max(0.2, Math.min(0.88 * halfW / rx, 0.80 * halfH / ry));
  }

  function textTargets(text, count) {
    if (!globalThis.HotaruMask || typeof HotaruMask.sampleMask !== 'function') return null;
    var r = rasterize(text);
    var lit = 0;
    for (var i = 0; i < r.alpha.length; i++) if (r.alpha[i] > 16) lit++;
    if (!lit) return null;
    return HotaruMask.sampleMask(r.alpha, r.w, r.h, count, {
      seed: 0x3C5A11, threshold: 16, fit: fitForMask(r.w, r.h), jitter: 0.55, depth: 0.018
    });
  }

  /* ── scenes ────────────────────────────────────────────────────────── */

  /* A portrait phone sees a far narrower slice of the world than a laptop, so
     the camera is pushed back until the field fits the *tighter* of the two
     axes. Without this the cloud spills off both sides and reads as flat noise. */
  function frameDistance(base, radius) {
    var c = $('stage');
    var aspect = Math.max(c.clientWidth, 1) / Math.max(c.clientHeight, 1);
    var t = Math.tan(FOV / 2);
    return Math.max(base, radius / (0.86 * t), radius / (0.86 * t * aspect));
  }

  function applyPreset(name, tweak, radius) {
    var p = PRESET[name], k;
    for (k in p) if (Object.prototype.hasOwnProperty.call(p, k)) engine.params[k] = p[k];
    if (tweak) for (k in tweak) if (Object.prototype.hasOwnProperty.call(tweak, k)) engine.params[k] = tweak[k];
    if (radius) engine.params.dist = frameDistance(engine.params.dist, radius);
  }

  /* The heart costs ~0.5s for a million points, so hold on to the last two
     buffers rather than rebuilding on every trip back to a shape. */
  var cacheKeys = [], cacheVals = [];
  function shapeTargets(key, count) {
    var ck = key + ':' + count;
    var hit = cacheKeys.indexOf(ck);
    if (hit > -1) return cacheVals[hit];
    var out;
    var S = globalThis.HotaruShapes;
    if (key === 'free') out = cloud(count, 7);
    else if (S && typeof S[key] === 'function') {
      try { out = S[key](count, { seed: 0x5EED }); } catch (e) { out = cloud(count, 11); }
    } else out = cloud(count, 11);
    cacheKeys.unshift(ck); cacheVals.unshift(out);
    cacheKeys.length = Math.min(cacheKeys.length, 2);
    cacheVals.length = Math.min(cacheVals.length, 2);
    return out;
  }

  function sceneRadius(key) { return key === 'free' ? 1.30 : 1.02; }

  function setScene(key) {
    scene = key;
    word = '';
    $('word').value = '';
    mode = 'free';
    applyPreset(key === 'free' ? 'free' : 'shape', SHAPE_TWEAK[key], sceneRadius(key));
    engine.setTargets(shapeTargets(key, engine.count));
    paintChips();
  }

  function setWord(text) {
    word = text;
    if (!text) { setScene(scene); return; }
    var pts = textTargets(text, engine.count);
    if (!pts) { setScene(scene); return; }
    mode = 'text';
    applyPreset('text');
    engine.setTargets(pts);
    paintChips();
  }

  function refreshTargets() {
    if (mode === 'text' && word) {
      var pts = textTargets(word, engine.count);
      if (pts) { engine.setTargets(pts); return; }
    }
    engine.setTargets(shapeTargets(scene, engine.count));
  }

  function paintChips() {
    var host = $('chips');
    var bs = host.querySelectorAll('button');
    for (var i = 0; i < bs.length; i++) {
      bs[i].setAttribute('aria-pressed',
        String(mode === 'free' && bs[i].getAttribute('data-key') === scene));
    }
    var rel = $('release');
    if (!engine || !engine.hasGravity()) { rel.hidden = true; return; }
    rel.hidden = false;
    rel.textContent = mode === 'gravity' ? '戻す' : '放つ';
    rel.className = mode === 'gravity' ? 'release on' : 'release';
  }

  /* ── release ───────────────────────────────────────────────────────── */

  var releasedAt = 0;
  var gravSampleAt = 0;
  var distTarget = 0;

  function release() {
    if (!engine.hasGravity()) return;
    prevMode = mode; prevWord = word;
    mode = 'gravity';
    applyPreset('gravity', null, 1.55);
    // A little random dispersion so the cloud fragments into filaments instead
    // of falling into one ball, and a little spin so the remnant forms a disc.
    engine.setGravity(true, { sigma: 0.050, spin: 0.160 });
    releasedAt = performance.now();
    gravSampleAt = 0;
    distTarget = 0;
    paintChips();
  }

  function unrelease() {
    engine.setGravity(false);
    mode = 'free';
    releasedAt = 0;
    if (prevMode === 'text' && prevWord) { $('word').value = prevWord; setWord(prevWord); }
    else setScene(scene);
    paintChips();
  }

  var prevMode = 'free', prevWord = '';

  /* ── hud ───────────────────────────────────────────────────────────── */

  var hudAt = 0;
  function hud(t) {
    if (t - hudAt < 400) return;
    hudAt = t;
    var extra = mode === 'gravity'
      ? '<br>重力 ' + ((t - releasedAt) / 1000).toFixed(0) + ' 秒'
      : '';
    $('hud').innerHTML =
      '<b>' + engine.count.toLocaleString('ja-JP') + '</b> 粒<br>' +
      Math.round(engine.fps) + ' fps' + extra;
  }

  /* ── pointer ───────────────────────────────────────────────────────── */

  function ndc(e) {
    var c = $('stage');
    var r = c.getBoundingClientRect();
    return [
      ((e.clientX - r.left) / Math.max(r.width, 1)) * 2 - 1,
      -(((e.clientY - r.top) / Math.max(r.height, 1)) * 2 - 1)
    ];
  }

  function bindPointer() {
    var c = $('stage');
    var down = false;

    c.addEventListener('pointerdown', function (e) {
      down = true;
      if (c.setPointerCapture) { try { c.setPointerCapture(e.pointerId); } catch (err) {} }
      var p = ndc(e);
      engine.setPointer(p[0], p[1], 1);
      finePointer = e.pointerType === 'mouse';
    });

    c.addEventListener('pointermove', function (e) {
      var p = ndc(e);
      if (down) engine.setPointer(p[0], p[1], 1);
      else if (e.pointerType === 'mouse') engine.setPointer(p[0], p[1], 0.32);
    });

    var release = function () { down = false; engine.setPointer(engine.pointer.x, engine.pointer.y, 0); };
    c.addEventListener('pointerup', release);
    c.addEventListener('pointercancel', release);
    c.addEventListener('pointerleave', function (e) { if (!down) release(); });

    c.addEventListener('dblclick', function (e) {
      var p = ndc(e);
      engine.setPointer(p[0], p[1], 3.2);
      burstUntil = performance.now() + 220;
    });
  }

  /* ── loop ──────────────────────────────────────────────────────────── */

  function tick(t) {
    requestAnimationFrame(tick);
    if (!running || !engine) return;
    var dt = last ? t - last : 16;
    last = t;

    if (burstUntil && t > burstUntil) { burstUntil = 0; engine.setPointer(engine.pointer.x, engine.pointer.y, 0); }

    /* Follow the collapse. The cloud shrinks by a factor of five on its way
       in, so a fixed camera either clips it at the start or loses it at the
       end. Radius comes from a small readback twice a second. */
    if (mode === 'gravity') {
      if (t - gravSampleAt > 500) {
        gravSampleAt = t;
        try {
          var rad = engine.sampleRadius(32);
          if (isFinite(rad.rms) && rad.rms > 0) {
            distTarget = frameDistance(1.85, Math.max(0.3, rad.rms * 2.0));
          }
        } catch (err) { distTarget = 0; }
      }
      if (distTarget) {
        engine.params.dist += (distTarget - engine.params.dist) * Math.min(1, dt / 1000 * 0.7);
      }
    }

    if (mode === 'text') {
      // Bring the field face-on so the word stays readable.
      var s = engine.spin % (Math.PI * 2);
      if (s > Math.PI) s -= Math.PI * 2;
      if (s < -Math.PI) s += Math.PI * 2;
      engine.spin -= s * Math.min(1, dt / 1000 * 3.2);
    }

    engine.resize();
    engine.frame(dt);
    hud(t);

    // Give the GPU a few seconds, then step down if it is clearly struggling.
    if (!probeAt) probeAt = t + 3000;
    else if (t > probeAt && drops < 3 && mode !== 'gravity') {
      if (engine.fps < 40 && tier < TIERS.length - 1) {
        tier++; drops++;
        engine.setCount(TIERS[tier]);
        refreshTargets();
      }
      probeAt = t + 2600;
    }
  }

  /* ── boot ──────────────────────────────────────────────────────────── */

  function fail(err) {
    $('fail').hidden = false;
    $('fail-msg').textContent = /WebGL2/.test(err.message)
      ? 'この端末またはブラウザが WebGL2 に対応していないようです。'
      : '描画の初期化に失敗しました。';
    $('fail-detail').textContent = err.message;
  }

  function buildChips() {
    var host = $('chips');
    var S = globalThis.HotaruShapes;
    var html = '';
    for (var i = 0; i < SHAPES.length; i++) {
      var s = SHAPES[i];
      if (s.key !== 'free' && !(S && typeof S[s.key] === 'function')) continue;
      html += '<button type="button" data-key="' + s.key + '" aria-pressed="' +
        (s.key === scene) + '">' + s.label + '</button>';
    }
    host.innerHTML = html;
    paintChips();
    host.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('button[data-key]') : null;
      if (b) setScene(b.getAttribute('data-key'));
    });
  }

  function bindInput() {
    var input = $('word');
    var timer;
    input.addEventListener('input', function () {
      clearTimeout(timer);
      var v = input.value.trim();
      timer = setTimeout(function () {
        if (v) setWord(v); else setScene(scene);
      }, 280);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        clearTimeout(timer);
        var v = input.value.trim();
        if (v) setWord(v);
        input.blur();
      }
    });
    $('release').addEventListener('click', function () {
      if (mode === 'gravity') unrelease(); else release();
    });
    $('clear').addEventListener('click', function () {
      input.value = '';
      setScene(scene);
      input.focus();
    });
  }

  // ?n=65536 (or #n=65536) pins the particle count — used when driving the page
  // from a test harness, and handy on a machine you want to go easy on.
  function requestedCount() {
    var m = /[?#&]n=(\d+)/.exec(location.search + location.hash);
    if (!m) return 0;
    var n = parseInt(m[1], 10);
    return n >= 1024 && n <= TIERS[0] ? n : 0;
  }

  function start() {
    var coarse = false;
    try { coarse = window.matchMedia('(pointer: coarse)').matches; } catch (e) {}
    var small = Math.min(window.innerWidth, window.innerHeight) < 620;
    tier = (coarse || small) ? 2 : 0;

    var pinned = requestedCount();
    try {
      engine = new HotaruEngine($('stage'), {
        count: pinned || TIERS[tier],
        // Only a harness needs to read the canvas back after a frame is presented.
        preserveBuffer: /[?#&]grab=1/.test(location.search + location.hash)
      });
    } catch (err) {
      fail(err);
      return;
    }
    if (pinned) drops = 3;

    globalThis.__hotaru = {
      get engine() { return engine; },
      get mode() { return mode; },
      get scene() { return scene; },
      setWord: setWord,
      setScene: setScene
    };

    applyPreset('free', null, sceneRadius('free'));
    engine.setTargets(shapeTargets('free', engine.count));

    buildChips();
    bindInput();
    bindPointer();

    document.addEventListener('visibilitychange', function () {
      running = !document.hidden;
      last = 0;
    });

    // How large a word can be drawn depends on the viewport, so re-cut it when
    // the window changes shape (or a phone is turned on its side).
    var resizeTimer;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        if (mode === 'text' && word) setWord(word);
        else applyPreset(scene === 'free' ? 'free' : 'shape', SHAPE_TWEAK[scene], sceneRadius(scene));
      }, 260);
    });

    // Re-cut the glyphs once the real typeface has arrived.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { if (mode === 'text' && word) setWord(word); });
    }

    requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
