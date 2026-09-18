// Independent check: do sampled points actually land on the lit pixels?
require('../hotaru/mask.js');
const { sampleMask } = globalThis.HotaruMask;

function mask(mw, mh, fn) {
  const a = new Uint8Array(mw * mh);
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) a[y * mw + x] = fn(x, y) ? 255 : 0;
  return a;
}

function check(name, mw, mh, fn, count) {
  const alpha = mask(mw, mh, fn);
  const fit = 1.0;
  const out = sampleMask(alpha, mw, mh, count, { seed: 5, threshold: 16, fit, jitter: 0.5, depth: 0 });
  const long = Math.max(mw, mh);
  const ex = (mw / long) * fit, ey = (mh / long) * fit;
  let hit = 0, near = 0, outside = 0;
  let minx = 9, maxx = -9, miny = 9, maxy = -9;
  for (let i = 0; i < count; i++) {
    const X = out[i * 3], Y = out[i * 3 + 1];
    minx = Math.min(minx, X); maxx = Math.max(maxx, X);
    miny = Math.min(miny, Y); maxy = Math.max(maxy, Y);
    const px = Math.floor((X + ex) / (2 * ex) * mw);
    const py = Math.floor((ey - Y) / (2 * ey) * mh);
    if (px < 0 || py < 0 || px >= mw || py >= mh) { outside++; continue; }
    if (alpha[py * mw + px] > 16) hit++;
    else {
      let ok = false;
      for (let dy = -1; dy <= 1 && !ok; dy++) for (let dx = -1; dx <= 1 && !ok; dx++) {
        const qx = px + dx, qy = py + dy;
        if (qx >= 0 && qy >= 0 && qx < mw && qy < mh && alpha[qy * mw + qx] > 16) ok = true;
      }
      if (ok) near++; else outside++;
    }
  }
  const pct = (n) => ((n / count) * 100).toFixed(2) + '%';
  console.log(name.padEnd(26),
    'exact=' + pct(hit).padStart(7),
    'within1px=' + pct(near).padStart(7),
    'OFF=' + pct(outside).padStart(7),
    ' bbox x[' + minx.toFixed(3) + ',' + maxx.toFixed(3) + '] y[' + miny.toFixed(3) + ',' + maxy.toFixed(3) + ']',
    ' expect x±' + ex.toFixed(3) + ' y±' + ey.toFixed(3));
  return hit + near;
}

const N = 60000;
check('vertical bar 12px', 400, 200, (x) => x >= 60 && x < 72, N);
check('thin diagonal', 300, 300, (x, y) => Math.abs(x - y) < 2, N);
check('ring', 300, 300, (x, y) => { const d = Math.hypot(x - 150, y - 150); return d > 90 && d < 110; }, N);
check('L (asymmetric)', 400, 200, (x, y) => (x < 40 && y < 180) || (y > 140 && y < 180 && x < 240), N);
check('two separated blobs', 400, 160, (x, y) => Math.hypot(x - 80, y - 80) < 40 || Math.hypot(x - 320, y - 80) < 40, N);
check('tall text-ish', 160, 400, (x, y) => x > 40 && x < 80 && y > 40 && y < 360, N);
