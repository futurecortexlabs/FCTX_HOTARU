/* Does the cloud actually fall inward?
   Starts from a sphere with no noise and no release kick, turns gravity on, and
   measures the cloud radius once a second. Collapse must show a falling RMS;
   a rising one means the force is repulsive or the solver is wrong. */
const path = require('path');
const pup = require('puppeteer-core');
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const FILE = path.join(__dirname, '..', 'docs', 'preview.html').split(path.sep).join('/');

const GM = Number((process.argv.find((a) => a.startsWith('--gm=')) || '--gm=0.056').slice(5));
const N = (process.argv.find((a) => a.startsWith('--n=')) || '--n=262144').slice(4);
const SECONDS = Number((process.argv.find((a) => a.startsWith('--t=')) || '--t=16').slice(4));

(async () => {
  const b = await pup.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--ignore-gpu-blocklist', '--enable-gpu-rasterization',
      '--use-angle=default', '--window-size=1280,800'],
    defaultViewport: { width: 1280, height: 800 },
  });
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', (e) => errs.push(e.message));
  await p.goto('file:///' + FILE + '?n=' + N, { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, 2500));

  // A clean initial condition: a sphere, at rest, with nothing but gravity.
  await p.evaluate((gm) => {
    const h = globalThis.__hotaru;
    h.setScene('sphere');
    return new Promise((res) => setTimeout(() => {
      h.engine.params.GM = gm;
      h.engine.params.noiseAmp = 0;
      h.engine.params.spring = 0;
      h.engine.params.damp = 1.0;
      h.engine.setGravity(true, { sigma: 0, spin: 0 });
      res();
    }, 2200));
  }, GM);

  const probe = () => p.evaluate(() => {
    const e = globalThis.__hotaru.engine;
    const r = e.sampleRadius(64);
    return { rms: +r.rms.toFixed(4), mean: +r.mean.toFixed(4), max: +r.max.toFixed(4),
      bad: r.nonFinite, fps: Math.round(e.fps), n: e.count, gm: e.params.GM, jac: e.params.jacobi };
  });

  const first = await probe();
  console.log('GM=' + GM + '  n=' + first.n + '  jacobi=' + first.jac + '  fps=' + first.fps);
  console.log('  t      rms     mean      max   nonfinite');
  console.log('  0.0  ' + String(first.rms).padStart(7) + String(first.mean).padStart(9) +
    String(first.max).padStart(9) + String(first.bad).padStart(8));
  let prev = first.rms;
  for (let t = 1; t <= SECONDS; t++) {
    await new Promise((r) => setTimeout(r, 1000));
    const s = await probe();
    const arrow = s.rms < prev - 1e-4 ? ' contracting' : s.rms > prev + 1e-4 ? ' EXPANDING' : ' flat';
    console.log('  ' + String(t).padStart(3) + '.0  ' + String(s.rms).padStart(7) +
      String(s.mean).padStart(9) + String(s.max).padStart(9) + String(s.bad).padStart(8) + arrow);
    prev = s.rms;
  }
  console.log('page errors:', errs.length, errs.slice(0, 2));
  await b.close();
})();
