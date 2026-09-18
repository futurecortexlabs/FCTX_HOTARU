/* Focused check: at a million particles, does the text field keep what was
   typed, and does the field agree with what the particles are spelling? */
const path = require('path');
const pup = require('puppeteer-core');
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const FILE = path.join(__dirname, '..', 'docs', 'preview.html').split(path.sep).join('/');

(async () => {
  const b = await pup.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--ignore-gpu-blocklist', '--enable-gpu-rasterization',
      '--use-angle=default', '--window-size=1440,900'],
    defaultViewport: { width: 1440, height: 900 },
  });
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', (e) => errs.push(e.message));
  await p.goto('file:///' + FILE + '?n=1048576', { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, 3000));

  const snap = async (tag) => {
    const s = await p.evaluate(() => ({
      value: document.getElementById('word').value,
      mode: globalThis.__hotaru.mode,
      scene: globalThis.__hotaru.scene,
      fps: Math.round(globalThis.__hotaru.engine.fps),
      n: globalThis.__hotaru.engine.count,
    }));
    console.log(tag.padEnd(24), JSON.stringify(s));
    return s;
  };

  await snap('boot');
  await p.click('#word');
  await p.type('#word', 'HOTARU', { delay: 30 });
  await snap('just typed');
  await new Promise((r) => setTimeout(r, 4000));
  await snap('4s later');
  await p.screenshot({ path: path.join(__dirname, '..', 'docs', 'shots', 'probe-typed.png') });

  // Reproduce exactly what the harness did before the odd screenshot.
  await p.$eval('#word', (el) => { el.value = ''; el.dispatchEvent(new Event('input', { bubbles: true })); });
  await new Promise((r) => setTimeout(r, 600));
  await p.click('#word');
  await p.type('#word', 'ゆうき', { delay: 30 });
  await snap('cleared + retyped');
  await new Promise((r) => setTimeout(r, 4000));
  await snap('4s later');
  await p.screenshot({ path: path.join(__dirname, '..', 'docs', 'shots', 'probe-retyped.png') });

  console.log('page errors:', errs.length, errs.slice(0, 3));
  await b.close();
})();
