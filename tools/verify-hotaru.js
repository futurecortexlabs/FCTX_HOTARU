/* Drives dist/hotaru.html in the installed Chrome with a real (software) WebGL2
   stack, so shader compile errors, framebuffer problems and JS faults surface
   here instead of in front of an audience.

   node verify-hotaru.js [--headful] [--n=65536] */
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const ROOT = path.join(__dirname, '..');
const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  process.env.LOCALAPPDATA + '/Google/Chrome/Application/chrome.exe',
].find((p) => p && fs.existsSync(p));

const argv = process.argv.slice(2);
const headful = argv.includes('--headful');
const nArg = (argv.find((a) => a.startsWith('--n=')) || '--n=65536').slice(4);

const SHOTS = path.join(ROOT, 'docs', 'shots');

(async () => {
  if (!CHROME) throw new Error('Chrome not found');
  fs.mkdirSync(SHOTS, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: headful ? false : 'new',
    // --gpu runs on the real adapter to measure honest frame rates; the default
    // is SwiftShader, which is deterministic and catches shader errors but is
    // far too slow to say anything about performance.
    args: (argv.includes('--gpu')
      ? ['--no-sandbox', '--disable-dev-shm-usage', '--ignore-gpu-blocklist',
         '--enable-gpu-rasterization', '--use-angle=default']
      : ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader',
         '--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist']
    ).concat(['--window-size=1440,900']),
    defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
  });

  const page = await browser.newPage();
  const logs = [];
  const errors = [];
  page.on('console', (m) => logs.push(m.type().toUpperCase() + ': ' + m.text()));
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('requestfailed', (r) => {
    const u = r.url();
    if (!u.startsWith('data:')) logs.push('REQFAIL: ' + u + ' — ' + (r.failure() || {}).errorText);
  });

  // The preview build carries the same wrapper the Artifact skeleton adds, so
  // what runs here is what ships.
  const url = 'file:///' + path.join(ROOT, 'docs', 'preview.html').replace(/\\/g, '/') + '?n=' + nArg + '&grab=1';
  await page.goto(url, { waitUntil: 'load', timeout: 60000 });
  await new Promise((r) => setTimeout(r, 3500));

  const probe = async () => page.evaluate(() => {
    const h = globalThis.__hotaru;
    const failCard = document.getElementById('fail');
    const canvas = document.getElementById('stage');
    let glInfo = null;
    try {
      const gl = canvas.getContext('webgl2');
      const dbg = gl && gl.getExtension('WEBGL_debug_renderer_info');
      glInfo = gl ? {
        renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        floatRender: !!gl.getExtension('EXT_color_buffer_float'),
        drawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
        vertTexUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
        pointRange: Array.from(gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) || []),
        glError: gl.getError(),
      } : null;
    } catch (e) { glInfo = { error: e.message }; }
    return {
      failVisible: failCard ? !failCard.hidden : null,
      failText: failCard && !failCard.hidden ? document.getElementById('fail-detail').textContent : null,
      count: h && h.engine ? h.engine.count : null,
      fps: h && h.engine ? Math.round(h.engine.fps) : null,
      frames: h && h.engine ? h.engine.frames : null,
      mode: h ? h.mode : null,
      scene: h ? h.scene : null,
      simFmt: h && h.engine ? (h.engine.floatRender ? 'RGBA32F' : 'RGBA16F') : null,
      hud: (document.getElementById('hud') || {}).textContent,
      chips: Array.from(document.querySelectorAll('#chips button')).map((b) => b.textContent),
      shapesLoaded: typeof globalThis.HotaruShapes === 'object',
      maskLoaded: typeof globalThis.HotaruMask === 'object',
      noiseLoaded: typeof globalThis.HotaruNoise === 'object',
      usingRealNoise: !!(globalThis.HotaruNoise && String(globalThis.HotaruNoise.PRELUDE || '').includes('curlNoise')),
      glInfo,
    };
  });

  // Is anything actually being drawn? Count non-black pixels in the canvas.
  const luminance = async () => page.evaluate(() => {
    const c = document.getElementById('stage');
    const off = document.createElement('canvas');
    off.width = 240; off.height = 150;
    const ctx = off.getContext('2d');
    ctx.drawImage(c, 0, 0, off.width, off.height);
    const d = ctx.getImageData(0, 0, off.width, off.height).data;
    let lit = 0, sum = 0, max = 0;
    for (let i = 0; i < d.length; i += 4) {
      const l = (d[i] * 0.2126 + d[i + 1] * 0.7152 + d[i + 2] * 0.0722) / 255;
      sum += l; if (l > max) max = l;
      if (l > 0.08) lit++;
    }
    return { litFraction: +(lit / (d.length / 4)).toFixed(4), meanLuma: +(sum / (d.length / 4)).toFixed(4), maxLuma: +max.toFixed(3) };
  });

  const steps = [];
  const record = async (name) => {
    const p = await probe();
    const l = await luminance();
    await page.screenshot({ path: path.join(SHOTS, name + '.png') });
    steps.push({ step: name, ...p, ...l });
  };

  await record('01-idle');

  // Drag across the field.
  await page.mouse.move(500, 420);
  await page.mouse.down();
  for (let i = 0; i < 24; i++) await page.mouse.move(500 + i * 18, 420 + Math.sin(i / 3) * 90);
  await new Promise((r) => setTimeout(r, 400));
  await record('02-drag');
  await page.mouse.up();

  const setWord = async (text, settle) => {
    await page.$eval('#word', (el) => {
      el.value = '';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await new Promise((r) => setTimeout(r, 600));
    await page.click('#word');
    await page.type('#word', text, { delay: 30 });
    await new Promise((r) => setTimeout(r, settle || 3600));
  };

  await setWord('ゆうき');
  await record('03-text');

  await setWord('HOTARU');
  await record('04-text-latin');

  // Each shape chip.
  const chips = await page.$$('#chips button');
  for (let i = 1; i < chips.length; i++) {
    const label = await page.evaluate((el) => el.textContent, chips[i]);
    await chips[i].click();
    await new Promise((r) => setTimeout(r, 2300));
    await record('05-shape-' + i + '-' + label);
  }

  // Release: spring off, self-gravity on. Sample the collapse over time.
  await setWord('HOTARU', 2600);
  const relBtn = await page.$('#release');
  if (relBtn) {
    const hidden = await page.evaluate((el) => el.hidden, relBtn);
    if (hidden) {
      steps.push({ step: '07-gravity-UNAVAILABLE', failVisible: false, litFraction: 0, meanLuma: 0, maxLuma: 0 });
    } else {
      await relBtn.click();
      const t0 = Date.now();
      for (const at of [2000, 5000, 9000, 14000, 20000, 28000]) {
        const wait = at - (Date.now() - t0);
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        await record('07-gravity-' + String(at / 1000).padStart(2, '0') + 's');
      }
      // Drag during the collapse: the pointer is an attractor now.
      await page.mouse.move(700, 430);
      await page.mouse.down();
      for (let i = 0; i < 20; i++) await page.mouse.move(700 - i * 14, 430 + i * 6);
      await new Promise((r) => setTimeout(r, 1200));
      await record('08-gravity-drag');
      await page.mouse.up();
      await page.click('#release');
      await new Promise((r) => setTimeout(r, 2500));
      await record('09-after-reset');
    }
  }

  // Phone viewport.
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await new Promise((r) => setTimeout(r, 1600));
  await record('06-phone');

  await browser.close();

  const report = { steps, logs, errors };
  fs.writeFileSync(path.join(ROOT, 'docs', 'shots', 'verify.json'), JSON.stringify(report, null, 2));

  console.log('\n── environment ──');
  console.log(JSON.stringify(steps[0].glInfo, null, 2));
  console.log('modules  noise=' + steps[0].noiseLoaded + ' (real curl: ' + steps[0].usingRealNoise + ')' +
    '  shapes=' + steps[0].shapesLoaded + '  mask=' + steps[0].maskLoaded);
  console.log('chips    ' + JSON.stringify(steps[0].chips));

  console.log('\n── steps ──');
  for (const s of steps) {
    console.log(
      s.step.padEnd(22),
      'fail=' + String(s.failVisible).padEnd(6),
      'n=' + String(s.count).padEnd(8),
      'frames=' + String(s.frames).padEnd(6),
      'mode=' + String(s.mode).padEnd(6),
      'lit=' + String(s.litFraction).padEnd(8),
      'mean=' + String(s.meanLuma).padEnd(8),
      'max=' + s.maxLuma
    );
    if (s.failText) console.log('   FAIL: ' + s.failText);
  }

  console.log('\n── console (' + logs.length + ') ──');
  for (const l of logs.slice(0, 40)) console.log('  ' + l);
  console.log('\n── page errors (' + errors.length + ') ──');
  for (const e of errors) console.log('  ' + e);

  const dark = steps.filter((s) => s.litFraction < 0.002);
  const failed = steps.filter((s) => s.failVisible);
  console.log('\nVERDICT ' + JSON.stringify({
    pageErrors: errors.length,
    failCardShown: failed.length,
    blankSteps: dark.map((s) => s.step),
    shots: 'docs/shots',
  }));
  process.exit(errors.length || failed.length ? 1 : 0);
})().catch((e) => { console.error('HARNESS FAILED: ' + e.stack); process.exit(2); });
