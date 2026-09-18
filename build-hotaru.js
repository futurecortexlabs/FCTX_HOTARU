/* Assembles Hotaru into one self-contained HTML file at dist/hotaru.html.
   Every module is an IIFE publishing a global, so the bundle is concatenation.
   Optional parts are allowed to be missing while they are still being built —
   the engine carries a fallback noise field and the app falls back to its own
   point cloud, so a partial build still runs and can be smoke-tested. */
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const MODULES = [
  { file: 'hotaru/noise.js',  required: false, note: 'simplex + curl noise (engine falls back to value noise)' },
  { file: 'hotaru/shapes.js', required: false, note: 'procedural shapes (app falls back to a drifting cloud)' },
  { file: 'hotaru/mask.js',   required: false, note: 'glyph mask sampler (text forming is disabled without it)' },
  { file: 'hotaru/atlas.js',  required: false, note: '3D-grid-in-2D-texture mapping (self-gravity is disabled without it)' },
  { file: 'hotaru/engine.js', required: true,  note: 'WebGL2 GPGPU particle field' },
];

const read = (rel) => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const has = (rel) => fs.existsSync(path.join(ROOT, rel));
const guard = (s) => s.replace(/<\/script/gi, '<\\/script');

const parts = [];
const missing = [];
let bytes = 0;

for (const m of MODULES) {
  if (!has(m.file)) {
    if (m.required) throw new Error('missing required module: ' + m.file);
    missing.push(m);
    parts.push('/* ' + m.file + ' — not present at build time (' + m.note + ') */');
    continue;
  }
  const src = read(m.file);
  bytes += Buffer.byteLength(src);
  const rule = '─'.repeat(Math.max(3, 60 - m.file.length));
  parts.push('/* ─── ' + m.file + ' ' + rule + ' */\n' + src.trim());
}

const shell = read('web/hotaru-shell.html');
const app = read('web/hotaru-app.js');
if (!shell.includes('/*__HOTARU_BUNDLE__*/')) throw new Error('shell is missing the bundle marker');
if (!shell.includes('/*__HOTARU_APP__*/')) throw new Error('shell is missing the app marker');

const out = shell
  .replace('/*__HOTARU_BUNDLE__*/', () => guard(parts.join('\n\n')))
  .replace('/*__HOTARU_APP__*/', () => guard(app.trim()));

fs.mkdirSync(path.join(ROOT, 'docs'), { recursive: true });
const dest = path.join(ROOT, 'docs', 'index.html');
fs.writeFileSync(dest, out, 'utf8');

/* The published page is wrapped by the Artifact skeleton, which supplies the
   charset, the viewport meta (including viewport-fit=cover) and a small reset.
   Locally there is no wrapper, so without this the phone media queries never
   match and the harness tests a layout nobody will ever see. */
const preview = [
  '<!doctype html>',
  '<html lang="ja"><head>',
  '<meta charset="utf-8">',
  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
  '<style>',
  ':root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}',
  'body{margin:0;font:14px system-ui,sans-serif;background:#fbfbfa}',
  'img{max-width:100%}[hidden]{display:none!important}',
  '</style>',
  '</head><body>',
  out,
  '</body></html>',
].join('\n');
fs.writeFileSync(path.join(ROOT, 'docs', 'preview.html'), preview, 'utf8');

const kb = (n) => (n / 1024).toFixed(1) + ' KB';
for (const m of MODULES) {
  const state = has(m.file) ? String(read(m.file).split('\n').length).padStart(6) + ' lines' : '     — missing';
  console.log('  ' + m.file.padEnd(20) + state);
}
console.log('  ' + 'web/hotaru-app.js'.padEnd(20) + String(app.split('\n').length).padStart(6) + ' lines');
if (missing.length) console.log('\nPARTIAL BUILD — still missing: ' + missing.map((m) => m.file).join(', '));
console.log('\nbuilt docs/index.html  ' + kb(Buffer.byteLength(out)) + '  (modules ' + kb(bytes) + ')');
