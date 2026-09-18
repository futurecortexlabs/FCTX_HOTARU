<h1 align="center">蛍 &nbsp;Hotaru</h1>

<p align="center">
  A million particles become the word you type — then collapse under their own gravity.
</p>

<p align="center">
  <a href="https://futurecortexlabs.github.io/FCTX_HOTARU/"><b>▶ Live demo</b></a>
  &nbsp;·&nbsp;
  <a href="README.ja.md">日本語</a>
  &nbsp;·&nbsp;
  <a href="#the-gravity-is-real">The physics</a>
</p>

<p align="center">
  <img src="docs/media/collapse.gif" width="720" alt="The word HOTARU, written in a million particles, collapsing under self-gravity into filaments and a rotating disc">
</p>

<p align="center">
  <sub>Nothing here is keyframed. The spring holding the letters together is switched off and
  self-gravity switched on; 15 seconds of simulation, played at 2×.</sub>
</p>

---

## What it is

**One HTML file.** No dependencies, no bundler, no three.js. Open it and it runs.

- **Type anything** — a million GPU particles fly into that shape. Latin, Japanese, kanji, emoji.
- **Drag across the field** — they scatter, swirl, and settle back.
- **Press 放つ (Release)** — the spring is cut and the particles begin pulling on *each other*.
  The word sags under its own weight, fragments into gravitationally bound clumps, draws
  filaments between them, and settles into a rotating disc.
- **Seven procedural shapes** — sphere, galaxy, torus knot, wave, ring, heart, double helix.

Released twice, the same word never lands the same way. It is a simulation, not a playback.

<table>
<tr>
<td width="50%"><img src="docs/media/text-ja.png" alt="Japanese kana rendered in particles"></td>
<td width="50%"><img src="docs/media/galaxy.png" alt="A procedural spiral galaxy"></td>
</tr>
<tr>
<td><img src="docs/media/idle.png" alt="The resting state: a drifting cloud of embers"></td>
<td align="center"><img src="docs/media/mobile.png" width="240" alt="Running on a phone"></td>
</tr>
</table>

---

## Try it

**Online** — [futurecortexlabs.github.io/FCTX_HOTARU](https://futurecortexlabs.github.io/FCTX_HOTARU/)

**Offline** — clone and open `docs/index.html`. There is no server and no install step.

**From source**

```bash
node build-hotaru.js    # six modules + shell + glue  ->  docs/index.html
npm test                # 5,335 checks
```

Node is the only requirement. `puppeteer-core` is a dev dependency used solely by the
verification harness, which drives your installed Chrome.

Append `?n=65536` to the URL to pin the particle count on a modest machine.

---

## How it works

Every frame, on the GPU:

```
position / velocity textures  (RGBA32F, 1024x1024)
        |
        +- 1. deposit ---- a million particles drawn additively into a 64^3 grid,
        |                  held as z-slice tiles in one 512x512 texture
        +- 2. solve ------ grad^2 phi = 4 pi G rho, 24 Chebyshev-accelerated passes,
        |                  warm-started from the previous frame
        +- 3. force ------ -grad(phi) sampled back trilinearly to each particle
        |
        +- 4. integrate -- one fragment shader writes position AND velocity
        |                  through multiple render targets
        +- 5. draw ------- gl.POINTS pulled by gl_VertexID; no vertex buffer exists
        +- 6. post ------- auto-exposure -> bright pass -> separable blur -> ACES
```

There is no particle array and no vertex buffer. The whole field is one
`drawArrays(POINTS, 0, 1048576)` call whose vertex shader fetches its own particle from a
texture by `gl_VertexID`.

Exposure is automatic: the scene's mipmap chain is reduced to a single texel, smoothed over
time in a 1×1 buffer, and applied as gain at composite. A simulation whose density changes by
two orders of magnitude cannot be exposed by hand.

---

## The gravity is real

A direct million-body sum is 10¹² interactions per frame. Instead this uses the
**Particle-Mesh** method — the same approach cosmological N-body codes use.

1. **Deposit** each particle into its nearest grid cell, additively.
2. **Solve** the discrete Poisson equation for the potential. A periodic box has no unique
   solution, so the mean density is subtracted (the Jeans swindle).
3. **Differentiate** the potential and interpolate the force back to the particles.

### Why Chebyshev, and how that number was found

Plain Jacobi relaxation converges slowest for exactly the long-wavelength modes gravity cares
about. Rather than guess a pass count, this repository carries an **exact CPU reference** —
[`hotaru/pm.js`](hotaru/pm.js), a 3D-FFT Poisson solver — and
[`test/pm.test.js`](test/pm.test.js) measures what the GPU actually needs:

Passes per frame, warm-started from the previous frame, scored as the relative L2 error of the
acceleration interpolated back to the particles — against the FFT solution of the *same*
stencil, so this is iteration error alone:

| grid | Jacobi → 3% | Jacobi → 1% | Chebyshev → 3% | Chebyshev → 1% |
|:--|--:|--:|--:|--:|
| 32³ | ~70 | >128 | **~9** | **~15** |
| 64³ | ~94 | ~210 | **~10** | **~26** |
| 128³ | ~158 | >256 | **~15** | **~42** |

This is the operator's spectrum, not a tuning failure: a Jacobi sweep damps its slowest mode by
only `(2 + cos(2π/N))/3` per pass — 0.9984 at 64³ — and that mode carries most of the
potential's power, because φ<sub>k</sub> ~ ρ<sub>k</sub>/k². Chebyshev semi-iteration over the
same sweep converges at 0.9449 per pass instead: the square-root-of-condition-number speedup,
fully parallel, with no red/black parity to work out inside a z-slice atlas.

The stencil is identical. Only what gets written back differs:

```
x[k+1] = alpha[k] * (c1 * sweep(x[k]) - c2 * x[k]) - beta[k] * x[k-1]
```

Three ping-pong buffers instead of two; `beta[0] = 0`, so the first pass never reads
`x[k-1]`. `c1` and `c2` depend only on the grid size, `alpha` and `beta` only on the grid size
and the pass index — two float uniforms per pass, computed once. This ships **24 passes** at
64³: about 1% force error, which plain Jacobi would need some 210 passes to reach.

Subtracting the mean density is not optional and is free. Relax against the raw density and
every pass shifts the mean potential by a fixed amount that never cancels, because the box has
net mass — measured at −0.409 after 200 passes at 32³, matching prediction to 1e-9 and sliding
linearly forever. In float32 the useful field would end up in the low bits of a large number.
The mean needs no GPU reduction: CIC deposition conserves mass bit-exactly, so it is just
`totalMass / L³`.

**This is the part worth measuring.** Starved of passes, the solver does not produce visible
noise. It produces a smooth, coherent bias spread over many cells: a plausible-looking field
that is quietly the wrong one. You cannot catch that by looking at the screen.

### Does the cloud actually fall?

Collapse and expansion are indistinguishable by eye once the field fills the frame, so
[`tools/probe-gravity.js`](tools/probe-gravity.js) reads the position texture back and measures
the cloud radius directly. A sphere at rest, no noise, no initial velocity:

```
  t=0   rms 1.002
  t=3   rms 0.887   contracting
  t=5   rms 0.654   contracting
  t=7   rms 0.168   contracting
  t=8   rms 0.255   EXPANDING   <- passes through the centre and rebounds
```

Predicted free-fall time 6.2 s, measured 7 s — and a dissipationless cold collapse that
overshoots and rebounds, exactly as the textbook says it should.

---

## Verification

`npm test` runs **5,335 checks**.

| suite | checks | what it proves |
|:--|--:|:--|
| noise | 182 | range, continuity, curl divergence under 10⁻³, GLSL and the JS mirror agree |
| shapes | 539 | length, finiteness, radius bounds, determinism, and per-shape structure — the Fibonacci sphere's nearest-neighbour spacing has a coefficient of variation of 0.015; the galaxy's arms show 24× contrast in an angular histogram |
| mask | 4,290 | every sampled point lands on a lit pixel, aspect preserved, orientation correct, coverage uniform |
| atlas | 259 | cell↔texel bijection, and a linear field reproduced across both slice seams and the periodic boundary |
| pm | 65 | mass and momentum conservation, a two-body circular orbit, Plummer-sphere equilibrium, self-force — and the pass-count measurement above |

Separately, [`tools/verify-hotaru.js`](tools/verify-hotaru.js) launches real Chrome and drives
20 states — idle, drag, Japanese input, Latin input, all seven shapes, gravity at 2–28 s, and a
phone viewport — checking each for shader failures, JavaScript errors and blank frames.

---

## Performance

| | |
|:--|:--|
| particles | 1,048,576, stepping down automatically; phones start at 262,144 |
| frame rate | 60 fps on an RTX 5070 Ti, pinned to vsync |
| per frame | a million-point deposition plus 24 Poisson passes over 512×512 |
| size | ~170 KB, single self-contained HTML file |

---

## Known limits

Stated plainly, because a demo that hides its approximations is not worth reading.

- **Deposition is nearest-grid-point; interpolation is trilinear.** The kernels do not match, so
  a small self-force exists. Cloud-in-cell deposition would need eight passes, since a point
  sprite can only write one texel — not a trade worth making for a visual piece. The reference
  puts the trustworthy range at roughly four cells and beyond.
- **The box is periodic**, so a residue of force from distant images remains. The box is several
  times the size of the cloud, but the residue is not zero.
- **An artificial containment force** acts beyond radius 2.4, to keep particles from wrapping
  around the periodic box. That is bookkeeping, not physics.
- **Additive blending into 32-bit float targets needs `EXT_float_blend`.** Without it the mass
  grid falls back to 16-bit, where accumulation saturates past roughly 2,048 particles per cell.
- **No dependencies is a constraint, not a boast** — adding one would end the single-file
  distribution that makes this openable by anyone.

---

## Repository

| path | contents |
|:--|:--|
| [`hotaru/engine.js`](hotaru/engine.js) | all the WebGL2: GPGPU integration, the gravity pipeline, bloom, auto-exposure |
| [`hotaru/atlas.js`](hotaru/atlas.js) | the 3D-grid-in-2D-texture mapping, as GLSL plus a JS mirror that proves it |
| [`hotaru/pm.js`](hotaru/pm.js) | the exact CPU reference solver (FFT / Jacobi / Chebyshev / SOR). Not shipped to the browser |
| [`hotaru/noise.js`](hotaru/noise.js) | simplex noise and a curl field verified divergence-free |
| [`hotaru/shapes.js`](hotaru/shapes.js) | eight deterministic procedural generators |
| [`hotaru/mask.js`](hotaru/mask.js) | even point sampling from a rasterised glyph |
| [`web/`](web) | the page: shell and glue |
| [`tools/`](tools) | the Chrome-driven verification harness and the physics probes |

Every module is an IIFE publishing one global, so the build is plain concatenation.

---

## Licence

[Apache License 2.0](LICENSE)

The 3D simplex noise is a transcription of the implementation by Stefan Gustavson and
Ashima Arts (MIT).
