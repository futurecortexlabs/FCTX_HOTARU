<h1 align="center">蛍 &nbsp;Hotaru</h1>

<p align="center">
  百万個の光の粒が、打ち込んだ言葉のかたちに集まる。放てば、自らの重力で崩れる。<br>
  A million particles become the word you type — then collapse under their own gravity.
</p>

<p align="center">
  <a href="https://futurecortexlabs.github.io/FCTX_HOTARU/"><b>▶ デモを開く / Live demo</b></a>
  &nbsp;·&nbsp;
  <a href="#自己重力--the-gravity-is-real">物理の話 / The physics</a>
</p>

<p align="center">
  <img src="docs/media/collapse.gif" width="720" alt="HOTARU という言葉が百万の粒子で描かれ、自己重力で崩れてフィラメントと回転円盤になる様子 — the word HOTARU collapsing under self-gravity into filaments and a rotating disc">
</p>

<p align="center">
  <sub>
  一コマもキーフレームを打っていません。文字を保持していたバネを切り、自己重力を入れ、最初の1フレームだけ小さな速度のばらつきと回転を与えています。以後の形はすべて計算結果です。15秒ぶんのシミュレーションを2倍速で再生。<br>
  Nothing here is keyframed. The spring holding the letters together is switched off, self-gravity is switched on, and the cloud is given one frame of small random dispersion and a gentle spin — the spin is what flattens the remnant into a disc. Everything after that is solved. 15 seconds of simulation, played at 2×.
  </sub>
</p>

---

## これは何か / What it is

**HTML 1ファイル**です。依存ライブラリなし、バンドラなし、three.js なし。開けば動きます。

**One HTML file.** No dependencies, no bundler, no three.js. Open it and it runs.

| 操作 / what you do | 起きること / what happens |
|:--|:--|
| **言葉を打つ**<br>Type anything | 百万個のGPU粒子がその形へ飛ぶ。ラテン文字、ひらがな、漢字、絵文字<br>A million GPU particles fly into that shape. Latin, Japanese, kanji, emoji |
| **ドラッグ**<br>Drag the field | 粒子が散り、渦を巻き、また戻る<br>They scatter, swirl, and settle back |
| **放つ**<br>Release | バネが切れ、粒子が互いに引き合いはじめる。言葉は自重でたわみ、重力で束ねられた塊に分裂し、塊の間にフィラメントを引き、やがて回転円盤になる<br>The spring is cut and the particles begin pulling on *each other*. The word sags under its own weight, fragments into gravitationally bound clumps, draws filaments between them, and settles into a rotating disc |
| **7種の形**<br>Seven shapes | 球・銀河・結び目・波・環・ハート・螺旋<br>Sphere, galaxy, torus knot, wave, ring, heart, double helix |

同じ言葉を二度放っても、同じ形にはなりません。再生ではなくシミュレーションだからです。

Released twice, the same word never lands the same way. It is a simulation, not a playback.

<table>
<tr>
<td width="50%"><img src="docs/media/text-ja.png" alt="ひらがなを粒子で描いたところ / Japanese kana rendered in particles"></td>
<td width="50%"><img src="docs/media/galaxy.png" alt="手続き的に生成した渦巻銀河 / a procedural spiral galaxy"></td>
</tr>
<tr>
<td><img src="docs/media/idle.png" alt="待機状態。漂う光の雲 / the resting state, a drifting cloud of embers"></td>
<td align="center"><img src="docs/media/mobile.png" width="240" alt="スマートフォンでの表示 / running on a phone"></td>
</tr>
</table>

---

## 動かす / Try it

**オンライン** — [futurecortexlabs.github.io/FCTX_HOTARU](https://futurecortexlabs.github.io/FCTX_HOTARU/)

**手元で** — clone して `docs/index.html` をブラウザで開くだけ。サーバもインストールも不要です。

**Online** — the link above. **Offline** — clone and open `docs/index.html`. There is no server and no install step.

**ソースから / From source**

```bash
node build-hotaru.js    # 5モジュール + シェル + グルー / 5 modules + shell + glue -> docs/index.html
npm test                # 5,335 件のテスト / 5,335 checks
```

必要なのは Node だけです。`puppeteer-core` は検証ハーネス専用の開発依存で、インストール済みの Chrome を駆動するために使います。非力な端末では、URL に `?n=65536` を付けると粒子数を固定できます。

Node is the only requirement. `puppeteer-core` is a dev dependency used solely by the verification harness, which drives your installed Chrome. Append `?n=65536` to the URL to pin the particle count on a modest machine.

---

## 仕組み / How it works

毎フレーム、以下が GPU 上で走ります。 &nbsp; Every frame, on the GPU:

```
position / velocity textures  (RGBA32F, 1024x1024)
        |
        +- 1. deposit    堆積    a million particles drawn additively into a 64^3 grid,
        |                        held as z-slice tiles in one 512x512 texture
        +- 2. solve      求解    grad^2 phi = 4 pi G rho, 24 Chebyshev-accelerated passes,
        |                        warm-started from the previous frame
        +- 3. force      力      -grad(phi) sampled back trilinearly to each particle
        |
        +- 4. integrate  積分    one fragment shader writes position AND velocity
        |                        through multiple render targets
        +- 5. draw       描画    gl.POINTS pulled by gl_VertexID; no vertex buffer exists
        +- 6. post       ポスト  auto-exposure -> bright pass -> separable blur -> ACES
```

粒子の配列も頂点バッファも存在しません。描画は `drawArrays(POINTS, 0, 1048576)` の1回だけで、頂点シェーダが `gl_VertexID` から自分の粒子をテクスチャ読みします。

There is no particle array and no vertex buffer. The whole field is one `drawArrays(POINTS, 0, 1048576)` call whose vertex shader fetches its own particle from a texture by `gl_VertexID`.

露出は自動です。シーンのミップマップ連鎖を1テクセルまで縮約し、1×1バッファで時間平滑して、合成時のゲインに使います。密度が2桁変わるシミュレーションを手で露出調整するのは無理だからです。

Exposure is automatic: the scene's mipmap chain is reduced to a single texel, smoothed over time in a 1×1 buffer, and applied as gain at composite. A simulation whose density changes by two orders of magnitude cannot be exposed by hand.

---

## 自己重力 / The gravity is real

百万体の直接和は毎フレーム 10¹² 回の計算になり不可能です。そこで宇宙論の N 体計算と同じ **Particle-Mesh 法**を使います。

A direct million-body sum is 10¹² interactions per frame. Instead this uses the **Particle-Mesh** method — the same approach cosmological N-body codes use.

1. **堆積 / Deposit** — 粒子を最近傍セルへ加算描画する。 Each particle is drawn additively into its nearest grid cell.
2. **求解 / Solve** — 離散ポアソン方程式を解いてポテンシャルを得る。周期境界では解が一意に定まらないため平均密度を差し引く（Jeans の swindle）。 The discrete Poisson equation is solved for the potential; a periodic box has no unique solution, so the mean density is subtracted (the Jeans swindle).
3. **微分 / Differentiate** — ポテンシャルの勾配を取り、力を粒子へ補間して戻す。 The potential is differentiated and the force interpolated back to the particles.

### なぜ Chebyshev なのか、そしてその数字の出どころ / Why Chebyshev, and how that number was found

素朴なヤコビ法は、重力が最も必要とする長波長成分の収束が極端に遅い。パス数を勘で決める代わりに、このリポジトリは**厳密な CPU 参照実装** [`hotaru/pm.js`](hotaru/pm.js)（3次元 FFT によるポアソン解法）を持ち、[`test/pm.test.js`](test/pm.test.js) が GPU に必要なパス数を実測します。

Plain Jacobi relaxation converges slowest for exactly the long-wavelength modes gravity cares about. Rather than guess a pass count, this repository carries an **exact CPU reference** — [`hotaru/pm.js`](hotaru/pm.js), a 3D-FFT Poisson solver — and [`test/pm.test.js`](test/pm.test.js) measures what the GPU actually needs.

前フレームから暖機始動した状態での、毎フレームの必要パス数。評価指標は「粒子へ補間して戻した加速度の相対L2誤差」で、**同一ステンシル**のFFT解と突き合わせているため、反復誤差だけを見ています。

Passes per frame, warm-started from the previous frame, scored as the relative L2 error of the acceleration interpolated back to the particles — against the FFT solution of the *same* stencil, so this is iteration error alone:

| 格子 / grid | ヤコビ / Jacobi → 3% | ヤコビ / Jacobi → 1% | Chebyshev → 3% | Chebyshev → 1% |
|:--|--:|--:|--:|--:|
| 32³ | ~70 | >128 | **~9** | **~15** |
| 64³ | ~94 | ~210 | **~10** | **~26** |
| 128³ | ~158 | >256 | **~15** | **~42** |

これは調整の失敗ではなく、作用素のスペクトルそのものです。ヤコビ掃引は最も遅いモードを1パスあたり `(2 + cos(2π/N))/3` — 64³ なら 0.9984 — しか減衰させられません。そして φ<sub>k</sub> ~ ρ<sub>k</sub>/k² である以上、そのモードこそがポテンシャルの power の大半を担っています。同じ掃引に Chebyshev 半反復法をかけると 1パスあたり 0.9449 になる。条件数の平方根ぶんの高速化で、完全に並列、しかも Z スライス配置の中で赤黒のパリティを考える必要もありません。

This is the operator's spectrum, not a tuning failure: a Jacobi sweep damps its slowest mode by only `(2 + cos(2π/N))/3` per pass — 0.9984 at 64³ — and that mode carries most of the potential's power, because φ<sub>k</sub> ~ ρ<sub>k</sub>/k². Chebyshev semi-iteration over the same sweep converges at 0.9449 per pass instead: the square-root-of-condition-number speedup, fully parallel, with no red/black parity to work out inside a z-slice atlas.

ステンシルは完全に同一で、違うのは「何を書き戻すか」だけです。 &nbsp; The stencil is identical. Only what gets written back differs:

```
x[k+1] = alpha[k] * (c1 * sweep(x[k]) - c2 * x[k]) - beta[k] * x[k-1]
```

ピンポンバッファが2枚から3枚になるだけ。`beta[0] = 0` なので初回パスは `x[k-1]` を読みません。`c1`・`c2` は格子サイズのみ、`alpha`・`beta` は格子サイズとパス番号のみに依存するので、パスごとに float ユニフォームが2個、一度計算すれば済みます。本実装は 64³ で **24 パス**。力の誤差およそ1%で、素朴なヤコビなら約210パスを要する水準です。

Three ping-pong buffers instead of two; `beta[0] = 0`, so the first pass never reads `x[k-1]`. `c1` and `c2` depend only on the grid size, `alpha` and `beta` only on the grid size and the pass index — two float uniforms per pass, computed once. This ships **24 passes** at 64³: about 1% force error, which plain Jacobi would need some 210 passes to reach.

平均密度の差し引きは省略不可で、しかもコストゼロです。生の密度で緩和すると、箱に正味の質量がある以上、毎パス平均ポテンシャルが一定量ずつずれ続けて相殺されません（32³ で200パス後に −0.409、予測と 1e-9 で一致）。float32 では有用な場が大きな数の下位ビットに埋もれます。この平均に GPU 側の縮約は要りません。最近傍セル堆積は1粒子あたりちょうど 1.0 を書き込むので質量が厳密に保存され、平均は `totalMass / L³` という既知の定数だからです。

Subtracting the mean density is not optional and is free. Relax against the raw density and every pass shifts the mean potential by a fixed amount that never cancels, because the box has net mass — measured at −0.409 after 200 passes at 32³, matching prediction to 1e-9. In float32 the useful field would end up in the low bits of a large number. The mean needs no GPU reduction: nearest-grid-point deposition writes exactly 1.0 per particle, so mass is conserved exactly and the mean is just `totalMass / L³`.

> **ここが実測する価値のあった部分です。** パス数が足りないとき、ソルバは目に見えるノイズを出しません。多数のセルにまたがる滑らかで一貫したバイアスを出します。もっともらしく見えたまま、静かに間違った場になる。画面を眺めていても気づけません。
>
> **This is the part worth measuring.** Starved of passes, the solver does not produce visible noise. It produces a smooth, coherent bias spread over many cells: a plausible-looking field that is quietly the wrong one. You cannot catch that by looking at the screen.

### 本当に落ちているのか / Does the cloud actually fall?

場が画面を覆うと崩壊と膨張は目で区別できません。そこで [`tools/probe-gravity.js`](tools/probe-gravity.js) が位置テクスチャを読み戻し、雲の半径を直接測ります。初期条件は半径1の薄い殻、ノイズも初速もゼロ、減衰も 1.0 に戻して重力だけを残しています。

Collapse and expansion are indistinguishable by eye once the field fills the frame, so [`tools/probe-gravity.js`](tools/probe-gravity.js) reads the position texture back and measures the cloud radius directly. The initial condition is a thin shell of radius 1 at rest, with the noise and the damping switched off so that only gravity acts:

```
  t=0   rms 1.002
  t=3   rms 0.887   収縮 / contracting
  t=5   rms 0.654   収縮 / contracting
  t=7   rms 0.168   収縮 / contracting
  t=8   rms 0.255   膨張 / EXPANDING   <- 中心を通過して跳ね返る / passes through the centre and rebounds
```

理論値は 6.6 秒です。薄い殻では各要素が殻自身の重力を実効 GM/2 として受けるので、落下時間は (π/2)√(R³/GM) = 6.64 秒。実測 7 秒。散逸のない冷たい崩壊が中心を通り抜けて跳ね返るところまで、教科書どおりです。

Predicted 6.6 s: an element of a thin shell feels the shell's own gravity as an effective GM/2, giving (π/2)√(R³/GM) = 6.64 s at GM = 0.056. Measured 7 s — and a dissipationless cold collapse that overshoots and rebounds, exactly as the textbook says it should.

---

## 検証 / Verification

`npm test` が **5,335 件**を実行します。 &nbsp; `npm test` runs **5,335 checks**.

| スイート / suite | 件数 / checks | 何を証明するか / what it proves |
|:--|--:|:--|
| noise | 182 | 値域・連続性・カール場の発散が 10⁻³ 未満・GLSL と JS ミラーの一致<br>Range, continuity, curl divergence under 10⁻³, GLSL and the JS mirror agree |
| shapes | 539 | 長さ・有限性・半径境界・決定性・形状ごとの構造。フィボナッチ球の最近傍間隔の変動係数は 0.015、銀河の腕は角度ヒストグラムで 24 倍のコントラスト<br>Length, finiteness, radius bounds, determinism, and per-shape structure — the Fibonacci sphere's nearest-neighbour spacing has a coefficient of variation of 0.015; the galaxy's arms show 24× contrast in an angular histogram |
| mask | 4,290 | 全サンプル点が発光画素上に着地・アスペクト保存・上下の向き・被覆の一様性<br>Every sampled point lands on a lit pixel, aspect preserved, orientation correct, coverage uniform |
| atlas | 259 | セル↔テクセルの全単射、スライス境界と周期境界をまたぐ線形場の再現<br>Cell↔texel bijection, and a linear field reproduced across both slice seams and the periodic boundary |
| pm | 65 | 質量・運動量保存、2体円軌道、Plummer 球の平衡、自己力、そして上記のパス数測定<br>Mass and momentum conservation, a two-body circular orbit, Plummer-sphere equilibrium, self-force — and the pass-count measurement above |

これとは別に [`tools/verify-hotaru.js`](tools/verify-hotaru.js) が実際の Chrome を起動し、20の状態（待機・ドラッグ・日本語入力・ラテン入力・7形状・重力2〜28秒・携帯ビューポート）を撮影して、シェーダ失敗・JSエラー・空画面を検査します。

Separately, [`tools/verify-hotaru.js`](tools/verify-hotaru.js) launches real Chrome and drives 20 states — idle, drag, Japanese input, Latin input, all seven shapes, gravity at 2–28 s, and a phone viewport — checking each for shader failures, JavaScript errors and blank frames.

---

## 性能 / Performance

| | |
|:--|:--|
| 粒子数 / particles | 1,048,576。自動的に段階を下げる。携帯は 262,144 から開始<br>1,048,576, stepping down automatically; phones start at 262,144 |
| フレームレート / frame rate | RTX 5070 Ti で 60 fps、垂直同期の上限に張り付き<br>60 fps on an RTX 5070 Ti, pinned to vsync |
| 毎フレーム / per frame | 百万粒の堆積 + 512×512 上で 24 パスのポアソン緩和<br>a million-point deposition plus 24 Poisson passes over 512×512 |
| サイズ / size | HTML 1ファイルで約 170 KB。実行時の外部取得は Google Fonts の書体2つだけで、オフラインではシステムフォントに落ちます<br>~170 KB in one HTML file. The only runtime fetch is two Google Fonts; offline it falls back to system fonts |

---

## 既知の限界 / Known limits

近似を隠すデモは読む価値がないので、正直に書きます。 &nbsp; Stated plainly, because a demo that hides its approximations is not worth reading.

- **重力モードには非物理的なリミッタが3つ**あります。加速度は 6.0 で頭打ち（`forceClamp`、崩壊の底で実際に効きます）、速度は 8.0 で頭打ち、放った後の速度には毎秒約 8.6% の減衰がかかります（`damp: 0.9985`）。数値的な安定と見た目のための措置で、物理ではありません。上の自由落下の測定は、この減衰を 1.0 に戻した状態で行っています。
  <br>**The gravity mode applies three non-physical limiters.** Acceleration is capped at 6.0 (`forceClamp`, and it does bind at the bottom of a collapse), speed is capped at 8.0, and velocities decay by about 8.6% per second after release (`damp: 0.9985`). These are for numerical stability and for the look, not physics. The free-fall measurement above is taken with that damping returned to 1.0.
- **堆積は最近傍セル（NGP）、補間は三重線形**でカーネルが一致していません。厳密にはわずかな自己力が生じます。点スプライトは1テクセルにしか書けないため CIC 堆積には8パス必要で、視覚作品としては割に合わないと判断しました。参照実装によれば、力が信頼できるのは約4セル以上離れた相手に対してです。
  <br>**Deposition is nearest-grid-point; interpolation is trilinear.** The kernels do not match, so a small self-force exists. Cloud-in-cell deposition would need eight passes, since a point sprite can only write one texel — not a trade worth making for a visual piece. The reference puts the trustworthy range at roughly four cells and beyond.
- **箱は周期境界**なので、遠方の鏡像からの力がわずかに残ります。箱は雲の数倍の大きさですが、ゼロではありません。
  <br>**The box is periodic**, so a residue of force from distant images remains. The box is several times the size of the cloud, but the residue is not zero.
- **半径 2.4 を超えると人工的な封じ込め力**が働きます。周期境界の巻き込みを防ぐための事務処理であって、物理ではありません。
  <br>**An artificial containment force** acts beyond radius 2.4, to keep particles from wrapping around the periodic box. That is bookkeeping, not physics.
- **32bit float への加算ブレンドには `EXT_float_blend` が必要**です。無い環境では質量格子が 16bit にフォールバックし、1セルあたり約 2,048 粒子を超えると加算が飽和します。
  <br>**Additive blending into 32-bit float targets needs `EXT_float_blend`.** Without it the mass grid falls back to 16-bit, where accumulation saturates past roughly 2,048 particles per cell.
- **依存ゼロは矜持ではなく制約です。** ひとつ足した時点で、誰でも開ける1ファイル配布が終わります。
  <br>**No dependencies is a constraint, not a boast** — adding one would end the single-file distribution that makes this openable by anyone.

---

## 構成 / Repository

| パス / path | 中身 / contents |
|:--|:--|
| [`hotaru/engine.js`](hotaru/engine.js) | WebGL2 の全て。GPGPU 積分、重力パイプライン、ブルーム、自動露出<br>All the WebGL2: GPGPU integration, the gravity pipeline, bloom, auto-exposure |
| [`hotaru/atlas.js`](hotaru/atlas.js) | 3D格子を2Dテクスチャに畳む写像。GLSL と、それを検証する JS ミラー<br>The 3D-grid-in-2D-texture mapping, as GLSL plus a JS mirror that proves it |
| [`hotaru/pm.js`](hotaru/pm.js) | CPU 側の厳密な参照ソルバ（FFT / ヤコビ / Chebyshev / SOR）。ブラウザには載りません<br>The exact CPU reference solver (FFT / Jacobi / Chebyshev / SOR). Not shipped to the browser |
| [`hotaru/noise.js`](hotaru/noise.js) | Simplex ノイズと、発散ゼロを数値検証したカール場<br>Simplex noise and a curl field verified divergence-free |
| [`hotaru/shapes.js`](hotaru/shapes.js) | 8種の決定的な手続き的生成器<br>Eight deterministic procedural generators |
| [`hotaru/mask.js`](hotaru/mask.js) | ラスタライズした字形から偏りなく点を取るサンプラー<br>Even point sampling from a rasterised glyph |
| [`web/`](web) | ページ本体。シェルとグルーコード<br>The page: shell and glue |
| [`tools/`](tools) | Chrome を駆動する検証ハーネスと物理プローブ<br>The Chrome-driven verification harness and the physics probes |

各モジュールは「グローバルを1つ公開する IIFE」なので、ビルドはただの連結です。

Every module is an IIFE publishing one global, so the build is plain concatenation.

---

## ライセンス / Licence

[Apache License 2.0](LICENSE)

3次元 Simplex ノイズは Stefan Gustavson と Ashima Arts による実装の移植です（MIT）。

The 3D simplex noise is a transcription of the implementation by Stefan Gustavson and Ashima Arts (MIT).
