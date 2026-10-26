"""Baut paper/PreCap_paper.pdf. Aufruf aus dem Repo-Wurzelordner: python paper/build_paper.py"""
import pathlib, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle,
                                KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
R = pathlib.Path(__file__).resolve().parents[1]; P = R / 'paper'; F = P / 'figures'; EQ = P / 'eq'; EQ.mkdir(exist_ok=True)
LF = '/usr/share/fonts/truetype/liberation/LiberationSerif-'
for n, f in [('Serif', 'Regular'), ('Serif-B', 'Bold'), ('Serif-I', 'Italic'), ('Serif-BI', 'BoldItalic')]:
    pdfmetrics.registerFont(TTFont(n, LF + f + '.ttf'))
pdfmetrics.registerFont(TTFont('Mono', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('Serif', normal='Serif', bold='Serif-B', italic='Serif-I', boldItalic='Serif-BI')
TEAL = colors.HexColor('#01696F'); INK = colors.HexColor('#28251D'); MUTED = colors.HexColor('#7A7974')
body = ParagraphStyle('b', fontName='Serif', fontSize=10.2, leading=13.4, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
small = ParagraphStyle('s', parent=body, fontSize=8.6, leading=11, spaceAfter=3, alignment=0)
cap = ParagraphStyle('c', parent=body, fontSize=8.6, leading=10.8, textColor=INK, spaceBefore=2, spaceAfter=9)
h1 = ParagraphStyle('h1', parent=body, fontName='Serif-B', fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, alignment=0)
h2 = ParagraphStyle('h2', parent=body, fontName='Serif-B', fontSize=10.4, leading=13, spaceBefore=6, spaceAfter=3, alignment=0)
title = ParagraphStyle('t', parent=body, fontName='Serif-B', fontSize=16.5, leading=20, alignment=TA_CENTER, spaceAfter=6)
cen = ParagraphStyle('cen', parent=body, alignment=TA_CENTER, spaceAfter=2)
absS = ParagraphStyle('a', parent=body, fontSize=9.6, leading=12.6, leftIndent=1.1*cm, rightIndent=1.1*cm)
refS = ParagraphStyle('r', parent=small, leftIndent=0.55*cm, firstLineIndent=-0.55*cm)
cell = ParagraphStyle('cell', parent=body, fontSize=8.4, leading=10, alignment=TA_CENTER, spaceAfter=0)

def eq(name, tex, fs=15, w=None):
    fig = plt.figure(figsize=(0.01, 0.01)); plt.rcParams['mathtext.fontset'] = 'stix'
    fig.text(0, 0, tex, fontsize=fs); p = EQ / f'{name}.png'
    fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.03, transparent=True); plt.close(fig)
    from PIL import Image as PI
    im = PI.open(p); ww, hh = im.size; width = ww / 300 * 72 * 0.8
    if w and width > w: width = w
    return Image(str(p), width=width, height=width * hh / ww)
def fig(name, width, caption):
    from PIL import Image as PI
    im = PI.open(F / name); ww, hh = im.size
    return KeepTogether([Image(str(F / name), width=width, height=width * hh / ww), Paragraph(caption, cap)])
def table(rows, widths, bold_cells=(), head=1):
    data = [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths, hAlign='CENTER')
    st = [('LINEABOVE', (0, 0), (-1, 0), 0.8, INK), ('LINEBELOW', (0, head-1), (-1, head-1), 0.5, INK),
          ('LINEBELOW', (0, -1), (-1, -1), 0.8, INK), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    t.setStyle(TableStyle(st)); return t
A = lambda u, t=None: f'<a href="{u}" color="#01696F">{t or u}</a>'
S = []; p = lambda t, s=body: S.append(Paragraph(t, s))

# ---------- Daten (keine abgetippten Zahlen) ----------
P3 = R / 'code/phase3/p3a3_chi1/analysis'
conf = pd.read_csv(P3 / 'p3a3_confirmatory.csv')
def row(d, L, c, r):
    x = conf[(conf.dataset == d) & (conf.depth == L) & (conf.cand == c) & (conf.ref == r)].iloc[0]; return x
cells = [('fashion', 50), ('fashion', 100), ('mnist', 50), ('mnist', 100)]
cname = {('fashion', 50): 'FashionMNIST, 50', ('fashion', 100): 'FashionMNIST, 100', ('mnist', 50): 'MNIST, 50', ('mnist', 100): 'MNIST, 100'}
med = {}
for d, L in cells:
    for c, r in [('spreccap_cs', 'selu'), ('spreccap_cs', 'spreccap'), ('spreccap_cs', 'elu'), ('spreccap_e03', 'selu'), ('spreccap_cs_ls', 'spreccap_cs')]:
        x = row(d, L, c, r); med[(d, L, c)] = 100 * x.med_err_cand; med[(d, L, r)] = 100 * x.med_err_ref
fmt = lambda v: f'{v:.2f}'

# ---------- Titel ----------
p('PreCap: A Bounded, Self-Normalizing Activation Function<br/>that Matches SELU in Deep Normalization-Free MLPs', title)
p('Aaron Schmitt', cen)
p('Independent researcher, Wiesbaden, Germany · ORCID ' + A('https://orcid.org/0009-0009-8689-5848', '0009-0009-8689-5848'), ParagraphStyle('x', parent=cen, fontSize=9, textColor=MUTED))
p('Preprint, version 0.1 · 26 October 2026 · Code and data: ' + A('https://github.com/hermi-labs/precap-activation', 'github.com/hermi-labs/precap-activation'), ParagraphStyle('x', parent=cen, fontSize=9, textColor=MUTED, spaceAfter=10))
p('<b>Abstract.</b> We introduce PreCap, a smooth, monotonic activation function that is bounded on both sides: an exponential unit that '
  'switches to its tangent line above a threshold (CapExp, which is exactly a shifted and scaled ELU) is applied after a soft tanh clip. '
  'SPreCap scales PreCap so that, like SELU, zero mean and unit variance are a fixed point, and adds an input gain <i>b</i> that sets the '
  'mean-field gradient gain χ<sub>1</sub>. In a preregistered study on 50- and 100-layer MLPs without normalization or skip connections '
  '(MNIST and FashionMNIST, per-activation learning-rate tuning, 15 confirmation seeds, Holm correction), SPreCap-cs with χ<sub>1</sub> equal to SELU '
  f'matches SELU: no cell is significantly worse, with median paired differences between {min(row(d,L,"spreccap_cs","selu").med_diff_pp for d,L in cells):+.2f} '
  f'and {max(row(d,L,"spreccap_cs","selu").med_diff_pp for d,L in cells):+.2f} percentage points. Test error decreases monotonically as χ<sub>1</sub> approaches 1 in all four cells. '
  f'The variant SPreCap-e03 (χ<sub>1</sub> = 1.03) beats SELU on MNIST at depth 100 ({fmt(med[("mnist",100,"spreccap_e03")])} % vs. {fmt(med[("mnist",100,"selu")])} %, '
  f'Holm-adjusted p = {row("mnist",100,"spreccap_e03","selu").holm_all:.4f}, secondary test), while plain ELU is about as strong overall. '
  'PreCap is not more robust to the learning rate. Earlier large advantages at a fixed learning rate disappeared under fair tuning, and we report these negative results. '
  'We release fused CUDA and Apple MLX kernels that run within about 1.5× of native SELU in float32 and an installable package.', absS)
S.append(Spacer(1, 4))

# ---------- 1 ----------
p('1  Introduction', h1)
p('Self-normalizing networks [1] showed that an activation function alone can keep the forward signal of deep fully connected networks at zero mean and unit '
  'variance. SELU achieves this with an unbounded positive branch. Bounded activations promise numerical safety, for example at low precision, '
  'and in deep networks without normalization layers, but classic bounded functions such as tanh saturate and are hard to train at depth [9]. '
  'Recent work replaces normalization layers with bounded point-wise functions (DyT [6], Derf [7], SALU [5]).')
p('This work started from a simple idea: use the expressiveness of exponential growth as long as it is harmless, and cap it before anything explodes. '
  'It asks a narrow question. <i>Can a bounded, smooth exponential-type activation be made self-normalizing and match SELU in very deep MLPs '
  'without normalization?</i> Our contributions are:')
for t in ['the PreCap/SPreCap family and its constants, including a third parameter that sets χ<sub>1</sub> (Section 2);',
          'a preregistered, hash-frozen evaluation with separate tuning and confirmation seeds, including all negative intermediate results (Sections 4–5);',
          'evidence that, for this bounded family, test error falls monotonically as χ<sub>1</sub> → 1, an independent confirmation of the slightly super-critical design in bSELU [2];',
          'fused forward/backward kernels for CUDA and Apple MLX and a pip-installable package (Section 6).']:
    p('• ' + t, ParagraphStyle('li', parent=body, leftIndent=0.5*cm, firstLineIndent=-0.3*cm, spaceAfter=2))

# ---------- 2 ----------
p('2  Definition', h1)
p('<b>CapExp.</b> For a threshold <i>c</i>, CapExp is exponential below <i>c</i> and continues along its tangent above it:')
S.append(eq('capexp', r'$\mathrm{capexp}(w)=e^{w}-1\ \ \mathrm{for}\ w\leq c,\qquad \mathrm{capexp}(w)=e^{c}-1+e^{c}\,(w-c)\ \ \mathrm{for}\ w>c$'))
p('CapExp is exactly <i>e<super>c</super></i>·ELU(<i>w</i>−<i>c</i>) + <i>e<super>c</super></i> − 1 (verified numerically to 4·10<super>−15</super>). '
  'With biases and a following linear layer it therefore represents the same functions as ELU [8]. The new element is the soft clip in front of it.')
p('<b>PreCap and SPreCap.</b> With the clip <i>w</i> = <i>s</i>·tanh(<i>bx</i>/<i>s</i>), fixed <i>s</i> = 1.5 and <i>c</i> = 1:')
S.append(eq('precap', r'$f(x)=\lambda\cdot\{1\ \mathrm{if}\ x>0;\ \ \alpha\ \mathrm{if}\ x\leq 0\}\cdot\mathrm{capexp}\left(s\,\tanh(bx/s)\right)$'))
p('PreCap0 is λ = α = <i>b</i> = 1. It has slope 1 at the origin and is close to 2·SiLU for small inputs. For SPreCap, (λ, α) are chosen as in SELU so that '
  'E[<i>f</i>(<i>z</i>)] = 0 and E[<i>f</i>(<i>z</i>)<super>2</super>] = 1 for <i>z</i> ~ N(0, 1), computed with Gauss–Hermite quadrature. '
  'The gain <i>b</i> sets χ<sub>1</sub> = E[<i>f</i>′(<i>z</i>)<super>2</super>], which controls how gradients grow from the output to the input layer '
  'at initialization: log<sub>10</sub>(<i>g</i><sub>1</sub>/<i>g<sub>L</sub></i>) ≈ (<i>L</i>−1)/2 · log<sub>10</sub> χ<sub>1</sub> [3, 9]. '
  'Since tanh is bounded, the output is bounded on both sides (Table 1). The function is monotonic and continuous, its derivative jumps at 0 by the factor α as in SELU, '
  'and its second derivative is discontinuous at <i>w</i> = <i>c</i>.')
S.append(KeepTogether([table([['Variant', 'λ', 'α', 'b', 'χ<sub>1</sub>', 'Range'],
       ['SPreCap', '0.80676869644', '2.39695130922', '1', '1.1305', '[−1.50, 2.48]'],
       ['SPreCap-cs', '1.102702500', '2.053776267', '0.71188112', '1.0716 (= SELU)', '[−1.76, 3.39]'],
       ['SPreCap-e03', '1.985552225', '1.608136397', '0.4118896262', '1.0300', '[−2.48, 6.11]'],
       ['PreCap0', '1', '1', '1', '–', '[−0.78, 3.08]']], [2.4*cm, 2.4*cm, 2.4*cm, 2.3*cm, 2.6*cm, 2.4*cm]),
       Paragraph('<b>Table 1.</b> Constants (<i>s</i> = 1.5, <i>c</i> = 1). Derivations: <font name="Mono" size="7.5">math/p1_spreccap_exact.py</font>, '
                 '<font name="Mono" size="7.5">math/p3a3_chi1_family.py</font>.', cap)]))
S.append(fig('fig1_function.png', 15.5*cm, '<b>Figure 1.</b> SPreCap-cs and SPreCap-e03 compared with SELU and ELU. Unlike SELU and ELU, the PreCap variants saturate for large positive inputs.'))

# ---------- 3 ----------
p('3  Related work', h1)
p('<b>Self-normalization.</b> SELU [1] introduced the mean/variance fixed point. Chen et al. [2] generalize it with explicit tolerances, '
  'E[<i>f</i>] = ε<sub>μ</sub>, E[<i>f</i><super>2</super>] = 1 − ε<sub>q</sub>, φ(1) = 1 + ε<sub>φ</sub>, and report that bSELU with a slightly '
  'super-critical gradient term beats SELU on a 56-layer CNN without shortcuts (88.54 % vs. 84.81 % on CIFAR-10). Our χ<sub>1</sub> family follows the same idea '
  'for a bounded function. Lu et al. [4] obtain bidirectional self-normalization with Gauss–Poincaré normalized activations and orthogonal weights. '
  '<b>Bounded and normalization-free.</b> DyT [6] and Derf [7] replace normalization layers in Transformers with tanh(α<i>x</i>) and erf(α<i>x</i> + <i>s</i>). '
  'SaluNet [5] uses the bounded, learnable SALU(<i>x</i>) = <i>ax</i>/√(1 + <i>abx</i><super>2</super>) to train ResNets and Transformers without normalization. '
  '<b>Signal propagation.</b> Edge-of-chaos theory [3] links χ<sub>1</sub> = 1 to trainable depth. Bukva et al. [9] show that saturation can impede training even at criticality, '
  'which matches our observation that the critical point of PreCap0 is fragile. <b>Similar shapes.</b> TanhExp, <i>x</i>·tanh(<i>e<super>x</super></i>) [10], combines tanh and exp in the opposite order '
  'and is unbounded. A survey of over 400 activations [11] contains no identical function. To our knowledge the exact PreCap form is new, but every building block is known. '
  'The contribution is therefore small and specific.')

# ---------- 4 ----------
p('4  Experimental protocol', h1)
p('Every study was preregistered before it ran: hypotheses, tests, decision rules, and SHA-256 hashes of all scripts. Hashes were verified before and after each run. '
  'Tuning seeds and confirmation seeds are disjoint across all studies. Comparisons are paired by seed and use an exact sign-flip permutation version of the '
  'Wilcoxon signed-rank test, which does not depend on the scipy version. Holm correction is applied per test family.')
p('<b>Phase 3a-3 (main study).</b> MLP, width 256, depth 50 and 100, no normalization, no residual connections; MNIST and FashionMNIST, standardized inputs, the last 5000 '
  'training images as a fixed validation split; Adam, batch 128, 1720 steps; weights <i>W</i> ~ N(0, <i>g</i><super>2</super>/fan_in) with a variance-preserving gain '
  '<i>g</i> = 1/√E[<i>f</i>(<i>z</i>)<super>2</super>] for every activation (≈ 1 for SELU and SPreCap). Learning rates {1, 2, 5}·10<super>−5</super>, {1, 2, 5}·10<super>−4</super>; '
  'the rate with the highest median validation accuracy over 4 tuning seeds is selected per activation and cell, then evaluated on 15 confirmation seeds. '
  'Activations: SPreCap, SPreCap-cs, SPreCap-e03, SPreCap-cs with a learnable clip <i>s</i> per layer, SELU, and ELU. In total 936 runs. All runs were finite, and the analysis '
  'was reproduced independently from the raw data.')
p('<b>Primary hypothesis and v0.1 decision rule.</b> SPreCap-cs vs. SELU in four cells, Holm over 4. SPreCap-cs counts as validated if it is significantly worse in no cell '
  'and the median difference is at most +0.5 percentage points in at least 3 of 4 cells. Secondary tests (24, one Holm family) include e03 vs. SELU and cs vs. ELU.')

# ---------- 5 ----------
p('5  Results', h1)
rows = [['Cell (data, depth)', 'SPreCap', 'SPreCap-cs', 'cs, learn. <i>s</i>', 'SPreCap-e03', 'SELU', 'ELU']]
for d, L in cells:
    vals = [med[(d, L, 'spreccap')], med[(d, L, 'spreccap_cs')], med[(d, L, 'spreccap_cs_ls')], med[(d, L, 'spreccap_e03')], med[(d, L, 'selu')], med[(d, L, 'elu')]]
    best = min(vals); rows.append([cname[(d, L)]] + [f'<b>{fmt(v)}</b>' if v == best else fmt(v) for v in vals])
S.append(KeepTogether([table(rows, [3.2*cm, 2*cm, 2.1*cm, 2.1*cm, 2.2*cm, 1.7*cm, 1.7*cm]),
    Paragraph('<b>Table 2.</b> Median test error (%) over 15 confirmation seeds at the selected learning rate. Best per row in bold.', cap)]))
rows = [['Cell', 'cs − SELU (pp)', 'cs better / worse', 'Holm p', 'e03 − SELU (pp)', 'Holm p']]
for d, L in cells:
    a = row(d, L, 'spreccap_cs', 'selu'); b = row(d, L, 'spreccap_e03', 'selu')
    rows.append([cname[(d, L)], f'{a.med_diff_pp:+.2f}', f'{a.wins} / {a.losses}', f'{a.holm_primary:.3f}', f'{b.med_diff_pp:+.2f}',
                 f'<b>{b.holm_all:.4f}</b>' if b.holm_all < 0.05 else f'{b.holm_all:.3f}'])
S.append(KeepTogether([table(rows, [3.2*cm, 2.5*cm, 2.6*cm, 1.7*cm, 2.6*cm, 1.7*cm]),
    Paragraph('<b>Table 3.</b> Paired tests. Differences are medians of per-seed differences, negative means lower error than SELU. '
              'Left: primary family (Holm over 4). Right: secondary family (Holm over 24).', cap)]))
p('<b>Primary result.</b> SPreCap-cs is significantly worse than SELU in none of the four cells, and all median differences are within ±0.2 pp, so the preregistered '
  'v0.1 rule is met. It is not better than SELU either: all four primary tests are inconclusive. In the secondary family SPreCap-cs is better than the original SPreCap '
  f'in 4 of 4 cells and worse than ELU on MNIST at depth 50 ({row("mnist",50,"spreccap_cs","elu").med_diff_pp:+.2f} pp, Holm p = {row("mnist",50,"spreccap_cs","elu").holm_all:.3f}). '
  f'SPreCap-e03 is better than SELU on MNIST at depth 100 ({row("mnist",100,"spreccap_e03","selu").med_diff_pp:+.2f} pp, 13 of 15 seeds) and better than SPreCap-cs there. '
  'Mean ranks over the four cells: e03 1.5, ELU 1.75, learnable <i>s</i> 3.25, cs 4.25, SELU 4.25, SPreCap 6.0.')
S.append(fig('fig2_boxplots.png', 16*cm, '<b>Figure 2.</b> Test error per confirmation seed (15 seeds) at the selected learning rate. The original SPreCap (χ<sub>1</sub> = 1.13) lies far above and is omitted, see Figure 3.'))
p('<b>χ<sub>1</sub> dose-response.</b> Within the family, the median error decreases strictly as χ<sub>1</sub> goes from 1.13 to 1.07 to 1.03 in all four cells (Figure 3). '
  'The largest gain comes from 1.13 to 1.07. Measured gradient growth at initialization agrees with the mean-field prediction to within 0.17 orders of magnitude (Figure 4), '
  'and SPreCap-cs is indistinguishable from SELU at initialization by construction.')
S.append(fig('fig3_chi1.png', 16*cm, '<b>Figure 3.</b> Median test error of the SPreCap family against χ<sub>1</sub>, with SELU and ELU for reference.'))
from PIL import Image as _PI
_w, _h = _PI.open(F / 'fig4_gradprofile.png').size
S.append(KeepTogether([Image(str(F / 'fig4_gradprofile.png'), width=7.6*cm, height=7.6*cm * _h / _w),
    Paragraph('<b>Figure 4.</b> Gradient growth from the last to the first layer at initialization, at depth 50 (lower point) and 100 (upper point), against the prediction (<i>L</i>−1)/2·log<sub>10</sub> χ<sub>1</sub>. The line marks equality.', cap)]))
p('<b>Learning rate.</b> PreCap is not more tolerant of large learning rates (Figure 5). At depth 50 and learning rate 5·10<super>−4</super>, SELU and ELU still reach '
  '59–72 % validation accuracy, all PreCap variants 10–15 %. At depth 100 only the original SPreCap still learns at 10<super>−4</super>. This shows a trade-off: '
  'larger χ<sub>1</sub> tolerates larger steps but reaches a worse optimum. The learnable clip <i>s</i> moved to a median of about 1.26–1.34 and helped only on MNIST at depth 50.')
S.append(fig('fig5_lr.png', 16*cm, '<b>Figure 5.</b> Median validation accuracy over the 4 tuning seeds for every learning rate in the grid.'))
p('<b>History and negative results.</b> The project began with 1D, 2D and 3D regression tasks (more than 19,000 runs). At a fixed learning rate of 10<super>−3</super>, PreCap was far better than SiLU and GELU '
  'in residual networks. This advantage vanished once every activation received its own tuned learning rate (0 of 30 tests better, 2 worse), and a 3D transfer study '
  '(7200 runs) confirmed this. The explanation is scale: PreCap has slope 1 at the origin, SiLU and GELU 0.5. One robust property remained. The bound keeps undamped 128-layer residual networks '
  'trainable where SiLU and GELU diverge. In the first deep-MLP study (Phase 3a, 2430 runs) PreCap0 failed to learn at depth 100. In Phase 3a-2 (1092 runs) the original SPreCap was significantly worse than SELU in '
  '4 of 4 cells. Its larger χ<sub>1</sub> (1.13 vs. 1.07) inflated input-side gradients by about 600× at depth 100, compared with 35× for SELU. This diagnosis motivated the χ<sub>1</sub> family above. '
  'All of these studies, including the failed ones, are documented in the repository.')

# ---------- 6 ----------
p('6  Kernels and package', h1)
p('An unfused PreCap needs about ten element-wise operations per direction. We provide fused forward and backward kernels: CUDA C++ compiled at run time through '
  '<font name="Mono" size="8.5">torch.cuda.jiterator</font> (no CUDA toolkit required) and a Metal kernel through <font name="Mono" size="8.5">mx.fast.metal_kernel</font> with a custom VJP for MLX. '
  'On an RTX 4070 all 17 checks pass (values and gradients in fp32/fp64, gradcheck, edge values up to ±10<super>4</super>, a 20-layer MLP trained with kernel vs. reference), '
  'with a maximum relative fp32 error of 1.1·10<super>−6</super>. A second, extended run on the same GPU passes 14 further checks: fp16 and bf16 values and gradients for all four presets (maximum relative error 3.8·10<super>−4</super> in fp16 and 2.9·10<super>−3</super> in bf16, i.e. at the rounding level of the format), NaN propagation, ±Inf mapped to the two bounds, non-contiguous tensors, and identical results of the package module and the reference. Speed varies between runs: forward plus backward over 16M fp32 elements took 0.91 ms vs. 0.89 ms for native SELU in the first run (Figure 6) and 1.06 ms vs. 0.73 ms in the second, where ReLU, SiLU, GELU and ELU also took 0.73 ms, so the fused kernel costs up to about 1.5× a native activation. In fp16 the kernel is slower (1.61 ms vs. 0.34 ms for SELU) because inputs are converted to fp32 and back in separate passes. On an M1 Pro all 12 checks pass. The package installs with')
p('<font name="Mono" size="8.3">pip install "precap[torch] @ git+https://github.com/hermi-labs/precap-activation"</font>', ParagraphStyle('code', parent=body, alignment=TA_CENTER))
p('and exposes <font name="Mono" size="8.5">from precap import PreCap</font> (the CUDA kernel is used automatically on CUDA tensors) and <font name="Mono" size="8.5">from precap.mlx import PreCap</font>.')
S.append(fig('fig6_kernels.png', 15*cm, '<b>Figure 6.</b> Time for forward plus backward over 16M float32 elements. The MLX measurement ran alongside a training job and is likely pessimistic.'))

# ---------- 7 ----------
p('7  Limitations', h1)
p('Only fully connected networks on MNIST and FashionMNIST, one width (256), one optimizer (Adam) and short training (1720 steps). No CNNs, Transformers, or large datasets. '
  'The advantage of SPreCap-e03 rests on one cell of a secondary family and should be treated as a hypothesis. The study was not powered to separate e03 from ELU. '
  'The benefit of a bounded range for fp16/fp8/int8 inference is plausible but was not measured. The CUDA kernel has no native fp16/bf16 path and is about 4.7× slower than native SELU in fp16. '
  'The work was done by a single independent researcher on consumer hardware with AI assistance for code and analysis (see acknowledgments).')

# ---------- 8 ----------
p('8  Conclusion', h1)
p('A bounded, smooth exponential-type activation can be made self-normalizing and then matches SELU in 50- and 100-layer MLPs without normalization. '
  'The decisive quantity is not the bound but the gradient gain χ<sub>1</sub>: a value slightly above 1 gives the best results, in line with bSELU. '
  'PreCap is not a general improvement over SELU or ELU. It is a bounded alternative with the same accuracy in this setting, open for tests where a bounded range should matter: '
  'low-precision inference, normalization-free deep networks, and depth-dependent χ<sub>1</sub> = 1 + <i>k</i>/<i>L</i>.')
p('<b>Acknowledgments.</b> Idea, function and all experiments by the author. Implementation, statistical analysis, literature research and writing were supported by '
  'Perplexity Computer with Claude models by Anthropic. See <font name="Mono" size="8.5">docs/Danksagung_und_Werkzeuge.md</font>.', small)
p('<b>Reproducibility.</b> Preregistrations, frozen scripts with SHA-256 hashes, raw data (or hashes), analysis outputs, figure and paper sources '
  '(<font name="Mono" size="8.5">paper/make_figures.py</font>, <font name="Mono" size="8.5">paper/build_paper.py</font>) are in the repository under Apache-2.0.', small)

# ---------- Refs ----------
p('References', h1)
refs = [('Klambauer, Unterthiner, Mayr, Hochreiter. Self-Normalizing Neural Networks. NeurIPS 2017.', 'https://arxiv.org/abs/1706.02515'),
        ('Chen, Zhao, Deng, Ding, Wen, Li, Xie. Large-scale self-normalizing neural networks (bSELU). Journal of Automation and Intelligence 3, 101–110, 2024.', 'https://www.sciopen.com/article/10.1016/j.jai.2024.05.001'),
        ('Hayou, Doucet, Rousseau. On the Impact of the Activation Function on Deep Neural Networks Training. ICML 2019.', 'https://arxiv.org/abs/1902.06853'),
        ('Lu, Gould, Ajanthan. Bidirectionally Self-Normalizing Neural Networks. 2020/2021.', 'https://arxiv.org/abs/2006.12169'),
        ('Zaied. SaluNet: Enabling Total Plasticity in Normalization-Free Networks. 2026.', 'https://arxiv.org/abs/2606.02927'),
        ('Zhu, Chen, He, LeCun, Liu. Transformers without Normalization (DyT). CVPR 2025.', 'https://arxiv.org/abs/2503.10622'),
        ('Chen, Lu, Zhu, Sun, Liu. Stronger Normalization-Free Transformers (Derf). 2025.', 'https://arxiv.org/abs/2512.10938'),
        ('Clevert, Unterthiner, Hochreiter. Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). ICLR 2016.', 'https://arxiv.org/abs/1511.07289'),
        ('Bukva, de Gier, Grosvenor, Jefferson, Schalm, Schwander. Criticality versus uniformity in deep neural networks. 2023.', 'https://arxiv.org/abs/2304.04784'),
        ('Liu, Di. TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks. IET Computer Vision.', 'https://arxiv.org/abs/2003.09855'),
        ('Kunc, Kléma. Three Decades of Activations: A Comprehensive Survey of 400 Activation Functions for Neural Networks. 2024.', 'https://arxiv.org/abs/2402.09092')]
for i, (t, u) in enumerate(refs, 1): p(f'[{i}] {t} {A(u)}', refS)

def deco(c, d):
    c.saveState(); c.setFont('Serif', 8); c.setFillColor(MUTED)
    c.drawString(2*cm, 1.2*cm, 'Schmitt · PreCap · preprint v0.1'); c.drawRightString(A4[0]-2*cm, 1.2*cm, str(d.page)); c.restoreState()
doc = BaseDocTemplate(str(P / 'PreCap_paper.pdf'), pagesize=A4, leftMargin=2.1*cm, rightMargin=2.1*cm, topMargin=2*cm, bottomMargin=2*cm,
                      title='PreCap: A Bounded, Self-Normalizing Activation Function that Matches SELU in Deep Normalization-Free MLPs',
                      author='Aaron Schmitt', creator='Perplexity Computer', subject='Preprint v0.1, 26 October 2026')
doc.addPageTemplates([PageTemplate(frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')], onPage=deco)])
doc.build(S); print('ok')
