# Abbildungen fuer den Preprint v0.4 (mit Claude). Liest nur Konstanten, Analyse-Ausgaben und Roh-CSVs der Studien in code/v04/.
# (23.11.: neu nummeriert; Abb. 1 Formen und Abb. 4 Seed-Paare neu)
# Abb. 3: Testfehler fp32 und w8a4a_best am fruehen Stopp, Median ueber 12 Seeds, sieben Formen im selben Netz (L = 50).
#         Quelle: code/v04/v2h/analysis/v2h_result.json["descriptive"] (enthaelt alle sieben Formen, historische Arme gepaart).
# Abb. 2: V2k, Verhaeltnis BPreCap/ctanh je Seed fuer die Stoerung an den Logits, nur Abschneiden (D^C), nur Runden (D^R)
#         und voller 4-Bit-Fehler (D^Q), bei mse4 und bei max. Quelle: code/v04/v2k/workers/w*.csv.
# Farben: Slots der validierten Standardpalette (wie v0.3), BPreCap hervorgehoben, Rest neutral.
# Aufruf: python paper/make_figures_v04.py (matplotlib, numpy)
import csv, json, math, statistics as stt
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parent.parent; ST = ROOT / 'code' / 'v04'; OUT = Path(__file__).resolve().parent / 'figures_v04'; OUT.mkdir(exist_ok=True)
BLUE, GREEN, ORANGE = '#2a78d6', '#1baf7a', '#eb6834'
INK, MUTED, GRID, NEUTRAL = '#28251D', '#7A7974', '#D4D1CA', '#9B9890'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'axes.edgecolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})

# ---------- Abb. 2 (Fehler je Form) ----------
d = json.load(open(ST / 'v2h/analysis/v2h_result.json'))['descriptive']
NAME = {'bprecap': 'BPreCap', 'selu': 'SELU', 'ctanh': 'calibrated tanh', 'tat': 'TAT', 'htanh': 'critical tanh, Hayou point',
        'xtanh': 'critical tanh, near-linear point', 'tanh': 'tanh'}
rows = sorted(((v['es_test_err_w8a4a_best'] * 100, v['es_test_err_fp32'] * 100, k, v['n_ok'], v['preis_4bit_pp_median']) for k, v in d.items()), reverse=True)
assert len(rows) == 7 and all(r[3] == 12 for r in rows), rows
fig, ax = plt.subplots(figsize=(6.6, 3.3))
for i, (q4, fp, k, _, price) in enumerate(rows):
    col = BLUE if k == 'bprecap' else NEUTRAL
    ax.plot([fp, q4], [i, i], color=col, lw=2, solid_capstyle='round', zorder=1)
    ax.scatter([fp], [i], s=34, facecolor='white', edgecolor=col, lw=1.6, zorder=2)
    ax.scatter([q4], [i], s=34, color=col, zorder=3)
    ax.text(max(fp, q4) + 0.8, i, f'{q4:.1f} %  (cost {price:.1f} pp)', va='center', fontsize=7.5, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[r[2]] for r in rows], color=INK)
ax.set_xlim(38, 78); ax.set_xlabel('test error (%), median over 12 seeds'); ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
ax.scatter([], [], s=34, facecolor='white', edgecolor=MUTED, lw=1.6, label='fp32'); ax.scatter([], [], s=34, color=MUTED, label='w8a4a_best (4-bit activations)')
ax.legend(fontsize=7.5, loc='upper right')
fig.text(0.01, -0.05, 'Plain CNN, L = 50, CIFAR-10, early-stopping point. Label: median 4-bit error; cost = median over seeds of (w8a4a_best − fp32).', fontsize=7, color=MUTED)
fig.savefig(OUT / 'fig3_error_by_form.png'); plt.close(fig)

# ---------- Abb. 1 (Abschneiden und Runden) ----------
R = {}
for f in sorted((ST / 'v2k/workers').glob('w*.csv')):
    for r in csv.DictReader(open(f)):
        assert r['status'] == 'ok', r['run_id']
        R[(r['act'], r['seed'], r['cal'])] = r
seeds = sorted({k[1] for k in R}); assert len(seeds) == 9 and len(R) == 36
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
PARTS = (('D_allC', 'clipping only', BLUE), ('D_allR', 'rounding only', GREEN), ('D_allQ', 'full 4-bit error', NEUTRAL))
for j, cal in enumerate(('mse4', 'max')):
    for i, (col, lab, c) in enumerate(PARTS):
        q = [float(R[('bprecap', s, cal)][col]) / float(R[('ctanh', s, cal)][col]) for s in seeds]
        xs = [i + (n - 4) * 0.045 for n in range(9)]
        ax[j].scatter(xs, q, s=14, color=c, zorder=3)
        m = stt.median(q); ax[j].plot([i - .25, i + .25], [m, m], color=INK, lw=1.4, zorder=4)
        ax[j].text(i + .28, m, f'{m:.2f}', va='center', fontsize=7.5, color=INK)
    ax[j].axhline(1, color=MUTED, lw=.8, ls='--', zorder=1)
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels([p[1] for p in PARTS], fontsize=8, color=INK)
    ax[j].set_title({'mse4': 'MSE-optimal clipping range', 'max': 'min/max range (almost no clipping)'}[cal], color=INK)
    ax[j].set_yscale('log'); ax[j].set_yticks([0.3, 0.5, 0.7, 1.0, 1.4]); ax[j].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f')); ax[j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[j].grid(axis='y', color=GRID, lw=.6, which='both'); ax[j].set_axisbelow(True)
ax[0].set_ylabel('perturbation at the logits,\nBPreCap / calibrated tanh')
fig.text(0.01, -0.06, 'All layers perturbed at once. Dots: 9 seeds; bar: median; below 1 = smaller for BPreCap. Log scale. With min/max, the clipping part\nis below 0.3 % of the error energy, so its ratio does not affect the full error.', fontsize=7, color=MUTED)
fig.savefig(OUT / 'fig2_clipping_rounding.png'); plt.close(fig)

# ---------- Abb. 1 (Formen und Ableitungen) ----------
import numpy as np
KC = json.load(open(ST / 'v2a2/kL2B_d50.json')); TATP = json.load(open(ST / 'v2a2/v2a_tat_params.json'))['50']
CT_LAM, CT_B = 2.2948782077912586, 0.5328431680484216   # code/v04/v2d/v2d_run.py
def bprecap(x, lam=KC['lam'], al=KC['alpha'], b=KC['b'], rho=0.8, M=3.0, s=1.5, c=1.0):
    w = s * np.tanh(b * x / s); ec = math.e ** c
    ce = np.where(w <= c, np.exp(np.minimum(w, c)) - 1, ec - 1 + ec * (w - c)); y = lam * np.where(x > 0, ce, al * ce)
    u = y / M; a = np.abs(u); return M * np.where(a <= rho, u, np.sign(u) * (rho + (1 - rho) * np.tanh((a - rho) / (1 - rho))))
FORMS = [('BPreCap (L = 50)', bprecap, BLUE, '-', 2.2), ('calibrated tanh', lambda x: CT_LAM * np.tanh(CT_B * x), ORANGE, '-', 1.6),
         ('SELU', lambda x: 1.0507 * np.where(x > 0, x, 1.67326 * (np.exp(np.minimum(x, 0)) - 1)), GREEN, '--', 1.4),
         ('TAT (L = 50)', lambda x: TATP['scale'] * np.where(x > 0, x, TATP['neg'] * x), INK, ':', 1.4),
         ('tanh', np.tanh, NEUTRAL, '-.', 1.4)]
x = np.linspace(-6, 6, 4801); fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for lab, f, c, ls, lw in FORMS:
    y = f(x); ax[0].plot(x, y, color=c, ls=ls, lw=lw, label=lab, zorder=3 if c == BLUE else 2)
    ax[1].plot(x, np.gradient(y, x), color=c, ls=ls, lw=lw, zorder=3 if c == BLUE else 2)
for a_ in ax: a_.axhline(0, color=GRID, lw=.6, zorder=0); a_.axvline(0, color=GRID, lw=.6, zorder=0); a_.grid(color=GRID, lw=.4); a_.set_axisbelow(True); a_.set_xlabel('pre-activation x')
ye = bprecap(np.array([-80.0, 80.0])); assert abs(ye[1] - KC['hi']) < 1e-6 and abs(ye[0] - KC['lo']) < 1e-6, ye   # gleiche Rechnung wie der Loeser
ax[0].axhspan(KC['lo'], 3.0, color=BLUE, alpha=.06, zorder=0); ax[0].set_ylim(-3.2, 4.6); ax[0].set_ylabel('f(x)'); ax[0].set_title('activation', color=INK)
ax[0].text(-5.8, 3.12, 'BPreCap range [−2.29, 3.00]', fontsize=7, color=BLUE); ax[0].legend(fontsize=7, loc='lower right')
ax[1].set_ylim(-0.05, 1.9); ax[1].set_ylabel("f'(x)"); ax[1].set_title('slope', color=INK)
fig.text(0.01, -0.05, 'Constants as in Table 1 and Section 4.2. The shaded band is the image of BPreCap. The critical tanh networks of Section 5.6 use plain tanh with a different initialization.', fontsize=7, color=MUTED)
fig.savefig(OUT / 'fig1_shapes.png'); plt.close(fig)

# ---------- Abb. 4 (gepaarte Differenzen je Seed, w8a4a_best) ----------
def rows_of(study, act):
    out = {}
    for f in sorted((ST / study / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if r['act'] == act and r.get('stage', 'confirm') == 'confirm' and r['status'] == 'ok': out[int(r['seed'])] = float(r['es_test_err_w8a4a_best'])
    return out
B = rows_of('v2a2', 'bprecap'); assert len(B) == 12
COMP = [('SELU (V2a2)', 'v2a2', 'selu'), ('calibrated tanh (V2d)', 'v2d', 'ctanh'), ('TAT (V2a2)', 'v2a2', 'tat'),
        ('critical tanh, Hayou point (V2h)', 'v2h', 'htanh'), ('critical tanh, near-linear (V2gb)', 'v2gb', 'xtanh'), ('tanh (V2c)', 'v2c', 'tanh')]
fig, ax = plt.subplots(figsize=(6.6, 3.0)); med = {}
for i, (lab, st_, act) in enumerate(COMP):
    O = rows_of(st_, act); assert sorted(O) == sorted(B), (st_, act)
    dlt = [100 * (B[s] - O[s]) for s in sorted(B)]; med[lab] = stt.median(dlt)
    ax.scatter(dlt, [i + (n - 5.5) * 0.03 for n in range(12)], s=13, color=BLUE, alpha=.85, zorder=3)
    ax.plot([med[lab]] * 2, [i - .3, i + .3], color=INK, lw=1.6, zorder=4)
    ax.text(1.0, i, f'median {med[lab]:.2f} pp, {sum(v < 0 for v in dlt)}/12'.replace('-', '−'), va='center', fontsize=7.5, color=INK)
ax.axvline(0, color=MUTED, lw=.8, ls='--'); ax.set_yticks(range(len(COMP))); ax.set_yticklabels([c[0] for c in COMP], color=INK); ax.invert_yaxis()
ax.set_xlim(-30, 14); ax.set_xlabel('BPreCap − baseline, test error at w8a4a_best (pp); left = BPreCap better'); ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
fig.text(0.01, -0.05, 'Plain CNN, L = 50, CIFAR-10. Dots: the 12 test seeds (paired, same initial weights and batch order); bar: median. All baselines share the BPreCap runs of V2a2.', fontsize=7, color=MUTED)
fig.savefig(OUT / 'fig4_paired_seeds.png'); plt.close(fig)
print('ok', [r[2] for r in rows], {k: round(v, 2) for k, v in med.items()})
