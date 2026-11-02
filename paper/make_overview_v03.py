# Uebersichtsgrafik zu v0.3 (mit Claude). Liest nur die Roh-CSVs in code/v03 (B1, B10-B14, B15).
# Aufruf aus dem Repo-Ordner: python paper/make_overview_v03.py   -> paper/figures_v03/v03_overview.png
import csv, statistics as stt
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ST = HERE.parent / 'code' / 'v03'; OUT = HERE / 'figures_v03'; OUT.mkdir(exist_ok=True)
BLUE, ORANGE, BASE = '#2a78d6', '#eb6834', '#8C8A84'   # wie im Preprint: BPreCap/glatt blau, stueckweise linear orange; Basislinien grau
INK, MUTED, GRID = '#28251D', '#7A7974', '#D4D1CA'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.edgecolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
col = lambda s, n: n if s == 6880 else f'c{s}_{n}'
def es(r): return min(CK, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s))
def read(study):
    out = []
    for f in sorted((ST / study / 'workers').glob('w*.csv')): out += list(csv.DictReader(open(f)))
    return out
def tint(h, a=0.38):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5)); return '#%02x%02x%02x' % tuple(int(255 - (255 - v) * a) for v in (r, g, b))

# (a) MNIST, Tiefe 200 (B1): fp32 und 4 Bit (w8a4_best), kL2 aus v0.2 gegen BPreCap, Median ueber 12 Seeds
B1 = [r for r in read('b1') if r['dataset'] == 'mnist' and int(r['depth']) == 200 and r['status'] == 'ok']
A = {}
for act in ('kL2', 'kL2B'):
    R = [r for r in B1 if r['act'] == act]; assert len(R) == 12, (act, len(R))
    A[act] = (stt.median(float(r['test_err_fp32']) * 100 for r in R), stt.median(float(r['test_err_w8a4_best']) * 100 for r in R))
# (c) CIFAR-10, Tiefe 200 (B15): fp32 und w8a4a am Stopp-Punkt, SELU gegen BPreCap
B15 = [r for r in read('b15') if int(r['depth']) == 200 and r['status'] == 'ok']
Cv = {}
for act in ('selu_tr', 'kL2B_tr'):
    R = [r for r in B15 if r['act'] == act and r['dataset'] == 'cifar10']; R = [r for r in R if 42192 <= int(r['seed'])] if len(R) > 12 else R
    assert len(R) == 12, (act, len(R))
    Cv[act] = (stt.median(float(r[col(es(r), 'test_err_fp32')]) * 100 for r in R), stt.median(float(r[col(es(r), 'test_err_w8a4a_best')]) * 100 for r in R))
# (b) 4-Bit-Preis je Form (wie Abb. 4 des Preprints, B10-B14), Median ueber 12 Seeds
def rows_sel(study, act, lr=None, wd=None):
    out = [r for r in read(study) if r['dataset'] == 'cifar10' and int(r['depth']) == 100 and r['act'] == act and int(r['seed']) in range(42168, 42180) and r['status'] == 'ok'
           and (lr is None or float(r['lr']) == lr) and (wd is None or float(r['wd']) == wd)]
    assert len(out) == 12, (study, act, lr, wd, len(out)); return out
ITEMS = [('BPreCap', 'b10', 'kL2B_tr', None, None, 'g'), ('linear core + soft hull', 'b10', 'LB_tr', None, None, 'g'), ('SELU', 'b12', 'selu_tr', None, None, 'g'),
         ('PreCap core + hard clip', 'b13', 'kL2Bh_tr', None, None, 'g'), ('linear + hard clip', 'b13', 'LBh_tr', None, None, 'p'),
         ('TAT, WD 0.3', 'b10b', 'tat', 1e-4, 0.3, 'p'), ('TAT, fp32-tuned', 'b10', 'tat', None, None, 'p'), ('ReLU', 'b12', 'relu_tr', None, None, 'p')]
Bv = []
for lab, st_, act, lr, wd, grp in ITEMS:
    R = rows_sel(st_, act, lr, wd); Bv.append((lab, stt.median((float(r[col(es(r), 'test_err_w8a4a_best')]) - float(r[col(es(r), 'test_err_fp32')])) * 100 for r in R), grp))
Bv.sort(key=lambda v: v[1])

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.45, 1], wspace=0.62))
def pair(axi, data, names, colors, title, ylab):
    for i, (k, nm, c) in enumerate(zip(data, names, colors)):
        f32, w4 = data[k]
        axi.bar(i - 0.19, f32, 0.36, color=tint(c), edgecolor='white', linewidth=1); axi.bar(i + 0.19, w4, 0.36, color=c, edgecolor='white', linewidth=1)
        axi.text(i - 0.19, f32 + 1, f'{f32:.1f}', ha='center', va='bottom', fontsize=8, color=INK); axi.text(i + 0.19, w4 + 1, f'{w4:.1f}', ha='center', va='bottom', fontsize=8, color=INK)
    axi.set_xticks(range(len(names))); axi.set_xticklabels(names, color=INK); axi.set_ylabel(ylab); axi.set_title(title, loc='left', color=INK)
    axi.grid(axis='y', color=GRID, lw=.6); axi.set_axisbelow(True); axi.set_ylim(0, max(max(v) for v in data.values()) * 1.18)
pair(ax[0], A, ['kL2 (v0.2)', 'BPreCap'], [BASE, BLUE], '(a) MNIST, 200 layers', 'test error (%)')
pair(ax[2], Cv, ['SELU', 'BPreCap'], [BASE, BLUE], '(c) CIFAR-10, 200 layers', 'test error (%)')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=tint(INK, .3), label='fp32 (light)'), Patch(color=INK, label='4-bit activations (solid)')], fontsize=8.5, loc='upper right', ncol=2, bbox_to_anchor=(0.99, 1.06))
for i, (lab, m, grp) in enumerate(Bv):
    c = BLUE if grp == 'g' else ORANGE; ax[1].barh(i, m, 0.62, color=c); ax[1].text(m + 0.4, i, f'+{m:.1f}', va='center', fontsize=8, color=INK)
ax[1].set_yticks(range(len(Bv))); ax[1].set_yticklabels([v[0] for v in Bv], fontsize=8.5, color=INK); ax[1].invert_yaxis()
ax[1].set_xlabel('4-bit cost (pp, 4-bit − fp32 test error)'); ax[1].set_title('(b) CIFAR-10, 100 layers: cost of 4 bits', loc='left', color=INK)
ax[1].grid(axis='x', color=GRID, lw=.6); ax[1].set_axisbelow(True); ax[1].set_xlim(0, max(v[1] for v in Bv) * 1.22)
ax[1].legend(handles=[Patch(color=BLUE, label='smooth saturation'), Patch(color=ORANGE, label='piecewise linear')], fontsize=8, loc='center right')
fig.text(0.01, -0.06, 'MLPs without normalization or skip connections, width 256. Post-training quantization: weights int8 per channel, activations int4 per tensor. '
         'Medians over 12 paired seeds. (a) B1, symmetric 4 bit; (b) B10–B14, TAT recipe, AdamW + cosine, LR/WD tuned per form; (c) B15, asymmetric 4 bit. '
         'Preprint v0.3, code/v03.', fontsize=6.8, color=MUTED, wrap=True)
fig.suptitle('PreCap v0.3 – BPreCap: depth calibration and a soft fixed bound in one activation', x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK, y=1.04)
fig.savefig(OUT / 'v03_overview.png'); print('A', {k: tuple(round(x, 2) for x in v) for k, v in A.items()}, 'C', {k: tuple(round(x, 2) for x in v) for k, v in Cv.items()}, 'B', [(l, round(m, 2)) for l, m, _ in Bv])
