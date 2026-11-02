# Abbildungen 2-4 fuer das Preprint v0.3 (mit Claude). Liest nur die Roh-CSVs in code/v03. Aufruf aus dem Repo-Ordner: python paper/make_figures_v03.py
# Abb. 2: Bereich (Median ueber Schichten der maximalen |Aktivierung|, dann Median ueber 12 Seeds) ueber die Messpunkte, B9 / B10 / B11.
# Abb. 3: 4-Bit-Preis (w8a4a - fp32, Testfehler in pp) am Punkt des fruehen Stoppens, Median ueber 12 Seeds je Form und Studie.
# Farben: Slots 1-3 der validierten Standardpalette des dataviz-Skills in fester Reihenfolge; Form zusaetzlich ueber Linienart und Marker.
# Aufruf: .venv/bin/python -I paper_v03/make_figures_v03.py
import csv, statistics as stt
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent; ST = ROOT / 'code' / 'v03'; OUT = Path(__file__).resolve().parent / 'figures_v03'; OUT.mkdir(exist_ok=True)
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
FORMS = ('kL2B_tr', 'LB_tr', 'tat'); N = {'kL2B_tr': 'BPreCap', 'LB_tr': 'LB', 'tat': 'TAT'}
C = {'kL2B_tr': '#2a78d6', 'LB_tr': '#1baf7a', 'tat': '#eb6834'}; LS = {'kL2B_tr': '-', 'LB_tr': '--', 'tat': '-'}; MK = {'kL2B_tr': 'o', 'LB_tr': 's', 'tat': '^'}
INK, MUTED, GRID = '#28251D', '#7A7974', '#D4D1CA'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'axes.edgecolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
STUDIES = [('b9', 'cifar10', range(42156, 42168), 'B9: CIFAR-10\nAdam, constant LR'),
           ('b10', 'cifar10', range(42168, 42180), 'B10: CIFAR-10\nAdamW, cosine, tuned WD'),
           ('b11', 'cifar100', range(42180, 42192), 'B11: CIFAR-100\nAdamW, cosine, tuned WD')]

def rows(study, ds, seeds):
    R = {}
    for f in sorted((ST / study / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if r['dataset'] == ds and int(r['depth']) == 100 and int(r['seed']) in seeds and r['act'] in FORMS and r['status'] == 'ok':
                R.setdefault(r['act'], []).append(r)
    for a in FORMS: assert len(R.get(a, [])) == 12, (study, a, len(R.get(a, [])))
    return R
col = lambda s, n: n if s == 6880 else f'c{s}_{n}'
def es(r): return min(CK, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s))

data = {s: rows(s, ds, seeds) for s, ds, seeds, _ in STUDIES}

# Abb. 2
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
for i, (s, _, _, title) in enumerate(STUDIES):
    for a in FORMS:
        med = [stt.median(float(r[col(k, 'act_range_max_median')]) for r in data[s][a]) for k in CK]
        lo = [min(float(r[col(k, 'act_range_max_median')]) for r in data[s][a]) for k in CK]
        hi = [max(float(r[col(k, 'act_range_max_median')]) for r in data[s][a]) for k in CK]
        ax[i].fill_between(CK, lo, hi, color=C[a], alpha=.15, lw=0)
        ax[i].plot(CK, med, LS[a], color=C[a], marker=MK[a], ms=4, lw=2, label=N[a])
        ax[i].annotate(f'{med[-1]:.1f}', (CK[-1], med[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7, color=INK)
    ax[i].set_yscale('log'); ax[i].set_title(title, color=INK); ax[i].set_xlabel('training step'); ax[i].grid(axis='y', color=GRID, lw=.6, which='both')
    ax[i].set_xticks([860, 3440, 6880]); ax[i].set_yticks([2, 3, 5, 10, 20, 40]); ax[i].set_yticklabels(['2', '3', '5', '10', '20', '40']); ax[i].minorticks_off()
ax[0].set_ylabel('activation range (max |a|, median over layers)')
ax[0].legend(loc='upper left', fontsize=8)
fig.text(0.01, -0.04, 'Lines: median over 12 confirmation seeds; bands: min-max over seeds. Depth 100, width 256, TAT recipe. Log scale.', fontsize=7, color=MUTED)
fig.savefig(OUT / 'fig2_range.png'); plt.close(fig)

# Abb. 3
fig, ax = plt.subplots(figsize=(7.4, 3.2)); w = .26; summ = {}
for j, (s, _, _, title) in enumerate(STUDIES):
    for k, a in enumerate(FORMS):
        cost = [(float(r[col(es(r), 'test_err_w8a4a_best')]) - float(r[col(es(r), 'test_err_fp32')])) * 100 for r in data[s][a]]
        v = stt.median(cost); summ[(s, a)] = v
        b = ax.bar(j + (k - 1) * (w + .02), v, w, color=C[a], label=N[a] if j == 0 else None, hatch='///' if a == 'LB_tr' else None, edgecolor='white', lw=.5)
        ax.text(j + (k - 1) * (w + .02), v + .25, f'{v:.1f}', ha='center', fontsize=7, color=INK)
# B10b: TAT mit Gewichtsabfall (a: LR 1e-4/WD 0,3, b: 5e-5/WD 3,0) gegen BPreCap aus B10, gleiche Seeds; Bereich am Stopp-Punkt als Beschriftung
tb = {}
for f in sorted((ST / 'b10b' / 'workers').glob('w*.csv')):
    for r in csv.DictReader(open(f)):
        if r['act'] == 'tat' and int(r['seed']) in range(42168, 42180) and r['status'] == 'ok' and float(r['wd']) > 0: tb.setdefault((float(r['lr']), float(r['wd'])), []).append(r)
assert sorted(len(v) for v in tb.values()) == [12, 12], {k: len(v) for k, v in tb.items()}
j = 3; bars = [('kL2B_tr', data['b10']['kL2B_tr'], None, 'BPreCap'), ('tat', tb[(1e-4, 0.3)], '..', 'TAT, WD 0.3'), ('tat', tb[(5e-5, 3.0)], 'xx', 'TAT, WD 3.0')]
for k, (a, R, h, lab) in enumerate(bars):
    v = stt.median((float(r[col(es(r), 'test_err_w8a4a_best')]) - float(r[col(es(r), 'test_err_fp32')])) * 100 for r in R)
    rg = stt.median(float(r[col(es(r), 'act_range_max_median')]) for r in R); summ[('b10b', lab)] = (round(v, 2), round(rg, 2))
    x = j + (k - 1) * (w + .02); ax.bar(x, v, w, color=C[a], hatch=h, edgecolor='white', lw=.5)
    ax.text(x, v + .25, f'{v:.1f}', ha='center', fontsize=7, color=INK); ax.text(x, -0.4, ('' if a != 'tat' else lab.split(', ')[1] + '\n') + f'range\n{rg:.2g}', ha='center', va='top', fontsize=5.8, color=MUTED)
ax.set_xticks(range(4)); ax.set_xticklabels([t.split('\n')[0].replace(': ', '\n') for s, _, _, t in STUDIES] + ['B10b\nTAT with WD'])
ax.set_ylim(-3.6, None); ax.set_yticks(range(0, 16, 2)); ax.axhline(0, color=MUTED, lw=.6)
ax.set_ylabel('4-bit cost (pp, w8a4a − fp32)'); ax.grid(axis='y', color=GRID, lw=.6); ax.set_axisbelow(True); ax.legend(fontsize=8, ncol=3, loc='upper left')
fig.text(0.01, -0.06, 'Median over 12 seeds of the per-run test-error increase at the early-stopping checkpoint.', fontsize=7, color=MUTED)
fig.savefig(OUT / 'fig3_cost.png'); plt.close(fig)
for k, v in summ.items(): print(k, v if isinstance(v, tuple) else round(v, 2))

# Abb. 4: 4-Bit-Preis aller Formen im B10-Regime (CIFAR-10, d100, AdamW, Kosinus, getunt), Seeds 42168-42179, Stopp-Punkt.
# Quelle je Form: die Studie, in der sie bestaetigt wurde. GELU fehlt (trainiert nicht, B12). Farbe nach Form: glatt (blau), stueckweise linear (orange).
def rows_sel(study, act, lr=None, wd=None):
    out = []
    for f in sorted((ST / study / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if r['dataset'] == 'cifar10' and int(r['depth']) == 100 and r['act'] == act and int(r['seed']) in range(42168, 42180) and r['status'] == 'ok' \
               and (lr is None or float(r['lr']) == lr) and (wd is None or float(r['wd']) == wd): out.append(r)
    assert len(out) == 12, (study, act, lr, wd, len(out)); return out
ITEMS = [('BPreCap (soft hull)', 'b10', 'kL2B_tr', None, None, 'g'), ('LB (soft hull)', 'b10', 'LB_tr', None, None, 'g'), ('SELU', 'b12', 'selu_tr', None, None, 'g'),
         ('PreCap core, hard clip', 'b13', 'kL2Bh_tr', None, None, 'g'), ('PreCap core, clip (PACT style)', 'b14', 'kL2Bp_tr', None, None, 'g'),
         ('linear, hard clip', 'b13', 'LBh_tr', None, None, 'p'), ('linear, clip (PACT style)', 'b14', 'LBp_tr', None, None, 'p'),
         ('TAT, WD 0.3', 'b10b', 'tat', 1e-4, 0.3, 'p'), ('TAT, WD 3.0', 'b10b', 'tat', 5e-5, 3.0, 'p'), ('TAT, no WD (fp32-tuned)', 'b10', 'tat', None, None, 'p'),
         ('ReLU', 'b12', 'relu_tr', None, None, 'p')]
vals = []
for lab, st_, act, lr, wd, grp in ITEMS:
    R = rows_sel(st_, act, lr, wd); c = sorted((float(r[col(es(r), 'test_err_w8a4a_best')]) - float(r[col(es(r), 'test_err_fp32')])) * 100 for r in R)
    vals.append((lab, stt.median(c), c[2], c[-3], grp))
vals.sort(key=lambda v: v[1])
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for i, (lab, m, lo, hi, grp) in enumerate(vals):
    colr = '#2a78d6' if grp == 'g' else '#eb6834'
    ax.barh(i, m, 0.62, color=colr); ax.plot([lo, hi], [i, i], color=INK, lw=1); ax.text(max(m, hi) + 0.4, i, f'{m:.1f}', va='center', fontsize=7.5, color=INK)
ax.set_yticks(range(len(vals))); ax.set_yticklabels([v[0] for v in vals], fontsize=8); ax.set_xlabel('4-bit cost (pp, w8a4a − fp32), median and 3rd–10th of 12 seeds')
ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#2a78d6', label='smooth nonlinearity (soft hull, PreCap core, SELU)'), Patch(color='#eb6834', label='piecewise linear')], fontsize=8, loc='lower right')
fig.text(0.01, -0.03, 'CIFAR-10, depth 100, TAT recipe, AdamW + cosine, LR and WD tuned per form, early-stopping checkpoint. GELU omitted (does not train).', fontsize=6.5, color=MUTED)
fig.savefig(OUT / 'fig4_cost_by_shape.png'); plt.close(fig)
for v in vals: print('fig4', v[0], round(v[1], 2))
