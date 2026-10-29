"""Erzeugt die neuen Abbildungen 7-9 fuer Preprint v0.2 aus den Rohdaten der Studien A, Q1b und Q3.
Aufruf aus dem Repo-Wurzelordner: python paper/make_figures_v02.py
Stil wie paper/make_figures.py (v0.1). Gibt die Mediane zur Kontrolle gegen die Berichte aus."""
import glob, pathlib, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]; OUT = R / 'paper' / 'figures'; S = R / 'code' / 'v02'
# v0.1-Farben fuer cs, e03, SELU; neue Varianten aus der Standardpalette des dataviz-Skills
C = {'spreccap_cs': '#01696F', 'spreccap_e03': '#DA7101', 'selu': '#28251D', 'kL1': '#6250d6', 'kL2': '#2a78d6', 'tat': '#e34948'}
N = {'spreccap_cs': 'SPreCap-cs', 'spreccap_e03': 'SPreCap-e03', 'selu': 'SELU', 'kL1': 'kL1', 'kL2': 'kL2', 'tat': 'TAT'}
M = {'spreccap_cs': 'o', 'spreccap_e03': 's', 'selu': 'D', 'kL1': '^', 'kL2': 'v', 'tat': 'P'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5,
                     'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
GRID = '#D4D1CA'
load = lambda s: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(S / s / 'workers' / 'w*.csv')))])
dn = lambda d: 'MNIST' if d == 'mnist' else 'FashionMNIST'

# 7 Testfehler gegen Tiefe (Studie A, Bestaetigungslaeufe)
a = load('a'); a = a[a.status == 'ok']; assert len(a) == 450, len(a)
med = a.groupby(['dataset', 'depth', 'act']).test_err_fp32.median().mul(100)
print('Abb. 7, Mediane Testfehler (%):'); print(med.unstack('act').round(2))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for i, d in enumerate(['mnist', 'fashion']):
    for k in ['selu', 'spreccap_cs', 'spreccap_e03', 'kL1', 'kL2']:
        y = med[d][:, k] if False else med.xs((d, k), level=('dataset', 'act'))
        ax[i].plot(y.index, y.values, '--' if k == 'selu' else '-', color=C[k], marker=M[k], ms=5, lw=1.6, label=N[k])
    ax[i].set_yscale('log'); ax[i].set_xticks([50, 100, 200]); ax[i].set_xlabel('Depth L (hidden layers)')
    ticks = [3, 5, 10, 20, 40] if d == 'mnist' else [12, 15, 20, 30, 40]
    ax[i].set_yticks(ticks); ax[i].set_yticklabels([str(t) for t in ticks]); ax[i].minorticks_off()
    ax[i].grid(axis='y', color=GRID, lw=.6); ax[i].set_title(dn(d)); ax[i].set_ylabel('Median test error (%)')
ax[0].legend(fontsize=7.5, loc='upper left')
fig.savefig(OUT / 'fig7_depth.png'); plt.close(fig)

# 8 fp32 gegen 4 Bit, symmetrisch und asymmetrisch (Studie Q3)
q = load('q3'); q = q[q.status == 'ok']; assert len(q) == 180, len(q)
m = q.groupby(['dataset', 'depth', 'act'])[['test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best']].median().mul(100)
print('\nAbb. 8, Mediane (%):'); print(m.round(2))
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6)); ax = ax.ravel()
for i, (d, L) in enumerate([('mnist', 50), ('fashion', 50), ('mnist', 100), ('fashion', 100)]):
    lo, hi = 1e9, 0
    for k in ['tat', 'selu', 'spreccap_cs']:
        r = m.loc[(d, L, k)]; x = r.test_err_fp32
        ax[i].plot([x, x], [r.test_err_w8a4_best, r.test_err_w8a4a_best], color=C[k], lw=1.2, alpha=.6)
        ax[i].plot(x, r.test_err_w8a4_best, M[k], color=C[k], ms=7, label=f'{N[k]}, symmetric' if i == 0 else None)
        ax[i].plot(x, r.test_err_w8a4a_best, M[k], mfc='white', mec=C[k], mew=1.5, ms=7, label=f'{N[k]}, asymmetric' if i == 0 else None)
        lo = min(lo, x); hi = max(hi, r.test_err_w8a4_best)
    pad = .08 * (hi - lo); ax[i].plot([lo - pad, hi + pad], [lo - pad, hi + pad], ':', color=MUTED if False else '#7A7974', lw=1)
    ax[i].set_xlim(lo - 2 * pad, None); ax[i].set_ylim(lo - pad, hi + 2 * pad)
    ax[i].set_title(f'{dn(d)}, L = {L}'); ax[i].grid(color=GRID, lw=.6)
    ax[i].set_xlabel('Test error, fp32 (%)'); ax[i].set_ylabel('Test error, w8a4_best (%)')
ax[0].text(.98, .04, 'dotted line: no loss\nthrough quantization', transform=ax[0].transAxes, ha='right', fontsize=7, color='#7A7974')
fig.legend(loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(.5, -0.04)); fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig(OUT / 'fig8_tradeoff.png'); plt.close(fig)

# 9 4-Bit-Verlust mit Max-Kalibrierung gegen beste Kalibrierung (Studie Q1b, wiederholt Q1 bitgleich)
b = load('q1b'); b = b[(b.status == 'ok') & b.act.isin(['spreccap_cs', 'selu'])]; assert len(b) == 120, len(b)
b = b.assign(l_max=100 * (b.test_err_w8a4 - b.test_err_fp32), l_best=100 * (b.test_err_w8a4_best - b.test_err_fp32))
lm = b.groupby(['dataset', 'depth', 'act'])[['l_max', 'l_best']].median()
print('\nAbb. 9, mediane 4-Bit-Verluste (pp):'); print(lm.round(2))
cells = [('mnist', 50), ('fashion', 50), ('mnist', 100), ('fashion', 100)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
w = .36
for j, (col, title) in enumerate([('l_max', 'Max calibration (Q1)'), ('l_best', 'Best of four calibrations (Q1b)')]):
    for o, k in [(-w / 2 - .01, 'selu'), (w / 2 + .01, 'spreccap_cs')]:
        v = [lm.loc[(d, L, k), col] for d, L in cells]
        bars = ax[j].bar([i + o for i in range(4)], v, w, color=C[k], label=N[k] if j == 0 else None)
        for bb, vv in zip(bars, v): ax[j].text(bb.get_x() + w / 2, vv + .4, f'{vv:.1f}', ha='center', fontsize=6.8, color='#28251D')
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels([f"{'MNIST' if d=='mnist' else 'Fashion'}\nL = {L}" for d, L in cells], fontsize=7.8)
    ax[j].set_title(title); ax[j].grid(axis='y', color=GRID, lw=.6); ax[j].set_axisbelow(True)
ax[0].set_ylabel('Median loss through w8a4 (pp)'); ax[0].legend(fontsize=7.5, loc='upper left')
fig.savefig(OUT / 'fig9_calibration.png'); plt.close(fig)
print('\nok')
