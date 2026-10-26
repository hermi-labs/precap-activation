"""Erzeugt alle Paper-Grafiken aus den Daten im Repo. Aufruf aus dem Repo-Wurzelordner:
    python paper/make_figures.py  [Pfad zu entpackten 3a-3-Rohdaten, Standard /tmp/p3raw/workers]"""
import sys, glob, json, math, pathlib, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / 'src'))
from precap import precap_ref, PRESETS
OUT = R / 'paper' / 'figures'; P3 = R / 'code/phase3/p3a3_chi1'
RAW = sys.argv[1] if len(sys.argv) > 1 else '/tmp/p3raw/workers'
C = {'spreccap_cs': '#01696F', 'spreccap_e03': '#DA7101', 'spreccap': '#9BB7B9', 'spreccap_cs_ls': '#006494',
     'selu': '#28251D', 'elu': '#7A7974'}
N = {'spreccap_cs': 'SPreCap-cs', 'spreccap_e03': 'SPreCap-e03', 'spreccap': 'SPreCap (χ₁=1.13)',
     'spreccap_cs_ls': 'SPreCap-cs, learnable s', 'selu': 'SELU', 'elu': 'ELU'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5,
                     'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
cells = [('fashion', 50), ('fashion', 100), ('mnist', 50), ('mnist', 100)]
cn = lambda d, L: f"{'Fashion' if d=='fashion' else 'MNIST'}, L = {L}"

# 1 Funktion und Ableitung
x = torch.linspace(-5, 5, 2001, dtype=torch.float64, requires_grad=True)
selu = torch.nn.functional.selu; elu = torch.nn.functional.elu
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k, f in [('spreccap_cs', lambda t: precap_ref(t, **PRESETS['spreccap_cs'])), ('spreccap_e03', lambda t: precap_ref(t, **PRESETS['spreccap_e03'])),
             ('selu', selu), ('elu', elu)]:
    y = f(x); g, = torch.autograd.grad(y.sum(), x)
    ls = '--' if k in ('selu', 'elu') else '-'
    ax[0].plot(x.detach(), y.detach(), ls, color=C[k], lw=1.6, label=N[k]); ax[1].plot(x.detach(), g, ls, color=C[k], lw=1.6)
ax[0].set_title('Activation f(x)'); ax[1].set_title("Derivative f'(x)"); ax[0].legend(fontsize=8)
for a in ax: a.axhline(0, color='#D4D1CA', lw=.8); a.axvline(0, color='#D4D1CA', lw=.8); a.set_xlabel('x')
ax[0].set_ylim(-2.8, 6.4); fig.savefig(OUT / 'fig1_function.png'); plt.close(fig)

# Rohdaten Bestaetigung
df = pd.concat([pd.read_csv(f) for f in glob.glob(RAW + '/w*.csv')])
conf = df[(df.seed >= 42010) & (df.step == 1720)].copy(); conf['err'] = 100 * (1 - conf.test_acc)
sel = pd.read_csv(P3 / 'analysis/p3a3_lr_selection.csv')
conf = conf.merge(sel, on=['dataset', 'depth', 'act']); conf = conf[np.isclose(conf.lr, conf.lr_sel)]
assert len(conf) == 360, len(conf)

# 2 Boxplots (altes SPreCap weggelassen, liegt weit darueber, siehe Abb. 3)
order = ['spreccap', 'spreccap_cs', 'spreccap_cs_ls', 'spreccap_e03', 'selu', 'elu']
ob = order[1:]
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.9))
for a, (d, L) in zip(ax, cells):
    data = [conf[(conf.dataset == d) & (conf.depth == L) & (conf.act == k)].err.values for k in ob]
    b = a.boxplot(data, widths=.6, patch_artist=True, showfliers=True, medianprops=dict(color='white', lw=1.4),
                  flierprops=dict(markersize=2.5))
    for p, k in zip(b['boxes'], ob): p.set_facecolor(C[k]); p.set_edgecolor(C[k])
    a.set_title(cn(d, L), fontsize=8.5); a.set_xticks([])
ax[0].set_ylabel('test error (%)')
fig.legend([plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ob], [N[k] for k in ob], ncol=5, fontsize=7,
           loc='lower center', bbox_to_anchor=(.5, -.02))
fig.subplots_adjust(bottom=.1, wspace=.35); fig.savefig(OUT / 'fig2_boxplots.png'); plt.close(fig)

# 3 chi1 Dosis-Wirkung
chi = {'spreccap': 1.1305, 'spreccap_cs': 1.0716, 'spreccap_e03': 1.0300}
med = conf.groupby(['dataset', 'depth', 'act']).err.median()
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.4), sharex=True)
for a, (d, L) in zip(ax, cells):
    xs = [chi[k] for k in chi]; ys = [med[(d, L, k)] for k in chi]
    a.plot(xs, ys, '-o', color=C['spreccap_cs'], ms=4, lw=1.4, label='SPreCap family')
    a.axhline(med[(d, L, 'selu')], color=C['selu'], ls='--', lw=1, label='SELU (χ₁=1.07)')
    a.axhline(med[(d, L, 'elu')], color=C['elu'], ls=':', lw=1.2, label='ELU')
    a.set_title(cn(d, L), fontsize=8.5); a.set_xlabel('χ₁')
ax[0].set_ylabel('median test error (%)')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, fontsize=7, loc='lower center', bbox_to_anchor=(.5, -.12))
fig.subplots_adjust(wspace=.4, bottom=.2); fig.savefig(OUT / 'fig3_chi1.png'); plt.close(fig)

# 4 Gradientenprofil
gp = {'spreccap': (1.346, 2.804), 'spreccap_cs': (0.703, 1.540), 'selu': (0.699, 1.558), 'spreccap_e03': (0.342, 0.572)}
chi_all = {**chi, 'selu': 1.0716}
fig, ax = plt.subplots(figsize=(3.4, 2.8)); mx = 3.0
ax.plot([0, mx], [0, mx], color='#D4D1CA', lw=1)
for k, (m50, m100) in gp.items():
    p = [(L - 1) / 2 * math.log10(chi_all[k]) for L in (50, 100)]
    ax.scatter(p, [m50, m100], color=C[k], s=22, label=N[k], zorder=3, marker='o' if k != 'selu' else 's')
ax.set_xlabel('predicted  (L−1)/2 · log₁₀ χ₁'); ax.set_ylabel('measured  log₁₀(g₁ / g_L)')
ax.set_title('Gradient growth at initialization'); ax.legend(fontsize=7); fig.savefig(OUT / 'fig4_gradprofile.png'); plt.close(fig)

# 5 LR-Kurven
tune = pd.read_csv(P3 / 'analysis/p3a3_tune_median_val_acc_by_lr.csv'); lrs = [c for c in tune.columns if c[0].isdigit()]
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.4), sharey=True)
for a, (d, L) in zip(ax, cells):
    for k in order:
        r = tune[(tune.dataset == d) & (tune.depth == L) & (tune.act == k)]
        if len(r): a.plot([float(v) for v in lrs], 100 * r[lrs].values[0], marker='o', ms=2.5, lw=1.2, color=C[k],
                          ls='--' if k in ('selu', 'elu') else '-', label=N[k])
    a.set_xscale('log'); a.set_title(cn(d, L), fontsize=8.5); a.set_xlabel('learning rate')
ax[0].set_ylabel('median val. accuracy (%)')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=6, fontsize=7, loc='lower center', bbox_to_anchor=(.5, -.1))
fig.subplots_adjust(bottom=.22, wspace=.12); fig.savefig(OUT / 'fig5_lr.png'); plt.close(fig)

# 6 Kernel
kc = json.load(open(R / 'kernels/test_cuda_ergebnis_rtx4070.json'))['tempo_ms_16M_fwd_bwd']
km = json.load(open(R / 'kernels/test_mlx_ergebnis_m1pro.json'))['tempo_ms_16M_fwd_bwd']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
for a, (t, k, sk) in zip(ax, [('RTX 4070 (CUDA)', kc, 'selu_pytorch'), ('M1 Pro (MLX)', km, 'selu_mlx')]):
    labs = ['PreCap kernel', 'unfused reference', 'native SELU']; vals = [k['kernel'], k['referenz'], k[sk]]
    a.barh(labs, vals, color=[C['spreccap_cs'], '#BAB9B4', C['selu']])
    for i, v in enumerate(vals): a.text(v, i, f' {v:.2f} ms', va='center', fontsize=7.5)
    a.set_title(t); a.invert_yaxis(); a.set_xlim(0, max(vals) * 1.3); a.set_xlabel('ms, 16M elements, fwd+bwd')
fig.subplots_adjust(wspace=.9); fig.savefig(OUT / 'fig6_kernels.png'); plt.close(fig)
print(med.unstack().round(2).to_string())
