# Abb. 1 fuer das Preprint v0.3 (mit Claude): Formen aus dem Paket in src/.
# Aufruf aus dem Repo-Ordner: python paper/make_fig1_v03.py
import sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'src'))
import torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from precap.torch import bprecap_ref, lb_ref, precap_ref
from precap._presets import PRESETS
INK, MUTED, GRID = '#28251D', '#7A7974', '#D4D1CA'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
x = torch.linspace(-8, 8, 3201, dtype=torch.float64, requires_grad=True)
a = 0.5704; s = math.sqrt(2 / (1 + a * a))     # TAT d100 (dks, eta 0.9)
k = PRESETS['spreccap_kl2_d100']
F = [('BPreCap, L = 100', lambda t: bprecap_ref(t, 'bprecap_d100'), '#2a78d6', '-'),
     ('BPreCap, L = 200', lambda t: bprecap_ref(t, 'bprecap_d200'), '#2a78d6', ':'),
     ('LB, L = 100', lambda t: lb_ref(t, 'lb_d100'), '#1baf7a', '--'),
     ('TAT Leaky ReLU, L = 100', lambda t: s * torch.where(t > 0, t, a * t), '#eb6834', '-'),
     ('kL2, L = 100 (v0.2)', lambda t: precap_ref(t, **k), MUTED, '-.')]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for name, f, c, ls in F:
    y = f(x); g, = torch.autograd.grad(y.sum(), x)
    ax[0].plot(x.detach(), y.detach(), ls, color=c, lw=2, label=name); ax[1].plot(x.detach(), g, ls, color=c, lw=2)
for i, t in enumerate(['f(x)', "f'(x)"]):
    ax[i].axhline(0, color=GRID, lw=.6); ax[i].axvline(0, color=GRID, lw=.6); ax[i].set_xlabel('x'); ax[i].set_title(t, color=INK); ax[i].grid(color=GRID, lw=.4)
ax[0].set_ylim(-6, 9); ax[0].axhline(3, color='#2a78d6', lw=.6, alpha=.4); ax[0].axhline(-3, color='#2a78d6', lw=.6, alpha=.4)
ax[0].legend(fontsize=7.5, loc='upper left')
fig.savefig(HERE / 'figures_v03' / 'fig1_function.png'); plt.close(fig); print('ok', PRESETS['spreccap_kl2_d100'])
