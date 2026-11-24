# C7 Aequivalenztest, Teil 1 (MLX, privat, mit Claude): Referenzwerte aus der eingefrorenen studien/v2a2/v2a2_run.py.
# Aufruf: ~/precap_mlx_venv/bin/python -I mlx_referenz_c7.py [--ohne-training]
import json, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; S = H.parent.parent; sys.path.insert(0, str(S / 'v2a2'))
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
import v2a2_run as R
seed = 42350; out = {}; meta = {}
xtr, ytr, xv, yv, xte, yte = R.load()
out['x_ausschnitt'] = np.concatenate([xtr[:200], xv[:100], xte[:100]]).astype(np.float32)          # NHWC
out['x_hash'] = np.array([float(np.sum(xtr, dtype=np.float64)), float(np.sum(xv, dtype=np.float64)), float(np.sum(xte, dtype=np.float64))])
r = np.random.default_rng(seed + 777); b = r.permutation(len(xtr))[:128]; out['batch_idx'] = b
grid = np.linspace(-12, 12, 48001).astype(np.float32)
for a in ('bprecap', 'tat', 'selu'):
    m = R.CNN(R.make_act(a), seed); mx.eval(m.parameters())
    out[f'w0_{a}'] = np.array(m.convs[0].weight); out[f'w25_{a}'] = np.array(m.convs[25].weight); out[f'wo_{a}'] = np.array(m.out.weight)
    out[f'form_{a}'] = np.array(R.make_act(a)(mx.array(grid)))
    out[f'logits_{a}'] = np.array(m(mx.array(xtr[b])))
    lf = lambda m_, u, v: nn.losses.cross_entropy(m_(u), v, reduction='mean')
    _, g = nn.value_and_grad(m, lf)(m, mx.array(xtr[b]), mx.array(ytr[b])); g = dict(tree_flatten(g))
    for i in range(R.L): out[f'grad_{a}_{i}'] = np.array(g[f'convs.{i}.weight'])
    if a != 'selu':
        Rc = R.calib(m, xtr[:R.NCAL], seed); meta[f'calib_{a}'] = {k: Rc[k] for k in ('max', 'max_abs', 'p999', 'p9999', 'p99999', 'mse4')}
rg = np.random.default_rng(1); p0 = rg.normal(0, 0.1, (32, 32)).astype(np.float32); G = rg.normal(0, 1e-3, (3, 32, 32)).astype(np.float32)
opt = optim.AdamW(learning_rate=optim.cosine_decay(1e-4, 3440), weight_decay=3.0); P = {'w': mx.array(p0)}
for k in range(3): P = opt.apply_gradients({'w': mx.array(G[k])}, P); mx.eval(P)
out['adam_p0'] = p0; out['adam_G'] = G; out['adam_p3'] = np.array(P['w'])
A = rg.normal(0, 1.3, (4000,)).astype(np.float32); Wt = rg.normal(0, 0.1, (32, 3, 3, 32)).astype(np.float32)
out['qA'] = A; out['qW'] = Wt; out['qw'] = np.array(R.qw(mx.array(Wt))); out['qa_sym'] = np.array(R.qa_sym(mx.array(A), 8, 3.7))
out['qa_asym'] = np.array(R.qa_asym(mx.array(A), 4, (-2.1, 3.3))); meta['mse_asym'] = list(R.mse_asym(A, (float(A.min()), float(A.max()))))
if '--ohne-training' not in sys.argv:
    for a, lr in (('bprecap', 3e-5), ('tat', 1e-4)):
        m, ok_, tl, kurve, best = R.train(a, lr, 0.0, 'cos', seed, 172); meta[f'kurz_{a}'] = dict(ok=ok_, kurve=kurve, train_loss=tl)
np.savez(H / 'mlx_ref_c7.npz', **out); json.dump(meta, open(H / 'mlx_ref_c7.json', 'w'), indent=1); print('MLX-Referenz geschrieben', len(out), 'Felder')
