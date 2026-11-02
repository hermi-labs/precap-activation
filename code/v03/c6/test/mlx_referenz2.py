# C6 Aequivalenztest, Teil 1b (MLX): Gradienten bei der Init auf dem ersten Batch, AdamW an synthetischen Gradienten, Datenausschnitte.
import importlib.util, sys
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
H = Path(__file__).resolve().parent; ST = H.parent.parent
sp = importlib.util.spec_from_file_location('b10_run', ST / 'b10' / 'b10_run.py'); B = importlib.util.module_from_spec(sp); sys.modules['b10_run'] = B; sp.loader.exec_module(B)
mx.set_cache_limit(B.CACHE_LIMIT); out = {}
xtr, ytr, xv, yv, xte, yte = B.load('cifar10'); P = B.data_for('cifar10', 'kL2B_tr')
out['raw_rows'] = xtr[:3]; out['pex_rows'] = P[0][:3]; out['pex_test_rows'] = P[4][:3]
for act in ('kL2B_tr', 'tat', 'selu_tr', 'LB_tr'):
    D = B.data_for('cifar10', act); x, y = D[0], D[1]
    mx.random.seed(42168); m = B.Net(100, act, 42168, din=3072); mx.eval(m.parameters())
    perm = np.random.default_rng(42168 + 777).permutation(len(x)); b = perm[:B.BATCH]
    loss, g = nn.value_and_grad(m, B.loss_fn)(m, mx.array(x[b]), mx.array(y[b]))
    out[f'{act}_g0'] = np.array(g['hid'][0]['weight']); out[f'{act}_g50'] = np.array(g['hid'][50]['weight']); out[f'{act}_gout'] = np.array(g['out']['weight']); out[f'{act}_loss'] = np.array([float(loss)])
# AdamW an festen synthetischen Gradienten, LR-Spitze 1e-4 mit Kosinus ueber 6880, WD 0,3, 3 Schritte
r = np.random.default_rng(5); p0 = r.normal(0, 0.06, (64, 64)).astype(np.float32); G = [r.normal(0, 1e-3, (64, 64)).astype(np.float32) for _ in range(3)]
opt = optim.AdamW(learning_rate=optim.cosine_decay(1e-4, B.STEPS), betas=[0.9, 0.999], eps=1e-8, weight_decay=0.3, bias_correction=True)
params = {'w': mx.array(p0)}
for gi in G: params = opt.apply_gradients({'w': mx.array(gi)}, params); mx.eval(params)
out['adam_p0'] = p0; out['adam_G'] = np.stack(G); out['adam_p3'] = np.array(params['w'])
np.savez(H / 'mlx_ref2.npz', **out); print('MLX-REFERENZ2 OK', len(out))
