# V2a-Entwurf (privat, mit Claude): Laufzeit und Signalfluss eines CNN ohne Normierung und ohne Residuen auf dem M1 Pro messen.
# Kein Studienlauf, nur Messung vor der Planung. Netz: Conv 3x3 (3 -> C), dann L-1 Conv 3x3 (C -> C, gleiche Aufloesung 32x32),
# globales Mittel, Linear 10. Init Delta-orthogonal (Xiao et al. 2018): nur der mittlere Abgriff traegt eine orthogonale Matrix.
# Aktivierung BPreCap mit chi1 = 1 + 2/L (Konstanten aus v2a_konstanten.json) oder SELU. Gemessen: ms je Trainingsschritt (Batch 128,
# Adam), Varianz der Vor-Aktivierungen je Schicht bei der Init (soll etwa 1 bleiben). Aufruf: python v2a_probe.py
import json, math, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
H = Path(__file__).resolve().parent; K = json.load(open(H / 'v2a_konstanten.json')); S_, C_, RHO, M = 1.5, 1.0, 0.8, 3.0
def bprecap(L):
    p = K[f'kL2B_d{L}']; lam, al, b = p['lam'], p['alpha'], p['b']; ec = math.exp(C_)
    def f(x):
        w = S_ * mx.tanh(b * x / S_); ce = mx.where(w <= C_, mx.exp(mx.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_)); y = lam * mx.where(x > 0, ce, al * ce)
        u = y / M; a = mx.abs(u); return M * mx.where(a <= RHO, u, mx.sign(u) * (RHO + (1 - RHO) * mx.tanh((a - RHO) / (1 - RHO))))
    return f
def delta_orth(r, cin, cout):   # MLX-Gewicht (cout, 3, 3, cin); mittlerer Abgriff orthogonal (cout x cin), sonst 0
    A = r.normal(size=(max(cin, cout), max(cin, cout))); Q, R_ = np.linalg.qr(A); Q = Q * np.sign(np.diag(R_))
    w = np.zeros((cout, 3, 3, cin), dtype=np.float32); w[:, 1, 1, :] = Q[:cout, :cin] * (math.sqrt(cout / cin) if cin < cout else 1.0); return w
class CNN(nn.Module):
    def __init__(self, L, C, act, seed=0):
        super().__init__(); r = np.random.default_rng(seed); dims = [3] + [C] * L
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in zip(dims[:-1], dims[1:])]
        for c, (a, b) in zip(self.convs, zip(dims[:-1], dims[1:])): c.weight = mx.array(delta_orth(r, a, b)); c.bias = mx.zeros((b,))
        self.out = nn.Linear(C, 10); self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(C), size=(10, C)).astype(np.float32)); self.out.bias = mx.zeros((10,))
        self.act = act
    def __call__(self, x, trace=None):
        for c in self.convs:
            z = c(x)
            if trace is not None: trace.append(float(mx.mean(z * z)))
            x = self.act(z)
        return self.out(mx.mean(x, axis=(1, 2)))
def probe(L, C, name, steps=30):
    act = bprecap(L) if name == 'bprecap' else nn.selu; m = CNN(L, C, act); mx.eval(m.parameters())
    r = np.random.default_rng(1); x = mx.array(r.normal(size=(128, 32, 32, 3)).astype(np.float32)); y = mx.array(r.integers(0, 10, 128))
    tr = []; m(x, trace=tr)
    opt = optim.Adam(learning_rate=1e-4); lf = lambda m_, a, b: nn.losses.cross_entropy(m_(a), b, reduction='mean')
    vg = nn.value_and_grad(m, lf); state = [m.state, opt.state]
    def step(a, b):
        l, g = vg(m, a, b); opt.update(m, g); return l
    step = mx.compile(step, inputs=state, outputs=state)
    for _ in range(5): mx.eval(step(x, y), state)
    t = time.time()
    for _ in range(steps): l = step(x, y); mx.eval(state)
    ms = (time.time() - t) / steps * 1000
    return dict(L=L, C=C, act=name, ms_per_step=round(ms, 1), min_per_6880=round(ms * 6880 / 60000, 1), min_per_1720=round(ms * 1720 / 60000, 1),
                q_first=round(tr[0], 3), q_mid=round(tr[len(tr) // 2], 3), q_last=round(tr[-1], 3), loss=round(float(l), 3))
if __name__ == '__main__':
    print('Geraet:', mx.default_device(), flush=True); res = []
    for L, C in ((20, 32), (32, 32), (32, 64), (50, 64)):
        for name in ('bprecap', 'selu'):
            o = probe(L, C, name); res.append(o); print(o, flush=True)
    json.dump(res, open(H / 'v2a_probe_M1Pro.json', 'w'), indent=1)
