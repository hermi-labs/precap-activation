# V2a-Pilot (privat, mit Claude, explorativ, nicht praeregistriert): Trainiert ein CNN ohne Normierung und Residuen ueberhaupt?
# Netz und Init aus v2a_probe.py (L = 32, 32 Kanaele, Delta-orthogonal), CIFAR-10 wie B5 (Standardisierung ueber den Trainingssatz),
# Adam mit konstanter LR, 1720 Schritte, Batch 128, Seed 42000. Formen BPreCap (chi1 = 1 + 2/32) und SELU, LR 1e-4 und 1e-3.
# Gemessen: Val-Fehler fp32 am Ende, Verlauf des Trainingsverlusts, ms je Schritt. Aufruf: python v2a_pilot.py
import json, math, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); import v2a_probe as P
DATA = Path.home() / 'PyCharmMiscProject' / 'data' / 'cifar10' / 'cifar-10-batches-bin'
def rd(fs):
    a = np.concatenate([np.fromfile(DATA / f, dtype=np.uint8).reshape(-1, 3073) for f in fs]); x = a[:, 1:].astype(np.float32) / 255
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), a[:, 0].astype(np.int64)
xtr, ytr = rd([f'data_batch_{i}.bin' for i in range(1, 6)]); xv, yv = xtr[-5000:], ytr[-5000:]; xtr, ytr = xtr[:-5000], ytr[:-5000]
m_, s_ = xtr.mean(), xtr.std(); xtr = (xtr - m_) / s_; xv = (xv - m_) / s_
def run(name, lr, L=32, C=32, steps=1720, seed=42000):
    mx.random.seed(seed); act = P.bprecap(L) if name == 'bprecap' else nn.selu; m = P.CNN(L, C, act, seed); mx.eval(m.parameters())
    opt = optim.Adam(learning_rate=lr); lf = lambda mm, a, b: nn.losses.cross_entropy(mm(a), b, reduction='mean')
    vg = nn.value_and_grad(m, lf); state = [m.state, opt.state]
    def step(a, b):
        l, g = vg(m, a, b); opt.update(m, g); return l
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(seed + 777); perm = r.permutation(len(xtr)); pos = 0; trace = []; t0 = time.time()
    for s in range(1, steps + 1):
        if pos + 128 > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + 128]; pos += 128; l = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s % 172 == 0: trace.append(round(float(l), 3))
        if not math.isfinite(float(l)): break
    sec = time.time() - t0; ac = 0
    for i in range(0, len(xv), 500): ac += int((mx.argmax(m(mx.array(xv[i:i + 500])), 1) == mx.array(yv[i:i + 500])).sum())
    return dict(act=name, lr=lr, L=L, C=C, steps=steps, val_err=round(1 - ac / len(xv), 4), loss_trace=trace, min=round(sec / 60, 1))
if __name__ == '__main__':
    res = []
    for lr in (1e-4, 1e-3):
        for name in ('bprecap', 'selu'):
            o = run(name, lr); res.append(o); print(o, flush=True); json.dump(res, open(H / 'v2a_pilot_M1Pro.json', 'w'), indent=1)
