# V2a-Pilot, Teil 4 (privat, mit Claude, explorativ, nicht praeregistriert, 04.11.2026): PTQ im CNN ohne Normierung und Residuen.
# Frage: Deutet sich BPreCaps 4-Bit-Vorteil (w8a4a) gegen TAT und SELU im CNN ueberhaupt an, bevor die Studie auf dem neuen Mac laeuft?
# Netz, Init, Daten, Training wie v2a_pilot.py (Adam, konstante LR, 1720 Schritte, Batch 128). Nach dem Training:
#   Gewichte je Ausgangskanal symmetrisch int8 (wie qw in lagma9_run.py), Aktivierungen = Eingang jeder Faltung und der Eingang der
#   Linearschicht (globales Mittel) je Tensor: w8a8 symmetrisch (max), w8a4a_max asymmetrisch (min/max, Nullpunkt, wie SCULPT/Lagma-8),
#   w8a4a_p9999 (Quantile 0,005 % / 99,995 %). Kalibrierung: erste 1000 Trainingsbilder. Quantile auf einer festen Stichprobe von
#   2 Mio. Werten je Tensor (Speicher), Min/Max auf allen Werten. Ausgewertet auf dem Val-Teil (5000 Bilder, wie Pilot 1-3).
# Aufruf: python v2a_pilot4.py --test   (Mini-Lauf: L=16, 8 Kanaele, 40 Schritte, prueft Quantisierung gegen fp32 und w8a8-Naehe)
#         python v2a_pilot4.py --phase A|B|C   (A: L=32, 2 Seeds; B: L=50, 2 LR je Form; C: L=50, beste LR aus B, Seed 42001; D: Raster bei L=50 vervollstaendigt)
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); import v2a_probe as P, v2a_pilot as V
T = json.load(open(H / 'v2a_tat_params.json'))
NCAL, NSUB = 1000, 2_000_000
def act_for(name, L):
    if name == 'bprecap': return P.bprecap(L)
    if name == 'selu': return nn.selu
    if name == 'tat':
        a, s = T[str(L)]['neg'], T[str(L)]['scale']; return lambda x: s * mx.where(x > 0, x, a * x)
    raise ValueError(name)
def train(name, lr, L, C, steps, seed):
    mx.random.seed(seed); m = P.CNN(L, C, act_for(name, L), seed); mx.eval(m.parameters())
    opt = optim.Adam(learning_rate=lr); lf = lambda mm, a, b: nn.losses.cross_entropy(mm(a), b, reduction='mean')
    vg = nn.value_and_grad(m, lf); state = [m.state, opt.state]
    def step(a, b):
        l, g = vg(m, a, b); opt.update(m, g); return l
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(seed + 777); perm = r.permutation(len(V.xtr)); pos = 0; trace = []; ok = True; t0 = time.time()
    for s in range(1, steps + 1):
        if pos + 128 > len(V.xtr): perm = r.permutation(len(V.xtr)); pos = 0
        b = perm[pos:pos + 128]; pos += 128; l = step(mx.array(V.xtr[b]), mx.array(V.ytr[b])); mx.eval(state)
        if s % max(1, steps // 10) == 0: trace.append(round(float(l), 3))
        if not math.isfinite(float(l)): ok = False; break
    return m, ok, trace, time.time() - t0
def qw(W):  # je Ausgangskanal symmetrisch int8; Faltung (cout,3,3,cin) bzw. Linear (cout,cin)
    ax = tuple(range(1, W.ndim)); am = mx.maximum(mx.max(mx.abs(W), axis=ax, keepdims=True), 1e-12); sc = am / 127
    return mx.clip(mx.round(W / sc), -127, 127) * sc
def qa_sym(a, bits, rng):
    qm = 2 ** (bits - 1) - 1; sc = max(rng, 1e-12) / qm; return mx.clip(mx.round(a / sc), -qm, qm) * sc
def qa_asym(a, bits, lohi):
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 2 ** bits - 1; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc)
    return (mx.clip(mx.round(a / sc) + zp, 0, n) - zp) * sc
def layer_inputs(m, x):  # Eingaenge aller Faltungen und der Linearschicht (fp32), als Liste von mx-Arrays (in Teilen gerechnet)
    outs = None
    for i in range(0, len(x), 250):
        h = mx.array(x[i:i + 250]); cur = []
        for c in m.convs: cur.append(h); h = m.act(c(h))
        cur.append(mx.mean(h, axis=(1, 2))); mx.eval(cur)
        cur = [np.array(t, dtype=np.float32).ravel() for t in cur]
        outs = cur if outs is None else [np.concatenate([o, c_]) for o, c_ in zip(outs, cur)]
    return outs
def calib(m, xc, seed):
    r = np.random.default_rng(seed); R = {'max_abs': [], 'max': [], 'p9999': []}
    for a in layer_inputs(m, xc):
        R['max_abs'].append(float(np.abs(a).max())); R['max'].append((float(a.min()), float(a.max())))
        s = a if len(a) <= NSUB else a[r.choice(len(a), NSUB, replace=False)]
        R['p9999'].append((float(np.quantile(s, 0.00005)), float(np.quantile(s, 0.99995))))
    return R
def qforward(m, mode, R):
    g = (lambda W: W) if mode == 'fp32' else qw   # fp32: Pruefweg ohne jede Rundung, muss m(x) exakt treffen
    Wc = [(g(c.weight), c.bias) for c in m.convs]; Wo = (g(m.out.weight), m.out.bias)
    def qx(h, i):
        if mode == 'fp32': return h
        if mode == 'w8a8': return qa_sym(h, 8, R['max_abs'][i])
        if mode == 'w8a4a_max': return qa_asym(h, 4, R['max'][i])
        if mode == 'w8a4a_p9999': return qa_asym(h, 4, R['p9999'][i])
        raise ValueError(mode)
    def f(x):
        h = x
        for i, (W, b) in enumerate(Wc): h = m.act(mx.conv2d(qx(h, i), W, padding=1) + b)
        h = qx(mx.mean(h, axis=(1, 2)), len(Wc)); return h @ Wo[0].T + Wo[1]
    return f
def err(f, x, y):
    ac = 0
    for i in range(0, len(x), 500): ac += int((mx.argmax(f(mx.array(x[i:i + 500])), 1) == mx.array(y[i:i + 500])).sum())
    return 1 - ac / len(x)
MODES = ['w8a8', 'w8a4a_max', 'w8a4a_p9999']
def run(name, lr, L, C, steps, seed, xv=None, yv=None):
    xv = V.xv if xv is None else xv; yv = V.yv if yv is None else yv
    m, ok, trace, sec = train(name, lr, L, C, steps, seed); o = dict(act=name, lr=lr, L=L, C=C, steps=steps, seed=seed, ok=ok, loss_trace=trace, min_train=round(sec / 60, 1))
    if not ok:
        for k in ['fp32'] + MODES: o[f'val_err_{k}'] = 0.9
        return o
    o['val_err_fp32'] = round(err(m, xv, yv), 4); o['val_err_fp32_ref'] = round(err(qforward(m, 'fp32', None), xv, yv), 4)
    R = calib(m, V.xtr[:NCAL], seed)
    for k in MODES: o[f'val_err_{k}'] = round(err(qforward(m, k, R), xv, yv), 4)
    o['act_hi_max_median'] = round(float(np.median([h for _, h in R['max'][1:-1]])), 3); o['act_lo_max_median'] = round(float(np.median([l_ for l_, _ in R['max'][1:-1]])), 3)
    o['act_hi_p9999_median'] = round(float(np.median([h for _, h in R['p9999'][1:-1]])), 3)
    return o
PHASES = {   # (Form, LR, L, C, Schritte, Seed)
    'A': [(n, lr, 32, 32, 1720, s) for s in (42000, 42001) for n, lr in (('bprecap', 1e-4), ('tat', 1e-4), ('selu', 3e-5))],
    'B': [(n, lr, 50, 32, 1720, 42000) for n, lrs in (('bprecap', (1e-4, 3e-4)), ('tat', (1e-4, 3e-5)), ('selu', (3e-5, 1e-5))) for lr in lrs],
    # D (nachgetragen 04.11., 08:55, nach Phase B): Raster bei L = 50 vervollstaendigen auf {3e-5, 1e-4, 3e-4} je Form, Seed 42000
    'D': [('bprecap', 3e-5, 50, 32, 1720, 42000), ('tat', 3e-4, 50, 32, 1720, 42000), ('selu', 1e-4, 50, 32, 1720, 42000)],
    # E (nachgetragen 04.11., 11:15): zweiter Seed fuer BPreCap bei der besten LR aus D (L = 50); TAT und SELU haben ihn schon aus C
    'E': [('bprecap', 3e-5, 50, 32, 1720, 42001)],
}
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--test', action='store_true'); ap.add_argument('--phase'); a = ap.parse_args()
    if a.test:
        res = []
        for n in ('bprecap', 'tat', 'selu'):
            o = run(n, 1e-4, 16, 8, 40, 1, V.xv[:500], V.yv[:500]); res.append(o); print(o, flush=True)
            assert o['ok'] and o['val_err_fp32'] == o['val_err_fp32_ref'], 'fp32-Weg weicht ab'
            assert abs(o['val_err_w8a8'] - o['val_err_fp32']) <= 0.03, 'w8a8 zu weit von fp32'
        print('TEST OK'); sys.exit(0)
    out = H / f'v2a_pilot4_{a.phase}_M1Pro.json'
    if out.exists(): sys.exit(f'{out.name} existiert schon, Abbruch')
    if a.phase == 'C':   # L = 50: je Form die LR mit dem kleinsten fp32-Val-Fehler aus B (Gleichstand: kleinere LR), zweiter Seed
        B = json.load(open(H / 'v2a_pilot4_B_M1Pro.json')); PHASES['C'] = []
        for n in ('bprecap', 'tat', 'selu'):
            best = min((o for o in B if o['act'] == n), key=lambda o: (o['val_err_fp32'], o['lr'])); PHASES['C'].append((n, best['lr'], 50, 32, 1720, 42001))
        print('Phase C:', PHASES['C'], flush=True)
    res = []
    for cfg in PHASES[a.phase]:
        o = run(*cfg); res.append(o); print(time.strftime('%H:%M'), o, flush=True); json.dump(res, open(out, 'w'), indent=1)
    print('FERTIG', time.strftime('%H:%M'), flush=True)
