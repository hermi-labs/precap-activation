# C6 Aequivalenztest, Teil 1 (MLX-Umgebung): Referenzwerte aus dem eingefrorenen studien/b10/b10_run.py (unveraendert geladen).
# Schreibt test/mlx_ref.npz: Daten-Hash, Init-Gewichte (Auszug), Logits bei der Init, Kalibrierung und w8a4a bei der Init,
# Gewichte nach 1 und 20 Trainingsschritten (AdamW, Kosinus wie B10), Kennzahlen nach 172 Schritten.
import importlib.util, sys, hashlib, math
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
H = Path(__file__).resolve().parent; ST = H.parent.parent
sp = importlib.util.spec_from_file_location('b10_run', ST / 'b10' / 'b10_run.py'); B = importlib.util.module_from_spec(sp); sys.modules['b10_run'] = B; sp.loader.exec_module(B)
mx.set_cache_limit(B.CACHE_LIMIT)
out = {}
for act, lr, wd in (('kL2B_tr', 1e-4, 0.3), ('tat', 1e-4, 0.3), ('selu_tr', 2e-4, 0.3), ('LB_tr', 1e-4, 0.3)):
    D = B.data_for('cifar10', act); xtr, ytr, xv, yv, xte, yte = D
    out[f'{act}_datahash'] = np.frombuffer(hashlib.sha256(np.ascontiguousarray(xtr).tobytes() + np.ascontiguousarray(xte).tobytes()).digest(), np.uint8)
    mx.random.seed(42168); m = B.Net(100, act, 42168, din=3072); mx.eval(m.parameters()); m.eval()
    out[f'{act}_W0'] = np.array(m.hid[0].weight); out[f'{act}_W50'] = np.array(m.hid[50].weight); out[f'{act}_Wout'] = np.array(m.out.weight)
    out[f'{act}_logits0'] = np.array(m(mx.array(xte[:512])))
    RA = B.calib_asym(m, xtr[:B.NCAL])
    for cal in B.CALS4: out[f'{act}_RA_{cal}'] = np.array(RA[cal], dtype=np.float64)
    out[f'{act}_w4a_mse4_err2500'] = np.array([1 - B.evaluate(B.qforward_asym(m, 4, RA['mse4']), xte[:2500], yte[:2500])[1]])
    # Training wie B.execute (ohne Zwischenmessungen), Schnappschuesse nach 1 und 20 Schritten, Kennzahlen nach 172 Schritten
    r0 = np.random.default_rng(42168 + 777); b0 = r0.permutation(len(xtr))[:B.BATCH]; _ = nn.value_and_grad(m, B.loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    m.train(); opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, B.STEPS), betas=[0.9, 0.999], eps=1e-8, weight_decay=wd, bias_correction=True)
    vg = nn.value_and_grad(m, B.loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(42168 + 777); perm = r.permutation(len(xtr)); pos = 0
    for s in range(1, 173):
        if pos + B.BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + B.BATCH]; pos += B.BATCH; loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s in (1, 20): out[f'{act}_W0_s{s}'] = np.array(m.hid[0].weight); out[f'{act}_W50_s{s}'] = np.array(m.hid[50].weight); out[f'{act}_loss_s{s}'] = np.array([float(loss)])
    m.eval(); out[f'{act}_test_err_172'] = np.array([1 - B.evaluate(lambda x: m(x), xte, yte)[1]]); out[f'{act}_lr_end_172'] = np.array([float(opt.learning_rate)])
    print(act, 'ok', float(out[f'{act}_test_err_172'][0]), flush=True)
np.savez(H / 'mlx_ref.npz', **out); print('MLX-REFERENZ OK', len(out))
