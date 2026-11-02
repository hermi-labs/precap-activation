# Mechanik-Test B8 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen und Rezept gleich B6 (jetzt bei d200), orthogonale Init mal Gain, Signal-Varianz ueber 200 Schichten, Raster, Laufzahlen,
# Seeds, LR von TAT aus B3 d200, execute bitgleich zu B6 (kL2B_tr, LB_tr bei d200) und zu B3 (tat bei d200), Kontrollpfad bitgleich zu B1.
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b8_run.py', 'B8'); B6 = load(H.parent / 'b6' / 'b6_run.py', 'B6'); B3 = load(H.parent / 'b3' / 'b3_run.py', 'B3'); B1 = load(H.parent / 'b1' / 'b1_run.py', 'B1')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Formen gleich B6 bei d200
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'LB_tr', 'tat'):
    ok(f'{n}_form_wie_B6', np.array_equal(np.array(B.make_act(n, 200)(z)), np.array(B6.make_act(n, 200)(z))))
# 2. Init orthogonal mal Gain (d200), gleich B6
g = B.GAINS[('kL2B', 200)]; m = B.Net(200, 'kL2B_tr', 42144, din=3072)
W = np.array(m.hid[5].weight); ok('tr_init_orthogonal_mal_gain', np.allclose(W @ W.T, g * g * np.eye(256), atol=1e-4), round(g, 6))
for n in ('kL2B_tr', 'LB_tr', 'tat'):
    a, b = B.Net(200, n, 42144, din=3072), B6.Net(200, n, 42144, din=3072)
    ok(f'{n}_init_wie_B6', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
# 3. Signal: Vor-Aktivierungs-Varianz ueber 200 Schichten
dt = B.data_for('cifar10', 'kL2B_tr')
def prevar(net, x):
    h = mx.array(x); v = []
    for l in net.hid: y = l(h); v.append(float(mx.mean(y * y))); h = net.post(0, y)
    return v
for n in ('kL2B', 'LB'):
    vt = prevar(B.Net(200, n + '_tr', 42144, din=3072), dt[0][:256]); ok(f'{n}_tr_varianz_stabil_d200', 0.7 < vt[-1] < 1.4, (round(vt[0], 3), round(vt[99], 3), round(vt[-1], 3)))
# 4. Raster, Laufzahlen, Seeds, LR
ok('raster', B.LR_GRID == [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4] and B.TUNE_SEEDS == B6.TUNE_SEEDS and B.DEPTHS == [200])
ok('n_tune', len(list(B.tune_configs())) == 48); ok('n_confirm', len(B.REPRO) + len(B.SEEDS) * len(B.ACTS) == 37)
used = set(range(42000, 42004)) | set(range(42010, 42144))
ok('seeds_neu', set(B.SEEDS) == set(range(42144, 42156)) and not (set(B.SEEDS) & used))
B.TUNEDIR = Path(tempfile.mkdtemp()); sel = B.lrsel(); ok('lr_tat_aus_B3_d200', sel[('cifar10', 200, 'tat')] == 2e-5)
# 5. execute bitgleich (60 Schritte, d200)
for M in (B, B6, B3, B1): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
for act, ref in (('kL2B_tr', B6), ('LB_tr', B6), ('tat', B3)):
    c = dict(dataset='cifar10', depth=200, act=act, lr=2e-5, seed=42144); o1 = B.execute(dict(c), 0); o2 = ref.execute(dict(c), 0)
    ok(f'{act}_d200_bitgleich', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys), round(o1['test_err_fp32'], 4))
c = dict(dataset='synth', depth=100, act='kL2B', lr=2e-5, seed=42060); o1 = B.execute(dict(c), 0); o2 = B1.execute(dict(c), 0)
ok('kontrolle_bitgleich_B1', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in ('train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best')))
# 6. Cache-Grenze (nur in main gesetzt): getrennte Prozesse mit und ohne Grenze, CIFAR-10 d200 kL2B_tr, 60 Schritte, muessen bitgleich sein
import json, subprocess
PROBE = ("import importlib.util, json, sys, tempfile\nfrom pathlib import Path\nimport mlx.core as mx\n"
         "sp = importlib.util.spec_from_file_location('B8', sys.argv[1]); B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B)\n"
         "if sys.argv[2] == '1': mx.set_cache_limit(B.CACHE_LIMIT)\n"
         "B.STEPS = 60; B.OUTDIR = Path(tempfile.mkdtemp())\n"
         "o = B.execute(dict(dataset='cifar10', depth=200, act='kL2B_tr', lr=2e-5, seed=42144), 0)\n"
         "print(json.dumps(dict(o={k: o[k] for k in %r}, cache=mx.get_cache_memory())))" % keys)
res = {}
for lim in ('0', '1'):
    out = subprocess.run([sys.executable, '-c', PROBE, str(H / 'b8_run.py'), lim], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    res[lim] = json.loads(out)
ok('cache_grenze_bitgleich', res['0']['o'] == res['1']['o'], f"Cache ohne {res['0']['cache'] / 1e9:.2f} GB, mit {res['1']['cache'] / 1e9:.2f} GB")
# MLX gibt den Cache erst bei der naechsten Anforderung frei, die Grenze kann daher kurz um einen Puffer ueberschritten werden
ok('cache_grenze_wirkt', res['1']['cache'] <= 1.25 * B.CACHE_LIMIT and res['1']['cache'] < 0.5 * res['0']['cache'], (res['1']['cache'], res['0']['cache']))

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
