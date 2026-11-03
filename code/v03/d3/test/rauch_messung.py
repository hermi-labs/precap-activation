# Rauchtest der D3-Messung (mit Claude): untrainierte Netze (d200 BPreCap und SELU, d100 BPreCap), echte CIFAR-10-Daten.
# Prueft, dass alle Kennzahlen endlich sind, und misst die Laufzeit. Ausgabe je Netz eine Zeile mit noise_gain_median.
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location('d3', H / 'd3_diag.py'); D3 = importlib.util.module_from_spec(sp); sys.modules['d3'] = D3; sp.loader.exec_module(D3)
def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)
ok = True
for act, depth, mod in (('kL2B_tr', 200, 'b15_run'), ('selu_tr', 200, 'b15_run'), ('kL2B_tr', 100, 'b10_run')):
    B = D3.MODS[mod]; D3.B = B; t0 = time.time()
    D = B.data_for('cifar10', B.base(act) + '_tr'); m = B.Net(depth, act, 7, din=3072); mx.eval(m.parameters()); m.eval()
    RA = B.calib_asym(m, D[0][:B.NCAL]); o = D3.measure(m, D, {4: RA['mse4'], 8: RA['max']}, depth)
    f = finite({k: v for k, v in o.items()}); ok &= f
    print(act, depth, 'endlich' if f else 'NICHT ENDLICH', 'noise_gain_median', round(o['noise_gain_median'], 3), 'gain4', round(o['gain4_median'], 3),
          'gain8', round(o['gain8_median'], 3), 'eps4', round(o['eps4_median'], 5), 'eps8', round(o['eps8_median'], 7),
          'accum_r4', [round(a['r4'], 3) for a in o['accum']], f'{time.time() - t0:.1f}s', flush=True)
print('RAUCHTEST OK' if ok else 'RAUCHTEST FEHLER'); sys.exit(0 if ok else 1)
