# Rauchtest D3b (mit Claude): kurzes Training mit Anlauf (60 Schritte, Anlauf 20) fuer BPreCap und SELU bei d200, dann die volle D3-Messung.
# Prueft, dass Training, Plan und alle Kennzahlen endlich sind. Aendert nur Kopien der Konstanten im geladenen Modul, keine Datei.
import importlib.util, math, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location('d3b', H / 'd3b_diag.py'); M = importlib.util.module_from_spec(sp); sys.modules['d3b'] = M; sp.loader.exec_module(M)
def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)
M.V.STEPS, M.V.WARM = 60, 20; ok = True
for c in (M.RUNS[0], M.RUNS[2]):
    t0 = time.time(); m, status, D = M.train(c); RA = M.V.calib_asym(m, D[0][:M.V.NCAL]); o = M.D3.measure(m, D, {4: RA['mse4'], 8: RA['max']}, 200)
    f = status == 'ok' and finite(o); ok &= f
    print(c['name'], status, 'endlich' if f else 'NICHT ENDLICH', 'eps4', round(o['eps4_median'], 5), 'gain4', round(o['gain4_median'], 3),
          'frac_soft', round(o['frac_soft_median'], 4), f'{time.time() - t0:.1f}s', flush=True)
print('RAUCHTEST OK' if ok else 'RAUCHTEST FEHLER'); sys.exit(0 if ok else 1)
