# Rauchtest D1: Messfunktionen an untrainierten Netzen (Init) mit echten CIFAR-10-Daten, ohne Training
import sys, json, time; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); import d1_diag as D
import mlx.core as mx; mx.set_cache_limit(D.B.CACHE_LIMIT)
for act in ('kL2B_tr', 'tat'):
    t0 = time.time(); Dd = D.B.data_for('cifar10', D.B.base(act) + ('_tr' if D.B.recipe(act) else ''))
    mx.random.seed(1); m = D.B.Net(100, act, 1, din=Dd[0].shape[1]); mx.eval(m.parameters()); m.eval()
    R = D.B.calib_asym(m, Dd[0][:D.B.NCAL])['mse4']; o = D.measure(m, Dd, R)
    print(act, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ('per_layer', 'propagation')}), 'sec', round(time.time() - t0, 1))
    print('  prop', [(p['layer'], round(p['eps_local'], 5), round(p['gain'], 3), round(p['noise_gain'], 3)) for p in o['propagation']])
