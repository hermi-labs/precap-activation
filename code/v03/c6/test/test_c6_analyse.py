# Test der Auswertung c6_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: BPreCap vorn, TAT vorn, Null (20x), Ausfall ohne Messpunkt (90 %), Wahl des Stopp-Punkts, Plattformvergleich mit den echten MLX-Studien.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; C6 = H.parent; PY = sys.executable
MET = ['val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median', 'act_range_p9999_median']
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020)
COLS = ['run_id', 'act', 'lr', 'wd', 'seed', 'source', 'status', 'device', 'torch'] + MET + [f'c{k}_{x}' for k in CK for x in MET]
FORMS = [('kL2B_tr', 1e-4, 0.3), ('LB_tr', 1e-4, 0.3), ('tat', 1e-4, 0.3), ('tat', 2e-4, 0.0), ('selu_tr', 2e-4, 0.3)]
def make(scn, seed=0, fail=None):
    rng = np.random.default_rng(seed); rows = []
    for s in range(42168, 42180):
        for a, lr, wd in FORMS:
            good = a == 'kL2B_tr'; cost = 3.0 if (good or scn == 'null') else 8.0
            if scn == 'hinten': cost = 8.0 if good else 3.0
            vals = []
            for j in range(8):
                f = 55 - 4 * min(j, 5) / 5 + rng.normal(0, 0.3); w = f + cost + rng.normal(0, 0.3)
                vals.append(dict(val_err_fp32=(f + rng.normal(0, 0.1)) / 100, test_err_fp32=f / 100, test_err_w8a4_best=(w + 1) / 100, test_err_w8a4a_best=w / 100,
                                 test_err_w8a3_best=(w + 10) / 100, act_range_max_median=3.0, act_range_p9999_median=2.9))
            r = dict(run_id=f'c6_{a}_lr{lr:g}_wd{wd:g}_s{s}', act=a, lr=lr, wd=wd, seed=s, source='x', status='ok', device='cuda', torch='t', **vals[-1])
            for j, k in enumerate(CK): r.update({f'c{k}_{x}': vals[j][x] for x in MET})
            if fail == (a, lr, s): r = {k: (v if k in ('run_id', 'act', 'lr', 'wd', 'seed', 'source') else '') for k, v in r.items()}; r['status'] = 'nonfinite_loss'
            rows.append(r)
    return rows
def run(rows, st=None):
    d = Path(tempfile.mkdtemp()) / 'c6'; (d / 'workers').mkdir(parents=True)
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rows]
    subprocess.run([PY, str(C6 / 'c6_analyze.py'), str(d), str(st or d.parent)], check=True, capture_output=True)
    return json.load(open(d / 'analysis' / 'c6_result.json'))
ok = True
o = run(make('vorn')); c = o['success'] and o['secondary_verdicts']['S2_bp_vs_t0_w4a'] == 'besser' and o['primary']['wins'] == 12; ok &= c; print('BPreCap vorn ->', o['primary']['verdict'], o['secondary_verdicts'], c)
o = run(make('hinten')); c = o['primary']['verdict'] == 'schlechter'; ok &= c; print('TAT vorn ->', o['primary']['verdict'], c)
nulls = [run(make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); fs = sum(any(v != 'unklar' for v in x['secondary_verdicts'].values()) for x in nulls)
c = fp <= 3 and fs <= 2; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, 'sekundaer', fs, c)
o = run(make('vorn', fail=('tat', 1e-4, 42170))); c = o['failed'] == 1 and min(o['primary']['diffs']) < -30; ok &= c; print('Ausfall TAT a zaehlt 90 % ->', o['failed'], round(min(o['primary']['diffs']), 2), c)
o = run(make('vorn')); c = all(s >= 5160 for s in o['descriptive']['bp']['es_steps']); ok &= c; print('Stopp-Punkte ab Index 5 ->', set(o['descriptive']['bp']['es_steps']), c)
o = run(make('vorn'), st=(C6 / 'mlx_vergleich') if (C6 / 'mlx_vergleich').exists() else C6.parent); m = o['descriptive']['bp'].get('mlx'); c = m is not None and abs(m['test_err_fp32'] - 46.52) < 0.01 and 'mlx' in o['descriptive']['ta'] and 'mlx' in o['descriptive']['se']
ok &= c; print('Plattformvergleich mit echten MLX-Studien ->', m, c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
