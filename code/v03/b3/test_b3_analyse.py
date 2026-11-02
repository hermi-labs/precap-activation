# Test der Auswertung b3_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: Erfolg (kL2B besser als LB in fp32), Null (6 Wiederholungen), falsche Kontrolle, Ausfall, Aufruf mit "." aus dem Ergebnisordner.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'test_err_fp32', 'test_err_w8a8', 'test_err_w8a4_best',
        'test_err_w8a4a_best', 'test_err_w8a3_best', 'test_err_w8a3a_best', 'act_range_max_median', 'grad_log10_first_last', 'ms_per_step', 'sat_rate']
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    # (fp32-Verschiebung, w4-Verschiebung) gegen LB, in pp
    eff = dict(erfolg={'kL2B': (-1.5, 2), 'kL2': (-3, 30), 'tat': (-4, 3)}, null={a: (0, 0) for a in ('kL2B', 'kL2', 'tat')})[scn]
    for d in (100, 200):
        for s in range(42084, 42096):
            f0 = 55 + rng.normal(0, 0.5); w0 = 62 + rng.normal(0, 1); n = lambda sd: rng.normal(0, sd)
            for act in ('LB',) + tuple(eff):
                df, dw = (0, 0) if act == 'LB' else eff[act]
                rows.append(dict(run_id=f'cifar10_d{d}_{act}_s{s}', dataset='cifar10', depth=d, act=act, lr=1e-4, seed=s, status='ok', train_loss_last50=1.2,
                                 test_err_fp32=(f0 + df + n(0.3)) / 100, test_err_w8a8=0.55, test_err_w8a4_best=(w0 + dw + n(0.8)) / 100,
                                 test_err_w8a4a_best=(w0 + dw + n(0.8)) / 100, test_err_w8a3_best=(w0 + 8 + n(1)) / 100, test_err_w8a3a_best=(w0 + 6 + n(1)) / 100,
                                 act_range_max_median=3, grad_log10_first_last=0.4, ms_per_step=25, sat_rate=0.05 if act in ('kL2B', 'LB') else ''))
    rep = [dict(rows[0], run_id='mnist_d100_kL2B_s42060', dataset='mnist', seed=42060, act='kL2B')]
    return rows + rep, [dict(r) for r in rep]
def run(rows, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b3', rows), ('b1', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    with open(d / 'b3' / 'lr_tuned.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'depth', 'act', 'lr_sel', 'med_val_acc', 'at_edge', 'n_seeds']); w.writeheader()
        for dd in (100, 200):
            for a in ('kL2B', 'LB', 'kL2', 'tat'): w.writerow(dict(dataset='cifar10', depth=dd, act=a, lr_sel=1e-4, med_val_acc=0.45, at_edge=False, n_seeds=4))
    if relativ: subprocess.run([PY, str(H / 'b3_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b3')
    else: subprocess.run([PY, str(H / 'b3_analyze.py'), str(d / 'b3'), str(d / 'b1')], check=True, capture_output=True)
    return json.load(open(d / 'b3' / 'analysis' / 'b3_result.json'))
ok = True
o = run(*make('erfolg')); c = o['success'] and o['repro_ok'] and o['verdict'] == 'besser als LB' and o['p8_prediction']['d100']['p8_order_holds'] and len(o['lr_tuned']) == 8
ok &= c; print('erfolg ->', o['verdict'], o['repro_ok'], o['secondary_verdicts'], o['p8_prediction'], c)
nulls = [run(*make('null', k))['verdict'] for k in range(6)]; c = all(v == 'kein Effekt' for v in nulls); ok &= c; print('null ->', nulls, c)
rows, old = make('erfolg'); old[0] = dict(old[0], train_loss_last50=old[0]['train_loss_last50'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle ->', not o['repro_ok'], c)
rows, old = make('erfolg'); rows[1] = dict(rows[1], status='nonfinite_loss'); o = run(rows, old); c = o['failed'] == 1; ok &= c; print('Ausfall ->', o['failed'], c)
o = run(*make('erfolg'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B1 ->', o['repro_ok'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
