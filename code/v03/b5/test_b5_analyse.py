# Test der Auswertung b5_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: Erfolg (kL2B besser als LB in fp32), "LB holt auf" (kein Effekt), Null (20 Wiederholungen, Fehlalarmrate), falsche Kontrolle B1,
# falsche Tuning-Kontrolle B3, Ausfall, Aufruf mit "." aus dem Ergebnisordner.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a8', 'test_err_w8a4_best',
        'test_err_w8a4a_best', 'test_err_w8a3_best', 'test_err_w8a3a_best', 'act_range_max_median', 'grad_log10_first_last', 'ms_per_step', 'sat_rate']
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    eff = dict(erfolg={'kL2B': (-4, 0), 'kL2': (-8, 10), 'tat': (-12, -3)}, aufholen={'kL2B': (0, 0), 'kL2': (-4, 10), 'tat': (-10, -3)},
               null={a: (0, 0) for a in ('kL2B', 'kL2', 'tat')})[scn]
    for s in range(42108, 42120):
        f0 = 62 + rng.normal(0, 0.5); w0 = 70 + rng.normal(0, 1); n = lambda sd: rng.normal(0, sd)
        for act in ('LB',) + tuple(eff):
            df, dw = (0, 0) if act == 'LB' else eff[act]
            rows.append(dict(run_id=f'cifar10_d200_{act}_s{s}', dataset='cifar10', depth=200, act=act, lr=5e-6, seed=s, status='ok', train_loss_last50=1.7,
                             val_err_fp32=0.6, test_err_fp32=(f0 + df + n(0.3)) / 100, test_err_w8a8=0.6, test_err_w8a4_best=(w0 + dw + n(0.8)) / 100,
                             test_err_w8a4a_best=(w0 + dw + n(0.8)) / 100, test_err_w8a3_best=0.85, test_err_w8a3a_best=0.83,
                             act_range_max_median=3, grad_log10_first_last=0.7, ms_per_step=45, sat_rate=0.06 if act in ('kL2B', 'LB') else ''))
    rep = [dict(rows[0], run_id='mnist_d100_kL2B_s42060', dataset='mnist', depth=100, seed=42060, act='kL2B')]
    tune = []
    for act in ('kL2B', 'LB', 'kL2'):
        for s in range(42000, 42004):
            for lr in (3e-6, 5e-6, 7e-6, 1e-5):
                tune.append(dict(rows[0], run_id=f'cifar10_d200_{act}_s{s}_lr{lr:g}', act=act, seed=s, lr=lr, val_err_fp32=round(0.6 + rng.normal(0, 0.01), 4)))
    t3 = [dict(r) for r in tune if r['lr'] == 1e-5] + [dict(tune[0], run_id='cifar10_d200_kL2B_s42000_lr2e-05', lr=2e-5)]
    return rows + rep, [dict(r) for r in rep], tune, t3
def run(rows, old, tune, t3, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b5', rows), ('b1', old), ('b5/tune', tune), ('b3/tune', t3)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    with open(d / 'b5' / 'lr_tuned.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'depth', 'act', 'lr_sel', 'med_val_acc', 'at_edge', 'n_seeds']); w.writeheader()
        for a, l in (('kL2B', 5e-6), ('LB', 3e-6), ('kL2', 1e-5)): w.writerow(dict(dataset='cifar10', depth=200, act=a, lr_sel=l, med_val_acc=0.4, at_edge=False, n_seeds=4))
    if relativ: subprocess.run([PY, str(H / 'b5_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b5')
    else: subprocess.run([PY, str(H / 'b5_analyze.py'), str(d / 'b5'), str(d / 'b1'), str(d / 'b3')], check=True, capture_output=True)
    return json.load(open(d / 'b5' / 'analysis' / 'b5_result.json'))
ok = True
o = run(*make('erfolg')); sv = o['secondary_verdicts']
c = (o['success'] and o['repro_ok'] and o['tune_repro_ok'] and sv['S3_kL2_vs_LB_fp32'] == 'besser' and sv['S6_kL2B_vs_tat_w4a'] == 'schlechter'
     and o['p8_prediction']['p8_order_holds'] and o['lr_tuned']['LB_d200']['at_lower_edge'] and not o['lr_tuned']['kL2B_d200']['at_lower_edge'] and len(o['tune_curve']) == 12)
ok &= c; print('erfolg ->', o['primary']['verdict'], o['repro_ok'], o['tune_repro_ok'], sv, o['p8_prediction'], c)
o = run(*make('aufholen')); c = not o['success'] and o['primary']['verdict'] == 'unklar' and o['secondary_verdicts']['S3_kL2_vs_LB_fp32'] == 'besser'
ok &= c; print('LB holt auf ->', o['primary']['verdict'], o['primary']['med_diff_pp'], c)
# Null: Primaer ist eine Zelle mit alpha 0,05, also etwa 5 % Fehlalarme; Sekundaer Holm ueber 8. Geprueft wird die Rate ueber 20 Wiederholungen.
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); fs = sum(any(v != 'unklar' for v in x['secondary_verdicts'].values()) for x in nulls)
c = fp <= 3 and fs <= 2; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, 'sekundaer', fs, c)
rows, old, tune, t3 = make('erfolg'); old[0] = dict(old[0], train_loss_last50=old[0]['train_loss_last50'] + 1e-6); o = run(rows, old, tune, t3); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle B1 ->', not o['repro_ok'], c)
rows, old, tune, t3 = make('erfolg'); t3[3] = dict(t3[3], val_err_fp32=0.123); o = run(rows, old, tune, t3); c = not o['tune_repro_ok']; ok &= c
print('falsche Tuning-Kontrolle B3 ->', not o['tune_repro_ok'], c)
rows, old, tune, t3 = make('erfolg'); rows[1] = dict(rows[1], status='nonfinite_loss'); o = run(rows, old, tune, t3); c = o['failed'] == 1; ok &= c; print('Ausfall ->', o['failed'], c)
o = run(*make('erfolg'), relativ=True); c = o['repro_ok'] and o['tune_repro_ok']; ok &= c; print('relativer Pfad findet B1 und B3 ->', o['repro_ok'], o['tune_repro_ok'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
