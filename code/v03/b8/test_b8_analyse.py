# Test der Auswertung b8_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: "haelt" (kL2B_tr bei w4a besser als TAT, fp32 knapp schlechter), "TAT vorn" (Primaer schlechter), Null (20 Wiederholungen,
# Fehlalarmrate), falsche Kontrolle B1, Ausfall, Aufruf mit "." aus dem Ergebnisordner.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a8', 'test_err_w8a4_best',
        'test_err_w8a4a_best', 'test_err_w8a3_best', 'test_err_w8a3a_best', 'act_range_max_median', 'grad_log10_first_last', 'ms_per_step', 'sat_rate']
ACTS = ('kL2B_tr', 'LB_tr', 'tat')
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    # (fp32, w4) in pp gegen kL2B
    # (fp32, w4) in pp gegen TAT
    eff = dict(hilft={'kL2B_tr': (1, -1.5), 'LB_tr': (2.5, -1.5)}, egal={'kL2B_tr': (1, 3), 'LB_tr': (2.5, 3)},
               null={a: (0, 0) for a in ACTS if a != 'tat'})[scn]
    for s in range(42144, 42156):
        f0 = 56 + rng.normal(0, 0.5); w0 = 61 + rng.normal(0, 1); n = lambda sd: rng.normal(0, sd)
        for act in ACTS:
            df, dw = (0, 0) if act == 'tat' else eff[act]
            rows.append(dict(run_id=f'cifar10_d200_{act}_s{s}', dataset='cifar10', depth=200, act=act, lr=2e-5, seed=s, status='ok', train_loss_last50=1.5,
                             val_err_fp32=0.56, test_err_fp32=(f0 + df + n(0.3)) / 100, test_err_w8a8=0.56, test_err_w8a4_best=(w0 + dw + n(0.8)) / 100,
                             test_err_w8a4a_best=(w0 + dw + n(0.8)) / 100, test_err_w8a3_best=0.8, test_err_w8a3a_best=0.75,
                             act_range_max_median=3, grad_log10_first_last=0.7, ms_per_step=25, sat_rate=0.06 if act in ('kL2B_tr', 'LB_tr') else ''))
    rep = [dict(rows[2], run_id='mnist_d100_kL2B_s42060', dataset='mnist', depth=100, seed=42060, act='kL2B')]
    tune = [dict(rows[0], run_id=f'cifar10_d200_{a}_s{s}_lr{lr:g}', act=a, seed=s, lr=lr, val_err_fp32=round(0.56 + rng.normal(0, 0.01), 4))
            for a in ('kL2B_tr', 'LB_tr') for s in range(42000, 42004) for lr in (5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4)]
    return rows + rep, [dict(r) for r in rep], tune
def run(rows, old, tune, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b8', rows), ('b1', old), ('b8/tune', tune)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    with open(d / 'b8' / 'lr_tuned.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'depth', 'act', 'lr_sel', 'med_val_acc', 'at_edge', 'n_seeds']); w.writeheader()
        for a, l, e in (('kL2B_tr', 2e-5, False), ('LB_tr', 5e-6, True)): w.writerow(dict(dataset='cifar10', depth=200, act=a, lr_sel=l, med_val_acc=0.45, at_edge=e, n_seeds=4))
    if relativ: subprocess.run([PY, str(H / 'b8_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b8')
    else: subprocess.run([PY, str(H / 'b8_analyze.py'), str(d / 'b8'), str(d / 'b1')], check=True, capture_output=True)
    return json.load(open(d / 'b8' / 'analysis' / 'b8_result.json'))
ok = True
o = run(*make('hilft')); sv = o['secondary_verdicts']
c = o['success'] and o['repro_ok'] and sv['S1_kL2Btr_vs_tat_fp32'] == 'schlechter' and sv['S4_kL2Btr_vs_LBtr_fp32'] == 'besser' and sv['S3_LBtr_vs_tat_w4a'] == 'besser' \
    and o['lr_tuned']['LB_tr_d200']['at_edge'] and not o['lr_tuned']['kL2B_tr_d200']['at_edge'] and len(o['tune_curve']) == 12 and o['order_fp32'][0] == 'tat'
ok &= c; print('haelt ->', o['primary']['verdict'], o['repro_ok'], sv, o['order_fp32'], c)
o = run(*make('egal')); sv = o['secondary_verdicts']
c = not o['success'] and o['primary']['verdict'] == 'schlechter' and sv['S2_kL2Btr_vs_tat_w4'] == 'schlechter' and sv['S8_kL2Btr_vs_LBtr_w4'] == 'unklar'
ok &= c; print('TAT vorn ->', o['primary']['verdict'], sv, c)
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); fs = sum(any(v != 'unklar' for v in x['secondary_verdicts'].values()) for x in nulls)
c = fp <= 3 and fs <= 2; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, 'sekundaer', fs, c)
rows, old, tune = make('hilft'); old[0] = dict(old[0], train_loss_last50=old[0]['train_loss_last50'] + 1e-6); o = run(rows, old, tune); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle B1 ->', not o['repro_ok'], c)
rows, old, tune = make('hilft'); rows[1] = dict(rows[1], status='nonfinite_loss'); o = run(rows, old, tune); c = o['failed'] == 1; ok &= c; print('Ausfall ->', o['failed'], c)
o = run(*make('hilft'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B1 ->', o['repro_ok'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
