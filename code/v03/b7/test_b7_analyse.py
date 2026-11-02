# Test der Auswertung b7_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: "haelt" (kL2B bleibt vor LB), "holt auf" (Vorsprung bei 1720, weg bei 6880), Null (20 Wiederholungen, Fehlalarmrate),
# falsche Kontrolle B3, Ausfall vor der Zwischenmessung, Aufruf mit "." aus dem Ergebnisordner.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
sys.path.insert(0, str(H)); CK = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'test_err_w8a8', 'test_err_w8a3a_best', 'sat_rate'] + CK + [f'c{k}_{x}' for k in (1720, 3440) for x in CK]
ACTS = ('kL2B', 'LB', 'tat')
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    # fp32 in pp gegen kL2B je Messzeitpunkt (1720, 3440, 6880); w4a-Preis in pp
    eff = dict(haelt={'LB': (3, 3, 3), 'tat': (-5, -5, -5)}, holtauf={'LB': (3, 1.5, 0), 'tat': (-5, -4, -3)},
               null={'LB': (0, 0, 0), 'tat': (0, 0, 0)})[scn]
    costg = dict(haelt=(4, 5, 6), holtauf=(4, 5, 6), null=(4, 4, 4))[scn]
    for s in range(42132, 42144):
        f0 = [56 + rng.normal(0, 0.5), 53 + rng.normal(0, 0.5), 51 + rng.normal(0, 0.5)]
        for act in ACTS:
            d = (0, 0, 0) if act == 'kL2B' else eff[act]
            vals = []
            for j in range(3):
                f = (f0[j] + d[j] + rng.normal(0, 0.3)) / 100; w = f + (costg[j] + rng.normal(0, 0.3)) / 100
                vals.append(dict(train_loss_last50=1.5 - 0.2 * j, val_err_fp32=f, test_err_fp32=f, test_err_w8a4_best=w + 0.01, test_err_w8a4a_best=w, test_err_w8a3_best=0.8, act_range_max_median=3.0))
            r = dict(run_id=f'cifar10_d100_{act}_s{s}', dataset='cifar10', depth=100, act=act, lr=2e-5, seed=s, status='ok', test_err_w8a8=0.5, test_err_w8a3a_best=0.75, sat_rate=0.06, **vals[2])
            for j, k in enumerate((1720, 3440)): r.update({f'c{k}_{x}': vals[j][x] for x in CK})
            rows.append(r)
    old = []
    for act in ('kL2B', 'LB'):
        x = {c: round(float(rng.uniform(0.3, 0.7)), 6) for c in CK}
        rows.append(dict(run_id=f'cifar10_d100_{act}_s42084', dataset='cifar10', depth=100, act=act, lr=2e-5, seed=42084, status='ok', **{f'c1720_{c}': x[c] for c in CK}, **{c: 0.5 for c in CK}))
        old.append(dict(run_id=f'cifar10_d100_{act}_s42084', dataset='cifar10', depth=100, act=act, lr=2e-5, seed=42084, status='ok', **x))
    return rows, old
def run(rows, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b7', rows), ('b3', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ: subprocess.run([PY, str(H / 'b7_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b7')
    else: subprocess.run([PY, str(H / 'b7_analyze.py'), str(d / 'b7'), str(d / 'b3')], check=True, capture_output=True)
    return json.load(open(d / 'b7' / 'analysis' / 'b7_result.json'))
ok = True
o = run(*make('haelt')); sv = o['secondary_verdicts']
c = o['success'] and o['repro_ok'] and sv['S3_gap_kL2B_LB_fp32_6880_minus_1720'] == 'unklar' and sv['S4_kL2B_vs_tat_fp32'] == 'schlechter' \
    and sv['S7_cost4a_kL2B_6880_minus_1720'] == 'schlechter' and abs(o['gaps']['kL2B_vs_LB_fp32_6880'] + 3) < 0.5 and len(o['descriptive']) == 9
ok &= c; print('haelt ->', o['primary']['verdict'], o['repro_ok'], sv, c)
o = run(*make('holtauf')); sv = o['secondary_verdicts']
c = not o['success'] and o['primary']['verdict'] == 'unklar' and sv['S3_gap_kL2B_LB_fp32_6880_minus_1720'] == 'schlechter' and sv['S8_gap_kL2B_tat_fp32_6880_minus_1720'] == 'besser' \
    and abs(o['gaps']['kL2B_vs_LB_fp32_1720'] + 3) < 0.5
ok &= c; print('holt auf ->', o['primary']['verdict'], sv, c)
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); fs = sum(any(v != 'unklar' for v in x['secondary_verdicts'].values()) for x in nulls)
c = fp <= 3 and fs <= 2; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, 'sekundaer', fs, c)
rows, old = make('haelt'); old[1] = dict(old[1], test_err_w8a4a_best=old[1]['test_err_w8a4a_best'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle B3 ->', not o['repro_ok'], c)
rows, old = make('haelt'); rows[1] = dict(rows[1], status='nonfinite_loss', **{f'c{k}_{x}': '' for k in (1720, 3440) for x in CK}); o = run(rows, old)
c = o['failed'] == 1 and -7 < o['secondary']['S3_gap_kL2B_LB_fp32_6880_minus_1720']['diffs'][0] < -3 and o['primary']['diffs'][0] < -20
ok &= c; print('Ausfall ->', o['failed'], o['primary']['diffs'][0], c)
o = run(*make('haelt'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B3 ->', o['repro_ok'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
