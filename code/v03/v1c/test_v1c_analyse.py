# Test der Auswertung v1c_analyze.py an kuenstlichen Daten (mit Claude, Code aus test_v1_analyse.py). Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: "BPreCap vorn und mit Anlauf besser als B15", "SELU vorn", Null (20 Wiederholungen), falsche Kontrolle (Ende, Messpunkt), Ausfall.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
CK = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
STEPS = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'wd', 'sched', 'seed', 'status'] + CK + [f'c{k}_{x}' for k in STEPS[:-1] for x in CK]
ACTS = ('kL2B_tr', 'selu_tr'); SEEDS = range(42192, 42204)
def row(a, s, sched, f, c, rng):
    vals = []
    for j in range(8):
        ff = f - 4 * min(j, 5) / 5 + rng.normal(0, 0.3); w = ff + c + rng.normal(0, 0.3)
        vals.append(dict(train_loss_last50=1.5 - 0.1 * j, val_err_fp32=round(ff + rng.normal(0, 0.1), 3) / 100, test_err_fp32=ff / 100, test_err_w8a4_best=(w + 1) / 100,
                         test_err_w8a4a_best=w / 100, test_err_w8a3_best=0.8, act_range_max_median=3.0))
    r = dict(run_id=f'cifar10_d200_{a}_s{s}', dataset='cifar10', depth=200, act=a, lr=1e-4, wd=0.3, sched=sched, seed=s, status='ok', **vals[-1])
    for j, k in enumerate(STEPS[:-1]): r.update({f'c{k}_{x}': vals[j][x] for x in CK})
    return r
def make(scn, seed=0):
    rng = np.random.default_rng(seed); v1, b10 = [], []
    for s in SEEDS:
        for a in ACTS:
            base = 52 if a == 'kL2B_tr' else 60; cost = 7.0 if a == 'kL2B_tr' else 4.0
            b10.append(row(a, s, 'cos', base, cost, rng))
            if scn == 'vorn': f, c = (base - 1.0, cost) if a == 'kL2B_tr' else (base, cost)
            elif scn == 'hinten': f, c = (base + 9, 9.0) if a == 'kL2B_tr' else (50, 3.0)
            else: f, c = 52, 3.0
            v1.append(row(a, s, 'wcos', f, c, rng))
    for a in ('kL2B_tr', 'selu_tr'):   # Kontrolle: gleicher Lauf wie in B15
        r = next(x for x in b10 if x['act'] == a and x['seed'] == 42192); v1.append(dict(r, run_id=r['run_id'] + '_kontrolle'))
    return v1, b10
def run(v1, b10):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('v1', v1), ('b10', b10)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    subprocess.run([PY, str(H / 'v1c_analyze.py'), str(d / 'v1'), str(d / 'b10')], check=True, capture_output=True)
    return json.load(open(d / 'v1' / 'analysis' / 'v1c_result.json'))
ok = True
o = run(*make('vorn')); sv = o['secondary_verdicts']
c = o['success'] and o['repro_ok'] and sv['S2_kL2Btr_V1c_vs_B15_fp32_es'] == 'besser' and sv['S4_selu_V1c_vs_B15_fp32_es'] == 'unklar'
ok &= c; print('vorn ->', o['primary']['verdict'], o['repro_ok'], sv, c)
o = run(*make('hinten')); c = o['primary']['verdict'] == 'schlechter' and o['secondary_verdicts']['S5_selu_V1c_vs_B15_w4a_es'] == 'besser'
ok &= c; print('hinten ->', o['primary']['verdict'], o['secondary_verdicts']['S5_selu_V1c_vs_B15_w4a_es'], c)
nulls = [run(*make('null', k)) for k in range(20)]
fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); ok &= fp <= 3; print('null (20x) -> Fehlalarme primaer', fp)
v1, b10 = make('vorn'); i = next(j for j, r in enumerate(v1) if r['run_id'].endswith('_kontrolle')); v1[i] = dict(v1[i], test_err_w8a4a_best=v1[i]['test_err_w8a4a_best'] + 1e-6)
o = run(v1, b10); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Ende) ->', c)
v1, b10 = make('vorn'); i = next(j for j, r in enumerate(v1) if r['run_id'].endswith('_kontrolle')); v1[i] = dict(v1[i], c3440_val_err_fp32=v1[i]['c3440_val_err_fp32'] + 1e-6)
o = run(v1, b10); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Messpunkt) ->', c)
v1, b10 = make('vorn'); i = next(j for j, r in enumerate(v1) if r['act'] == 'kL2B_tr' and r['seed'] == 42193 and r['sched'] == 'wcos')
v1[i] = dict(v1[i], status='nonfinite_loss', **{x: '' for x in CK}, **{f'c{k}_{x}': '' for k in STEPS[:-1] for x in CK})
o = run(v1, b10); c = o['failed'] == 1 and 90.0 - 1 in [round(x) for x in []] or (o['failed'] == 1 and max(o['primary']['diffs']) > 20); ok &= c; print('Ausfall ->', o['failed'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
