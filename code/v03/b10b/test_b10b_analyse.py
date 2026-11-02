# Test der Auswertung b10b_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: "BPreCap vorn", "TAT mit WD vorn", Null (20 Wiederholungen), falsche Kontrolle (Ende und Messpunkt), Ausfall ohne Messpunkt, relativer Pfad.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
CK = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
STEPS = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'wd', 'sched', 'seed', 'status'] + CK + [f'c{k}_{x}' for k in STEPS[:-1] for x in CK]
SEEDS = range(42168, 42180)
def row(act, lr, wd, s, f0, cost, rng, rng_range=3.0):
    vals = []
    for j in range(8):
        f = f0 - 4 * min(j, 5) / 5 + rng.normal(0, 0.3); w = f + cost + rng.normal(0, 0.3)
        vals.append(dict(train_loss_last50=1.5 - 0.1 * j, val_err_fp32=round(f + rng.normal(0, 0.1), 3) / 100, test_err_fp32=f / 100,
                         test_err_w8a4_best=(w + 1) / 100, test_err_w8a4a_best=w / 100, test_err_w8a3_best=0.8, act_range_max_median=rng_range))
    r = dict(run_id=f'cifar10_d100_{act}_s{s}_lr{lr:g}_wd{wd:g}', dataset='cifar10', depth=100, act=act, lr=lr, wd=wd, sched='cos', seed=s, status='ok', **vals[-1])
    for j, k in enumerate(STEPS[:-1]): r.update({f'c{k}_{x}': vals[j][x] for x in CK})
    return r
def make(scn, seed=0):
    rng = np.random.default_rng(seed); old, new = [], []
    ca = {'vorn': 8.0, 'hinten': 1.0, 'null': 3.0}[scn]
    for s in SEEDS:
        old.append(row('kL2B_tr', 1e-4, 0.3, s, 50, 3.0, rng))
        old.append(row('tat', 2e-4, 0.0, s, 50, 14.0, rng, 28.0))
        new.append(row('tat', 1e-4, 0.3, s, 50, ca, rng, 8.0))
        new.append(row('tat', 5e-5, 3.0, s, 51, ca + 0.5, rng, 6.0))
    c = next(r for r in old if r['act'] == 'tat' and r['seed'] == 42168); new.append(dict(c))   # Kontrolle: identische Kopie
    return new, old
def run(new, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b10b', new), ('b10', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ: subprocess.run([PY, str(H / 'b10b_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b10b')
    else: subprocess.run([PY, str(H / 'b10b_analyze.py'), str(d / 'b10b'), str(d / 'b10')], check=True, capture_output=True)
    return json.load(open(d / 'b10b' / 'analysis' / 'b10b_result.json'))
ok = True
o = run(*make('vorn')); sv = o['secondary_verdicts']
c = o['success'] and o['repro_ok'] and sv['S1_kL2Btr_vs_tatb_w4a'] == 'besser' and sv['S7_tata_vs_tat0_w4a'] == 'besser' and o['n_rows'] == 25 and abs(o['descriptive']['tat_a']['range_es'] - 8.0) < 1e-9
ok &= c; print('BPreCap vorn ->', o['primary']['verdict'], o['repro_ok'], sv, c)
o = run(*make('hinten')); sv = o['secondary_verdicts']
c = o['primary']['verdict'] == 'schlechter' and sv['S1_kL2Btr_vs_tatb_w4a'] == 'unklar' or (o['primary']['verdict'] == 'schlechter' and sv['S1_kL2Btr_vs_tatb_w4a'] == 'schlechter')
ok &= c; print('TAT mit WD vorn ->', o['primary']['verdict'], sv, c)
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls)
c = fp <= 3; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, c)
new, old = make('vorn'); new[-1] = dict(new[-1], test_err_w8a4a_best=new[-1]['test_err_w8a4a_best'] + 1e-6); o = run(new, old); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Ende) ->', c)
new, old = make('vorn'); new[-1] = dict(new[-1], c2580_val_err_fp32=new[-1]['c2580_val_err_fp32'] + 1e-6); o = run(new, old); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Messpunkt) ->', c)
new, old = make('vorn'); i = next(j for j, r in enumerate(new) if r['lr'] == 1e-4 and r['seed'] == 42170)
new[i] = dict(new[i], status='nonfinite_loss', **{x: '' for x in CK}, **{f'c{k}_{x}': '' for k in STEPS[:-1] for x in CK})
o = run(new, old); d = o['primary']['diffs'][2]; c = o['failed'] == 1 and d < -30 and 0 in o['descriptive']['tat_a']['es_steps']; ok &= c
print('Ausfall TAT a ohne Messpunkt -> 90 %:', o['failed'], round(d, 2), c)
o = run(*make('vorn'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B10 ->', c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
