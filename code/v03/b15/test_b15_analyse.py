# Test der Auswertung b15_analyze.py (aus test_b11_analyse.py; CIFAR-10 d200, kL2B_tr gegen selu_tr, Kontrolle d100 gegen B10) an kuenstlichen Daten.
# Szenarien: BPreCap vorn, SELU vorn, Null (20 Wiederholungen), falsche Kontrolle (Ende und Messpunkt), Ausfall ohne Messpunkt (90 %),
# Nullen in den Differenzen (exakter Test), Aufruf mit "." aus dem Ergebnisordner. Ausgabe "AUSWERTUNG OK" oder Fehler.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
CK = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
STEPS = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'wd', 'sched', 'seed', 'status', 'test_err_w8a8', 'test_err_w8a3a_best', 'sat_rate'] + CK + [f'c{k}_{x}' for k in STEPS[:-1] for x in CK]
ACTS = ('kL2B_tr', 'selu_tr')
def curve(a, scn, j):
    if scn == 'null': return 55 - 4 * min(j, 4) / 4, 4.0
    good = (a == 'kL2B_tr') == (scn == 'vorn')
    return (52 if good else 56) - 4 * min(j, 5) / 5, (3.0 if good else 6.0)
def make(scn, seed=0, fail=None):
    rng = np.random.default_rng(seed); rows = []
    for s in range(42192, 42204):
        for a in ACTS:
            vals = []
            for j in range(8):
                f, c = curve(a, scn, j); f += rng.normal(0, 0.3); w = f + c + rng.normal(0, 0.3)
                vals.append(dict(train_loss_last50=1.5 - 0.1 * j, val_err_fp32=round(f + rng.normal(0, 0.1), 3) / 100, test_err_fp32=f / 100, test_err_w8a4_best=(w + 1) / 100,
                                 test_err_w8a4a_best=w / 100, test_err_w8a3_best=(w + 10) / 100, act_range_max_median=3.0))
            r = dict(run_id=f'cifar10_d200_{a}_s{s}', dataset='cifar10', depth=200, act=a, lr=5e-5, wd=0.3, sched='cos', seed=s, status='ok', test_err_w8a8=0.5, test_err_w8a3a_best=0.75, sat_rate=0.05, **vals[-1])
            for j, k in enumerate(STEPS[:-1]): r.update({f'c{k}_{x}': vals[j][x] for x in CK})
            if fail == (a, s): r = {k: (v if k in ('run_id', 'dataset', 'depth', 'act', 'lr', 'wd', 'sched', 'seed') else '') for k, v in r.items()}; r['status'] = 'nonfinite_loss'
            rows.append(r)
    x = {c: round(float(rng.uniform(0.3, 0.7)), 6) for c in CK}; x.update({f'c{k}_{c}': round(float(rng.uniform(0.3, 0.7)), 6) for k in STEPS[:-1] for c in CK})
    ctl = dict(run_id='cifar10_d100_kL2B_tr_s42168', dataset='cifar10', depth=100, act='kL2B_tr', lr=1e-4, wd=0.3, sched='cos', seed=42168, status='ok', **x)
    return rows + [ctl], [dict(ctl)]
def run(rows, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b15', rows), ('b10', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ: subprocess.run([PY, str(H / 'b15_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b15')
    else: subprocess.run([PY, str(H / 'b15_analyze.py'), str(d / 'b15'), str(d / 'b10')], check=True, capture_output=True)
    return json.load(open(d / 'b15' / 'analysis' / 'b15_result.json'))
ok = True
o = run(*make('vorn')); sv = o['secondary_verdicts']
c = o['success'] and o['repro_ok'] and sv['S1_kL2Btr_vs_selutr_fp32_es'] == 'besser' and o['primary']['wins'] == 12; ok &= c; print('BPreCap vorn ->', o['primary']['verdict'], o['repro_ok'], sv, c)
o = run(*make('hinten')); c = o['primary']['verdict'] == 'schlechter' and o['secondary_verdicts']['S1_kL2Btr_vs_selutr_fp32_es'] == 'schlechter'; ok &= c; print('SELU vorn ->', o['primary']['verdict'], c)
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); fs = sum(any(v != 'unklar' for v in x['secondary_verdicts'].values()) for x in nulls)
c = fp <= 3 and fs <= 2; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, 'sekundaer', fs, c)
rows, old = make('vorn'); old[0] = dict(old[0], test_err_w8a4a_best=old[0]['test_err_w8a4a_best'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Ende) ->', c)
rows, old = make('vorn'); old[0] = dict(old[0], c3440_val_err_fp32=old[0]['c3440_val_err_fp32'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (3440) ->', c)
rows, old = make('vorn', fail=('selu_tr', 42195)); o = run(rows, old); dd = o['primary']['diffs']
c = o['failed'] == 1 and min(dd) < -20 and o['primary']['n'] == 12; ok &= c; print('Ausfall SELU ohne Messpunkt zaehlt 90 % ->', o['failed'], round(min(dd), 2), c)
rows, old = make('null', 3)
for r in rows:   # Nullen erzwingen: identische Werte fuer beide Formen in 4 Seeds
    if r['act'] == 'selu_tr' and r['seed'] in (42192, 42193, 42194, 42195):
        twin = next(q for q in rows if q['act'] == 'kL2B_tr' and q['seed'] == r['seed']); r.update({k: v for k, v in twin.items() if k not in ('run_id', 'act')})
o = run(rows, old); c = sum(v == 0 for v in o['primary']['diffs']) == 4 and 0 <= o['primary']['p'] <= 1; ok &= c; print('Nullen ->', sum(v == 0 for v in o['primary']['diffs']), round(o['primary']['p'], 4), c)
o = run(*make('vorn'), relativ=True); c = o['repro_ok'] and o['success']; ok &= c; print('relativer Pfad ->', c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER')
sys.exit(0 if ok else 1)
