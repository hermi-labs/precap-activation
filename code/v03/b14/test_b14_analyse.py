# Test der Auswertung b14_analyze.py an kuenstlichen Daten (aus dem B12/B13-Test, dazu die harten Formen aus B13 als dritte Quelle). Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: "BPreCap vorn" (eigene Formen bei 4 Bit schlechter), "eigene Formen vorn", Null (20 Wiederholungen), falsche Kontrolle (Ende, Messpunkt),
# Ausfall ohne Messpunkt (90 %), relativer Pfad.
import csv, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable; NAME = H.name
src = (H / f'{NAME}_analyze.py').read_text(); ns = {}
exec(src[src.index('# --- STUDIE ---'):src.index('# --- Ende STUDIE ---')], ns); OWN = ns['OWN']; PRIMARY = ns['PRIMARY']
CK = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
STEPS = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'wd', 'sched', 'seed', 'status'] + CK + [f'c{k}_{x}' for k in STEPS[:-1] for x in CK]
SEEDS = range(42168, 42180)
def row(act, s, f0, cost, rng, rr=3.0):
    vals = []
    for j in range(8):
        f = f0 - 4 * min(j, 5) / 5 + rng.normal(0, 0.3); w = f + cost + rng.normal(0, 0.3)
        vals.append(dict(train_loss_last50=1.5 - 0.1 * j, val_err_fp32=round(f + rng.normal(0, 0.1), 3) / 100, test_err_fp32=f / 100,
                         test_err_w8a4_best=(w + 1) / 100, test_err_w8a4a_best=w / 100, test_err_w8a3_best=(w + 10) / 100, act_range_max_median=rr))
    r = dict(run_id=f'cifar10_d100_{act}_s{s}', dataset='cifar10', depth=100, act=act, lr=1e-4, wd=0.3, sched='cos', seed=s, status='ok', **vals[-1])
    for j, k in enumerate(STEPS[:-1]): r.update({f'c{k}_{x}': vals[j][x] for x in CK})
    return r
def make(scn, seed=0):
    rng = np.random.default_rng(seed); old, new, hard = [], [], []
    co = {'vorn': 9.0, 'hinten': 0.5, 'null': 3.0}[scn]
    for s in SEEDS:
        old.append(row('kL2B_tr', s, 50, 3.0, rng)); old.append(row('LB_tr', s, 51, 2.5 if scn != 'null' else co, rng))
        hard.append(row('kL2Bh_tr', s, 50, 6.0, rng)); hard.append(row('LBh_tr', s, 51, 5.0, rng))
        for a in OWN: new.append(row(a, s, 50 if scn != 'hinten' else 47, co, rng, 9.0))
    new.append(dict(next(r for r in old if r['act'] == 'kL2B_tr' and r['seed'] == 42168)))   # Kontrolle: identische Kopie
    return new, old, hard
def run(new, old, hard, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in ((NAME, new), ('b10', old), ('b13', hard)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ: subprocess.run([PY, str(H / f'{NAME}_analyze.py'), '.'], check=True, capture_output=True, cwd=d / NAME)
    else: subprocess.run([PY, str(H / f'{NAME}_analyze.py'), str(d / NAME), str(d / 'b10'), str(d / 'b13')], check=True, capture_output=True)
    return json.load(open(d / NAME / 'analysis' / f'{NAME}_result.json'))
ok = True
o = run(*make('vorn')); sv = o['secondary_verdicts']
c = o['success'] and o['repro_ok'] and o['n_rows'] == 12 * len(OWN) + 1 and all(abs(o['descriptive'][a]['range_es'] - 9.0) < 1e-9 for a in OWN) and sv['S6_kL2Bp_tr_vs_kL2Bh_tr_w4a'] == 'schlechter' and o['descriptive']['kL2Bh_tr']['n'] == 12
ok &= c; print('BPreCap vorn ->', o['primary_verdicts'], o['repro_ok'], c)
o = run(*make('hinten')); c = all(v == 'schlechter' for v in o['primary_verdicts'].values()) and not o['success']
ok &= c; print('eigene Formen vorn ->', o['primary_verdicts'], c)
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(any(v != 'unklar' for v in x['primary_verdicts'].values()) for x in nulls)
c = fp <= 3; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, c)
new, old, hard = make('vorn'); new[-1] = dict(new[-1], test_err_w8a4a_best=new[-1]['test_err_w8a4a_best'] + 1e-6); o = run(new, old, hard); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Ende) ->', c)
new, old, hard = make('vorn'); new[-1] = dict(new[-1], c4300_val_err_fp32=new[-1]['c4300_val_err_fp32'] + 1e-6); o = run(new, old, hard); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle (Messpunkt) ->', c)
new, old, hard = make('vorn'); a0 = PRIMARY[0][1]; i = next(j for j, r in enumerate(new) if r['act'] == a0 and r['seed'] == 42170)
new[i] = dict(new[i], status='nonfinite_loss', **{x: '' for x in CK}, **{f'c{k}_{x}': '' for k in STEPS[:-1] for x in CK})
o = run(new, old, hard); k0 = next(iter(o['primary'])); d = o['primary'][k0]['diffs'][2]
c = o['failed'] == 1 and d < -30 and 0 in o['descriptive'][a0]['es_steps']; ok &= c; print('Ausfall ohne Messpunkt -> 90 %:', o['failed'], round(d, 2), c)
o = run(*make('vorn'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B10 ->', c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
