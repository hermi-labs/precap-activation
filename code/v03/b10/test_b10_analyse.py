# Test der Auswertung b10_analyze.py (aus test_b9_analyse.py) an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: "BPreCap vorn" (TAT ueberangepasst, Val-Minimum frueh), "TAT vorn", Null (20 Wiederholungen), Wahl des Stopp-Punkts,
# falsche Kontrolle B9 (auch nur an einem Messpunkt), Ausfall mit und ohne fruehere Messpunkte, Aufruf mit "." aus dem Ergebnisordner.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
CK = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
STEPS = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'wd', 'sched', 'seed', 'status', 'test_err_w8a8', 'test_err_w8a3a_best', 'sat_rate'] + CK + [f'c{k}_{x}' for k in STEPS[:-1] for x in CK]
ACTS = ('kL2B_tr', 'LB_tr', 'tat')
def curve(a, scn, j):   # Val/Test-Fehler fp32 (%) und w4a-Preis (pp) an Messpunkt j
    if scn == 'null': return 55 - 4 * min(j, 4) / 4, 4.0
    if a == 'tat':      # schnell, dann Ueberanpassung ab j = 3, Preis waechst
        f = 53 - 3 * min(j, 3) / 3 + 0.6 * max(0, j - 3); return f, 5 + 0.8 * j
    if a == 'kL2B_tr': return 55 - 4 * min(j, 5) / 5, 4.0 + (0 if scn == 'vorn' else 4)
    return 56 - 4 * min(j, 5) / 5, 3.5 + (0 if scn == 'vorn' else 4)
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for s in range(42168, 42180):
        for a in ACTS:
            vals = []
            for j in range(8):
                f, c = curve(a, scn, j); f += rng.normal(0, 0.3); w = f + c + rng.normal(0, 0.3)
                vals.append(dict(train_loss_last50=1.5 - 0.1 * j, val_err_fp32=round(f + rng.normal(0, 0.1), 3) / 100, test_err_fp32=f / 100, test_err_w8a4_best=(w + 1) / 100,
                                 test_err_w8a4a_best=w / 100, test_err_w8a3_best=0.8, act_range_max_median=3.0))
            r = dict(run_id=f'cifar10_d100_{a}_s{s}', dataset='cifar10', depth=100, act=a, lr=1e-4, wd=0.3, sched='cos', seed=s, status='ok', test_err_w8a8=0.5, test_err_w8a3a_best=0.75, sat_rate=0.05, **vals[-1])
            for j, k in enumerate(STEPS[:-1]): r.update({f'c{k}_{x}': vals[j][x] for x in CK})
            rows.append(r)
    old = []
    for a in ('kL2B_tr', 'tat'):
        x = {c: round(float(rng.uniform(0.3, 0.7)), 6) for c in CK}; x.update({f'c{k}_{c}': round(float(rng.uniform(0.3, 0.7)), 6) for k in STEPS[:-1] for c in CK})
        rows.append(dict(run_id=f'cifar10_d100_{a}_s42156', dataset='cifar10', depth=100, act=a, lr=5e-5, wd=0.0, sched='const', seed=42156, status='ok', **x))
        old.append(dict(run_id=f'cifar10_d100_{a}_s42156', dataset='cifar10', depth=100, act=a, lr=5e-5, wd=0.0, sched='const', seed=42156, status='ok', **x))
    return rows, old
def run(rows, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b10', rows), ('b9', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ: subprocess.run([PY, str(H / 'b10_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b10')
    else: subprocess.run([PY, str(H / 'b10_analyze.py'), str(d / 'b10'), str(d / 'b9')], check=True, capture_output=True)
    return json.load(open(d / 'b10' / 'analysis' / 'b10_result.json'))
ok = True
o = run(*make('vorn')); sv = o['secondary_verdicts']; es_t = o['descriptive']['tat']['es_steps']; es_k = o['descriptive']['kL2B_tr']['es_steps']
c = o['success'] and o['repro_ok'] and sv['S6_kL2Btr_vs_tat_w4a_6880'] == 'besser' and sv['S4_kL2Btr_vs_LBtr_fp32_es'] == 'besser' \
    and all(s in (3440, 4300) for s in es_t) and all(s >= 5160 for s in es_k)   # Minimum TAT bei Index 3 (3440), mit Rauschen auch 4300; kL2B_tr ab Index 5
ok &= c; print('BPreCap vorn ->', o['primary']['verdict'], o['repro_ok'], sv, 'Stopp tat', set(es_t), 'kL2B_tr', set(es_k), c)
o = run(*make('hinten')); sv = o['secondary_verdicts']
c = o['primary']['verdict'] == 'schlechter' and sv['S3_LBtr_vs_tat_w4a_es'] == 'schlechter'
ok &= c; print('TAT vorn ->', o['primary']['verdict'], sv, c)
nulls = [run(*make('null', k)) for k in range(20)]; fp = sum(x['primary']['verdict'] != 'unklar' for x in nulls); fs = sum(any(v != 'unklar' for v in x['secondary_verdicts'].values()) for x in nulls)
c = fp <= 3 and fs <= 2; ok &= c; print('null (20x) -> Fehlalarme primaer', fp, 'sekundaer', fs, c)
rows, old = make('vorn'); old[1] = dict(old[1], test_err_w8a4a_best=old[1]['test_err_w8a4a_best'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle B9 (Ende) ->', not o['repro_ok'], c)
rows, old = make('vorn'); old[0] = dict(old[0], c3440_val_err_fp32=old[0]['c3440_val_err_fp32'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle B9 (Messpunkt 3440) ->', not o['repro_ok'], c)
rows, old = make('vorn'); o = run(rows, old); c = o['repro_ok'] and o['descriptive']['tat']['wd'] == 0.3; ok &= c
print('richtige Kontrolle B9, WD gelesen ->', o['repro_ok'], o['descriptive']['tat']['wd'], c)
# Ausfall: tat Seed 42168 nach 3440 ausgefallen (Messpunkte bis 3440 da) -> Stopp bei 2580/3440, kein 90 %; kL2B_tr Seed 42169 ganz ohne Messpunkte -> 90 %
rows, old = make('vorn')
i = next(j for j, r in enumerate(rows) if r['act'] == 'tat' and r['seed'] == 42168); rows[i] = dict(rows[i], status='nonfinite_loss', **{x: '' for x in CK}, **{f'c{k}_{x}': '' for k in (4300, 5160, 6020) for x in CK})
i = next(j for j, r in enumerate(rows) if r['act'] == 'kL2B_tr' and r['seed'] == 42169); rows[i] = dict(rows[i], status='nonfinite_loss', **{x: '' for x in CK}, **{f'c{k}_{x}': '' for k in (860, 1720, 2580, 3440, 4300, 5160, 6020) for x in CK})
o = run(rows, old); d0, d1 = o['primary']['diffs'][0], o['primary']['diffs'][1]
c = o['failed'] == 2 and abs(d0) < 5 and d1 > 20 and 0 not in o['descriptive']['tat']['es_steps'] and 0 in o['descriptive']['kL2B_tr']['es_steps']
ok &= c; print('Ausfall ->', o['failed'], round(d0, 2), round(d1, 2), c)
o = run(*make('vorn'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B9 ->', o['repro_ok'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
