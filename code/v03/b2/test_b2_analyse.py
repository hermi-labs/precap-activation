# Test der Auswertung b2_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: Erfolg (weich besser bei 4 Bit, fp32 gleich), weich teuer in fp32, Null (6 Wiederholungen), falsche Kontrolle, Ausfall,
# Aufruf mit relativem Pfad "." aus dem Ergebnisordner (Lehre aus B1).
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'test_err_fp32', 'test_err_w8a8', 'test_err_w8a4_best',
        'test_err_w8a4a_best', 'test_err_w8a3_best', 'test_err_w8a3a_best', 'act_range_max_median', 'grad_log10_first_last', 'ms_per_step', 'sat_rate']
CELLS = [(ds, d) for d in (100, 200) for ds in ('mnist', 'fashion')]
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    # (fp32-Verschiebung je Tiefe, w4-Verschiebung) gegen LBh
    eff = dict(erfolg={'LB': ((0.1, 0.2), -3), 'kL2B': ((0.1, 0.2), 1), 'kL2Bh': ((0.1, 0.3), 2)},
               teuer={'LB': ((0.8, 1.5), -3), 'kL2B': ((0.1, 0.2), 1), 'kL2Bh': ((0.1, 0.3), 2)},
               null={a: ((0, 0), 0) for a in ('LB', 'kL2B', 'kL2Bh')})[scn]
    for ds, d in CELLS:
        for s in range(42072, 42084):
            f0 = 6 + rng.normal(0, 0.3); w0 = 11 + rng.normal(0, 1); n = lambda sd: rng.normal(0, sd)
            for act in ('LBh',) + tuple(eff):
                (a100, a200), dw = ((0, 0), 0) if act == 'LBh' else eff[act]; df = a100 if d == 100 else a200
                rows.append(dict(run_id=f'{ds}_d{d}_{act}_s{s}', dataset=ds, depth=d, act=act, lr=2e-5, seed=s, status='ok', train_loss_last50=0.1,
                                 test_err_fp32=(f0 + df + n(0.15)) / 100, test_err_w8a8=0.06, test_err_w8a4_best=(w0 + dw + n(0.7)) / 100,
                                 test_err_w8a4a_best=(w0 + dw + n(0.7)) / 100, test_err_w8a3_best=(w0 + 10 + dw + n(1)) / 100, test_err_w8a3a_best=(w0 + 8 + n(1)) / 100,
                                 act_range_max_median=2.5, grad_log10_first_last=0.4, ms_per_step=20, sat_rate=0.1))
    rep = [dict(rows[1], run_id='mnist_d100_LB_s42060', seed=42060), dict(rows[2], run_id='mnist_d100_kL2B_s42060', seed=42060)]
    return rows + rep, [dict(r) for r in rep]
def run(rows, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b2', rows), ('b1', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ:   # wie der Fehler in B1: Aufruf aus dem Ergebnisordner mit "." und Standard-Nachbarordner
        subprocess.run([PY, str(H / 'b2_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b2')
    else:
        subprocess.run([PY, str(H / 'b2_analyze.py'), str(d / 'b2'), str(d / 'b1')], check=True, capture_output=True)
    return json.load(open(d / 'b2' / 'analysis' / 'b2_result.json'))
ok = True
o = run(*make('erfolg'))
c = o['success'] and o['repro_ok'] and o['verdict'] == 'besser als LBh' and o['secondary_verdicts']['S6_kL2Bh_vs_LBh_fp32'] in ('kein Effekt', 'gemischt', 'schlechter als LBh')
ok &= c; print('erfolg ->', o['verdict'], o['H1'], o['success'], o['repro_ok'], o['secondary_verdicts'], c)
o = run(*make('teuer')); c = (not o['success']) and o['verdict'] == 'besser als LBh' and not o['H1'] and o['secondary_verdicts']['S1_LB_vs_LBh_fp32'] == 'schlechter als LBh'
ok &= c; print('teuer ->', o['verdict'], o['H1'], o['success'], o['secondary_verdicts']['S1_LB_vs_LBh_fp32'], c)
nulls = [run(*make('null', k))['verdict'] for k in range(6)]; c = all(v == 'kein Effekt' for v in nulls); ok &= c; print('null ->', nulls, c)
rows, old = make('erfolg'); old[0] = dict(old[0], test_err_w8a4a_best=old[0]['test_err_w8a4a_best'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle ->', not o['repro_ok'], c)
rows, old = make('erfolg'); rows[1] = dict(rows[1], status='nonfinite_loss'); o = run(rows, old); c = o['failed'] == 1 and 90.0 - 0 > max(o['primary'][0]['diffs']) - 100; ok &= c
print('Ausfall ->', o['failed'], o['primary'][0]['diffs'][:3], c)
o = run(*make('erfolg'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad "." findet B1 ->', o['repro_ok'], c)
o = run(*make('erfolg')); c = len(o['blumenfeld']) == 8 and all(abs(b['cost3_d100'] - 15) < 1.5 for b in o['blumenfeld'] if b['act'] == 'LBh')   # kuenstlich: w0 + 10 - f0 = 11 + 10 - 6; ok &= c
print('Blumenfeld-Tabelle ->', len(o['blumenfeld']), [round(b['cost3_d100'], 1) for b in o['blumenfeld']][:4], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
