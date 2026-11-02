# Test der Auswertung b4_analyze.py an kuenstlichen Daten. (abgeleitet aus test_b2_analyse.py) Ausgabe "AUSWERTUNG OK" oder Fehler.
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
    # Referenz LBhM; w4-Verschiebungen: 'erfolg' = Form zaehlt (weich -2 bei beiden Grenzen), 'bereich' = nur die enge Grenze zaehlt
    eff = dict(erfolg={'LB': ((0.1, 0.2), -2), 'LBh': ((0, 0), 1), 'LBM': ((0.1, 0.2), -1)},
               teuer={'LB': ((0.8, 1.5), -2), 'LBh': ((0, 0), 1), 'LBM': ((0.1, 0.2), -1)},
               bereich={'LB': ((0, 0), 0), 'LBh': ((0, 0), 1.5), 'LBM': ((0, 0), 1.5)},
               null={a: ((0, 0), 0) for a in ('LB', 'LBh', 'LBM')})[scn]
    for ds, d in CELLS:
        for s in range(42096, 42108):
            f0 = 6 + rng.normal(0, 0.3); w0 = 11 + rng.normal(0, 1); n = lambda sd: rng.normal(0, sd)
            for act in ('LBhM',) + tuple(eff):
                (a100, a200), dw = ((0, 0), 0) if act == 'LBhM' else eff[act]; df = a100 if d == 100 else a200
                rows.append(dict(run_id=f'{ds}_d{d}_{act}_s{s}', dataset=ds, depth=d, act=act, lr=2e-5, seed=s, status='ok', train_loss_last50=0.1,
                                 test_err_fp32=(f0 + df + n(0.15)) / 100, test_err_w8a8=0.06, test_err_w8a4_best=(w0 + dw + n(0.7)) / 100,
                                 test_err_w8a4a_best=(w0 + dw + n(0.7)) / 100, test_err_w8a3_best=(w0 + 10 + dw + n(1)) / 100, test_err_w8a3a_best=(w0 + 8 + n(1)) / 100,
                                 act_range_max_median=2.5, grad_log10_first_last=0.4, ms_per_step=20, sat_rate=0.1))
    rep = [dict(rows[1], run_id='mnist_d100_LB_s42072', seed=42072), dict(rows[2], run_id='mnist_d100_LBh_s42072', seed=42072, act='LBh')]
    return rows + rep, [dict(r) for r in rep]
def run(rows, old, relativ=False):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b4', rows), ('b2', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    if relativ:   # wie der Fehler in B1: Aufruf aus dem Ergebnisordner mit "." und Standard-Nachbarordner
        subprocess.run([PY, str(H / 'b4_analyze.py'), '.'], check=True, capture_output=True, cwd=d / 'b4')
    else:
        subprocess.run([PY, str(H / 'b4_analyze.py'), str(d / 'b4'), str(d / 'b2')], check=True, capture_output=True)
    return json.load(open(d / 'b4' / 'analysis' / 'b4_result.json'))
ok = True
o = run(*make('erfolg')); z = [e for e in o['zerlegung'] if e['metric'] == 'w8a4_best']
c = o['success'] and o['repro_ok'] and o['verdict'] == 'besser als LBhM' and all(e['form_med'] < -1 for e in z); ok &= c
print('Form zaehlt ->', o['verdict'], o['success'], o['repro_ok'], [round(e['form_med'], 2) for e in z], c)
o = run(*make('teuer')); c = (not o['success']) and o['verdict'] == 'besser als LBhM' and not o['H1']; ok &= c; print('weich teuer ->', o['verdict'], o['H1'], c)
o = run(*make('bereich')); z = [e for e in o['zerlegung'] if e['metric'] == 'w8a4_best']
c = o['verdict'] == 'kein Effekt' and all(e['grenze_med'] < -1 and abs(e['form_med']) < 0.6 for e in z) and o['secondary_verdicts']['S2_LB_vs_LBM_w4'] == 'besser als LBM'; ok &= c
print('nur Grenze zaehlt ->', o['verdict'], [(round(e['form_med'], 2), round(e['grenze_med'], 2)) for e in z], o['secondary_verdicts']['S2_LB_vs_LBM_w4'], c)
nulls = [run(*make('null', k))['verdict'] for k in range(6)]; c = all(v == 'kein Effekt' for v in nulls); ok &= c; print('null ->', nulls, c)
rows, old = make('erfolg'); old[0] = dict(old[0], test_err_w8a4a_best=old[0]['test_err_w8a4a_best'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c
print('falsche Kontrolle ->', not o['repro_ok'], c)
rows, old = make('erfolg'); rows[1] = dict(rows[1], status='nonfinite_loss'); o = run(rows, old); c = o['failed'] == 1; ok &= c; print('Ausfall ->', o['failed'], c)
o = run(*make('erfolg'), relativ=True); c = o['repro_ok']; ok &= c; print('relativer Pfad findet B2 ->', o['repro_ok'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
