# Test der Auswertung b1_analyze.py an kuenstlichen Daten. Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: Erfolg, fp32 teuer bei d200, Null (6 Wiederholungen), falsche Kontrolle, Ausfaelle.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; PY = sys.executable
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'test_err_fp32', 'test_err_w8a4', 'test_err_w8a8', 'test_err_w8a4_best',
        'test_err_w8a4a_best', 'act_range_max_median', 'act_range_p9999_median', 'grad_log10_first_last', 'ms_per_step', 'sat_rate', 'lg_c_median', 'lg_c_ratio_median', 'lg_intervention']
CELLS = [(ds, d) for d in (100, 200) for ds in ('mnist', 'fashion')]
def make(scn, seed=0):
    rng = np.random.default_rng(seed); rows = []
    # (fp32-Verschiebung in pp je Tiefe, w4-Verschiebung) gegen kL2
    eff = dict(erfolg={'kL2B': ((0.1, 0.2), -12), 'LB': ((0.1, 0.3), -11), 'kL2_LD100': ((0.2, 1.3), -5), 'spreccap_cs': ((1, 30), -15)},
               teuer={'kL2B': ((0.2, 1.5), -12), 'LB': ((0.2, 1.5), -11), 'kL2_LD100': ((0.2, 1.3), -5), 'spreccap_cs': ((1, 30), -15)},
               null={a: ((0, 0), 0) for a in ('kL2B', 'LB', 'kL2_LD100', 'spreccap_cs')})[scn]
    for ds, d in CELLS:
        for s in range(42060, 42072):
            f0 = 5 + rng.normal(0, 0.3); w0 = 29 + rng.normal(0, 2); n = lambda sd: rng.normal(0, sd)
            for act in ('kL2',) + tuple(eff):
                (a100, a200), dw = ((0, 0), 0) if act == 'kL2' else eff[act]; df = a100 if d == 100 else a200
                rows.append(dict(run_id=f'{ds}_d{d}_{act}_s{s}', dataset=ds, depth=d, act=act, lr=2e-5, seed=s, status='ok', train_loss_last50=0.1,
                                 test_err_fp32=(f0 + df + n(0.15)) / 100, test_err_w8a4=0.5, test_err_w8a8=0.05, test_err_w8a4_best=(w0 + dw + n(1)) / 100,
                                 test_err_w8a4a_best=(w0 + dw + n(1)) / 100, act_range_max_median=3, act_range_p9999_median=2.9, grad_log10_first_last=0.4,
                                 ms_per_step=10, sat_rate=0.01 if act in ('kL2B', 'LB') else '', lg_c_median=4 if act == 'kL2_LD100' else '',
                                 lg_c_ratio_median=0.9 if act == 'kL2_LD100' else '', lg_intervention=0.01 if act == 'kL2_LD100' else ''))
    rep = [dict(rows[0], run_id='mnist_d100_kL2_s42045', seed=42045), dict(rows[3], run_id='mnist_d100_kL2_LD100_s42045', seed=42045)]
    return rows + rep, [dict(r) for r in rep]
def run(rows, old):
    d = Path(tempfile.mkdtemp())
    for sub, rr in (('b1', rows), ('l5', old)):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rr]
    subprocess.run([PY, str(H / 'b1_analyze.py'), str(d / 'b1'), str(d / 'l5')], check=True, capture_output=True)
    return json.load(open(d / 'b1' / 'analysis' / 'b1_result.json'))
ok = True
o = run(*make('erfolg'))
c = o['success'] and o['repro_ok'] and o['verdict'] == 'besser als kL2_LD100'; ok &= c; print('erfolg ->', o['verdict'], o['H1'], o['success'], o['repro_ok'], o['secondary_verdicts'], c)
o = run(*make('teuer')); c = (not o['success']) and o['verdict'] == 'besser als kL2_LD100' and not o['H1']; ok &= c; print('teuer ->', o['verdict'], o['H1'], o['success'], c)
nulls = [run(*make('null', k))['verdict'] for k in range(6)]; c = all(v == 'kein Effekt' for v in nulls); ok &= c; print('null ->', nulls, c)
rows, old = make('erfolg'); old[0] = dict(old[0], test_err_fp32=old[0]['test_err_fp32'] + 1e-6); o = run(rows, old); c = not o['repro_ok']; ok &= c; print('falsche Kontrolle ->', not o['repro_ok'], c)
rows, old = make('erfolg'); rows[1] = dict(rows[1], status='nonfinite_loss'); o = run(rows, old); c = o['failed'] == 1; ok &= c; print('Ausfall ->', o['failed'], o['primary'][0]['diffs'][:3], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
