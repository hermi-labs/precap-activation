# Test der Auswertung v3_analyze.py an kuenstlichen Daten (mit Claude). Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: QAT klar besser (Primaer "schlechter"), BPreCap-PTQ besser (Primaer "besser"), Null (20x), Ausfall, Basis nicht bitgleich.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
COLS = ['run_id', 'stage', 'act', 'method', 'lrq', 'seed', 'status', 'base_valid', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'val_err_qat', 'test_err_qat', 'test_err_w8a4a_best', 'w8a4a_best_cal', 'range_med_start', 'range_med_end', 'range_move_med', 'sec', 'timestamp']
def make(eff, seed=0, fail=False, badbase=False):
    r = np.random.default_rng(seed); rows = []
    for s in range(42168, 42180):
        for a in ('kL2B_tr', 'tat'):
            f = 46 + r.normal(0, 0.5); c = 3.0 if a == 'kL2B_tr' else 12.0
            for mm in ('BASE', 'FT', 'LSQ', 'PACT'):
                q = f + (c if mm in ('BASE', 'FT') else eff.get((a, mm), c)) + r.normal(0, 0.4)
                rows.append(dict(run_id=f'confirm_{a}_{mm}_s{s}', stage='confirm', act=a, method=mm, lrq='' if mm in ('BASE', 'FT') else 0.001, seed=s, status='ok',
                                 base_valid=str(not (badbase and s == 42170)) if mm == 'BASE' else 'True', val_err_fp32=f / 100, test_err_fp32=f / 100,
                                 test_err_w8a4a_best=q / 100, test_err_qat=q / 100 if mm in ('LSQ', 'PACT') else '', val_err_qat='', range_move_med=0.05 if mm in ('LSQ', 'PACT') else ''))
    if fail: i = next(j for j, x in enumerate(rows) if x['act'] == 'tat' and x['method'] == 'LSQ' and x['seed'] == 42171); rows[i] = dict(rows[i], status='nonfinite_loss', test_err_qat='')
    d = Path(tempfile.mkdtemp()); (d / 'workers').mkdir()
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); [w.writerow(x) for x in rows]
    subprocess.run([sys.executable, str(H / 'v3_analyze.py'), str(d)], check=True, capture_output=True); return json.load(open(d / 'analysis' / 'v3_result.json'))
ok = True
o = make({('tat', 'LSQ'): 1.0, ('tat', 'PACT'): 2.0, ('kL2B_tr', 'LSQ'): 0.5, ('kL2B_tr', 'PACT'): 0.8}); sv = o['secondary_verdicts']
c = o['primary']['verdict'] == 'schlechter' and sv['S4_TAT_LSQ_vs_TAT_FT'] == 'besser' and sv['S5_BPreCap_LSQ_vs_BPreCap_FT'] == 'besser' and sv['S6_BPreCapFT_vs_TATFT_PTQ'] == 'besser' and o['base_all_valid']
ok &= c; print('QAT besser ->', o['primary']['verdict'], sv, c)
o = make({('tat', 'LSQ'): 8.0, ('tat', 'PACT'): 9.0, ('kL2B_tr', 'LSQ'): 3.0, ('kL2B_tr', 'PACT'): 3.0}); c = o['primary']['verdict'] == 'besser' and o['secondary_verdicts']['S2_BPreCap_LSQ_vs_TAT_LSQ'] == 'besser'
ok &= c; print('BPreCap besser ->', o['primary']['verdict'], c)
fp = sum(make({('tat', 'LSQ'): 3.0, ('tat', 'PACT'): 3.0, ('kL2B_tr', 'LSQ'): 3.0, ('kL2B_tr', 'PACT'): 3.0}, seed=k)['primary']['verdict'] != 'unklar' for k in range(20))
ok &= fp <= 3; print('Null 20x -> Fehlalarme', fp)
o = make({('tat', 'LSQ'): 3.0}, fail=True); c = o['failed'] == 1 and min(o['primary']['diffs']) < -30; ok &= c; print('Ausfall ->', o['failed'], c)
o = make({}, badbase=True); c = not o['base_all_valid'] and o['base_valid'] == '22/24'; ok &= c; print('Basis nicht bitgleich ->', o['base_valid'], c)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
