# V2a-1 Auswertungstest an kuenstlichen Daten (mit Claude): "BPreCap vorn" -> besser, "SELU vorn" -> schlechter,
# Null -> selten Fehlalarm (20 Wiederholungen), Ausfall zaehlt 90 %, fehlende Zeilen werden gemeldet.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; SEEDS = list(range(42330, 42342)); ok = True
COLS = ['run_id', 'stage', 'act', 'lr', 'seed', 'status', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a8', 'test_err_w8a4a_max', 'test_err_w8a4a_p999',
        'test_err_w8a4a_p9999', 'test_err_w8a4a_best', 'act_lo_max_median', 'act_hi_max_median', 'act_hi_p9999_median', 'train_loss_last50', 'w8a4a_best_cal']
def make(d, shift, seed, fail=None, drop=None):
    r = np.random.default_rng(seed); (d / 'workers').mkdir(parents=True)
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for s in SEEDS:
            base = r.normal(0.6, 0.02, len(COLS))
            for a in ('bprecap', 'selu'):
                if drop == (a, s): continue
                row = dict(run_id=f'confirm_{a}_s{s}', stage='confirm', act=a, lr='1e-4', seed=s, status='fail' if fail == (a, s) else 'ok')
                for i, c in enumerate(COLS[6:-1]): row[c] = base[i] + (shift if a == 'bprecap' and c.startswith('test_err') else 0) + r.normal(0, 0.005)
                row['w8a4a_best_cal'] = 'max'
                w.writerow(row)
def res(shift, seed, **kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); make(d, shift, seed, **kw)
        p = subprocess.run([sys.executable, '-I', str(H / 'v2a1_analyze.py'), str(d)], capture_output=True, text=True)
        if p.returncode: print(p.stderr); raise SystemExit(1)
        return json.load(open(d / 'analysis' / 'v2a1_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
r = res(-0.05, 1); chk('BPreCap vorn -> Erfolg', r['success'] and r['primary']['verdict'] == 'besser' and all(x['verdict'] == 'besser' for x in r['secondary'].values()))
r = res(+0.05, 2); chk('SELU vorn -> schlechter', r['primary']['verdict'] == 'schlechter' and not r['success'])
fa = sum(res(0.0, 10 + i)['primary']['verdict'] != 'unklar' for i in range(20)); chk('Null: Fehlalarme primaer <= 3 von 20', fa <= 3, str(fa))
r = res(-0.05, 3, fail=('selu', SEEDS[0])); chk('Ausfall SELU zaehlt 90 %', r['n_fail'] == 1 and r['primary']['diffs'][0] < -20)
r = res(-0.05, 4, drop=('bprecap', SEEDS[1])); chk('fehlende Zeile gemeldet, zaehlt 90 %', r['missing'] == [f'bprecap_{SEEDS[1]}'] and r['primary']['diffs'][1] > 20)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
