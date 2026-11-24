# V2c Auswertungstest an kuenstlichen Daten (mit Claude): BPreCap vorn -> "BPreCap besser"; tanh vorn -> "tanh besser"; Null -> selten Fehlalarm;
# S2/S3 erkennen tanh vor SELU/TAT; Ausfall zaehlt 90 %; fehlende Zeilen werden gemeldet; historische Arme werden aus dem V2a2-Ordner gelesen.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; SEEDS = list(range(42350, 42362)); ok = True
PT = ['test_err_fp32', 'test_err_w8a8', 'test_err_w8a4a_max', 'test_err_w8a4a_best', 'act_lo_max_median', 'act_hi_max_median', 'act_absmax_max']
ALL = ['run_id', 'stage', 'act', 'lr', 'wd', 'seed', 'status', 'val_err_end', 'val_err_es', 'es_step', 'train_loss_last50', 'es_w8a4a_best_cal'] + [p + c for p in ('es_', 'end_') for c in PT]
def make(d, shift, seed, fail=None, drop=None):
    r = np.random.default_rng(seed)
    for sub, acts in (('v2c', ('tanh',)), ('v2a2', ('bprecap', 'tat', 'selu'))):
        (d / sub / 'workers').mkdir(parents=True)
        with open(d / sub / 'workers' / 'w0.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=ALL); w.writeheader()
            for s in SEEDS:
                base = np.random.default_rng(seed * 1000 + s).normal(0.5, 0.02)
                for a in acts:
                    if drop == (a, s): continue
                    row = dict(run_id=f'confirm_{a}_s{s}', stage='confirm', act=a, lr='1e-4', wd='0.0', seed=s, status='fail' if fail == (a, s) else 'ok',
                               val_err_end=0.5, val_err_es=0.5, es_step=3440, train_loss_last50=1.0, es_w8a4a_best_cal='mse4')
                    for p in ('es_', 'end_'):
                        for c in PT: row[p + c] = base + r.normal(0, 0.005) + (shift.get(a, 0) if 'w8a4a' in c else 0)
                    w.writerow(row)
def res(shift, seed, **kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); make(d, shift, seed, **kw)
        p = subprocess.run([sys.executable, '-I', str(H / 'v2c_analyze.py'), str(d / 'v2c'), str(d / 'v2a2')], capture_output=True, text=True)
        if p.returncode: print(p.stderr); raise SystemExit(1)
        return json.load(open(d / 'v2c' / 'analysis' / 'v2c_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
r = res({'tanh': 0.05}, 1); chk('BPreCap vorn -> "BPreCap besser"', r['primary']['P1']['verdict'] == 'BPreCap besser', r['primary']['P1']['verdict'])
chk('fp32 ohne Effekt -> S1 unklar', r['secondary']['S1']['verdict'] == 'unklar')
r = res({'bprecap': 0.05, 'selu': 0.08, 'tat': 0.08}, 2); chk('tanh vorn -> "tanh besser"', r['primary']['P1']['verdict'] == 'tanh besser', r['primary']['P1']['verdict'])
chk('S2/S3: tanh besser als SELU und TAT', r['secondary']['S2']['verdict'] == 'besser' and r['secondary']['S3']['verdict'] == 'besser')
n = 0
for i in range(20): n += res({}, 10 + i)['primary']['P1']['verdict'] != 'unklar'
chk('Null: Fehlalarme P1 <= 3 von 20', n <= 3, str(n))
r = res({'tanh': 0.05}, 4, fail=('tanh', SEEDS[0])); chk('Ausfall tanh zaehlt 90 %', r['n_fail'] == 1 and r['primary']['P1']['diffs'][0] < -20)
r = res({'tanh': 0.05}, 5, drop=('bprecap', SEEDS[1])); chk('fehlende historische Zeile gemeldet, zaehlt 90 %', r['missing'] == [f'bprecap_{SEEDS[1]}'] and r['primary']['P1']['diffs'][1] > 20)
chk('historische Arme gelesen (35 ok)', r['n_hist_ok'] == 35, str(r['n_hist_ok']))
def make_price(d, seed):   # tanh in fp32 15 pp schlechter, Preis gleich -> P1 BPreCap besser, P2 unklar
    make(d, {}, seed)
    f = d / 'v2c' / 'workers' / 'w0.csv'; rows = list(csv.DictReader(open(f)))
    for r_ in rows:
        for p_ in ('es_', 'end_'):
            for c in PT:
                if c.startswith('test_err'): r_[p_ + c] = float(r_[p_ + c]) + 0.15
    with open(f, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ALL); w.writeheader(); w.writerows(rows)
with tempfile.TemporaryDirectory() as t:
    d = Path(t); make_price(d, 7)
    subprocess.run([sys.executable, '-I', str(H / 'v2c_analyze.py'), str(d / 'v2c'), str(d / 'v2a2')], capture_output=True, text=True, check=True)
    r = json.load(open(d / 'v2c' / 'analysis' / 'v2c_result.json'))
chk('tanh in fp32 schlechter, gleicher Preis -> P1 BPreCap besser, P2 unklar', r['primary']['P1']['verdict'] == 'BPreCap besser' and r['primary']['P2']['verdict'] == 'unklar', str((r['primary']['P1']['verdict'], r['primary']['P2']['verdict'])))
r = res({'tanh': 0.05}, 8); chk('tanh mit groesserem 4-Bit-Preis -> P2 BPreCap besser', r['primary']['P2']['verdict'] == 'BPreCap besser')
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
