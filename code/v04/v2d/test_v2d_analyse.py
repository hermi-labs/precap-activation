# V2d Auswertungstest an kuenstlichen Daten (mit Claude): BPreCap vorn -> "BPreCap besser"; ctanh vorn -> "ctanh besser"; Null -> selten Fehlalarm;
# S2-S4 erkennen ctanh vor tanh/SELU/TAT; Ausfall zaehlt 90 %; fehlende Zeilen werden gemeldet; historische Arme kommen aus V2a2 und V2c.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; SEEDS = list(range(42350, 42362)); ok = True
PT = ['test_err_fp32', 'test_err_w8a8', 'test_err_w8a4a_max', 'test_err_w8a4a_best', 'act_lo_max_median', 'act_hi_max_median', 'act_absmax_max']
ALL = ['run_id', 'stage', 'act', 'lr', 'wd', 'seed', 'status', 'val_err_end', 'val_err_es', 'es_step', 'train_loss_last50', 'es_w8a4a_best_cal'] + [p + c for p in ('es_', 'end_') for c in PT]
GROUPS = (('v2d', ('ctanh',)), ('v2a2', ('bprecap', 'tat', 'selu')), ('v2c', ('tanh',)))
def make(d, shift, seed, fail=None, drop=None, fp_shift=None):
    r = np.random.default_rng(seed)
    for sub, acts in GROUPS:
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
                        for c in PT: row[p + c] = base + r.normal(0, 0.005) + (shift.get(a, 0) if 'w8a4a' in c else 0) + ((fp_shift or {}).get(a, 0) if c.startswith('test_err') else 0)
                    w.writerow(row)
def res(shift, seed, **kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); make(d, shift, seed, **kw)
        p = subprocess.run([sys.executable, '-I', str(H / 'v2d_analyze.py'), str(d / 'v2d'), str(d / 'v2a2'), str(d / 'v2c')], capture_output=True, text=True)
        if p.returncode: print(p.stderr); raise SystemExit(1)
        return json.load(open(d / 'v2d' / 'analysis' / 'v2d_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
r = res({'ctanh': 0.05}, 1); chk('BPreCap vorn -> "BPreCap besser"', r['primary']['P1']['verdict'] == 'BPreCap besser', r['primary']['P1']['verdict'])
chk('fp32 ohne Effekt -> S1 unklar', r['secondary']['S1']['verdict'] == 'unklar')
r = res({'bprecap': 0.05, 'selu': 0.08, 'tat': 0.08, 'tanh': 0.08}, 2); chk('ctanh vorn -> "ctanh besser"', r['primary']['P1']['verdict'] == 'ctanh besser', r['primary']['P1']['verdict'])
chk('S2-S4: ctanh besser als tanh, SELU, TAT', all(r['secondary'][k]['verdict'] == 'besser' for k in ('S2', 'S3', 'S4')), str([r['secondary'][k]['verdict'] for k in ('S2', 'S3', 'S4')]))
n = 0
for i in range(20): n += res({}, 10 + i)['primary']['P1']['verdict'] != 'unklar'
chk('Null: Fehlalarme P1 <= 3 von 20', n <= 3, str(n))
r = res({'ctanh': 0.05}, 4, fail=('ctanh', SEEDS[0])); chk('Ausfall ctanh zaehlt 90 %', r['n_fail'] == 1 and r['primary']['P1']['diffs'][0] < -20)
r = res({'ctanh': 0.05}, 5, drop=('bprecap', SEEDS[1])); chk('fehlende historische Zeile gemeldet, zaehlt 90 %', r['missing'] == [f'bprecap_{SEEDS[1]}'] and r['primary']['P1']['diffs'][1] > 20)
chk('historische Arme gelesen (47 ok)', r['n_hist_ok'] == 47, str(r['n_hist_ok']))
r = res({}, 6, drop=('tanh', SEEDS[2])); chk('fehlende tanh-Zeile aus V2c gemeldet', r['missing'] == [f'tanh_{SEEDS[2]}'] and r['secondary']['S2']['diffs'][2] < -20)
r = res({}, 7, fp_shift={'ctanh': 0.15}); chk('ctanh in fp32 schlechter, gleicher Preis -> P1 BPreCap besser, P2 unklar', r['primary']['P1']['verdict'] == 'BPreCap besser' and r['primary']['P2']['verdict'] == 'unklar', str((r['primary']['P1']['verdict'], r['primary']['P2']['verdict'])))
r = res({'ctanh': 0.05}, 8); chk('ctanh mit groesserem 4-Bit-Preis -> P2 BPreCap besser', r['primary']['P2']['verdict'] == 'BPreCap besser')
r = res({'bprecap': 0.05}, 9); chk('BPreCap mit groesserem Preis -> P2 ctanh besser', r['primary']['P2']['verdict'] == 'ctanh besser')
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
