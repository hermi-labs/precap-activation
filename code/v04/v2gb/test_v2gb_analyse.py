# V2g Auswertungstest an kuenstlichen Daten (mit Claude, aus dem V2d-Test): BPreCap vorn -> "BPreCap besser"; xtanh vorn -> "xtanh besser";
# Null -> selten Fehlalarm; S2-S4 erkennen xtanh vor tanh/ctanh/SELU; Ausfall zaehlt 90 %; fehlende Zeilen gemeldet; historische Arme aus V2a2, V2c, V2d.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; SEEDS = list(range(42350, 42362)); ok = True
PT = ['test_err_fp32', 'test_err_w8a8', 'test_err_w8a4a_max', 'test_err_w8a4a_best', 'act_lo_max_median', 'act_hi_max_median', 'act_absmax_max']
ALL = ['run_id', 'stage', 'act', 'lr', 'wd', 'seed', 'status', 'val_err_end', 'val_err_es', 'es_step', 'train_loss_last50', 'es_w8a4a_best_cal'] + [p + c for p in ('es_', 'end_') for c in PT]
GROUPS = (('v2g', ('xtanh',)), ('v2a2', ('bprecap', 'tat', 'selu')), ('v2c', ('tanh',)), ('v2d', ('ctanh',)))
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
        p = subprocess.run([sys.executable, '-I', str(H / 'v2gb_analyze.py'), str(d / 'v2g'), str(d / 'v2a2'), str(d / 'v2c'), str(d / 'v2d')], capture_output=True, text=True)
        if p.returncode: print(p.stderr); raise SystemExit(1)
        return json.load(open(d / 'v2g' / 'analysis' / 'v2gb_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
r = res({'xtanh': 0.05}, 1); chk('BPreCap vorn -> "BPreCap besser"', r['primary']['P1']['verdict'] == 'BPreCap besser', r['primary']['P1']['verdict'])
chk('fp32 ohne Effekt -> S1 unklar', r['secondary']['S1']['verdict'] == 'unklar')
r = res({'bprecap': 0.05, 'selu': 0.08, 'ctanh': 0.08, 'tanh': 0.08}, 2); chk('xtanh vorn -> "xtanh besser"', r['primary']['P1']['verdict'] == 'xtanh besser', r['primary']['P1']['verdict'])
chk('S2-S4: xtanh besser als tanh, ctanh, SELU', all(r['secondary'][k]['verdict'] == 'besser' for k in ('S2', 'S3', 'S4')), str([r['secondary'][k]['verdict'] for k in ('S2', 'S3', 'S4')]))
n = 0
for i in range(20): n += res({}, 10 + i)['primary']['P1']['verdict'] != 'unklar'
chk('Null: Fehlalarme P1 <= 3 von 20', n <= 3, str(n))
r = res({'xtanh': 0.05}, 4, fail=('xtanh', SEEDS[0])); chk('Ausfall xtanh zaehlt 90 %', r['n_fail'] == 1 and r['primary']['P1']['diffs'][0] < -20)
r = res({'xtanh': 0.05}, 5, drop=('bprecap', SEEDS[1])); chk('fehlende historische Zeile gemeldet, zaehlt 90 %', r['missing'] == [f'bprecap_{SEEDS[1]}'] and r['primary']['P1']['diffs'][1] > 20)
chk('historische Arme gelesen (59 ok)', r['n_hist_ok'] == 59, str(r['n_hist_ok']))
r = res({}, 6, drop=('tanh', SEEDS[2])); chk('fehlende tanh-Zeile aus V2c gemeldet', r['missing'] == [f'tanh_{SEEDS[2]}'] and r['secondary']['S2']['diffs'][2] < -20)
r = res({}, 11, drop=('ctanh', SEEDS[3])); chk('fehlende ctanh-Zeile aus V2d gemeldet', r['missing'] == [f'ctanh_{SEEDS[3]}'] and r['secondary']['S3']['diffs'][3] < -20)
r = res({}, 7, fp_shift={'xtanh': 0.15}); chk('xtanh in fp32 schlechter, gleicher Preis -> P1 BPreCap besser, P2 unklar', r['primary']['P1']['verdict'] == 'BPreCap besser' and r['primary']['P2']['verdict'] == 'unklar', str((r['primary']['P1']['verdict'], r['primary']['P2']['verdict'])))
r = res({'xtanh': 0.05}, 8); chk('xtanh mit groesserem 4-Bit-Preis -> P2 BPreCap besser', r['primary']['P2']['verdict'] == 'BPreCap besser')
r = res({'bprecap': 0.05}, 9); chk('BPreCap mit groesserem Preis -> P2 xtanh besser', r['primary']['P2']['verdict'] == 'xtanh besser')
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
