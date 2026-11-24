# V2a2 Auswertungstest an kuenstlichen Daten (mit Claude): Effekt gegen beide -> Erfolg; nur gegen SELU -> teilweise; TAT vorn -> schlechter;
# Null -> selten Fehlalarm (20 Wiederholungen je Primaertest); Ausfall zaehlt 90 %; fehlende Zeilen werden gemeldet.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); SEEDS = list(range(42350, 42362)); ok = True
COLS = ['run_id', 'stage', 'act', 'lr', 'wd', 'seed', 'status', 'val_err_end', 'val_err_es', 'es_step', 'train_loss_last50']
PT = ['test_err_fp32', 'test_err_w8a8', 'test_err_w8a4a_max', 'test_err_w8a4a_best', 'act_hi_max_median']
ALL = COLS + [p + c for p in ('es_', 'end_', 'es_cle_') for c in PT] + ['es_w8a4a_best_cal', 'end_w8a4a_best_cal', 'es_cle_w8a4a_best_cal']
def make(d, shift, seed, fail=None, drop=None):
    r = np.random.default_rng(seed); (d / 'workers').mkdir(parents=True)
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ALL); w.writeheader()
        for s in SEEDS:
            base = r.normal(0.5, 0.02)
            for a in ('bprecap', 'tat', 'selu'):
                if drop == (a, s): continue
                row = dict(run_id=f'confirm_{a}_s{s}', stage='confirm', act=a, lr='1e-4', wd='0', seed=s, status='fail' if fail == (a, s) else 'ok',
                           val_err_end=0.5, val_err_es=0.5, es_step=3440, train_loss_last50=1.0, es_w8a4a_best_cal='max', end_w8a4a_best_cal='p999')
                if a == 'tat': row['es_cle_w8a4a_best_cal'] = 'mse4'
                for p in ('es_', 'end_') + (('es_cle_',) if a == 'tat' else ()):
                    for c in PT: row[p + c] = base + r.normal(0, 0.005) + (shift.get(a, 0) if 'w8a4a' in c else 0)
                w.writerow(row)
def res(shift, seed, **kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); make(d, shift, seed, **kw)
        p = subprocess.run([sys.executable, '-I', str(H / 'v2a2_analyze.py'), str(d)], capture_output=True, text=True)
        if p.returncode: print(p.stderr); raise SystemExit(1)
        return json.load(open(d / 'analysis' / 'v2a2_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
r = res({'selu': 0.05, 'tat': 0.05}, 1); chk('BPreCap vorn gegen beide -> Erfolg', r['summary'].startswith('Erfolg') and all(x['verdict'] == 'besser' for x in r['primary'].values()))
chk('Sekundaer w8a4a_max und TAT+CLE besser, fp32 unklar', r['secondary']['S3']['verdict'] == 'besser' and r['secondary']['S4']['verdict'] == 'besser' and r['secondary']['S5']['verdict'] == 'besser' and r['secondary']['S1']['verdict'] == 'unklar')
tw = sum(res({'selu': 0.05}, 40 + i)['summary'].startswith('teilweise: besser als SELU') for i in range(5)); chk('nur gegen SELU -> teilweise (mind. 4 von 5, Zufall im Null-Arm)', tw >= 4, str(tw))
r = res({'bprecap': 0.05}, 3); chk('BPreCap hinten -> schlechter gegen beide', r['primary']['P1']['verdict'] == 'schlechter' and r['primary']['P2']['verdict'] == 'schlechter' and r['summary'].startswith('mindestens'))
n1 = n2 = 0
for i in range(20):
    x = res({}, 10 + i); n1 += x['primary']['P1']['verdict'] != 'unklar'; n2 += x['primary']['P2']['verdict'] != 'unklar'
chk('Null: Fehlalarme je Primaertest <= 3 von 20', n1 <= 3 and n2 <= 3, f'{n1}, {n2}')
r = res({'selu': 0.05, 'tat': 0.05}, 4, fail=('tat', SEEDS[0])); chk('Ausfall TAT zaehlt 90 %', r['n_fail'] == 1 and r['primary']['P2']['diffs'][0] < -20)
r = res({'selu': 0.05, 'tat': 0.05}, 5, drop=('bprecap', SEEDS[1])); chk('fehlende Zeile gemeldet, zaehlt 90 %', r['missing'] == [f'bprecap_{SEEDS[1]}'] and r['primary']['P1']['diffs'][1] > 20)
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
