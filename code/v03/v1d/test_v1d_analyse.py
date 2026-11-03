# Test der V1d-Auswertung an kuenstlichen Daten (mit Claude). Aufruf: <python-mit-scipy> -I test_v1d_analyse.py
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020)
CKC = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
END = ['test_err_w8a4a_max', 'w8a4a_best_cal', 'test_err_w8a4_best', 'w8a4_best_cal', 'sat_rate', 'act_lo_max_median']
def row(rid, sched, warm, seed, fp, w4, status='ok'):
    r = dict(run_id=rid, dataset='cifar10', depth=200, act='kL2B_tr', lr=5e-5, wd=0.3, sched=sched, warm=warm, seed=seed, status=status,
             train_loss_last50=0.7, val_err_fp32=fp / 100, test_err_fp32=fp / 100, test_err_w8a4_best=w4 / 100 + 0.01, test_err_w8a4a_best=w4 / 100,
             test_err_w8a3_best=0.8, act_range_max_median=3.0, test_err_w8a4a_max=w4 / 100 + 0.02, w8a4a_best_cal='mse4', w8a4_best_cal='p999', sat_rate=0.02, act_lo_max_median=-2.9)
    for k in CK:   # Zwischenmessungen schlechter als das Ende -> Stopp am Ende
        for c in CKC: r[f'c{k}_{c}'] = (float(r[c]) + 0.05) if c != 'act_range_max_median' else 3.0
    return r
def write(d, rows):
    (d / 'workers').mkdir(parents=True); keys = list(rows[0].keys())
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
def run(w4fn, fpfn=lambda w, s: 45.0, break_repro=False, drop=None):
    t = Path(tempfile.mkdtemp()); S = range(42192, 42204)
    b15 = [row(f'cifar10_d200_kL2B_tr_s{s}', 'cos', '', s, fpfn(0, s), w4fn(0, s)) for s in S]
    v1c = [row(f'cifar10_d200_kL2B_tr_s{s}', 'wcos', '', s, fpfn(344, s), w4fn(344, s)) for s in S]
    v1c.append(row('cifar10_d200_kL2B_tr_s42192_kontrolle', 'cos', '', 42192, fpfn(0, 42192), w4fn(0, 42192)))
    new = [row('cifar10_d200_kL2B_tr_s42192_kontrolle', 'cos', 0, 42192, fpfn(0, 42192), w4fn(0, 42192) + (0.5 if break_repro else 0)),
           row('cifar10_d200_kL2B_tr_s42192_w344', 'wcos', 344, 42192, fpfn(344, 42192), w4fn(344, 42192))]
    for s in S:
        for w in (69, 172):
            if drop == (w, s): continue
            new.append(row(f'cifar10_d200_kL2B_tr_s{s}_w{w}', 'wcos', w, s, fpfn(w, s), w4fn(w, s)))
    write(t / 'v1d', new); write(t / 'v1c', v1c); write(t / 'b15', b15)
    pr = subprocess.run([sys.executable, '-I', str(H / 'v1d_analyze.py'), str(t / 'v1d'), str(t / 'v1c'), str(t / 'b15')], capture_output=True, text=True)
    if pr.returncode: print(pr.stderr); raise SystemExit('Auswertung abgebrochen')
    return json.load(open(t / 'v1d' / 'analysis' / 'v1d_result.json'))
rng = np.random.default_rng(1); N = {(w, s): rng.normal(0, 0.3) for w in (0, 69, 172, 344) for s in range(42192, 42204)}
ok = True
o = run(lambda w, s: 54 + 1.64 * w / 344 + N[(w, s)]); print('Dosis linear:', o['primary_verdicts'], o['overall'], o['repro_ok'])
ok &= o['primary_verdicts']['P1'] == 'besser' and o['primary_verdicts']['P2'] == 'schlechter' and o['overall'] == 'auch 1 % Anlauf kostet bei 4 Bit' and o['repro_ok']
o = run(lambda w, s: (55.64 if w == 344 else 54) + N[(w, s)]); print('nur 5 % kostet:', o['primary_verdicts'], o['overall'])
ok &= o['primary_verdicts']['P1'] == 'besser' and o['primary_verdicts']['P2'] == 'unklar' and o['overall'] == 'kurzer Anlauf vermeidet den 4-Bit-Preis'
o = run(lambda w, s: (54 if w == 0 else 55.64) + N[(w, s)]); print('jeder Anlauf kostet:', o['primary_verdicts'], o['overall'])
ok &= o['primary_verdicts']['P2'] == 'schlechter' and o['overall'] == 'auch 1 % Anlauf kostet bei 4 Bit'
nul = []
for i in range(10):
    r2 = np.random.default_rng(100 + i); M = {(w, s): r2.normal(0, 0.5) for w in (0, 69, 172, 344) for s in range(42192, 42204)}
    nul.append(run(lambda w, s: 54 + M[(w, s)])['overall'])
print('Null (10x):', nul); ok &= sum(x == 'nicht aufgeloest' for x in nul) >= 8
o = run(lambda w, s: 54 + N[(w, s)], break_repro=True); print('Kontrolle verfehlt:', o['repro_ok'], o['repro']); ok &= not o['repro_ok']
o = run(lambda w, s: 54 + N[(w, s)], drop=(69, 42195)); d = o['primary']['P2_w69_vs_w0_w4a_es']
print('Ausfall: Zeilen', d['n_rows'], 'Diff', d['diffs'][3]); ok &= d['n_rows'][0] == 11 and d['diffs'][3] > 30
o = run(lambda w, s: 54 + N[(w, s)], fpfn=lambda w, s: 45 - 0.6 * (w > 0) + N[(w, s)] * 0.2); print('fp32 S3:', o['secondary_verdicts']['S3_w69_vs_w0_fp32_es']); ok &= o['secondary_verdicts']['S3_w69_vs_w0_fp32_es'] == 'besser'
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
