# Test der V2e-Auswertung mit kuenstlichen Daten (mit Claude): (1) eingebauter Effekt in r_bar (BPreCap 0,8 x ctanh) muss "BPreCap kleiner" bei Pa
# und "kein klarer Unterschied" bei Pb geben, Deutung "Runden"; (2) ohne Effekt (Rauschen +-5 %) nichts; (3) ein nicht bitgleicher Lauf -> "unklar (Kontrolle)".
import csv, json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AN = HERE / 'v2e_analyze.py'; S = HERE.parent
sys.path.insert(0, str(HERE))
COLS = ['run_id', 'act', 'seed', 'status', 'kontrolle_n_diff', 'rbar', 'gbarQ', 'gbarG', 'D_allQ', 'D_allG', 'superpos_Q', 'superpos_G', 'flipQ', 'err_fp32_1000',
        'err_a4_1000', 'err_w8a4_1000', 'chi_emp_gm', 'eps_end_Q', 'margin_median', 'bias_anteil_median']
def make(d, eff_r, noise, bad=None, seed=0):
    rng = np.random.default_rng(seed); (d / 'workers').mkdir(parents=True)
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for a in ('bprecap', 'ctanh', 'selu', 'tanh', 'tat'):
            for s in (42350, 42351, 42352):
                e = lambda: 1 + rng.uniform(-noise, noise); r = 0.08 * (eff_r if a == 'bprecap' else 1) * e(); g = 0.3 * e()
                row = dict(run_id=f'{a}_{s}', act=a, seed=s, status='ok', kontrolle_n_diff=1 if (a, s) == bad else 0, rbar=r, gbarQ=g, gbarG=0.1 * e(), D_allQ=r * g * 10,
                           D_allG=0.2, superpos_Q=1, superpos_G=1.5, flipQ=0.1, err_fp32_1000=0.4, err_a4_1000=0.45, err_w8a4_1000=0.46, chi_emp_gm=1.0, eps_end_Q=0.5,
                           margin_median=0.1, bias_anteil_median=0.2)
                w.writerow(row)
def run(d):
    p = subprocess.run([sys.executable, '-I', str(AN), str(d), str(S / 'v2a2'), str(S / 'v2c'), str(S / 'v2d')], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr; return json.load(open(d / 'analysis' / 'v2e_result.json'))
t = Path(tempfile.mkdtemp())
try:
    make(t / 'eff', 0.8, 0.03); o = run(t / 'eff')
    assert o['Pa']['urteil'] == 'BPreCap kleiner', o['Pa']; assert o['Pb']['urteil'] == 'kein klarer Unterschied', o['Pb']; assert o['Deutung'] == 'Runden', o['Deutung']
    assert o['Pruefung_Messung']['erfuellt'], o['Pruefung_Messung']; assert o['Preis_12_Seeds']['ctanh'] is not None and len(o['S4_formen']) == 5
    make(t / 'null', 1.0, 0.05, seed=1); o = run(t / 'null')
    assert o['Pa']['urteil'] == 'kein klarer Unterschied' and o['Pb']['urteil'] == 'kein klarer Unterschied', (o['Pa'], o['Pb'])
    make(t / 'bad', 0.8, 0.03, bad=('ctanh', 42351)); o = run(t / 'bad')
    assert o['Pa']['urteil'] == 'unklar (Kontrolle)', o['Pa']
    print('AUSWERTUNGSTEST OK: Effekt erkannt, ohne Effekt nichts, Kontrolle greift')
finally:
    shutil.rmtree(t)
