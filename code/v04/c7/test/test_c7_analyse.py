# Test des C7-Auswertungsskripts mit kuenstlichen Daten (mit Claude). Aufruf: ~/PyCharmMiscProject/.venv/bin/python -I test_c7_analyse.py <scratch-Ordner>
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; A = H.parent / 'c7_analyze.py'; V = H.parent.parent / 'v2a2'; W = Path(sys.argv[1])
mlx = [r for r in csv.DictReader(open(V / 'workers' / 'w0.csv')) if r['stage'] == 'confirm']
COLS = ['run_id', 'act', 'seed', 'status', 'es_test_err_fp32', 'es_test_err_w8a8', 'es_test_err_w8a4a_max', 'es_test_err_w8a4a_best', 'es_cle_test_err_w8a4a_best']
def make(name, f, drop=None):
    d = W / name; (d / 'workers').mkdir(parents=True, exist_ok=False); rg = np.random.default_rng(len(name))
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction='ignore'); w.writeheader()
        for r in mlx:
            if drop and (r['act'], int(r['seed'])) == drop: continue
            o = {k: r.get(k, '') for k in COLS}; o['run_id'] = 'c7_' + r['run_id']; f(o, rg); w.writerow(o)
    p = subprocess.run([sys.executable, '-I', str(A), str(d), str(V)], capture_output=True, text=True); assert p.returncode == 0, p.stderr
    return json.load(open(d / 'analysis' / 'c7_result.json'))
ok = True
def chk(n, c, info=''):
    global ok; ok &= c; print(('OK     ' if c else 'FEHLER ') + n, info)
r = make('gleich', lambda o, rg: None)
chk('Kopie der MLX-Daten -> P1, P2 besser wie V2a2', r['primaer']['P1']['urteil'] == 'besser' and r['primaer']['P2']['urteil'] == 'besser' and abs(r['primaer']['P1']['med_diff_pp'] + 2.555) < 0.01, str({k: (v['med_diff_pp'], v['urteil']) for k, v in r['primaer'].items()}))
chk('Kopie -> CUDA = MLX je Form', all(v['diff_pp'] == 0 for a in r['cuda_gegen_mlx'].values() for k, v in a.items() if k != 'preis_4bit_pp'))
def kein_effekt(o, rg):
    if o['act'] == 'bprecap':
        s = [m for m in mlx if m['act'] == 'selu' and m['seed'] == o['seed']][0]; o['es_test_err_w8a4a_best'] = str(float(s['es_test_err_w8a4a_best']) + rg.normal(0, 0.003))
r = make('null_selu', kein_effekt); chk('BPreCap = SELU + Rauschen -> P1 unklar', r['primaer']['P1']['urteil'] == 'unklar', str(r['primaer']['P1']['p_holm']))
def umgekehrt(o, rg):
    if o['act'] == 'bprecap': o['es_test_err_w8a4a_best'] = str(float(o['es_test_err_w8a4a_best']) + 0.35)
r = make('umgekehrt', umgekehrt); chk('BPreCap +35 pp -> beide schlechter', r['primaer']['P1']['urteil'] == 'schlechter' and r['primaer']['P2']['urteil'] == 'schlechter')
def ausfall(o, rg):
    if o['act'] == 'selu' and o['seed'] == '42351': o['status'] = 'fail'
r = make('ausfall', ausfall, drop=('tat', 42352)); chk('Ausfall und fehlende Zeile zaehlen 90 %', r['n_ok'] == 34 and r['fehlend'] == ['tat_42352'] and min(r['primaer']['P2']['diffs']) < -30 and min(r['primaer']['P1']['diffs']) < -30)
print('ANALYSETEST', 'OK' if ok else 'FEHLER')
