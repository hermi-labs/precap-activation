# Test der V2f-Auswertung mit kuenstlichen Daten (mit Claude). Aufruf: python -I test_v2f_analyse.py (scipy-venv). Exit 0 = alles ok.
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AN = HERE / 'v2f_analyze.py'
SEEDS = range(42353, 42362); COLS = None
def make(d, eff, rng, broken=None):
    rows = []
    for s in SEEDS:
        for cal in ('mse4', 'max'):
            base = dict(rbar=0.08, gbarQ=0.25, superpos_Q=1.1, D_allQ=0.3, err_a4_1000=0.5, sqrt_sum_dQ2=0.27, err_fp32_1000=0.3)
            for a in ('bprecap', 'ctanh'):
                v = dict(base)
                if a == 'bprecap':
                    for k, f in eff.items(): v[k] = v[k] * f * float(np.exp(rng.normal(0, 0.02)))
                    for k in base:
                        if k not in eff: v[k] = v[k] * float(np.exp(rng.normal(0, 0.05)))
                ok = '0' if broken != (a, s) else '2'
                rows.append(dict(run_id=f'confirm_{a}_s{s}', act=a, seed=s, cal=cal, status='ok', kontrolle_n_diff=ok, **{k: f'{x:.6f}' for k, x in v.items()}))
    (d / 'workers').mkdir(parents=True);
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def run(eff, broken=None, seed=0):
    t = Path(tempfile.mkdtemp()); d = t / 'v2f'; (t / 'kal' / 'mess').mkdir(parents=True); make(d, eff, np.random.default_rng(seed), broken)
    subprocess.run([sys.executable, '-I', str(AN), str(d), str(t / 'kal')], check=True, capture_output=True)
    return json.load(open(d / 'analysis' / 'v2f_result.json'))
fails = []
def chk(name, cond):
    print(('ok  ' if cond else 'FEHLER ') + name); cond or fails.append(name)
r = run(dict(superpos_Q=0.92, gbarQ=0.8, rbar=1.1, D_allQ=0.85, err_a4_1000=0.95))
P = r['Primaer_mse4']
chk('Effekt: H1 bestaetigt', P['H1_Kohaerenz']['urteil'] == 'bestaetigt'); chk('Effekt: H2 bestaetigt', P['H2_Weitergabe']['urteil'] == 'bestaetigt')
chk('Effekt: H3 bestaetigt', P['H3_Rundung']['urteil'] == 'bestaetigt'); chk('Effekt: Deutung Kohaerenz', r['Deutung'].startswith('Vorteil durch weniger'))
chk('Effekt: S2 D_mse4 bestaetigt', r['S2']['D_mse4']['urteil'] == 'bestaetigt'); chk('Effekt: 9 Paare', len(r['gueltige_paare']) == 9)
chk('Effekt: min p einseitig 1/512', abs(P['H1_Kohaerenz']['p_einseitig'] - 1 / 512) < 1e-5)
nfp = 0
for k in range(20):
    r0 = run({}, seed=100 + k); nfp += sum(v['urteil'] == 'bestaetigt' for v in r0['Primaer_mse4'].values())
print('ohne Effekt: bestaetigt in', nfp, 'von 60 Tests'); chk('ohne Effekt: hoechstens 3/60 bestaetigt', nfp <= 3)
r1 = run(dict(superpos_Q=1.08, gbarQ=1.2, rbar=0.9))
chk('Gegenrichtung: nichts bestaetigt', all(v['urteil'] == 'nicht bestaetigt' for v in r1['Primaer_mse4'].values()))
chk('Gegenrichtung: zweiseitig klein', r1['Primaer_mse4']['H1_Kohaerenz']['p_zweiseitig'] < 0.01)
chk('Gegenrichtung: Deutung nicht bestaetigt', r1['Deutung'].startswith('Kohaerenz-Befund nicht'))
r2 = run(dict(superpos_Q=0.92, gbarQ=0.8, rbar=1.1), broken=('ctanh', 42357))
chk('Kontrolle: unklar', all(v['urteil'] == 'unklar (Kontrolle)' for v in r2['Primaer_mse4'].values()) and r2['Deutung'] == 'unklar (Kontrolle)')
# Holm von Hand: p = (0.01, 0.04, 0.03) -> (0.03, 0.06, 0.06)
sys.path.insert(0, str(HERE)); src = AN.read_text(); ns = {}
exec(src[src.index('def holm'):src.index('def block')], ns)
chk('Holm', [round(x, 6) for x in ns['holm']([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06])
print('FEHLER:', fails if fails else 'keine'); sys.exit(1 if fails else 0)
