# V2k Auswertungstest an kuenstlichen Daten (mit Claude): Effekt beim Abschneiden -> K1 und K2 bestaetigt; Effekt bei beidem gleich -> K1 ja, K2 nein;
# Null -> selten Fehlalarm; fehlendes Paar oder Abweichung von V2f -> "unklar (Kontrolle)"; S3 erkennt Kreuzterm-Unterschied; Profil und Kosinus lesbar.
import csv, json, math, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; SEEDS = list(range(42353, 42362)); ok = True
COLS = ['run_id', 'act', 'seed', 'cal', 'status', 'D_allQ', 'D_allC', 'D_allR', 'rbarQ', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'n_ohne_clip', 'frac_clip_median',
        'kappa', 'nu', 'superpos_Q', 'cos_nachbar_median', 'cos_mittel_innen', 'qc_abw', 'v2f_abw_dQ', 'v2f_abw_D', 'sec', 'timestamp']
def make(d, seed, fC=1.0, fR=1.0, dkappa=0.0, drop=None, abw=None, max_ohne_clip=False):
    r = np.random.default_rng(seed); (d / 'workers').mkdir(parents=True); (d / 'mess').mkdir()
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for s in SEEDS:
            for a in ('bprecap', 'ctanh'):
                for cal in ('mse4', 'max'):
                    if drop == (a, s, cal): continue
                    e = lambda: math.exp(r.normal(0, 0.05)); isb = a == 'bprecap'
                    row = dict(run_id=f'confirm_{a}_lr3e-05_wd0_s{s}', act=a, seed=s, cal=cal, status='ok', D_allQ=0.3 * e(), D_allC=0.2 * e() * (fC if isb else 1), D_allR=0.2 * e() * (fR if isb else 1),
                               rbarQ=0.08 * e(), rbarC=0.05 * e(), rbarR=0.06 * e(), gbarQ=0.2 * e(), gbarC=0.2 * e(), gbarR=0.2 * e(), n_ohne_clip=0, frac_clip_median=0.01,
                               kappa=0.0 + r.normal(0, 0.02) + (dkappa if isb else 0), nu=0.1 + r.normal(0, 0.01), superpos_Q=1.0 * e(), cos_nachbar_median=0.3, cos_mittel_innen=0.05,
                               qc_abw=0.0, v2f_abw_dQ=abw if (abw is not None and isb and s == SEEDS[0]) else 1e-6, v2f_abw_D=1e-6, sec=1, timestamp='x')
                    if max_ohne_clip and cal == 'max': row.update(D_allC=0.0, gbarC='nan', rbarC=0.0)
                    w.writerow(row)
                    nl = 51; dq = list(np.abs(r.normal(0.03, 0.005, nl))); rq = list(np.abs(r.normal(0.08, 0.01, nl))); rc = list(np.abs(r.normal(0.05, 0.01, nl)))
                    cos = np.eye(nl) + 0.0; cos[np.triu_indices(nl, 1)] = 0.1; cos = np.maximum(cos, cos.T)
                    json.dump(dict(dQ=dq, rQ=rq, rC=rc, cos=cos.tolist()), open(d / 'mess' / f"{row['run_id']}_{cal}.json", 'w'))
def res(seed, **kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); make(d, seed, **kw)
        p = subprocess.run([sys.executable, '-I', str(H / 'v2k_analyze.py'), str(d)], capture_output=True, text=True)
        if p.returncode: print(p.stderr); raise SystemExit(1)
        return json.load(open(d / 'analysis' / 'v2k_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
r = res(1, fC=0.7); u = (r['Primaer']['K1']['urteil'], r['Primaer']['K2']['urteil'])
chk('Effekt nur beim Abschneiden -> K1 und K2 bestaetigt', u == ('bestaetigt', 'bestaetigt'), str(u)); chk('Deutung gesetzt', 'Abschneiden' in r['Deutung'], r['Deutung'])
chk('Profil und Kosinus-Bloecke vorhanden', len(r['profil_bloecke']['mse4']) == 5 and len(r['kosinus_bloecke']['mse4']['bprecap']) == 5)
r = res(2, fC=0.7, fR=0.7); u = (r['Primaer']['K1']['urteil'], r['Primaer']['K2']['urteil']); chk('gleicher Effekt bei beidem -> K1 ja, K2 nein', u == ('bestaetigt', 'nicht bestaetigt'), str(u))
r = res(3, fC=1.3); chk('Gegenrichtung -> K1 nicht bestaetigt', r['Primaer']['K1']['urteil'] == 'nicht bestaetigt')
n1 = n2 = 0
for i in range(20):
    rr = res(10 + i); n1 += rr['Primaer']['K1']['urteil'] == 'bestaetigt'; n2 += rr['Sekundaer']['S3']['urteil'] != 'kein klarer Unterschied'
chk('Null: Fehlalarme K1 <= 3 von 20', n1 <= 3, str(n1)); chk('Null: Fehlalarme S3 <= 3 von 20', n2 <= 3, str(n2))
r = res(4, fC=0.7, dkappa=-0.2); chk('S3 erkennt kleineren Kreuzterm', r['Sekundaer']['S3']['urteil'] == 'Unterschied (BPreCap kleiner)', r['Sekundaer']['S3']['urteil'])
r = res(5, fC=0.7, drop=('ctanh', SEEDS[2], 'max')); chk('fehlendes Paar -> unklar (Kontrolle)', r['Primaer']['K1']['urteil'] == 'unklar (Kontrolle)' and len(r['gueltige_paare']) == 8)
r = res(6, fC=0.7, abw=0.01); chk('Abweichung von V2f -> unklar (Kontrolle)', r['Primaer']['K2']['urteil'] == 'unklar (Kontrolle)' and r['v2f_abw_max'] == 0.01)
r = res(7, fC=0.7, max_ohne_clip=True); chk('max ohne Abschneiden -> Primaer gueltig, max-K1 nicht berechenbar', r['Primaer']['K1']['urteil'] == 'bestaetigt' and isinstance(r['beschreibend_max']['K1'], str), str(r['beschreibend_max']['K1'])[:60])
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
