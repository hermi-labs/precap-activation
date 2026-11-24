# V2a-1 Mechanik-Test (mit Claude).
# 1. Gleichheit mit dem Pilot-Code (studien/v2a_entwurf/v2a_pilot4.py) bei 20 Schritten, beide Formen, alle Pilot-Messgroessen auf 500 Val-Bildern.
# 2. Mit --full: Kontrolle ueber die volle Laenge: BPreCap, LR 1e-4, Seed 42000, 1720 Schritte trifft die Pilot-Werte aus
#    v2a_pilot4_A_M1Pro.json bitgleich (val fp32, w8a8, w8a4a_max, w8a4a_p9999).
# 3. p999 aendert die anderen Kalibrierwerte nicht; Wahl w8a4a_best ist eine der drei Kalibrierungen; Seeds und Aufteilung.
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); sys.path.insert(0, str(H.parent / 'v2a_entwurf'))
import v2a1_run as R, v2a_pilot4 as P4
ok = True
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i, flush=True); ok = ok and bool(c)
xtr, ytr, xv, yv, xte, yte = R.load()
chk('Daten gleich Pilot', (xtr == P4.V.xtr).all() and (xv == P4.V.xv).all() and (ytr == P4.V.ytr).all())
def mine(name, lr, seed, steps, nv):
    m, okk, _ = R.train(name, lr, seed, steps); Rc = R.calib(m, xtr[:R.NCAL], seed); x, y = xv[:nv], yv[:nv]
    o = {f'val_err_{k}': round(R.err(R.qforward(m, k, Rc), x, y), 4) for k in ('fp32', 'w8a8', 'w8a4a_max', 'w8a4a_p9999')}; return o, Rc, m
for name, lr in (('bprecap', 1e-4), ('selu', 3e-5)):
    ref = P4.run(name, lr, 32, 32, 20, 7, P4.V.xv[:500], P4.V.yv[:500]); o, Rc, m = mine(name, lr, 7, 20, 500)
    for k, v in o.items(): chk(f'20 Schritte {name} {k}', v == ref[k], f'{v} gegen Pilot-Code {ref[k]}')
    R4 = P4.calib(m, xtr[:R.NCAL], 7); chk(f'Kalibrierung {name} ohne p999 gleich', all(R4[k] == Rc[k] for k in ('max_abs', 'max', 'p9999')))
if '--full' in sys.argv:
    ref = [o for o in json.load(open(H.parent / 'v2a_entwurf' / 'v2a_pilot4_A_M1Pro.json')) if o['act'] == 'bprecap' and o['seed'] == 42000][0]
    o, _, _ = mine('bprecap', 1e-4, 42000, 1720, len(xv))
    for k, v in o.items(): chk(f'Kontrolle volle Laenge {k}', v == ref[k], f'{v} gegen Pilot {ref[k]}')
c = R.configs(); chk('Konfigurationen', len(c) == 24 and {x['seed'] for x in c} == set(range(42330, 42342)) and all(x['lr'] == R.LR[x['act']] for x in c))
print('MECHANIK OK' if ok else 'MECHANIK FEHLER'); sys.exit(0 if ok else 1)
