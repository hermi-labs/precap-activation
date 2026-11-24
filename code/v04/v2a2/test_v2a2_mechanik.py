# V2a2 Mechanik-Test (mit Claude).
# 1. Daten gleich Pilot. 2. Kontrollpfad (konstante LR, WD 0) bei 20 Schritten, L = 50, alle drei Formen: Val-Fehler fp32, w8a8, w8a4a_max,
#    w8a4a_p9999 auf 500 Val-Bildern bitgleich zum Pilot-Code (v2a_pilot4.run); Kalibrierwerte max_abs, max, p9999 gleich.
# 3. Fruehes Stoppen: zurueckgeladene Parameter treffen den gemessenen Val-Fehler; Kosinus-Pfad laeuft. 4. mse4 und p99999 plausibel.
# 6. CLE fuer TAT: Funktion in fp32 gleich, Gewichtsbereiche je Kanalpaar angeglichen.
# 5. Mit --full: Kontrolle ueber die volle Laenge (BPreCap, LR 3e-5, Seed 42000, 1720 Schritte, L = 50) gegen v2a_pilot4_D_M1Pro.json.
import json, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); sys.path.insert(0, str(H.parent / 'v2a_entwurf'))
import v2a2_run as R, v2a_pilot4 as P4
import mlx.core as mx
from mlx.utils import tree_unflatten
ok = True
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i, flush=True); ok = ok and bool(c)
xtr, ytr, xv, yv, xte, yte = R.load()
chk('Daten gleich Pilot', (xtr == P4.V.xtr).all() and (xv == P4.V.xv).all() and (ytr == P4.V.ytr).all() and (yv == P4.V.yv).all())
def mine(name, lr, seed, steps, nv):
    m, okk, _, kurve, best = R.train(name, lr, 0.0, 'const', seed, steps); Rc = R.calib(m, xtr[:R.NCAL], seed); x, y = xv[:nv], yv[:nv]
    return {f'val_err_{k}': round(R.err(R.qforward(m, k, Rc), x, y), 4) for k in ('fp32', 'w8a8', 'w8a4a_max', 'w8a4a_p9999')}, Rc, m
for name, lr in (('bprecap', 3e-5), ('tat', 1e-4), ('selu', 3e-5)):
    ref = P4.run(name, lr, 50, 32, 20, 7, P4.V.xv[:500], P4.V.yv[:500]); o, Rc, m = mine(name, lr, 7, 20, 500)
    for k, v in o.items(): chk(f'20 Schritte {name} {k}', v == ref[k], f'{v} gegen Pilot-Code {ref[k]}')
    R4 = P4.calib(m, xtr[:R.NCAL], 7); chk(f'Kalibrierung {name} max_abs/max/p9999 gleich Pilot', all(R4[k] == Rc[k] for k in ('max_abs', 'max', 'p9999')))
    lo, hi = zip(*Rc['max']); lo5, hi5 = zip(*Rc['p99999']); lom, him = zip(*Rc['mse4'])
    chk(f'{name} p99999 und mse4 innerhalb min/max', all(a >= b - 1e-6 for a, b in zip(lo5, lo)) and all(a <= b + 1e-6 for a, b in zip(hi5, hi)) and
        all(a >= min(b, 0) - 1e-6 for a, b in zip(lom, lo)) and all(a <= max(b, 0) + 1e-6 for a, b in zip(him, hi)))
# mse4 ist auf der Stichprobe nie schlechter als min/max (k = 100 ist im Gitter)
s = np.random.default_rng(3).standard_t(3, 200_000).astype(np.float32); mm = (float(s.min()), float(s.max())); lm = R.mse_asym(s, mm)
e = lambda lh: float(mx.mean((R.qa_asym(mx.array(s), 4, lh) - mx.array(s)) ** 2)); chk('mse4 <= max auf Testverteilung', e(lm) <= e(mm) + 1e-9, f'{e(lm):.5f} <= {e(mm):.5f}')
# fruehes Stoppen und Kosinus-Pfad
m, okk, tl, kurve, best = R.train('bprecap', 1e-4, 3.0, 'cos', 5, 24)
chk('Kosinus-Lauf ok, 8 Messpunkte', okk and len(kurve) == 8 and kurve[-1][0] == 24, str(kurve))
chk('bester Punkt = kleinster Val-Fehler, Gleichstand frueher', best[1] == min(e_ for _, e_ in kurve) and best[0] == min(s_ for s_, e_ in kurve if e_ == best[1]))
m.update(tree_unflatten(list(best[2].items()))); mx.eval(m.parameters()); chk('zurueckgeladene Parameter treffen Val-Fehler', R.err(m, xv, yv) == best[1])
o = R.run_one(dict(stage='confirm', act='tat', lr=1e-4, wd=0.0, seed=6, steps=16))
chk('Bestaetigungszeile vollstaendig', o['status'] == 'ok' and all(o.get(c, '') != '' for c in R.COLS if c not in ('sec', 'timestamp')), str([c for c in R.COLS if o.get(c, '') == '']))
chk('w8a4a_best ist eine der Kalibrierungen', o['es_test_err_w8a4a_best'] == o[f"es_test_err_w8a4a_{o['es_w8a4a_best_cal']}"])
# CLE (nur TAT): Funktion in fp32 gleich, Bereiche je Kanalpaar angeglichen
m, okk, _, _, _ = R.train('tat', 1e-4, 0.0, 'cos', 8, 40); mc = R.cle(m); x = mx.array(xte[:500])
a, b = np.array(m(x)), np.array(mc(x)); rel = float(np.abs(a - b).max() / np.abs(a).max())
chk('CLE: Ausgabe in fp32 gleich (rel. < 1e-4)', rel < 1e-4, f'{rel:.2e}')
r1 = np.abs(np.array(mc.convs[10].weight)).max(axis=(1, 2, 3)); r2 = np.abs(np.array(mc.convs[11].weight)).max(axis=(0, 1, 2))
chk('CLE: Bereiche angeglichen (Paar 10/11)', float(np.abs(r1 / r2 - 1).max()) < 0.05, f'{float(np.abs(r1 / r2 - 1).max()):.3f}')
chk('CLE: fp32-Testfehler gleich (+-0,1 pp)', abs(R.err(m, xte, yte) - R.err(mc, xte, yte)) <= 0.001)
chk('Zeile TAT: es_cle_fp32_delta klein', abs(o['es_cle_fp32_delta']) <= 0.001, str(o['es_cle_fp32_delta']))
if '--full' in sys.argv:
    ref = [x for x in json.load(open(H.parent / 'v2a_entwurf' / 'v2a_pilot4_D_M1Pro.json')) if x['act'] == 'bprecap'][0]
    o, _, _ = mine('bprecap', 3e-5, 42000, 1720, len(xv))
    for k, v in o.items(): chk(f'Kontrolle volle Laenge {k}', v == ref[k], f'{v} gegen Pilot {ref[k]}')
print('MECHANIK OK' if ok else 'MECHANIK FEHLER'); sys.exit(0 if ok else 1)
