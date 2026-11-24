# V2d-Pilot (privat, mit Claude, 19.11.): kalibrierte tanh f(x) = lam * tanh(b x) im V2a2-Netz. lam, b nach demselben Rezept wie BPreCap:
# E f = 0, E f^2 = 1 bei z ~ N(0,1), chi1 = E[f'(z)^2] = 1 + 2/L = 1.04. Nur Val-Fehler fp32, zur Wahl des LR-Rasters fuer das Tuning.
# Code unveraendert aus studien/v2a2/v2a2_run.py (Import), nur make_act um 'ctanh' ergaenzt. BPreCap und Kalibrierungsrezept: Ideen Aarons.
import json, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent / 'v2a2'))
import v2a2_run as R
import mlx.core as mx
LAM, B = 2.2948782077912586, 0.5328431680484216
_orig = R.make_act
R.make_act = lambda n: (lambda x: LAM * mx.tanh(B * x)) if n == 'ctanh' else _orig(n)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1720; out = []
for lr in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
    t0 = time.time(); m, ok, tl, kurve, best = R.train('ctanh', lr, 0.0, 'cos', 42000, steps)
    o = dict(lr=lr, steps=steps, ok=ok, train_loss_last50=tl, val_kurve=kurve, val_err_end=kurve[-1][1] if ok else None, val_err_best=best[1] if best else None, sec=round(time.time() - t0, 1))
    out.append(o); print(time.strftime('%H:%M'), json.dumps(o), flush=True)
    json.dump(out, open(H / f'v2d_pilot_ctanh_{steps}.json', 'w'), indent=1)
