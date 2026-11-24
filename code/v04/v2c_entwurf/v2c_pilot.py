# V2c-Pilot (privat, mit Claude, 18.11.): tanh im V2a2-Netz (L = 50, Delta-orthogonal, Verstaerkung 1, Bias 0), nur Val-Fehler fp32,
# zur Wahl des LR-Rasters fuer das Tuning. Code unveraendert aus studien/v2a2/v2a2_run.py (Import), nur make_act um 'tanh' ergaenzt.
import json, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent / 'v2a2'))
import v2a2_run as R
import mlx.core as mx
_orig = R.make_act
R.make_act = lambda n: mx.tanh if n == 'tanh' else _orig(n)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1720; out = []
for lr in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
    t0 = time.time(); m, ok, tl, kurve, best = R.train('tanh', lr, 0.0, 'cos', 42000, steps)
    o = dict(lr=lr, steps=steps, ok=ok, train_loss_last50=tl, val_kurve=kurve, val_err_end=kurve[-1][1] if ok else None, val_err_best=best[1] if best else None, sec=round(time.time() - t0, 1))
    out.append(o); print(time.strftime('%H:%M'), json.dumps(o), flush=True)
    json.dump(out, open(H / f'v2c_pilot_tanh_{steps}.json', 'w'), indent=1)
