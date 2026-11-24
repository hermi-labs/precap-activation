# V2e-Nachmessung mit fester Kalibrierung (explorativ, nicht praeregistriert, mit Claude, 20.11.2026).
# Frage: Liegt der D^Q-Unterschied BPreCap/ctanh in V2e an der gewaehlten Kalibrierung (ctanh waehlte bei 42350/42352 p99999/max, BPreCap immer mse4)?
# Vorgehen: die gespeicherten V2e-Gewichte (fruehes Stoppen) laden und mit dem eingefrorenen v2e_run.measure bei mse4, max und p99999 messen.
# Kontrolle: bprecap 42350 mse4 muss die V2e-Messung exakt wiedergeben (D_allQ, r, dQ gleich).
import json, sys, time
from pathlib import Path
S = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(S / 'v2e'))
import mlx.core as mx
import v2e_run as E   # eingefroren, unveraendert
R = E.R
if hasattr(mx, 'set_cache_limit'): mx.set_cache_limit(2 * 1024 ** 3)
OUT = Path(__file__).resolve().parent / 'mess'; OUT.mkdir(exist_ok=True)
G = S / 'v2e' / 'gewichte'
def rid(a, s): return f'confirm_{a}_lr3e-05_wd0_s{s}'
jobs = [('bprecap', 42350, 'mse4')] + [(a, s, c) for a in ('bprecap', 'ctanh') for s in E.SEEDS for c in ('mse4', 'max', 'p99999')]
seen = set()
for a, s, c in jobs:
    p = OUT / f'{a}_{s}_{c}.json'
    if p.exists() or (a, s, c) in seen: continue
    seen.add((a, s, c)); t0 = time.time()
    m = R.CNN(R.make_act(a), s); m.load_weights(str(G / f'{rid(a, s)}.npz')); mx.eval(m.parameters())
    M = E.measure(m, s, c); M.update(E.summary(M)); json.dump(M, open(p, 'w'))
    print(time.strftime('%H:%M'), a, s, c, 'D %.4f rbar %.4f gQ %.4f sup %.3f err_a4 %.3f' % (M['D_allQ'], M['rbar'], M['gbarQ'], M['superpos_Q'], M['err_a4']), round(time.time() - t0, 1), flush=True)
ref = json.load(open(S / 'v2e' / 'mess' / f'{rid("bprecap", 42350)}.json')); new = json.load(open(OUT / 'bprecap_42350_mse4.json'))
ok = all(ref[k] == new[k] for k in ('r', 'dQ', 'dG', 'D_allQ', 'D_allG', 'err_a4'))
print('KONTROLLE bprecap 42350 mse4 gleich V2e:', ok, flush=True); print('fertig', time.strftime('%H:%M'), flush=True)
