# V2m (privat, mit Claude): Konstanten der fruehen Saettigung (Knie rho = 0,4) mit dem Loeser aus math/p4_bounded_depth.py (unveraendert importiert,
# nur dessen Modul-Konstante RHO gesetzt). Ziel wie V2a2 (kL2B_d50.json): E f = 0, E f^2 = 1, chi1 = 1 + 2/50 = 1,04, Grenze M = 3.
# Kontrolle: mit rho = 0,8 muss der Loeser die V2a2-Konstanten wiederfinden. Aufruf: <python mit scipy> -I v2m_konstanten.py
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent.parent / 'math'))
import p4_bounded_depth as P
KC = json.load(open(H.parent / 'v2a2' / 'kL2B_d50.json')); x0 = (KC['lam'], KC['alpha'], KC['b']); out = {}
for rho, name in ((0.8, 'kontrolle_rho08'), (0.4, 'rho04')):
    P.RHO = rho; p, res, ok = P.solve(1.04, 3.0, x0); r = P.props(p, 3.0, 1.04); r.update(rho=rho, M=3.0, resid=res, ok=bool(ok), L=50); out[name] = r
k = out['kontrolle_rho08']; out['kontrolle_abw'] = max(abs(k['lam'] - KC['lam']), abs(k['alpha'] - KC['alpha']), abs(k['b'] - KC['b']))
out['kontrolle_ok'] = out['kontrolle_abw'] < 1e-8
json.dump(out['rho04'], open(H / 'kL2B_d50_rho04.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
