# D3 Auswertung nach D3_Praeregistrierung.md (beschreibend, mit Claude). Aufruf: python3 -I d3_auswerten.py [results-Ordner]
# Lesarten F1-F5 nur fuer bp200, je Seed berechnet; gilt nur bei gleicher Lesart in beiden Seeds, sonst "uneinheitlich". F6 beschreibend.
import json, sys
from pathlib import Path
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results'
SEEDS = {'bp200': (42192, 42193), 'se200': (42192, 42193), 'bp100': (42168, 42169)}
TAB = ('err_fp32_2000', 'err_all4_2000', 'err_all8_2000', 'r_all4', 'r_all8', 'eps4_median', 'eps8_median', 'gain4_median', 'gain8_median',
       'noise_gain_median', 'kurt_median', 'kurt_q1', 'kurt_q4', 'max_over_p9999_q1', 'max_over_p9999_q4', 'frac_soft_median', 'eps4_q1', 'eps4_q4')

def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for t in range(i, j + 1): r[o[t]] = (i + j) / 2
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); va = sum((x - ma) ** 2 for x in ra); vb = sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5 if va > 0 and vb > 0 else float('nan')

def f1(o): g = o['gain4_median']; return 'haelt' if g <= 0.5 else ('schwaecher' if g <= 1.0 else 'verstaerkt')
def f2(o): q = o['eps4_q4'] / o['eps4_q1']; return 'waechst hinten' if q >= 1.5 else ('sinkt hinten' if q <= 1 / 1.5 else 'gleich')
def f3(o): return 'Ausreisser hinten' if (o['kurt_q4'] >= 2 * o['kurt_q1'] or o['kurt_q4'] > 5) else 'keine'
def f4(o): return 'verlustfrei' if o['err_all8_2000'] - o['err_fp32_2000'] <= 0.005 else f"Verlust {100 * (o['err_all8_2000'] - o['err_fp32_2000']):.2f} pp"
def f5(o, r):
    return {k: ('bei 200 groesser' if o[k] / r[k] >= 1.5 else 'aehnlich') for k in ('gain4_median', 'eps4_median', 'r_all4')}

def einig(per):
    vals = {json.dumps(v, sort_keys=True) for v in per.values()}
    return json.loads(vals.pop()) if len(vals) == 1 and 'ungueltig' not in per.values() else 'uneinheitlich'

def main():
    R = {}
    for n, ss in SEEDS.items():
        for s in ss:
            f = RES / f'd3_{n}_s{s}.json'; R[(n, s)] = json.load(open(f)) if f.exists() else None
    ok = lambda o: bool(o and o.get('status') == 'ok' and o.get('valid'))
    out = dict(valid={f'{n}_s{s}': (R[(n, s)] or {}).get('valid') for n, ss in SEEDS.items() for s in ss}, table={}, reading={}, f6={})
    for (n, s), o in R.items():
        if ok(o): out['table'][f'{n}_s{s}'] = {k: round(o[k], 6) for k in TAB}
    bp = [R[('bp200', s)] for s in SEEDS['bp200']]; ref = [R[('bp100', s)] for s in SEEDS['bp100']]
    for name, fn in (('F1', f1), ('F2', f2), ('F3', f3), ('F4', f4)):
        per = {s: (fn(o) if ok(o) else 'ungueltig') for s, o in zip(SEEDS['bp200'], bp)}
        out['reading'][name] = dict(per_seed=per, befund=einig(per))
    per = {s: (f5(o, r) if ok(o) and ok(r) else 'ungueltig') for s, o, r in zip(SEEDS['bp200'], bp, ref)}   # Seed-Paare nach Reihenfolge
    out['reading']['F5'] = dict(per_seed=per, befund=einig(per))
    for n, ss in SEEDS.items():
        rs = [spearman([p['frac_soft'] for p in R[(n, s)]['per_layer'][:-1]], [p['eps4'] for p in R[(n, s)]['per_layer'][:-1]]) for s in ss if ok(R[(n, s)])]
        out['f6'][n] = dict(spearman_frac_soft_eps4=[round(x, 3) for x in rs], median=round(sum(rs) / len(rs), 3) if rs else None)   # Median zweier Werte = Mittel
    print(json.dumps(out, indent=1)); return out

if __name__ == '__main__': main()
