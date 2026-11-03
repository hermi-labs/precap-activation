# D3b Auswertung nach D3b_Praeregistrierung.md (beschreibend, mit Claude). Aufruf: python3 -I d3b_auswerten.py [d3b-results] [d3-results]
# Lesarten L1-L4 fuer BPreCap: Verhaeltnis V1c (mit Anlauf) / B15 (ohne) je Seed. Ein Befund gilt nur bei gleicher Lesart in beiden Seeds.
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else H / 'results'
REF = Path(sys.argv[2]) if len(sys.argv) > 2 else H.parent / 'd3' / 'results'
SEEDS = (42192, 42193)
TAB = ('err_fp32_2000', 'err_all4_2000', 'err_all8_2000', 'r_all4', 'eps4_median', 'eps4_q1', 'eps4_q4', 'gain4_median', 'noise_gain_median',
       'frac_soft_median', 'frac_soft_q1', 'frac_soft_q4', 'kurt_median', 'max_over_p9999_median')
TH = 1.1   # L1, L2, L4: Verhaeltnis ab 1,1 bzw. bis 1/1,1 zaehlt als Unterschied
def lesart(q, hi, lo, th=TH): return hi if q >= th else (lo if q <= 1 / th else 'gleich')
READ = {'L1_eps4': ('eps4_median', 'lokal groesser', 'lokal kleiner', TH),
        'L2_gain4': ('gain4_median', 'Daempfung schwaecher', 'Daempfung staerker', TH),
        'L3_frac_soft': ('frac_soft_median', 'mehr im weichen Bereich', 'weniger im weichen Bereich', 1.25),
        'L4_r_all4': ('r_all4', 'Logit-Fehler groesser', 'Logit-Fehler kleiner', TH)}
def load(d, pre, n, s):
    f = d / f'{pre}_{n}_s{s}.json'; return json.load(open(f)) if f.exists() else None
def ok(o): return bool(o and o.get('status') == 'ok' and o.get('valid'))
def main():
    R = {('bp', 'w', s): load(RES, 'd3b', 'bp200w', s) for s in SEEDS} | {('se', 'w', s): load(RES, 'd3b', 'se200w', s) for s in SEEDS} | \
        {('bp', 'o', s): load(REF, 'd3', 'bp200', s) for s in SEEDS} | {('se', 'o', s): load(REF, 'd3', 'se200', s) for s in SEEDS}
    out = dict(valid={f'{a}_{w}_s{s}': ok(o) for (a, w, s), o in R.items()}, table={}, ratio={}, reading={}, cost4_pp={}, accum_r4={})
    for (a, w, s), o in R.items():
        if not ok(o): continue
        k = f"{a}200{'_anlauf' if w == 'w' else '_ohne'}_s{s}"
        out['table'][k] = {t: round(o[t], 6) for t in TAB}
        out['cost4_pp'][k] = round(100 * (o['err_all4_2000'] - o['err_fp32_2000']), 2)
        out['accum_r4'][k] = {str(x['upto']): round(x['r4'], 4) for x in o['accum']}
    for a in ('bp', 'se'):
        for name, (key, hi, lo, th) in READ.items():
            per = {}
            for s in SEEDS:
                w, o = R[(a, 'w', s)], R[(a, 'o', s)]
                if ok(w) and ok(o):
                    q = w[key] / o[key]; out['ratio'].setdefault(f'{a}_{name}', {})[s] = round(q, 4); per[s] = lesart(q, hi, lo, th)
                else: per[s] = 'ungueltig'
            v = set(per.values())
            befund = v.pop() if len(v) == 1 and 'ungueltig' not in per.values() else 'uneinheitlich'
            if a == 'bp': out['reading'][name] = dict(per_seed=per, befund=befund)
            else: out['reading'][f'SELU_{name}'] = dict(per_seed=per, befund=befund)   # SELU nur zur Einordnung
    print(json.dumps(out, indent=1)); return out
if __name__ == '__main__': main()
