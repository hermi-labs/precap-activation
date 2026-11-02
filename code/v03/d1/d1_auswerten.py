# D1 Auswertung nach D1_Praeregistrierung.md (beschreibend). Aufruf: python3 -I d1_auswerten.py [results-Ordner]
# Je Seed: Faktor_eps = eps_local_median(ta) / eps_local_median(bp bzw. lb), Faktor_gain = gain_median(ta) / gain_median(bp bzw. lb).
# Lesart: E1, wenn Faktor_eps >= 2 und Faktor_gain < 1,5; E2, wenn Faktor_eps < 1,5 und Faktor_gain >= 2; beide, wenn beide >= 2;
# keine, wenn beide < 1,5; sonst "zwischen". Befund nur, wenn beide Seeds dieselbe Lesart haben, sonst "uneinheitlich".
import json, sys
from pathlib import Path
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results'
SEEDS = (42168, 42169); NAMES = ('bp', 'lb', 'ta', 't0', 'se', 'lh')   # se (SELU) seit 01.11., 02:00 dabei, nur beschreibend

def lesart(fe, fg):
    if fe >= 2 and fg >= 2: return 'beide'
    if fe >= 2 and fg < 1.5: return 'E1'
    if fe < 1.5 and fg >= 2: return 'E2'
    if fe < 1.5 and fg < 1.5: return 'keine'
    return 'zwischen'

def main():
    R = {}
    for n in NAMES:
        for s in SEEDS:
            f = RES / f'd1_{n}_s{s}.json'
            R[(n, s)] = json.load(open(f)) if f.exists() else None
    out = dict(valid={f'{n}_s{s}': (R[(n, s)] or {}).get('valid') for n in NAMES for s in SEEDS}, table={}, reading={})
    for (n, s), o in R.items():
        if o and o.get('status') == 'ok':
            out['table'][f'{n}_s{s}'] = {k: round(o[k], 5) for k in ('eps_local_median', 'gain_median', 'noise_gain_median', 'kurt_median', 'r_all', 'err_all_2000', 'err_fp32_2000')}
    for ref in ('bp', 'lb', 'se'):   # ta_vs_se ist beschreibend (nicht in der urspruenglichen Lesart)
        per = {}
        for s in SEEDS:
            a, b = R[('ta', s)], R[(ref, s)]
            if not (a and b and a.get('valid') and b.get('valid')): per[s] = 'ungueltig'; continue
            fe = a['eps_local_median'] / b['eps_local_median']; fg = a['gain_median'] / b['gain_median']; fn = a['noise_gain_median'] / b['noise_gain_median']
            per[s] = dict(f_eps=round(fe, 3), f_gain=round(fg, 3), f_noise=round(fn, 3), lesart=lesart(fe, fg))
        ls = {v['lesart'] if isinstance(v, dict) else v for v in per.values()}
        out['reading'][f'ta_vs_{ref}'] = dict(per_seed=per, befund=(ls.pop() if len(ls) == 1 and 'ungueltig' not in ls else 'uneinheitlich'))
    # beschreibend (01.11., 03:50): gleicher linearer Kern, hart (lh) gegen weich (lb)
    per = {}
    for s_ in SEEDS:
        a_, b_ = R[('lh', s_)], R[('lb', s_)]
        if not (a_ and b_ and a_.get('valid') and b_.get('valid')): per[s_] = 'ungueltig'; continue
        fe = a_['eps_local_median'] / b_['eps_local_median']; fg = a_['gain_median'] / b_['gain_median']
        per[s_] = dict(f_eps=round(fe, 3), f_gain=round(fg, 3), f_noise=round(a_['noise_gain_median'] / b_['noise_gain_median'], 3), lesart=lesart(fe, fg))
    ls = {v['lesart'] if isinstance(v, dict) else v for v in per.values()}
    out['reading']['lh_vs_lb'] = dict(per_seed=per, befund=(ls.pop() if len(ls) == 1 and 'ungueltig' not in ls else 'uneinheitlich'))
    print(json.dumps(out, indent=1)); return out

if __name__ == '__main__': main()
