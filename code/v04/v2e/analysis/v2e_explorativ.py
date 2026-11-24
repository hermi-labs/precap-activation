# V2e explorativ (nicht praeregistriert, mit Claude, 20.11.2026). Aufruf: python v2e_explorativ.py <V2e-Ordner absolut>
# (1) Zerlegung log D^Q = log sqrt(sum d_i^2) + log Superposition, BPreCap gegen ctanh je Seed
# (2) Beitraege je Tiefenblock sqrt(sum d_i^2): Bild (i=0), 1-10, 11-20, 21-30, 31-40, 41-49, Linear (50)
# (3) gewaehlte Kalibrierung je Lauf; (4) 4-Bit-Preis BPreCap gegen ctanh ueber 12 Seeds bei jeder festen Kalibrierung (V2a2, V2d)
import csv, json, math, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]).resolve(); S = R.parent
BL = ((0, 1, 'Bild'), (1, 11, '1-10'), (11, 21, '11-20'), (21, 31, '21-30'), (31, 41, '31-40'), (41, 50, '41-49'), (50, 51, 'Linear'))
M = {}
for p in sorted((R / 'mess').glob('*.json')):
    a, s = p.stem.split('_')[1], int(p.stem[-5:]); M[(a, s)] = json.load(open(p))
out = []
pr = lambda *x: out.append(' '.join(str(y) for y in x))
pr('== (1) Zerlegung BPreCap/ctanh: log D = log sqrt(Sum d^2) + log Superposition ==')
for s in (42350, 42351, 42352):
    b, c = M[('bprecap', s)], M[('ctanh', s)]
    sb, sc = (math.sqrt(sum(x * x for x in m['dQ'])) for m in (b, c))
    pr(s, f"cal B={b['cal']} C={c['cal']}", f"logD={math.log(b['D_allQ'] / c['D_allQ']):+.3f}", f"log_sqrtSd2={math.log(sb / sc):+.3f}",
       f"log_superpos={math.log((b['D_allQ'] / sb) / (c['D_allQ'] / sc)):+.3f}", f"superpos B={b['D_allQ'] / sb:.3f} C={c['D_allQ'] / sc:.3f}")
pr('\n== (2) sqrt(Sum d_i^2) je Block, D^Q, Superposition ==')
pr('form seed cal', *[n for _, _, n in BL], 'D_allQ superpos')
for (a, s), m in sorted(M.items()):
    d = np.array(m['dQ']); sq = math.sqrt((d ** 2).sum())
    pr(a, s, m['cal'], *[f'{math.sqrt((d[i:j] ** 2).sum()):.3f}' for i, j, _ in BL], f"{m['D_allQ']:.3f}", f"{m['D_allQ'] / sq:.3f}")
pr('\n== (4) 4-Bit-Preis (pp) ueber 12 Seeds bei fester Kalibrierung, BPreCap (V2a2) gegen ctanh (V2d) ==')
def rows(f, a): return {r['seed']: r for r in csv.DictReader(open(f)) if r['act'] == a and r['stage'] == 'confirm' and r['status'] == 'ok'}
B, C = rows(S / 'v2a2' / 'workers' / 'w0.csv', 'bprecap'), rows(S / 'v2d' / 'workers' / 'w0.csv', 'ctanh'); K = sorted(set(B) & set(C))
for k in ('max', 'p999', 'p9999', 'p99999', 'mse4', 'best'):
    pb = np.array([100 * (float(B[s]['es_test_err_w8a4a_' + k]) - float(B[s]['es_test_err_fp32'])) for s in K])
    pc = np.array([100 * (float(C[s]['es_test_err_w8a4a_' + k]) - float(C[s]['es_test_err_fp32'])) for s in K])
    pr(f'{k:7s} n={len(K)} Median BPreCap {np.median(pb):.2f} ctanh {np.median(pc):.2f} Median Diff {np.median(pb - pc):+.2f} BPreCap kleiner {int((pb < pc).sum())}/{len(K)}')
pr('Wahl BPreCap', [B[s]['es_w8a4a_best_cal'] for s in K]); pr('Wahl ctanh  ', [C[s]['es_w8a4a_best_cal'] for s in K])
pr('mse4 gewaehlt: BPreCap', sum(B[s]['es_w8a4a_best_cal'] == 'mse4' for s in K), 'ctanh', sum(C[s]['es_w8a4a_best_cal'] == 'mse4' for s in K), 'von', len(K))
(R / 'analysis' / 'v2e_explorativ.txt').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
