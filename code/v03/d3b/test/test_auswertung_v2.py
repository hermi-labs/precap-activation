# Test der D3b-Auswertung v2 (Kopie des eingefrorenen Tests, nur der Dateiname geaendert) an kuenstlichen Ergebnisdateien (mit Claude). Aufruf: python3 -I test/test_auswertung.py
import json, sys, tempfile, importlib.util
from pathlib import Path
H = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location('aus', H / 'd3b_auswerten_v2.py'); A = importlib.util.module_from_spec(sp); sys.argv = ['x']; sp.loader.exec_module(A)
def fake(eps=0.0086, gain=0.42, fs=0.03, r=0.46, valid=True):
    o = dict(status='ok', valid=valid, err_fp32_2000=0.45, err_all4_2000=0.52, err_all8_2000=0.45, r_all4=r, eps4_median=eps, eps4_q1=eps, eps4_q4=eps,
             gain4_median=gain, noise_gain_median=1.0, frac_soft_median=fs, frac_soft_q1=fs, frac_soft_q4=fs, kurt_median=2.8, max_over_p9999_median=1.0,
             accum=[dict(upto=2, r4=0.1), dict(upto=201, r4=r)])
    return o
def run(w_bp, w_bp2=None, o_bp=None):
    d = Path(tempfile.mkdtemp()); a = d / 'a'; b = d / 'b'; a.mkdir(); b.mkdir()
    for s, wb in zip((42192, 42193), (w_bp, w_bp2 or w_bp)):
        json.dump(wb, open(a / f'd3b_bp200w_s{s}.json', 'w')); json.dump(fake(), open(a / f'd3b_se200w_s{s}.json', 'w'))
        json.dump(o_bp or fake(), open(b / f'd3_bp200_s{s}.json', 'w')); json.dump(fake(), open(b / f'd3_se200_s{s}.json', 'w'))
    A.RES, A.REF = a, b; return A.main()
ok = True
o = run(fake()); ok &= all(v['befund'] == 'gleich' for k, v in o['reading'].items())                                   # Null: alles gleich
o = run(fake(eps=0.012, fs=0.015, r=0.6)); ok &= o['reading']['L1_eps4']['befund'] == 'lokal groesser' and \
    o['reading']['L3_frac_soft']['befund'] == 'weniger im weichen Bereich' and o['reading']['L4_r_all4']['befund'] == 'Logit-Fehler groesser' \
    and o['reading']['L2_gain4']['befund'] == 'gleich'                                                                    # Hypothese
o = run(fake(gain=0.6), fake(gain=0.40)); ok &= o['reading']['L2_gain4']['befund'] == 'uneinheitlich'                 # Seeds uneinig
o = run(fake(valid=False)); ok &= o['reading']['L1_eps4']['befund'] == 'uneinheitlich' and not o['valid']['bp_w_s42192']  # ungueltig
o = run(fake(fs=0.025)); ok &= o['reading']['L3_frac_soft']['befund'] == 'gleich'                                     # 0,83 liegt innerhalb 1,25
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
