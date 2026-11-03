# Auswertungstest D3 an kuenstlichen Daten (mit Claude): Szenarien fuer F1-F5, uneinheitlich, ungueltig. Ausgabe AUSWERTUNGSTEST OK.
import importlib.util, json, sys, tempfile, contextlib, io
from pathlib import Path
H = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location('d3a', H / 'd3_auswerten.py'); A = importlib.util.module_from_spec(sp); sys.modules['d3a'] = A; sp.loader.exec_module(A)
def run(n, s, valid=True, **kw):
    per = [dict(layer=i + 2, frac_soft=0.01 * i, eps4=0.008 + 0.0001 * i * kw.get('slope', 0)) for i in range(20)] + [dict(layer=99, frac_soft=0, eps4=0)]
    o = dict(status='ok', valid=valid, per_layer=per, err_fp32_2000=0.46, err_all4_2000=0.5, err_all8_2000=0.46 + kw.get('l8', 0.002), r_all4=kw.get('ra', 0.3), r_all8=0.02,
             eps4_median=kw.get('e4', 0.008), eps8_median=3e-5, gain4_median=kw.get('g', 0.3), gain8_median=0.3, noise_gain_median=0.3,
             kurt_median=2.7, kurt_q1=2.7, kurt_q4=kw.get('kq4', 2.8), max_over_p9999_q1=1.0, max_over_p9999_q4=1.0, frac_soft_median=0.01,
             eps4_q1=0.008, eps4_q4=kw.get('eq4', 0.008))
    return o
def scen(bp, se=None, ref=None):
    d = Path(tempfile.mkdtemp())
    for (n, ss) in A.SEEDS.items():
        for i, s in enumerate(ss):
            o = {'bp200': bp, 'se200': se or [{}, {}], 'bp100': ref or [{}, {}]}[n][i]
            json.dump(run(n, s, **o) if o is not None else dict(status='ok', valid=False), open(d / f'd3_{n}_s{s}.json', 'w'))
    with contextlib.redirect_stdout(io.StringIO()): A.RES = d; return A.main()
R = {}
o = scen([{}, {}]); R['basis'] = (o['reading']['F1']['befund'], o['reading']['F2']['befund'], o['reading']['F3']['befund'], o['reading']['F4']['befund']) == ('haelt', 'gleich', 'keine', 'verlustfrei')
R['F5_aehnlich'] = o['reading']['F5']['befund'] == {'gain4_median': 'aehnlich', 'eps4_median': 'aehnlich', 'r_all4': 'aehnlich'}
o = scen([dict(g=0.8), dict(g=0.9)]); R['F1_schwaecher'] = o['reading']['F1']['befund'] == 'schwaecher'
o = scen([dict(g=1.2), dict(g=0.4)]); R['F1_uneinheitlich'] = o['reading']['F1']['befund'] == 'uneinheitlich'
o = scen([dict(eq4=0.02), dict(eq4=0.013)]); R['F2_waechst'] = o['reading']['F2']['befund'] == 'waechst hinten'
o = scen([dict(eq4=0.004), dict(eq4=0.005)]); R['F2_sinkt'] = o['reading']['F2']['befund'] == 'sinkt hinten'
o = scen([dict(kq4=6), dict(kq4=5.5)]); R['F3_ausreisser'] = o['reading']['F3']['befund'] == 'Ausreisser hinten'
o = scen([dict(l8=0.02), dict(l8=0.02)]); R['F4_verlust'] = o['reading']['F4']['befund'].startswith('Verlust 2.00')
o = scen([dict(g=0.6, ra=0.6), dict(g=0.6, ra=0.6)]); R['F5_groesser'] = o['reading']['F5']['befund'] == {'gain4_median': 'bei 200 groesser', 'eps4_median': 'aehnlich', 'r_all4': 'bei 200 groesser'}
o = scen([None, {}]); R['ungueltig'] = all(o['reading'][k]['befund'] == 'uneinheitlich' for k in ('F1', 'F2', 'F3', 'F4', 'F5'))
o = scen([dict(slope=1), dict(slope=1)]); R['F6_korrelation'] = abs(o['f6']['bp200']['median'] - 1.0) < 1e-9
for k, v in R.items(): print(('ok   ' if v else 'FEHLER ') + k)
print('AUSWERTUNGSTEST OK' if all(R.values()) else 'AUSWERTUNGSTEST FEHLER'); sys.exit(0 if all(R.values()) else 1)
