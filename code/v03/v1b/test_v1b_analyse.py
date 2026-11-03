# Auswertungstest V1b an kuenstlichen Daten (vor dem Freeze, mit Claude). Ausgabe "AUSWERTUNG OK" oder Fehler.
# Szenarien: Grenze bei 3e-4 (5 %) und 5e-4 (10 %), Kontrolle gleich / ungleich, Ausfall zaehlt als Einbruch, unterer Rand besser / gleich.
import importlib.util, sys
from pathlib import Path
H = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('A', H / 'v1b_analyze.py'); A = importlib.util.module_from_spec(sp); sp.loader.exec_module(A)
R = {}
def ok(n, c, info=''):
    R[n] = bool(c); print(('ok   ' if c else 'FEHLER ') + n, info)
def row(act, lr, wm, seed, v, st='ok', sched=None, wd=None):
    return dict(run_id=f'{act}{lr}{wm}{seed}', act=act, lr=str(lr), wd=str(A.WD[act] if wd is None else wd), seed=str(seed), status=st,
                sched=sched or f'wcos{wm}', val_err_fp32=str(v), test_err_w8a4a_best=str(v + 0.05), test_err_fp32=str(v), timestamp='t', sec_train='1')
def v1rows(lo=0.447):
    out = []
    for a in A.ACTS:
        for s in (42000, 42001):
            out.append(row(a, 2e-4, 344, s, 0.456 if a == 'kL2B_tr' else 0.453, sched='wcos'))
            out.append(row(a, 1e-4, 344, s, lo, sched='wcos'))
    return out
def v1brows(lim5=3e-4, lim10=5e-4, lo=0.44, fail=None):
    out = []
    for a in A.ACTS:
        for lr in A.LRS:
            for wm in A.WARMS:
                for s in (42000, 42001):
                    lim = lim5 if wm == 344 else lim10
                    v = 0.85 if lr > lim else (0.456 if (a == 'kL2B_tr' and lr == 2e-4) else (0.453 if lr == 2e-4 else (lo if lr == 5e-5 else 0.47)))
                    if lr == 2e-4 and wm == 688: v = 0.46
                    st = 'fail' if fail == (a, lr, wm, s) else 'ok'
                    out.append(row(a, lr, wm, s, v, st))
    return out
o = A.analyze(v1brows(), v1rows())
ok('n', o['n_rows'] == 32 and o['n_ok'] == 32, o['n_rows'])
ok('kontrolle_gleich', o['kontrolle_ok'], o['kontrolle'])
ok('grenze_5', o['grenze']['kL2B_tr|344'] == 3e-4 and o['grenze']['tat|344'] == 3e-4, o['grenze'])
ok('grenze_10', o['grenze']['kL2B_tr|688'] == 5e-4, o['grenze'])
ok('warm10_verschiebt', o['warm10']['kL2B_tr']['verschiebt'] and o['warm10']['tat']['verschiebt'], o['warm10'])
ok('unten_besser', o['unten']['kL2B_tr']['urteil'] == '5e-5 besser', o['unten'])
o = A.analyze(v1brows(lim5=3e-4, lim10=3e-4, lo=0.449), v1rows())
ok('warm10_verschiebt_nicht', not o['warm10']['kL2B_tr']['verschiebt'], o['warm10'])
ok('unten_gleich', o['unten']['kL2B_tr']['urteil'] == 'gleich', o['unten'])
o = A.analyze(v1brows(lim5=2e-4), v1rows())
ok('grenze_2e4', o['grenze']['kL2B_tr|344'] == 2e-4, o['grenze'])
o = A.analyze(v1brows(fail=('kL2B_tr', 3e-4, 344, 42001)), v1rows())
ok('ausfall_ist_einbruch', o['grenze']['kL2B_tr|344'] == 2e-4 and o['n_ok'] == 31, o['grenze'])
bad = v1rows(); bad[0]['test_err_fp32'] = '0.1'
o = A.analyze(v1brows(), bad)
ok('kontrolle_ungleich_erkannt', not o['kontrolle_ok'], o['kontrolle'])
print('AUSWERTUNG OK' if all(R.values()) else 'AUSWERTUNG FEHLER'); sys.exit(0 if all(R.values()) else 1)
