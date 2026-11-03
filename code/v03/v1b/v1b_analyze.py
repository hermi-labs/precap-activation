# V1b-Auswertung (privat, mit Claude, beschreibend): Lage der LR-Grenze mit Anlauf, Wirkung von 10 % Anlauf, unterer Rand 5e-5.
# Aufruf: python -I v1b_analyze.py <v1b-Ordner> <v1-Ordner>. Kriterien wie in V1b_Praeregistrierung.md, keine Signifikanztests (2 Seeds).
import csv, json, sys
from pathlib import Path
import numpy as np

EINBRUCH = 0.80    # Lauf gilt als eingebrochen, wenn der Val-Fehler am Ende ueber 80 % liegt
SCHWELLE = 0.005   # 0,5 pp: kleinere Unterschiede im Median gelten als gleich
ACTS = ['kL2B_tr', 'tat']; LRS = [5e-5, 2e-4, 3e-4, 5e-4]; WARMS = [344, 688]; WD = {'kL2B_tr': 0.3, 'tat': 0.0}
SKIP = {'run_id', 'sched', 'timestamp', 'sec_train', 'ms_per_step', 'sec_checks', 'backend'}   # Kontrolle: alles andere muss gleich sein

def load(d):
    rows = []
    for f in sorted((Path(d) / 'tune' / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    return rows

def warm(r): return int(r['sched'][4:] or 344)

def cell(rows, act, lr, wm):
    return [r for r in rows if r['act'] == act and abs(float(r['lr']) - lr) < 1e-12 and r['sched'].startswith('wcos') and warm(r) == wm]

def verr(r): return float(r['val_err_fp32']) if r['status'] == 'ok' else 0.9   # Ausfall zaehlt als Einbruch

def analyze(rows, v1rows):
    out = {'n_rows': len(rows), 'n_ok': sum(r['status'] == 'ok' for r in rows), 'cells': {}, 'kontrolle': [], 'grenze': {}, 'warm10': {}, 'unten': {}}
    for a in ACTS:
        for lr in LRS:
            for wm in WARMS:
                rs = cell(rows, a, lr, wm); v = [verr(r) for r in rs]
                q = [float(r['test_err_w8a4a_best']) for r in rs if r['status'] == 'ok' and r['test_err_w8a4a_best'] not in ('', 'nan')]
                out['cells'][f'{a}|{lr:g}|{wm}'] = dict(n=len(rs), val=[round(x * 100, 2) for x in v], med_val=round(float(np.median(v)) * 100, 2) if v else None,
                                                       einbrueche=sum(x > EINBRUCH for x in v), med_w8a4a=round(float(np.median(q)) * 100, 2) if q else None)
    # Kontrolle: LR 2e-4, Anlauf 344 gegen V1-Tuning (sched 'wcos'), alle Kennzahlen gleich
    for a in ACTS:
        for r in cell(rows, a, 2e-4, 344):
            m = [x for x in v1rows if x['act'] == a and x['seed'] == r['seed'] and abs(float(x['lr']) - 2e-4) < 1e-12 and abs(float(x['wd']) - WD[a]) < 1e-12 and x['sched'] == 'wcos']
            eq = len(m) == 1 and all(r[k] == m[0][k] for k in r if k not in SKIP and k in m[0])
            out['kontrolle'].append(dict(act=a, seed=r['seed'], equal=bool(eq)))
    out['kontrolle_ok'] = len(out['kontrolle']) == 4 and all(k['equal'] for k in out['kontrolle'])
    C = out['cells']
    for a in ACTS:
        for wm in WARMS:   # G1: hoechste LR ohne Einbruch in beiden Seeds (von unten, bis zum ersten Einbruch)
            g = None
            for lr in LRS[1:]:
                if C[f'{a}|{lr:g}|{wm}']['einbrueche'] == 0 and C[f'{a}|{lr:g}|{wm}']['n'] == 2: g = lr
                else: break
            out['grenze'][f'{a}|{wm}'] = g
        # G2: verschiebt 10 % die Grenze? (LR, bei der 344 einbricht und 688 nicht)
        out['warm10'][a] = dict(verschiebt=any(C[f'{a}|{lr:g}|344']['einbrueche'] > 0 and C[f'{a}|{lr:g}|688']['einbrueche'] == 0 for lr in LRS),
                                diff_2e4_pp=round(C[f'{a}|0.0002|688']['med_val'] - C[f'{a}|0.0002|344']['med_val'], 2) if C[f'{a}|0.0002|688']['med_val'] is not None else None)
        # G3: unterer Rand. 5e-5 mit 5 % Anlauf gegen die V1-Wahl 1e-4 (gleicher WD, V1-Tuning)
        ref = [float(x['val_err_fp32']) for x in v1rows if x['act'] == a and abs(float(x['lr']) - 1e-4) < 1e-12 and abs(float(x['wd']) - WD[a]) < 1e-12 and x['status'] == 'ok']
        lo = C[f'{a}|5e-05|344']['med_val']
        if ref and lo is not None:
            d = lo - float(np.median(ref)) * 100
            out['unten'][a] = dict(med_5e5=lo, med_1e4_v1=round(float(np.median(ref)) * 100, 2), diff_pp=round(d, 2),
                                   urteil='5e-5 besser' if d < -SCHWELLE * 100 else ('5e-5 schlechter' if d > SCHWELLE * 100 else 'gleich'))
    return out

def main():
    v1b, v1 = sys.argv[1], sys.argv[2]
    o = analyze(load(v1b), load(v1))
    print(f"Laeufe {o['n_ok']}/{o['n_rows']} ok | Kontrolle bitgleich zu V1: {o['kontrolle_ok']} {o['kontrolle']}")
    for a in ACTS:
        print(a)
        for lr in LRS:
            print('  ', f'{lr:g}', ' | '.join(f"Anlauf {wm}: {o['cells'][f'{a}|{lr:g}|{wm}']['val']} (w8a4a {o['cells'][f'{a}|{lr:g}|{wm}']['med_w8a4a']})" for wm in WARMS))
        print('   Grenze 5 % / 10 %:', o['grenze'][f'{a}|344'], '/', o['grenze'][f'{a}|688'], '| 10 % verschiebt:', o['warm10'][a], '| unten:', o['unten'].get(a))
    p = Path(v1b) / 'analysis'; p.mkdir(exist_ok=True); json.dump(o, open(p / 'v1b_result.json', 'w'), indent=1)

if __name__ == '__main__': main()
