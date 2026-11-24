# V2f (privat, mit Claude, praeregistriert in V2f_Praeregistrierung.md): Bestaetigung der V2e-Nachmessung an 9 neuen Seeds (42353-42361).
# BPreCap (V2a2) und ctanh (V2d) mit v2e_run.run_keep bitgleich nachgerechnet (Kontrolle gegen die historische CSV-Zeile), Gewichte am fruehen
# Stopp gespeichert, dann mit dem eingefrorenen v2e_run.measure bei fester Kalibrierung mse4 und max gemessen. PreCap, BPreCap, Huelle, Rezept: Ideen Aarons.
# Aufruf: --selfcheck | --smoke | --worker 0 --workers 1
import csv, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; S = ROOT.parent
sys.path.insert(0, str(S / 'v2e'))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
import v2e_run as E   # eingefroren; importiert v2a2_run, v2d_run, v2c_run unveraendert
R = E.R
SEEDS = tuple(range(42353, 42362)); FORMS = ('bprecap', 'ctanh'); CALS = ('mse4', 'max')
HIST = {'bprecap': 'v2a2', 'ctanh': 'v2d'}
def hist_rows():
    out = {}
    for a, d in HIST.items():
        for r in csv.DictReader(open(S / d / 'workers' / 'w0.csv')):
            if r['act'] == a and r['stage'] == 'confirm' and int(r['seed']) in SEEDS: out[(a, int(r['seed']))] = r
    return out
def cfgs():   # je Seed erst BPreCap, dann ctanh
    H = hist_rows(); out = []
    for s in SEEDS:
        for a in FORMS:
            h = H[(a, s)]; out.append(dict(stage='confirm', act=a, lr=float(h['lr']), wd=float(h['wd']), seed=s, _hist=h['run_id']))
    return out
COLS = ['run_id', 'act', 'seed', 'cal', 'hist_run_id', 'status', 'kontrolle_n_diff', 'kontrolle_diff', 'es_test_err_fp32', 'es_test_err_w8a4a_best',
        'es_w8a4a_best_cal', 'rbar', 'gbarQ', 'gbarG', 'D_allQ', 'D_allG', 'superpos_Q', 'superpos_G', 'sqrt_sum_dQ2', 'flipQ', 'err_fp32_1000', 'err_a4_1000',
        'err_w8a4_1000', 'chi_emp_gm', 'eps_end_Q', 'margin_median', 'bias_anteil_median', 'sec_train', 'sec_mess', 'timestamp']
def write(p, o):
    new = not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)
def work(outdir, cl, ntest=E.NTEST, check=True):
    wd_ = outdir / 'workers'; H = {h['run_id']: h for h in hist_rows().values()}
    have = {}
    if (wd_ / 'w0.csv').exists():
        for r in csv.DictReader(open(wd_ / 'w0.csv')): have.setdefault(r['run_id'], set()).add(r['cal'])
    for c in cl:
        rid = R.rid(c)
        if have.get(rid, set()) >= set(CALS): continue
        t0 = time.time(); o, m = E.run_keep(c); t1 = time.time()
        base = dict(run_id=rid, act=c['act'], seed=c['seed'], hist_run_id=c['_hist'], status=o['status'], sec_train=round(t1 - t0, 1))
        if check:
            diff = E.compare(o, H[c['_hist']]); base['kontrolle_n_diff'] = len(diff); base['kontrolle_diff'] = json.dumps(diff[:3])
        for k in ('es_test_err_fp32', 'es_test_err_w8a4a_best', 'es_w8a4a_best_cal'): base[k] = o.get(k, '')
        if m is not None:
            (outdir / 'gewichte').mkdir(parents=True, exist_ok=True); mx.savez(str(outdir / 'gewichte' / f'{rid}.npz'), **dict(tree_flatten(m.parameters())))
        for cal in CALS:
            if cal in have.get(rid, set()): continue
            row = dict(base, cal=cal, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')); t2 = time.time()
            if m is not None:
                M = E.measure(m, c['seed'], cal, ntest); (outdir / 'mess').mkdir(parents=True, exist_ok=True)
                json.dump(M, open(outdir / 'mess' / f'{rid}_{cal}.json', 'w')); row.update(E.summary(M))
                row['sqrt_sum_dQ2'] = float(np.sqrt(np.sum(np.square(M['dQ'])))); row['sec_mess'] = round(time.time() - t2, 1)
            write(wd_ / 'w0.csv', row)
            print(time.strftime('%H:%M'), rid, cal, row['status'], 'diff', row.get('kontrolle_n_diff'),
                  'rbar %.4f gQ %.3f sup %.3f D %.4f a4 %.3f' % tuple(row.get(k, -1) for k in ('rbar', 'gbarQ', 'superpos_Q', 'D_allQ', 'err_a4_1000')),
                  row['sec_train'], row.get('sec_mess'), flush=True)
        mx.clear_cache()
    print('Worker fertig', time.strftime('%H:%M'), flush=True)
def selfcheck():
    out = dict(n_runs=len(cfgs()), soll_zeilen=len(cfgs()) * len(CALS), forms=list(FORMS), seeds=list(SEEDS), cals=list(CALS), ntest=E.NTEST)
    H = hist_rows(); out['hist_vollstaendig'] = len(H) == 18 and all(r['status'] == 'ok' for r in H.values())
    out['seeds_neu_gegen_v2e'] = not (set(SEEDS) & set(E.SEEDS))
    out['run_ids_wie_hist'] = all(R.rid(c) == c['_hist'] for c in cfgs())
    out['act_ctanh_wie_v2d'] = R.make_act('ctanh') is E.V2D.ctanh
    X, Wq = np.polynomial.hermite_e.hermegauss(150); Wq = Wq / Wq.sum(); x = mx.array(X.astype(np.float32))
    for a in FORMS:
        y = np.array(R.make_act(a)(x), dtype=np.float64); out[f'Ef2_{a}'] = round(float(np.sum(Wq * y * y)), 3)
    # Messmechanik an einem untrainierten Netz, 50 Bilder, beide Kalibrierungen
    m = R.CNN(R.make_act('bprecap'), 7); mx.eval(m.parameters())
    for cal in CALS:
        M = E.measure(m, 7, cal, ntest=50, chunk=50); Sm = E.summary(M)
        out[f'mess_{cal}_endlich'] = bool(all(np.isfinite(M['dQ'] + M['dG'] + M['r'])) and np.isfinite(Sm['superpos_Q']))
        out[f'mess_{cal}_r_positiv'] = all(v > 0 for v in M['r'])
    # Reproduzierbarkeit: dieselbe Messung zweimal -> gleich
    a1 = E.measure(m, 7, 'mse4', ntest=50, chunk=50); a2 = E.measure(m, 7, 'mse4', ntest=50, chunk=50)
    out['mess_reproduzierbar'] = a1['dQ'] == a2['dQ'] and a1['D_allQ'] == a2['D_allQ']
    print(json.dumps(out, indent=1)); return out
def main():
    a = sys.argv[1:]; print('Geraet:', mx.default_device(), flush=True)
    if hasattr(mx, 'set_cache_limit'): mx.set_cache_limit(2 * 1024 ** 3)
    if '--selfcheck' in a: selfcheck(); return
    if '--smoke' in a:
        cl = [dict(c, steps=20) for c in cfgs() if c['seed'] == SEEDS[0]]
        work(ROOT / 'smoke', cl, ntest=250, check=False); return
    work(ROOT, cfgs())
if __name__ == '__main__': main()
