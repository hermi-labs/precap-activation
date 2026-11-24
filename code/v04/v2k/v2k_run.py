# V2k (privat, mit Claude, praeregistriert in V2k_Praeregistrierung.md): Kreuzterme und Zerlegung des 4-Bit-Fehlers in Abschneiden und Runden,
# gemessen an den gespeicherten V2f-Netzen (BPreCap und ctanh, Seeds 42353-42361, Gewichte am fruehen Stopp, kein Training).
# Je Netz und Kalibrierung (mse4, max) an 1000 Testbildern, nur Aktivierungen (Gewichte fp32), Eingaenge i = 0..L wie v2e_run.measure:
#   Fehler am Eingang i: e_Q = Q(h) - h = e_C + e_R mit e_C = clip(h, lo_eff, hi_eff) - h (Abschneiden) und e_R = Q(h) - clip(h) (Runden);
#   lo_eff, hi_eff = Grenzen des tatsaechlichen 4-Bit-Gitters von R.qa_asym. Einzeln je Schicht (Q, C, R) und alle Schichten zugleich (Q, C, R).
#   Kreuzterme: delta_i = Logit-Abweichung bei nur Eingang i gerundet (Vektor je Bild); kappa = (|sum delta_i|^2 - sum |delta_i|^2) / sum |delta_i|^2,
#   nu = |delta_all - sum delta_i| / |delta_all|, Kosinus-Matrix der delta_i. PreCap, BPreCap, Huelle, Kalibrierungsrezept: Ideen Aarons.
# Aufruf: --selfcheck | --smoke | --worker 0 --workers 1
import csv, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; S = ROOT.parent
sys.path.insert(0, str(S / 'v2e'))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten
import v2e_run as E   # eingefroren; importiert v2a2_run, v2d_run, v2c_run unveraendert (make_act kennt bprecap und ctanh)
R = E.R
SEEDS = tuple(range(42353, 42362)); FORMS = ('bprecap', 'ctanh'); CALS = ('mse4', 'max'); NTEST, CHUNK = 1000, 250
GW = S / 'v2f' / 'gewichte'; V2F_MESS = S / 'v2f' / 'mess'
INNER = slice(1, -1)   # innere Faltungseingaenge i = 1..L-1, wie v2e_run.gm
def rid(a, s): return f'confirm_{a}_lr3e-05_wd0_s{s}'
def cfgs(): return [dict(act=a, seed=s, run_id=rid(a, s), cal=c) for s in SEEDS for a in FORMS for c in CALS]
def load_net(a, s):
    m = R.CNN(R.make_act(a), s); w = mx.load(str(GW / f'{rid(a, s)}.npz')); m.update(tree_unflatten(list(w.items()))); mx.eval(m.parameters()); return m
def eff(lohi):   # Grenzen des Gitters wie in R.qa_asym
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 15; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc); return (0 - zp) * sc, (n - zp) * sc
def gm(v):
    v = np.asarray(v, dtype=np.float64)[INNER]; v = v[v > 0]; return float(np.exp(np.mean(np.log(v)))) if len(v) else float('nan')
def measure(m, seed, cal, ntest=NTEST, chunk=CHUNK, keep=True):
    xtr, _, _, _, xte, _ = R.load(); Rc = R.calib(m, xtr[:R.NCAL], seed); lohi = Rc[cal]; nl = len(m.convs) + 1; EF = [eff(lohi[i]) for i in range(nl)]
    Wc = [(c.weight, c.bias) for c in m.convs]; Wo, bo = m.out.weight, m.out.bias
    Qf = lambda t, i: R.qa_asym(t, 4, lohi[i]); Cf = lambda t, i: mx.clip(t, EF[i][0], EF[i][1])
    KIND = {'Q': Qf, 'C': Cf, 'R': lambda t, i: t + (Qf(t, i) - Cf(t, i))}
    def fwd(x, inj):
        h = x
        for i, (W, b) in enumerate(Wc):
            h = m.act(mx.conv2d(inj(i, h), W, padding=1) + b); mx.eval(h)
        h = inj(len(Wc), mx.mean(h, axis=(1, 2))); return h @ Wo.T + bo
    A = {k: np.zeros(nl) for k in ('h2', 'eQ2', 'eC2', 'eR2', 'dQ2', 'dC2', 'dR2')}; Dall = {k: 0.0 for k in 'QCR'}; z2 = 0.0; nu2 = sd2 = 0.0
    qc_abw = 0.0; frac_c = np.zeros(nl); cnt = np.zeros(nl); DEL = np.zeros((nl, ntest, 10), dtype=np.float32)
    for j in range(0, ntest, chunk):
        x = mx.array(xte[j:j + chunk]); h = x; z = fwd(x, lambda i, t: t); zn = np.array(z).astype(np.float64); z2 += float((zn ** 2).sum())
        for i in range(nl):   # sauberer Pfad: Fehleranteile je Eingang
            hi_ = h if i < nl - 1 else mx.mean(h, axis=(1, 2))
            q, c = Qf(hi_, i), Cf(hi_, i); eC, eR = c - hi_, q - c
            A['h2'][i] += float(mx.sum(hi_ * hi_)); A['eQ2'][i] += float(mx.sum((q - hi_) ** 2)); A['eC2'][i] += float(mx.sum(eC * eC)); A['eR2'][i] += float(mx.sum(eR * eR))
            qc_abw = max(qc_abw, float(mx.max(mx.abs(Qf(c, i) - q)))); frac_c[i] += float(mx.sum(eC != 0)); cnt[i] += hi_.size
            if i < nl - 1: h = m.act(mx.conv2d(hi_, Wc[i][0], padding=1) + Wc[i][1]); mx.eval(h)
        for k, f in KIND.items():   # alle Eingaenge zugleich
            dz = np.array(fwd(x, lambda ii, t, f=f: f(t, ii))).astype(np.float64) - zn; Dall[k] += float((dz ** 2).sum())
            if k == 'Q': dall = dz
        sdel = np.zeros_like(zn)
        for i in range(nl):   # nur Eingang i
            for k, f in KIND.items():
                dz = np.array(fwd(x, lambda ii, t, i=i, f=f: f(t, ii) if ii == i else t)).astype(np.float64) - zn; A['d' + k + '2'][i] += float((dz ** 2).sum())
                if k == 'Q': DEL[i, j:j + chunk] = dz; sdel += dz
        nu2 += float(((dall - sdel) ** 2).sum()); sd2 += float((sdel ** 2).sum()); mx.clear_cache()
    zr = math.sqrt(z2); out = dict(cal=cal, ntest=ntest, lo_eff=[e[0] for e in EF], hi_eff=[e[1] for e in EF], frac_clip=(frac_c / cnt).tolist(), qc_abw=qc_abw)
    for k in 'QCR':
        out['r' + k] = np.sqrt(A['e' + k + '2'] / A['h2']).tolist(); out['d' + k] = (np.sqrt(A['d' + k + '2']) / zr).tolist(); out['D_all' + k] = math.sqrt(Dall[k]) / zr
    D = DEL.reshape(nl, -1).astype(np.float64); G = D @ D.T; sum_d2 = float(np.trace(G))
    out['kappa'] = (float(G.sum()) - sum_d2) / sum_d2; out['nu'] = math.sqrt(nu2) / (out['D_allQ'] * zr); out['sum_delta_rel'] = math.sqrt(sd2) / zr
    nr = np.sqrt(np.diag(G)); C = G / np.maximum(np.outer(nr, nr), 1e-300); out['cos'] = C.tolist()
    if keep: out['_DEL'] = DEL
    return out
def summary(M):
    nl = len(M['dQ']); C = np.array(M['cos']); iu = np.triu_indices(nl, 1); inner = [(a, b) for a, b in zip(*iu) if 1 <= a and b <= nl - 2]
    g = {k: np.array(M['d' + k]) / np.maximum(np.array(M['r' + k]), 1e-300) for k in 'QCR'}
    for k in 'CR': g[k][np.array(M['r' + k]) == 0] = 0.0
    lin = math.sqrt(sum(d * d for d in M['dQ']))
    return dict(D_allQ=M['D_allQ'], D_allC=M['D_allC'], D_allR=M['D_allR'], rbarQ=gm(M['rQ']), rbarC=gm(M['rC']), rbarR=gm(M['rR']),
                gbarQ=gm(g['Q']), gbarC=gm(g['C']), gbarR=gm(g['R']), n_ohne_clip=int(sum(1 for v in M['rC'][INNER] if v == 0)),
                frac_clip_median=float(np.median(M['frac_clip'][INNER])), kappa=M['kappa'], nu=M['nu'], superpos_Q=M['D_allQ'] / lin,
                cos_nachbar_median=float(np.median([C[i, i + 1] for i in range(1, nl - 2)])), cos_mittel_innen=float(np.mean([C[a, b] for a, b in inner])), qc_abw=M['qc_abw'])
COLS = ['run_id', 'act', 'seed', 'cal', 'status', 'D_allQ', 'D_allC', 'D_allR', 'rbarQ', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'n_ohne_clip', 'frac_clip_median',
        'kappa', 'nu', 'superpos_Q', 'cos_nachbar_median', 'cos_mittel_innen', 'qc_abw', 'v2f_abw_dQ', 'v2f_abw_D', 'sec', 'timestamp']
def write(p, o):
    new = not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)
def v2f_abw(M, r, cal):   # Abweichung von der V2f-Messung desselben Netzes (gleiche Formel, gleiche Bilder)
    F = json.load(open(V2F_MESS / f'{r}_{cal}.json')); a = np.array(M['dQ']); b = np.array(F['dQ'])
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))), abs(M['D_allQ'] - F['D_allQ']) / F['D_allQ']
def work(outdir, cl, ntest=NTEST, chunk=CHUNK, check=True):
    p = outdir / 'workers' / 'w0.csv'; done = {(r['run_id'], r['cal']) for r in csv.DictReader(open(p))} if p.exists() else set(); net = None
    for c in cl:
        if (c['run_id'], c['cal']) in done: continue
        t0 = time.time(); row = dict(run_id=c['run_id'], act=c['act'], seed=c['seed'], cal=c['cal'])
        try:
            if net is None or net[0] != c['run_id']: net = (c['run_id'], load_net(c['act'], c['seed']))
            M = measure(net[1], c['seed'], c['cal'], ntest, chunk); DEL = M.pop('_DEL')
            (outdir / 'mess').mkdir(parents=True, exist_ok=True); json.dump(M, open(outdir / 'mess' / f"{c['run_id']}_{c['cal']}.json", 'w'))
            np.savez_compressed(outdir / 'mess' / f"{c['run_id']}_{c['cal']}_delta.npz", delta=DEL)
            row.update(summary(M)); fin = all(np.isfinite(float(row[k])) for k in ('D_allQ', 'D_allC', 'D_allR', 'kappa', 'nu', 'gbarQ', 'gbarR'))
            if check: row['v2f_abw_dQ'], row['v2f_abw_D'] = v2f_abw(M, c['run_id'], c['cal'])
            row['status'] = 'ok' if fin else 'nicht_endlich'
        except Exception as ex:
            row['status'] = 'fail'; print('FEHLER', c['run_id'], c['cal'], repr(ex), flush=True)
        row['sec'] = round(time.time() - t0, 1); row['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S'); write(p, row)
        print(time.strftime('%H:%M'), c['run_id'], c['cal'], row['status'], 'DQ %.4f DC %.4f DR %.4f kappa %.3f nu %.3f abw %s' % tuple(
            [float(row.get(k, float('nan'))) for k in ('D_allQ', 'D_allC', 'D_allR', 'kappa', 'nu')] + [row.get('v2f_abw_dQ')]), row['sec'], flush=True)
    print('Worker fertig', time.strftime('%H:%M'), flush=True)
def selfcheck():
    out = dict(n_mess=len(cfgs()), seeds=list(SEEDS), forms=list(FORMS), cals=list(CALS), ntest=NTEST)
    out['gewichte_da'] = bool(all((GW / f'{rid(a, s)}.npz').exists() for a in FORMS for s in SEEDS))
    out['v2f_mess_da'] = bool(all((V2F_MESS / f'{rid(a, s)}_{c}.json').exists() for a in FORMS for s in SEEDS for c in CALS))
    # Zerlegung exakt: Q(clip(h)) = Q(h), e_C + e_R = e_Q; Gittergrenzen liegen auf dem Gitter
    t = mx.array(np.random.default_rng(5).normal(0, 2, 20000).astype(np.float32)); lh = (-1.3, 2.7); lo, hi = eff(lh)
    q = R.qa_asym(t, 4, lh); c = mx.clip(t, lo, hi); out['q_clip_gleich'] = float(mx.max(mx.abs(R.qa_asym(c, 4, lh) - q)))
    out['zerlegung_summe'] = float(mx.max(mx.abs(((c - t) + (q - c)) - (q - t)))); out['grenzen_auf_gitter'] = bool(float(mx.max(q)) <= hi + 1e-6 and float(mx.min(q)) >= lo - 1e-6)
    out['zerlegung_ok'] = bool(out['q_clip_gleich'] < 1e-5 and out['zerlegung_summe'] < 1e-5 and out['grenzen_auf_gitter'])
    # an einem echten V2f-Netz, 50 Bilder: gleiche dQ und D_allQ wie v2e_run.measure; kappa aus den delta-Vektoren gegen direkte Summe; endlich
    for a in FORMS:
        m = load_net(a, SEEDS[0]); M = measure(m, SEEDS[0], 'mse4', ntest=50, chunk=50); Mv = E.measure(m, SEEDS[0], 'mse4', ntest=50, chunk=50)
        out[f'{a}_dQ_wie_v2e'] = float(np.max(np.abs(np.array(M['dQ']) - np.array(Mv['dQ'])) / np.maximum(np.abs(Mv['dQ']), 1e-12)))
        out[f'{a}_DallQ_wie_v2e'] = abs(M['D_allQ'] - Mv['D_allQ']) / Mv['D_allQ']
        D = M['_DEL'].reshape(len(M['dQ']), -1).astype(np.float64); s = D.sum(0); sd2 = float((D ** 2).sum())
        out[f'{a}_kappa_direkt_ok'] = bool(abs(((s ** 2).sum() - sd2) / sd2 - M['kappa']) < 1e-6)
        Sm = summary({k: v for k, v in M.items() if k != '_DEL'}); out[f'{a}_endlich'] = bool(all(np.isfinite(float(v)) for v in Sm.values()))
        out[f'{a}_dQ_ok'] = bool(out[f'{a}_dQ_wie_v2e'] < 1e-4 and out[f'{a}_DallQ_wie_v2e'] < 1e-4)
        out[f'{a}_summary'] = {k: round(float(v), 5) for k, v in Sm.items()}
        # Abschneiden + Runden ungefaehr Gesamtfehler je Schicht (Energie): rC^2 + rR^2 nahe rQ^2 (Kreuzterm e_C*e_R ist klein, nicht null)
        out[f'{a}_energie_quotient_median'] = round(float(np.median((np.array(M['rC']) ** 2 + np.array(M['rR']) ** 2) / np.array(M['rQ']) ** 2)), 4)
    out['alle_ok'] = bool(out['gewichte_da'] and out['v2f_mess_da'] and out['zerlegung_ok'] and all(out[f'{a}_dQ_ok'] and out[f'{a}_kappa_direkt_ok'] and out[f'{a}_endlich'] for a in FORMS))
    print(json.dumps(out, indent=1)); return out
def main():
    a = sys.argv[1:]; print('Geraet:', mx.default_device(), flush=True)
    if hasattr(mx, 'set_cache_limit'): mx.set_cache_limit(2 * 1024 ** 3)
    if '--selfcheck' in a: selfcheck(); return
    if '--smoke' in a: work(ROOT / 'smoke', [c for c in cfgs() if c['seed'] == SEEDS[0]], ntest=50, chunk=50, check=False); return
    work(ROOT, cfgs())
if __name__ == '__main__': main()
