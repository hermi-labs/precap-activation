# Mechanik-Test V1d (mit Claude): (1) Laufliste, (2) LR-Plan je Anlauflaenge, (3) kurze Laeufe (40 Schritte) V1d gegen V1c bitgleich
#   fuer cos und fuer den Anlauf-Pfad (bei 40 Schritten greift min(warm, STEPS/2) = 20 in beiden Skripten). Ausgabe in einen Temp-Ordner.
import csv, importlib.util, sys, tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.optimizers as optim
H = Path(__file__).resolve().parent
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); a = sys.argv; sys.argv = ['x']; sp.loader.exec_module(M); sys.argv = a; return M
D = load('v1d', H / 'v1d_run.py'); C = load('v1c', H.parent / 'v1c' / 'v1c_run.py')
ok = True
cf = list(D.configs()); ids = [D.rid(c) for c in cf]
ok &= len(cf) == 26 and len(set(ids)) == 26 and ids[0].endswith('_kontrolle') and ids[1].endswith('_w344') and all(c['lr'] == 5e-5 and c['wd'] == 0.3 for c in cf)
ok &= sorted({c['warm'] for c in cf[2:]}) == [69, 172] and {c['seed'] for c in cf[2:]} == set(range(42192, 42204))
print('Laufliste', len(cf), ids[:4], ok)
for w in (69, 172, 344):   # Plan wie im Skript: 0 am Start, Spitze bei w, danach fallend, ~0 am Ende
    lr = optim.join_schedules([optim.linear_schedule(0.0, 5e-5, w), optim.cosine_decay(5e-5, 6880 - w)], [w])
    v = np.array([float(lr(mx.array(s))) for s in range(0, 6881)]); pk = int(v.argmax())
    good = v[0] == 0 and abs(pk - w) <= 1 and abs(v.max() / 5e-5 - 1) < 1e-6 and np.all(np.diff(v[w:]) <= 1e-15) and v[-1] < 1e-9
    print(f'Plan w{w}: Spitze bei {pk}, Ende {v[-1]:.2e}', good); ok &= bool(good)
for M in (D, C): M.STEPS = 40; M.CHECKS = (10, 20); M.COLS = M.COLS_BASE + [f'c{k}_{x}' for k in M.CHECKS for x in M.CKCOLS] + ['sec_checks']; M.STAGE = 'confirm'
SKIP = {'sec_train', 'ms_per_step', 'timestamp', 'backend', 'sec_checks', 'run_id', 'warm'}
for sched, warm in (('cos', 0), ('wcos', 344)):
    t = Path(tempfile.mkdtemp()); D.OUTDIR = t / 'd'; C.OUTDIR = t / 'c'
    D.execute(dict(dataset='cifar10', depth=200, act='kL2B_tr', lr=5e-5, wd=0.3, sched=sched, warm=warm, seed=42192), 0)
    C.execute(dict(dataset='cifar10', depth=200, act='kL2B_tr', lr=5e-5, wd=0.3, sched=sched, seed=42192), 0)
    a = next(csv.DictReader(open(t / 'd' / 'workers' / 'w0.csv'))); b = next(csv.DictReader(open(t / 'c' / 'workers' / 'w0.csv')))
    diff = [k for k in b if k not in SKIP and a.get(k) != b[k]]
    print(f'{sched} warm {warm}: V1d {a["run_id"]} gegen V1c {b["run_id"]}:', 'bitgleich' if not diff else diff, '| w4a', a['test_err_w8a4a_best'])
    ok &= not diff and a['warm'] == str(warm)
print('MECHANIK OK' if ok else 'MECHANIK FEHLER'); sys.exit(0 if ok else 1)
