# Nachpruefung B2: zwei Laeufe, die ueber die Schlafpause liefen, neu rechnen und mit der Studien-CSV vergleichen.
import csv, glob, importlib.util, sys, tempfile
from pathlib import Path
H = Path.home() / 'PyCharmMiscProject' / 'studien' / 'b2'
sp = importlib.util.spec_from_file_location('B2', H / 'b2_run.py'); B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B)
B.OUTDIR = Path(tempfile.mkdtemp())
old = {r['run_id']: r for f in sorted(glob.glob(str(H / 'workers' / 'w*.csv'))) for r in csv.DictReader(open(f))}
cf = {B.rid(c): c for c in B.configs()}
keys = ['train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'test_err_w8a3a_best', 'act_range_max_median', 'sat_rate']
for rid in sys.argv[1:]:
    o = B.execute(dict(cf[rid]), 0); a = old[rid]
    eq = all(str(o[k]) == a[k] for k in keys)
    print(rid, 'BITGLEICH' if eq else 'ABWEICHUNG', {k: (o[k], a[k]) for k in keys[:3]}, flush=True)
