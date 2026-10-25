import csv, json, math, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path('/Users/aaronschmitt/PyCharmMiscProject/v10')
sys.path.insert(0, str(ROOT))
import capexp_v10 as base

OUT = Path(__file__).resolve().parent
META = OUT / 'meta'
LOGS = OUT / 'logs'
RESULTS = OUT / 'results'
META.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

FIELDS = ['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed','timestamp','test_loss','bad','fail_step','duration_sec']
KEYS = ['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed']
PARAM_KEYS = ['c','a','gamma','soft_s','init_k','lr']
TARGETS = ['sin3_quad','sin_mix','abs_wave']
DEPTHS = [32,64]
ACTS = ['precap','silu','gelu']
SEEDS = list(range(10,20))

SRC=[]
for f in sorted(ROOT.glob('v10_results_shard*.csv')):
    SRC.extend(csv.DictReader(open(f, newline='')))

def median(vals):
    vals=sorted(vals)
    n=len(vals)
    return vals[n//2] if n%2 else 0.5*(vals[n//2-1]+vals[n//2])

def best_cfg(act, target, depth):
    rows=[r for r in SRC if r['act']==act and r['target']==target and int(r['depth'])==depth and r['arch']=='res_inv']
    groups={}
    for r in rows:
        key=tuple(r[k] for k in PARAM_KEYS)
        groups.setdefault(key, []).append(float(r['test_loss']))
    if not groups:
        raise RuntimeError(f'No source rows for {act} {target} depth={depth}')
    best=min(groups, key=lambda k: median(groups[k]))
    c,a,g,s,init_k,lr = map(float, best)
    return dict(target=target, act=act, c=c, a=a, gamma=g, soft_s=s, depth=depth, arch='res_inv', init_k=init_k, lr=lr)

def frozen_configs():
    cfgs=[]
    for target in TARGETS:
        for depth in DEPTHS:
            for act in ACTS:
                cfgs.append(best_cfg(act, target, depth))
    return cfgs

def manifest_rows():
    rows=[]
    for cfg in frozen_configs():
        for seed in SEEDS:
            z=dict(cfg); z['seed']=seed; rows.append(z)
    return rows

def write_meta():
    cfgs=frozen_configs()
    rows=manifest_rows()
    with open(META/'frozen_configs.json','w') as f:
        json.dump(cfgs, f, indent=2)
    with open(META/'manifest.csv','w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader(); w.writerows(rows)
    summary={
        'targets': TARGETS,
        'depths': DEPTHS,
        'acts': ACTS,
        'holdout_seeds': SEEDS,
        'frozen_config_count': len(cfgs),
        'planned_runs': len(rows),
        'design': 'best-vs-best holdout with frozen v10 winners, 3 targets x 2 depths x 3 activations x 10 fresh seeds = 180 runs',
        'workspace_root': '/Users/aaronschmitt/PyCharmMiscProject',
        'source_root': '/Users/aaronschmitt/PyCharmMiscProject/v10',
        'output_root': str(OUT)
    }
    with open(META/'plan.json','w') as f:
        json.dump(summary, f, indent=2)

def load_done(outcsv):
    done=set()
    if outcsv.exists() and outcsv.stat().st_size>0:
        for r in csv.DictReader(open(outcsv, newline='')):
            done.add(tuple(str(r[k]) for k in KEYS))
    return done

def build_record(cfg, final, bad, fail, duration_sec):
    rec={k: cfg[k] for k in ['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed']}
    rec.update({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'test_loss': final if math.isfinite(final) else 'inf',
        'bad': bool(bad),
        'fail_step': int(fail),
        'duration_sec': float(duration_sec),
    })
    return rec

def run_shard(shard, nshards, smoke=False):
    rows=manifest_rows()[shard::nshards]
    if smoke:
        rows=rows[:1]
    outcsv=RESULTS/f'holdout_results_shard{shard}.csv'
    done=load_done(outcsv)
    newfile=not outcsv.exists()
    with open(outcsv, 'a', newline='') as fh:
        w=csv.DictWriter(fh, fieldnames=FIELDS)
        if newfile:
            w.writeheader(); fh.flush()
        total=len(rows)
        for i,cfg in enumerate(rows,1):
            key=tuple(str(cfg[k]) for k in KEYS)
            if key in done:
                continue
            t0=time.time()
            result = base.run(cfg)
            if not isinstance(result, (tuple, list)) or len(result) != 4:
                raise RuntimeError(f"Unexpected result payload: {type(result)} len={len(result) if hasattr(result, '__len__') else 'na'}")
            curve, final, bad, fail = result
            rec=build_record(cfg, final, bad, fail, time.time()-t0)
            w.writerow(rec); fh.flush()
            print(f'[{i}/{total}] {cfg["target"]} {cfg["act"]} d={cfg["depth"]} seed={cfg["seed"]} test={rec["test_loss"]} bad={rec["bad"]} sec={round(rec["duration_sec"],1)} checkpoints={len(curve)}', flush=True)

def validate():
    expected=manifest_rows()
    expected_keys={tuple(str(r[k]) for k in KEYS) for r in expected}
    seen=[]
    for f in sorted(RESULTS.glob('holdout_results_shard*.csv')):
        if f.stat().st_size == 0:
            continue
        seen.extend(list(csv.DictReader(open(f, newline=''))))
    seen_keys=[tuple(str(r[k]) for k in KEYS) for r in seen]
    report={
        'rows': len(seen),
        'expected': len(expected_keys),
        'missing': len(expected_keys - set(seen_keys)),
        'dups': len(seen_keys) - len(set(seen_keys)),
        'bad': sum(1 for r in seen if str(r['bad']).lower() == 'true')
    }
    with open(META/'validation.json','w') as f:
        json.dump(report, f, indent=2)
    if report['rows']==report['expected'] and report['missing']==0 and report['dups']==0:
        (OUT/'HOLDOUT_DONE').write_text(json.dumps(report, indent=2))
    return report

def selfcheck():
    write_meta()
    cfgs=frozen_configs()
    rows=manifest_rows()
    assert len(cfgs)==18, len(cfgs)
    assert len(rows)==180, len(rows)
    assert len({tuple((r[k] for k in KEYS)) for r in rows})==180
    counts={}
    for r in rows:
        key=(r['target'], r['depth'], r['act'])
        counts[key]=counts.get(key,0)+1
    assert all(v==10 for v in counts.values())
    return {'frozen_configs': len(cfgs), 'runs': len(rows), 'cells': len(counts), 'seeds_per_cell': sorted(set(counts.values()))}

if __name__ == '__main__':
    if len(sys.argv)>=2 and sys.argv[1]=='selfcheck':
        print(json.dumps(selfcheck(), indent=2)); sys.exit(0)
    if len(sys.argv)>=2 and sys.argv[1]=='validate':
        print(json.dumps(validate(), indent=2)); sys.exit(0)
    write_meta()
    shard=int(sys.argv[1]); nshards=int(sys.argv[2]); smoke=('--smoke' in sys.argv)
    run_shard(shard, nshards, smoke=smoke)
