import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "experiments_v2.csv")
df["c"] = df["c"].fillna(-1)
rng = np.random.default_rng(0)

def boot(x, n=2000):
    x = np.asarray(x); m = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return np.median(x), *np.percentile(m, [2.5, 97.5])

rows = []
for k, g in df.groupby(["use_layernorm", "depth", "activation", "c", "lr"]):
    med, lo, hi = boot(g.test_loss)
    rows.append(dict(zip(["ln","depth","act","c","lr"], k), n=len(g), med=med, lo=lo, hi=hi,
                     diverged=(g.test_loss > 1).mean(), gradmed=g.max_grad.median(),
                     dur=g.duration_sec.median()))
out = pd.DataFrame(rows).sort_values(["ln","depth","med"])
out.to_csv("summary.csv", index=False)
print(out.to_string(index=False))
