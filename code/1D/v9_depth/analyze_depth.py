import glob, os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{HERE}/v9_depth_results_shard[0-9].csv"))], ignore_index=True)
c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{HERE}/v9_depth_results_shard[0-9]_curves.csv"))], ignore_index=True)


def label(r):
    if r.act == "precap":
        return "PreCap A (c=0.5)" if r.c == 0.5 else "PreCap B (c=1.0)"
    return {"silu": "SiLU", "gelu": "GELU", "capexp": "CapExp (c=1.0)"}[r.act]


for df in (d, c):
    df["cand"] = df.apply(label, axis=1)
    df["bad"] = df["bad"].astype(str).str.lower().eq("true")

CANDS = ["SiLU", "GELU", "CapExp (c=1.0)", "PreCap A (c=0.5)", "PreCap B (c=1.0)"]
COL = dict(zip(CANDS, ["#1f77b4", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd"]))
TARGETS = sorted(d.target.unique())
DEPTHS = sorted(d.depth.unique())
d["loss"] = np.where(d.bad | ~np.isfinite(d.test_loss), np.inf, d.test_loss)


def boot_med(x, n=2000):
    x = np.asarray(x)
    m = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return np.percentile(m, [2.5, 97.5])


rows = []
for (t, dp, cd), g in d.groupby(["target", "depth", "cand"]):
    x = g.loss.values
    lo, hi = boot_med(x)
    fin = x[np.isfinite(x)]
    rows.append(dict(target=t, depth=dp, cand=cd, n=len(x), fails=int((~np.isfinite(x)).sum()),
                     fail_rate=float((~np.isfinite(x)).mean()), median=np.median(x),
                     q25=np.percentile(x, 25), q75=np.percentile(x, 75), ci_lo=lo, ci_hi=hi,
                     mean_finite=fin.mean() if len(fin) else np.nan))
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/summary.csv", index=False)

# rank of candidates within target/depth
S["rank"] = S.groupby(["target", "depth"])["median"].rank(method="min")

# depth change
dc = S.pivot_table(index=["target", "cand"], columns="depth", values="median").reset_index()
if len(DEPTHS) == 2:
    a, b = DEPTHS
    dc["factor"] = dc[b] / dc[a]
dc.to_csv(f"{OUT}/depth_change.csv", index=False)

# pairwise tests
pw = []
for (t, dp), g in d.groupby(["target", "depth"]):
    for a_, b_ in itertools.combinations(CANDS, 2):
        xa = g[g.cand == a_].loss.values
        xb = g[g.cand == b_].loss.values
        if len(xa) < 3 or len(xb) < 3:
            continue
        xa2 = np.where(np.isfinite(xa), xa, 1e9)
        xb2 = np.where(np.isfinite(xb), xb, 1e9)
        p = mannwhitneyu(xa2, xb2, alternative="two-sided").pvalue
        pw.append(dict(target=t, depth=dp, a=a_, b=b_, med_a=np.median(xa), med_b=np.median(xb), p=p,
                       better=a_ if np.median(xa) < np.median(xb) else b_))
P = pd.DataFrame(pw)
m = len(P)
order = P.p.values.argsort()
adj = np.empty(m)
run = 0.0
for i, idx in enumerate(order):
    run = max(run, min(1.0, (m - i) * P.p.values[idx]))
    adj[idx] = run
P["p_holm"] = adj
P["signif"] = P.p_holm < 0.05
P.to_csv(f"{OUT}/pairwise_tests.csv", index=False)

# failure analysis
F = d[d.bad].copy()
F.to_csv(f"{OUT}/failures.csv", index=False)

# ---------- figures ----------
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})

# 1 median loss bars with CI
fig, axes = plt.subplots(len(DEPTHS), len(TARGETS), figsize=(5 * len(TARGETS), 3.8 * len(DEPTHS)), squeeze=False)
for i, dp in enumerate(DEPTHS):
    for j, t in enumerate(TARGETS):
        ax = axes[i][j]
        for k, cd in enumerate(CANDS):
            r = S[(S.target == t) & (S.depth == dp) & (S.cand == cd)].iloc[0]
            if np.isfinite(r["median"]):
                ax.bar(k, r["median"], color=COL[cd])
                if np.isfinite(r.ci_hi):
                    ax.errorbar(k, r["median"], yerr=[[r["median"] - r.ci_lo], [r.ci_hi - r["median"]]], color="k", capsize=3)
            else:
                ax.text(k, 1e-5 * 1.5, f"FAIL\n{r.fails}/{r.n}", ha="center", color=COL[cd], fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(CANDS)))
        ax.set_xticklabels([x.replace(" (", "\n(") for x in CANDS], fontsize=6)
        ax.set_title(f"{t}, depth {dp}", fontsize=10)
        ax.set_ylabel("median test loss")
fig.suptitle("Median test loss (95% bootstrap CI), failures count as infinite", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/01_median_loss.png")
plt.close(fig)

# 2 failure rate
fig, axes = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 3.6), squeeze=False)
w = 0.38
for j, t in enumerate(TARGETS):
    ax = axes[0][j]
    for q, dp in enumerate(DEPTHS):
        vals = [S[(S.target == t) & (S.depth == dp) & (S.cand == cd)].fail_rate.iloc[0] * 100 for cd in CANDS]
        ax.bar(np.arange(len(CANDS)) + (q - 0.5) * w, vals, w, label=f"depth {dp}", alpha=0.9 if q else 0.5)
    ax.set_xticks(range(len(CANDS)))
    ax.set_xticklabels([x.replace(" (", "\n(") for x in CANDS], fontsize=6)
    ax.set_ylim(0, 105)
    ax.set_title(t)
    ax.set_ylabel("failure rate [%]")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/02_failure_rate.png")
plt.close(fig)

# 3 depth change
fig, axes = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 3.6), squeeze=False)
for j, t in enumerate(TARGETS):
    ax = axes[0][j]
    for cd in CANDS:
        y = [S[(S.target == t) & (S.depth == dp) & (S.cand == cd)]["median"].iloc[0] for dp in DEPTHS]
        yy = [v if np.isfinite(v) else np.nan for v in y]
        ax.plot(DEPTHS, yy, "o-", color=COL[cd], label=cd)
        for dp, v in zip(DEPTHS, y):
            if not np.isfinite(v):
                ax.plot(dp, ax.get_ylim()[1] if False else 1e-1, "x", color=COL[cd], ms=9)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels(DEPTHS)
    ax.set_title(f"{t} (x = all runs failed)")
    ax.set_xlabel("depth")
    ax.set_ylabel("median test loss")
    ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/03_depth_scaling.png")
plt.close(fig)

# 4 failure step histogram
if len(F):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for cd, g in F.groupby("cand"):
        ax.hist(np.log10(g.fail_step.clip(lower=1)), bins=np.linspace(0, 4, 21), alpha=0.7, label=f"{cd} (n={len(g)})", color=COL[cd])
    ax.set_xlabel("log10(failure step)")
    ax.set_ylabel("runs")
    ax.set_title("When do runs fail?")
    ax.legend(fontsize=7)
    ax = axes[1]
    cnt = F.groupby(["depth", "target"]).size().unstack(fill_value=0)
    cnt.plot.bar(ax=ax)
    ax.set_title("Failures by depth and target")
    ax.set_ylabel("runs")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_failure_steps.png")
    plt.close(fig)

# 5 learning curves (finite medians per step + IQR)
def band(g, col):
    fin = g[np.isfinite(g[col])]
    q = fin.groupby("step")[col].quantile([0.25, 0.5, 0.75]).unstack()
    return q


fig, axes = plt.subplots(len(DEPTHS), len(TARGETS), figsize=(5 * len(TARGETS), 3.6 * len(DEPTHS)), squeeze=False)
for i, dp in enumerate(DEPTHS):
    for j, t in enumerate(TARGETS):
        ax = axes[i][j]
        for cd in CANDS:
            g = c[(c.target == t) & (c.depth == dp) & (c.cand == cd) & ~c.bad]
            if len(g) == 0:
                continue
            q = band(g, "test_loss")
            ax.plot(q.index, q[0.5], color=COL[cd], label=cd)
            ax.fill_between(q.index, q[0.25], q[0.75], color=COL[cd], alpha=0.15)
        ax.set_yscale("log")
        ax.set_xscale("symlog")
        ax.set_title(f"{t}, depth {dp}", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("test loss")
        if i == 0 and j == 0:
            ax.legend(fontsize=6)
fig.suptitle("Learning curves (median, IQR, surviving runs only)", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/05_learning_curves.png")
plt.close(fig)

# 6 diagnostics
diag = [("grad_norm", "gradient norm"), ("act_std_mean", "mean activation std"), ("act_abs_max", "max |activation|"), ("cap_frac", "cap fraction")]
fig, axes = plt.subplots(len(diag), len(DEPTHS), figsize=(5 * len(DEPTHS), 3 * len(diag)), squeeze=False)
for r_, (col, nm) in enumerate(diag):
    for j, dp in enumerate(DEPTHS):
        ax = axes[r_][j]
        for cd in CANDS:
            g = c[(c.depth == dp) & (c.cand == cd) & ~c.bad]
            if len(g) == 0 or col not in g:
                continue
            q = band(g, col)
            if len(q) == 0:
                continue
            ax.plot(q.index, q[0.5], color=COL[cd], label=cd)
            ax.fill_between(q.index, q[0.25], q[0.75], color=COL[cd], alpha=0.12)
        if col != "cap_frac":
            ax.set_yscale("log")
        ax.set_xscale("symlog")
        ax.set_title(f"{nm}, depth {dp}", fontsize=9)
        ax.set_xlabel("step")
        if r_ == 0 and j == 0:
            ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/06_diagnostics.png")
plt.close(fig)

# failed runs early diagnostics (CapExp at step 1)
cap = c[c.cand == "CapExp (c=1.0)"].sort_values("step")
first = cap.groupby(["target", "depth", "seed"]).first().reset_index()
first.to_csv(f"{OUT}/capexp_first_step.csv", index=False)

# ---------- report ----------
def fmt(v):
    return "FAIL" if not np.isfinite(v) else f"{v:.2e}"


L = ["# v9_depth Auswertung", "", f"Runs: {len(d)} (Ausfaelle: {int(d.bad.sum())}); Ausfaelle zaehlen als unendlicher Loss.", "",
     "## Median Test-Loss, Ausfaelle, Rang", ""]
for t in TARGETS:
    for dp in DEPTHS:
        L += [f"### {t}, Tiefe {dp}", "", "| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |", "|---|---|---|---|---|---|"]
        for cd in CANDS:
            r = S[(S.target == t) & (S.depth == dp) & (S.cand == cd)].iloc[0]
            L.append(f"| {cd} | {fmt(r['median'])} | {fmt(r.q25)} - {fmt(r.q75)} | {fmt(r.ci_lo)} - {fmt(r.ci_hi)} | {r.fails}/{r.n} | {int(r['rank'])} |")
        L.append("")
L += ["## Tiefenwirkung (Median 128 / Median 64)", "", "| Ziel | Kandidat | 64 | 128 | Faktor |", "|---|---|---|---|---|"]
for _, r in dc.iterrows():
    L.append(f"| {r.target} | {r.cand} | {fmt(r[DEPTHS[0]])} | {fmt(r[DEPTHS[1]])} | {fmt(r.factor) if np.isfinite(r.factor) else 'n/a'} |")
L += ["", "## Signifikante Paarvergleiche (Mann-Whitney, Holm-korrigiert, p<0.05)", "", "| Ziel | Tiefe | Besser | Schlechter | Median besser | Median schlechter | p_holm |", "|---|---|---|---|---|---|---|"]
for _, r in P[P.signif].sort_values(["target", "depth", "p_holm"]).iterrows():
    worse = r.b if r.better == r.a else r.a
    mb = r.med_a if r.better == r.a else r.med_b
    mw = r.med_b if r.better == r.a else r.med_a
    L.append(f"| {r.target} | {r.depth} | {r.better} | {worse} | {fmt(mb)} | {fmt(mw)} | {r.p_holm:.1e} |")
L += ["", f"Nicht signifikant: {int((~P.signif).sum())} von {len(P)} Vergleichen.", "", "### Nicht signifikante Paare", "",
      "| Ziel | Tiefe | A | B | p_holm |", "|---|---|---|---|---|"]
for _, r in P[~P.signif].sort_values(["target", "depth"]).iterrows():
    L.append(f"| {r.target} | {r.depth} | {r.a} | {r.b} | {r.p_holm:.2f} |")
L += ["", "## Ausfaelle", "", "| Kandidat | Tiefe | Ausfaelle | davon Schritt<=1 | Median Ausfall-Schritt |", "|---|---|---|---|---|"]
for (cd, dp), g in F.groupby(["cand", "depth"]):
    L.append(f"| {cd} | {dp} | {len(g)} | {int((g.fail_step <= 1).sum())} | {g.fail_step.median():.0f} |")
L += ["", "## Dateien", "", "summary.csv, depth_change.csv, pairwise_tests.csv, failures.csv, capexp_first_step.csv, 01-06 PNG"]
open(f"{OUT}/report.md", "w").write("\n".join(L))
print("\n".join(L))
