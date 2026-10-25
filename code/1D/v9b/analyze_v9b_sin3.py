import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = "/Users/aaronschmitt/PyCharmMiscProject/v9b"
OUT = f"{ROOT}/analysis_sin3_quad"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)
d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/v9b_results_shard[0-9].csv")])
d = d[d.target.eq("sin3_quad")].copy()
d["bad"] = d.bad.astype(str).str.lower().eq("true")
vc = d.loc[~d.bad, "test_loss"].round(6).value_counts()
constant = float(vc.index[0]) if len(vc) and vc.iloc[0] >= 20 else None
if constant is not None:
    d["collapsed"] = (~d.bad) & np.isclose(d.test_loss, constant, rtol=1e-4, atol=1e-8)
else:
    d["collapsed"] = False
d["dead"] = d.bad | d.collapsed | ~np.isfinite(d.test_loss)
d["loss"] = np.where(d.dead, np.inf, d.test_loss)
name = {"silu": "SiLU", "gelu": "GELU", "relu": "ReLU", "capexp": "CapExp", "precap": "PreCap"}
d["family"] = d.act.map(name) + " " + d.arch.map({"plain": "plain", "res_inv": "res"})
F = [x for x in [f"{n} {a}" for n in name.values() for a in ["plain", "res"]] if x in set(d.family)]
D = sorted(d.depth.unique())
K = ["act", "arch", "c", "a", "gamma", "soft_s", "init_k", "lr"]
colors = {"SiLU": "#2878B5", "GELU": "#6E7781", "ReLU": "#A05A2C", "CapExp": "#C9302C", "PreCap": "#208A5B"}
color = lambda f: colors[f.split()[0]]
style = lambda f: "-" if f.endswith("res") else "--"

def ci(x):
    x = np.asarray(x)
    z = np.median(x[rng.integers(0, len(x), (3000, len(x)))], axis=1)
    return np.quantile(z, [.025, .975], method="nearest")

def fm(x):
    return "dead" if not np.isfinite(x) else f"{x:.2e}"

rows = []
for dep in D:
    for fam in F:
        g = d[(d.depth == dep) & (d.family == fam)]
        score = g[g.seed < 5].groupby(K, dropna=False).loss.median().sort_values()
        if score.empty:
            continue
        cfg = score.index[0]
        m = np.ones(len(g), bool)
        for k, v in zip(K, cfg):
            m &= (g[k].to_numpy() == v)
        e = g[m & (g.seed >= 5)].loss.to_numpy()
        allx = g[m].loss.to_numpy()
        lo, hi = ci(e)
        rows.append(dict(depth=dep, family=fam, median=np.median(e), ci_low=lo, ci_high=hi, n=len(e),
                         dead_eval=int(np.isinf(e).sum()), dead_all=np.isinf(allx).mean(),
                         config=dict(zip(K, cfg)), values=e))
S = pd.DataFrame(rows)
S.drop(columns="values").assign(config=lambda x: x.config.astype(str)).to_csv(f"{OUT}/best_config_holdout.csv", index=False)

C = d.groupby(["depth", "family"] + K, dropna=False).agg(n=("loss", "size"), median=("loss", "median"), dead_rate=("dead", "mean")).reset_index()
C.to_csv(f"{OUT}/config_medians.csv", index=False)
ops = []
for dep in D:
    ref = C[(C.depth == dep) & np.isfinite(C["median"])]["median"].min()
    for fam in F:
        x = C[(C.depth == dep) & (C.family == fam)]
        y = d[(d.depth == dep) & (d.family == fam)]
        ops.append(dict(depth=dep, family=fam, configs=len(x), alive_configs=(x.dead_rate == 0).mean(),
                        good_configs=((x.dead_rate == 0) & (x["median"] <= 3 * ref)).mean(), dead_runs=y.dead.mean(),
                        collapsed_runs=int(y.collapsed.sum()), overflow_runs=int(y.bad.sum())))
O = pd.DataFrame(ops)
O.to_csv(f"{OUT}/operating_region.csv", index=False)

T = []
for dep in D:
    ss = S[S.depth == dep]
    pc = ss[ss.family.str.startswith("PreCap")]
    if pc.empty:
        continue
    p = pc.loc[pc["median"].idxmin()]
    for _, b in ss[~ss.family.str.startswith(("PreCap", "CapExp"))].iterrows():
        T.append(dict(depth=dep, precap=p.family, baseline=b.family, median_precap=p["median"], median_baseline=b["median"],
                      p=mannwhitneyu(np.where(np.isfinite(p["values"]), p["values"], 1e9), np.where(np.isfinite(b["values"]), b["values"], 1e9), alternative="two-sided").pvalue))
T = pd.DataFrame(T)
if len(T):
    ix = T.p.argsort().to_numpy()
    adj = np.empty(len(T))
    last = 0
    for i, j in enumerate(ix):
        last = max(last, min(1, (len(T) - i) * T.p.iloc[j]))
        adj[j] = last
    T["p_holm"] = adj
    T["precap_better"] = T.median_precap < T.median_baseline
    T.to_csv(f"{OUT}/precap_vs_baselines.csv", index=False)

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.grid": True, "grid.alpha": .22,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, len(D), figsize=(5 * len(D), 4.8), sharey=True)
ax = np.atleast_1d(ax)
for a, dep in zip(ax, D):
    x = S[S.depth == dep].set_index("family").reindex(F).dropna(subset=["median"])
    for i, (fam, r) in enumerate(x.iterrows()):
        if np.isfinite(r["median"]):
            a.barh(i, r["median"], color=color(fam), hatch="//" if fam.endswith("plain") else "", edgecolor="white")
            a.plot([r.ci_low, r.ci_high], [i, i], "k-", lw=1.4)
            a.text(r["median"] * 1.15, i, f"{r['median']:.1e}", va="center", fontsize=8)
        else:
            a.text(1e-6, i, "all dead", color=color(fam), va="center")
    a.set_xscale("log"); a.set_yticks(range(len(x))); a.set_yticklabels(x.index); a.invert_yaxis()
    a.set_title(f"Tiefe {dep}"); a.set_xlabel("Holdout-Median Test-Loss")
fig.suptitle("sin3_quad: beste Einstellung je Familie (Auswahl Seeds 0-4, Bewertung 5-9)")
fig.tight_layout(); fig.savefig(f"{OUT}/01_best_holdout.png"); plt.close(fig)

pv = O.pivot(index="family", columns="depth", values="dead_runs").reindex(F) * 100
fig, a = plt.subplots(figsize=(6, 4.8))
im = a.imshow(pv, cmap="Reds", vmin=0, vmax=100, aspect="auto")
a.set_xticks(range(len(D))); a.set_xticklabels(D); a.set_yticks(range(len(F))); a.set_yticklabels(F); a.grid(False)
for i in range(len(F)):
    for j in range(len(D)):
        v = pv.iloc[i, j]
        a.text(j, i, f"{v:.0f}%", ha="center", va="center", color="white" if v > 50 else "black")
a.set_title("Tote Runs: Overflow oder konstante Ausgabe"); a.set_xlabel("Tiefe")
fig.colorbar(im, ax=a, label="Anteil [%]"); fig.tight_layout(); fig.savefig(f"{OUT}/02_dead_runs.png"); plt.close(fig)

fig, ax = plt.subplots(1, len(D), figsize=(5 * len(D), 4.8), sharey=True)
ax = np.atleast_1d(ax)
for a, dep in zip(ax, D):
    for i, fam in enumerate(F):
        x = C[(C.depth == dep) & (C.family == fam) & (C.dead_rate == 0) & np.isfinite(C["median"])]
        a.scatter(x["median"], np.full(len(x), i) + rng.uniform(-.17, .17, len(x)), s=20, color=color(fam), alpha=.72)
    a.set_xscale("log"); a.set_yticks(range(len(F))); a.set_yticklabels(F); a.invert_yaxis()
    a.set_title(f"Tiefe {dep}"); a.set_xlabel("Median je Hyperparameter-Config")
fig.suptitle("Breite des Betriebsbereichs: jeder Punkt ist eine vollstaendig stabile Config")
fig.tight_layout(); fig.savefig(f"{OUT}/03_operating_region.png"); plt.close(fig)

fig, ax = plt.subplots(1, len(D), figsize=(5 * len(D), 4.3), sharey=True)
ax = np.atleast_1d(ax)
for a, dep in zip(ax, D):
    for fam in F:
        x = C[(C.depth == dep) & (C.family == fam)].groupby("lr")["median"].min().replace(np.inf, np.nan)
        a.plot(x.index, x.values, style(fam), marker="o", lw=1.8, color=color(fam), label=fam)
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"Tiefe {dep}"); a.set_xlabel("Lernrate")
ax[0].set_ylabel("beste Config: Median Test-Loss"); ax[0].legend(fontsize=7, ncol=2)
fig.suptitle("Lernraten-Sensitivitaet"); fig.tight_layout(); fig.savefig(f"{OUT}/04_lr_sensitivity.png"); plt.close(fig)

fig, a = plt.subplots(figsize=(7, 4.5))
for fam in F:
    x = S[S.family == fam].sort_values("depth")
    a.plot(x.depth, x["median"].replace(np.inf, np.nan), style(fam), marker="o", lw=2, color=color(fam), label=fam)
a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks(D); a.set_xticklabels(D)
a.set_xlabel("Tiefe"); a.set_ylabel("Holdout-Median Test-Loss"); a.set_title("Skalierung mit Tiefe, beste Konfiguration")
a.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/05_depth_scaling.png"); plt.close(fig)

L = ["# v9b: sin3_quad Zwischenauswertung", "",
     f"Runs: {len(d)}. Overflow-Ausfaelle: {int(d.bad.sum())}. Konstante-Ausgabe-Kollaps: {int(d.collapsed.sum())} (Referenz-Loss: {constant}).", "",
     "## Beste Konfiguration je Familie", "", "Auswahl auf Seeds 0-4, unabhaengige Bewertung auf Seeds 5-9.", ""]
for dep in D:
    L += [f"### Tiefe {dep}", "", "| Familie | Holdout-Median | 95%-CI | Tote Holdout-Runs | Einstellung | Tote Runs, alle Seeds |", "|---|---:|---:|---:|---|---:|"]
    for _, r in S[S.depth == dep].sort_values("median").iterrows():
        z = r.config
        cfg = f"lr={z['lr']}, k={z['init_k']}" + (f", c={z['c']}, a={z['a']}, gamma={z['gamma']}" if z['act'] in ['capexp', 'precap'] else "")
        L.append(f"| {r.family} | {fm(r['median'])} | {fm(r.ci_low)} - {fm(r.ci_high)} | {r.dead_eval}/{r.n} | {cfg} | {r.dead_all:.0%} |")
    L.append("")
L += ["## Stabiler Betriebsbereich", "", "| Tiefe | Familie | Configs | Stabile Configs | Gute Configs | Tote Runs | Kollaps | Overflow |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for _, r in O.sort_values(['depth', 'good_configs'], ascending=[True, False]).iterrows():
    L.append(f"| {r.depth} | {r.family} | {r.configs} | {r.alive_configs:.0%} | {r.good_configs:.0%} | {r.dead_runs:.0%} | {r.collapsed_runs} | {r.overflow_runs} |")
if len(T):
    L += ["", "## Bestes PreCap gegen Baselines", "", "Mann-Whitney auf Holdout-Seeds; p-Werte Holm-korrigiert.", "", "| Tiefe | PreCap | Baseline | Median PreCap | Median Baseline | p Holm | PreCap besser |", "|---|---|---|---:|---:|---:|---|"]
    for _, r in T.iterrows():
        L.append(f"| {r.depth} | {r.precap} | {r.baseline} | {fm(r.median_precap)} | {fm(r.median_baseline)} | {r.p_holm:.3f} | {'ja' if r.precap_better else 'nein'} |")
open(f"{OUT}/report.md", "w").write("\n".join(L))
print("\n".join(L))
