import glob, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_full"); os.makedirs(OUT, exist_ok=True)
D = pd.concat(map(pd.read_csv, sorted(glob.glob(os.path.join(HERE, "v9b_results_shard[0-9].csv")))), ignore_index=True)
D["L"] = np.where(D.bad, np.inf, D.test_loss)
K = ["act", "c", "a", "gamma", "soft_s", "arch", "init_k", "lr"]
G = D.groupby(["target", "depth"] + K).agg(n=("seed", "count"), fails=("bad", "sum"),
        med=("L", "median"), q25=("L", lambda x: x.quantile(.25)), q75=("L", lambda x: x.quantile(.75))).reset_index()
G.to_csv(f"{OUT}/config_summary.csv", index=False)

fail = D.groupby(["act", "arch", "depth"]).bad.mean().unstack().round(3); fail.to_csv(f"{OUT}/fail_rates.csv")
best = G.loc[G.groupby(["target", "depth", "act", "arch"]).med.idxmin()]
best.to_csv(f"{OUT}/best_per_act_arch.csv", index=False)
typ = G[np.isfinite(G.med)].groupby(["target", "depth", "act", "arch"]).med.median().reset_index()
typ.to_csv(f"{OUT}/typical_per_act_arch.csv", index=False)
win = G.loc[G.groupby(["target", "depth"]).med.idxmin()]; win.to_csv(f"{OUT}/winners.csv", index=False)
G["rk"] = G.groupby(["target", "depth"]).med.rank()
R = G.groupby(["depth"] + K).agg(worst=("rk", "max"), mean=("rk", "mean"), nt=("rk", "count"), fails=("fails", "sum")).reset_index()
R = R[R.nt == 3]; R.to_csv(f"{OUT}/robust_ranking.csv", index=False)

# charts
cols = {"relu": "#bab9b4", "gelu": "#6a9fb5", "silu": "#2f5d7c", "capexp": "#da7101", "precap": "#01696f"}
fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for i, t in enumerate(["sin3_quad", "sin_mix", "abs_wave"]):
    for j, d in enumerate([16, 32, 64]):
        ax = axs[i, j]; s = best[(best.target == t) & (best.depth == d)]
        for k, (a, ar) in enumerate([(a, ar) for a in cols for ar in ["plain", "res_inv"]]):
            r = s[(s.act == a) & (s.arch == ar)]
            if len(r):
                v = max(float(r.med.iloc[0]), 1e-6)
                ax.bar(k, v, color=cols[a], alpha=1.0 if ar == "res_inv" else .45, edgecolor="k", linewidth=.3)
        ax.set_yscale("log"); ax.set_title(f"{t}, Tiefe {d}", fontsize=10)
        ax.set_xticks(range(10)); ax.set_xticklabels([f"{a}\n{'res' if ar=='res_inv' else 'pl'}" for a in cols for ar in ["plain", "res_inv"]], fontsize=6)
fig.suptitle("v9b: bester Median-Testloss je Aktivierung/Architektur (log, blass = plain)")
plt.tight_layout(); plt.savefig(f"{OUT}/best_loss.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
for k, (a, ar) in enumerate([(a, ar) for a in ["capexp", "precap", "silu", "gelu", "relu"] for ar in ["plain", "res_inv"]]):
    pass
fl = D.groupby(["act", "arch", "depth"]).bad.mean().reset_index()
for (a, ar), s in fl.groupby(["act", "arch"]):
    if s.bad.max() > 0: ax.plot(s.depth, s.bad * 100, marker="o", label=f"{a} {ar}", color=cols[a], ls="-" if ar == "res_inv" else "--")
ax.plot([], [], color="k", label="precap/silu/gelu/relu: 0 %"); ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64]); ax.set_xticklabels([16, 32, 64])
ax.set_xlabel("Tiefe"); ax.set_ylabel("Ausfallrate (%)"); ax.legend(); ax.set_title("Ausfaelle (NaN/Inf) je Aktivierung")
plt.tight_layout(); plt.savefig(f"{OUT}/fail_rate.png", dpi=140); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, t in zip(axs, ["sin3_quad", "sin_mix", "abs_wave"]):
    for (a, ar), s in typ[typ.target == t].groupby(["act", "arch"]):
        ax.plot(s.depth, s.med, marker="o", color=cols[a], ls="-" if ar == "res_inv" else "--", label=f"{a} {ar}")
    ax.set_yscale("log"); ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64]); ax.set_xticklabels([16, 32, 64]); ax.set_title(f"{t}: typische Konfiguration")
axs[0].set_ylabel("Median ueber Konfig-Mediane (Testloss)"); axs[2].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/typical_vs_depth.png", dpi=140); plt.close()

# markdown
def md(df, fmt="{:.6f}"):
    return "```\n" + df.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```"
L = ["# v9b komplette Auswertung", "", f"Runs: {len(D)} (3 Aufgaben x 3 Tiefen x 140 Konfigs x 10 Seeds). Ausfall = NaN/Inf, wird als Verlust unendlich gewertet.", "",
     "## Ausfallraten", "", md(fail.reset_index(), ), "", "## Sieger je Aufgabe und Tiefe", "", md(win[["target", "depth"] + K + ["fails", "med"]]), ""]
for d in [16, 32, 64]:
    L += [f"## Robustheit Tiefe {d} (mittlerer Rang ueber 3 Aufgaben, Top 8)", "", md(R[R.depth == d].nsmallest(8, "mean")[K + ["worst", "mean", "fails"]]), ""]
L += ["## Bester Median je Aktivierung/Architektur", "", md(best[["target", "depth", "act", "arch", "med", "fails"] + ["c", "a", "gamma", "soft_s", "init_k", "lr"]].sort_values(["target", "depth", "med"])), "",
      "## Typische Konfiguration (Median der Konfig-Mediane)", "", md(typ.sort_values(["target", "depth", "med"])), "",
      "Hinweis: Beste Einzelwerte sind durch die Auswahl aus 140 Konfigs optimistisch; Rangwerte und typische Werte sind robuster.", ""]
open(f"{OUT}/AUSWERTUNG_v9b_komplett.md", "w").write("\n".join(L))
print("done", os.listdir(OUT))
