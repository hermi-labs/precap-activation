import glob, os, sys, warnings, importlib.util
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DONE = os.path.exists(os.path.join(HERE, "v10_done.txt"))
OUT = os.path.join(HERE, "analysis_full" if DONE else "analysis_partial")
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(HERE, "v10_results_shard[0-9].csv")))
D = pd.concat([pd.read_csv(f).assign(shard=i) for i, f in enumerate(files)], ignore_index=True)
D["bad"] = D.bad.astype(str).str.lower().eq("true")
D["L"] = np.where(D.bad, np.inf, D.test_loss)
CK = ["target", "act", "c", "a", "gamma", "soft_s", "depth", "arch", "init_k", "lr"]
KEY = CK + ["seed"]

COL = {"relu": "#bab9b4", "gelu": "#6a9fb5", "silu": "#2f5d7c", "capexp": "#da7101", "precap": "#01696f"}
def col(a): return COL.get(a, "#7a39bb")

spec = importlib.util.spec_from_file_location("v10mod", os.path.join(HERE, "capexp_v10.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
expected = pd.DataFrame(list(mod.configs()))
EXP_N = len(expected)
ek = expected.rename(columns={"soft_s": "soft_s"})
ek = ek[KEY]
def norm(df):
    x = df[KEY].copy()
    for c in ["c", "a", "gamma", "soft_s", "init_k", "lr"]: x[c] = x[c].astype(float).round(6)
    x["depth"] = x.depth.astype(int); x["seed"] = x.seed.astype(int)
    return x
miss = norm(ek).merge(norm(D).assign(_d=1), on=KEY, how="left")
missing = miss[miss._d.isna()].drop(columns="_d")
missing.to_csv(f"{OUT}/missing_configs.csv", index=False)
dups = int(D.duplicated(KEY).sum())

integ = D.groupby("shard").agg(rows=("seed", "count"), bad=("bad", "sum"), mean_sec=("duration_sec", "mean")).reset_index()
integ.to_csv(f"{OUT}/schritt1_integritaet_shards.csv", index=False)
cover = D.groupby(["target", "depth", "arch", "act"]).size().rename("runs").reset_index()
cover.to_csv(f"{OUT}/schritt1_abdeckung.csv", index=False)

G = D.groupby(CK).agg(n=("seed", "count"), fails=("bad", "sum"), med=("L", "median"),
                      q25=("L", lambda x: x.quantile(.25)), q75=("L", lambda x: x.quantile(.75)),
                      mean_sec=("duration_sec", "mean")).reset_index()
G.to_csv(f"{OUT}/schritt2_config_summary.csv", index=False)
best = G.loc[G.groupby(["target", "depth", "act", "arch"]).med.idxmin()]
best.to_csv(f"{OUT}/schritt2_best_per_act_arch.csv", index=False)
typ = G[np.isfinite(G.med)].groupby(["target", "depth", "act", "arch"]).med.median().rename("typ").reset_index()
typ.to_csv(f"{OUT}/schritt2_typical_per_act_arch.csv", index=False)
win = G.loc[G.groupby(["target", "depth"]).med.idxmin()]
win.to_csv(f"{OUT}/schritt2_winners.csv", index=False)
G["rk"] = G.groupby(["target", "depth", "arch"]).med.rank()
R = G.groupby(["depth", "arch", "act", "c", "a", "gamma", "soft_s", "init_k", "lr"]).agg(
    worst=("rk", "max"), mean=("rk", "mean"), nt=("rk", "count"), fails=("fails", "sum")).reset_index()
R = R[R.nt == 3].sort_values(["depth", "arch", "mean"]); R.to_csv(f"{OUT}/schritt2_robust_ranking.csv", index=False)

# Schritt 3: Stabilitaet
fail = D.groupby(["act", "arch", "depth"]).bad.agg(["mean", "sum", "count"]).reset_index()
fail.to_csv(f"{OUT}/schritt3_fail_rates.csv", index=False)
cf = sorted(glob.glob(os.path.join(HERE, "v10_results_shard[0-9]_curves.csv")))
usecols = ["target", "act", "c", "a", "gamma", "soft_s", "depth", "arch", "init_k", "lr", "seed", "step", "train_loss", "test_loss", "grad_norm", "act_std_mean", "act_abs_max", "cap_frac"]
C = pd.concat([pd.read_csv(f, usecols=usecols) for f in cf], ignore_index=True)
last = C[C.step == C.step.max()]
stab = last.groupby(["act", "arch", "depth"]).agg(grad_final_med=("grad_norm", "median"), absmax_med=("act_abs_max", "median"),
        absmax_p95=("act_abs_max", lambda x: x.quantile(.95)), cap_frac_med=("cap_frac", "median"), gap_med=("train_loss", lambda x: 0)).reset_index().drop(columns="gap_med")
gmax = C.groupby(["act", "arch", "depth"]).grad_norm.agg(lambda x: x.quantile(.99)).rename("grad_p99_alle_steps").reset_index()
stab = stab.merge(gmax, on=["act", "arch", "depth"])
gen = last.assign(gap=last.test_loss - last.train_loss).groupby(["act", "arch", "depth"]).gap.median().rename("gen_gap_med").reset_index()
stab = stab.merge(gen, on=["act", "arch", "depth"]); stab.to_csv(f"{OUT}/schritt3_stabilitaet.csv", index=False)
early = C[C.step.isin([1, 5, 10, 20, 50, 100, 200, 400, 800, 1600, 3000])].groupby(["act", "arch", "depth", "step"]).test_loss.median().reset_index()
early.to_csv(f"{OUT}/schritt3_lernkurven_median.csv", index=False)

# Schritt 4 (vorlaeufig): faire Paare auf Seed-Ebene
pair_rows = []
for (t, d, ar), s in G.groupby(["target", "depth", "arch"]):
    p = s[s.act == "precap"]
    if p.empty: continue
    bp = p.loc[p.med.idxmin()]
    for base in ["silu", "gelu", "relu", "capexp"]:
        b = s[s.act == base]
        if b.empty: continue
        bb = b.loc[b.med.idxmin()]
        x = D[(D.target == t) & (D.depth == d) & (D.arch == ar) & (D.act == "precap") & (D.c == bp.c) & (D.a == bp.a) & (D.gamma == bp.gamma) & (D.soft_s == bp.soft_s) & (D.init_k == bp.init_k) & (D.lr == bp.lr)].set_index("seed").L
        y = D[(D.target == t) & (D.depth == d) & (D.arch == ar) & (D.act == base) & (D.c == bb.c) & (D.a == bb.a) & (D.gamma == bb.gamma) & (D.soft_s == bb.soft_s) & (D.init_k == bb.init_k) & (D.lr == bb.lr)].set_index("seed").L
        j = pd.concat([x, y], axis=1, keys=["p", "b"]).dropna()
        j = j[np.isfinite(j.p) & np.isfinite(j.b)]
        pv = np.nan
        if len(j) >= 6 and (j.p - j.b).abs().sum() > 0:
            try: pv = wilcoxon(np.log10(j.p), np.log10(j.b)).pvalue
            except Exception: pass
        pair_rows.append(dict(target=t, depth=d, arch=ar, gegner=base, precap_med=bp.med, gegner_med=bb.med,
                              faktor=bb.med / bp.med if bp.med > 0 else np.nan, seed_siege=float((j.p < j.b).mean()) if len(j) else np.nan,
                              n_paare=len(j), p_wilcoxon=pv))
pairs = pd.DataFrame(pair_rows); pairs.to_csv(f"{OUT}/schritt4_paarvergleich_best_vs_best.csv", index=False)

# Hyperparameter-Sensitivitaet PreCap (Tiefe 32/64)
P = G[(G.act == "precap") & (G.depth >= 32) & np.isfinite(G.med)].copy()
P["lg"] = np.log10(P.med)
P["z"] = P.lg - P.groupby(["target", "depth"]).lg.transform("mean")
sens = {k: P.groupby(k).z.mean() for k in ["c", "a", "gamma", "soft_s"]}
pd.concat({k: v for k, v in sens.items()}, names=["param", "wert"]).rename("mittl_log10_abweichung").reset_index().to_csv(f"{OUT}/precap_sensitivitaet.csv", index=False)

# Grafiken
order = [(a, ar) for a in ["relu", "gelu", "silu", "capexp", "precap"] for ar in ["plain", "res_inv"]]
targets = ["sin3_quad", "sin_mix", "abs_wave"]; depths = sorted(D.depth.unique())
fig, axs = plt.subplots(len(targets), len(depths), figsize=(3.6 * len(depths), 9), sharex=True)
for i, t in enumerate(targets):
    for j, d in enumerate(depths):
        ax = axs[i, j]; s = best[(best.target == t) & (best.depth == d)]
        for k, (a, ar) in enumerate(order):
            r = s[(s.act == a) & (s.arch == ar)]
            if len(r) and np.isfinite(r.med.iloc[0]):
                ax.bar(k, max(float(r.med.iloc[0]), 1e-7), color=col(a), alpha=1 if ar == "res_inv" else .45, edgecolor="k", linewidth=.3)
        ax.set_yscale("log"); ax.set_title(f"{t}, Tiefe {d}", fontsize=9)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{a}\n{'res' if ar == 'res_inv' else 'pl'}" for a, ar in order], fontsize=5)
fig.suptitle(f"v10: bester Median-Testloss je Aktivierung/Architektur (log, blass = plain) - {len(D)} Runs")
plt.tight_layout(); plt.savefig(f"{OUT}/g1_best_loss.png", dpi=140); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, t in zip(axs, targets):
    for (a, ar), s in typ[typ.target == t].groupby(["act", "arch"]):
        s = s.sort_values("depth"); ax.plot(s.depth, s.typ, marker="o", color=col(a), ls="-" if ar == "res_inv" else "--", label=f"{a} {ar}")
    ax.set_yscale("log"); ax.set_xscale("log", base=2); ax.set_xticks(depths); ax.set_xticklabels(depths); ax.set_title(f"{t}: typische Konfiguration"); ax.set_xlabel("Tiefe")
axs[0].set_ylabel("Median der Konfig-Mediane (Testloss)"); axs[2].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/g2_typisch_vs_tiefe.png", dpi=140); plt.close()

deep = D[(D.depth >= 32) & (D.arch == "res_inv")]
fig, axs = plt.subplots(len(targets), 2, figsize=(10, 9))
for i, t in enumerate(targets):
    for j, d in enumerate([32, 64]):
        ax = axs[i, j]; data, labs, cs = [], [], []
        for a in ["silu", "gelu", "precap"]:
            g = G[(G.target == t) & (G.depth == d) & (G.arch == "res_inv") & (G.act == a)]
            if g.empty: continue
            bc = g.loc[g.med.idxmin()]
            v = D[(D.target == t) & (D.depth == d) & (D.arch == "res_inv") & (D.act == a) & (D.c == bc.c) & (D.a == bc.a) & (D.gamma == bc.gamma) & (D.soft_s == bc.soft_s) & (D.init_k == bc.init_k) & (D.lr == bc.lr)].L
            data.append(np.log10(v[np.isfinite(v)].clip(lower=1e-8))); labs.append(a); cs.append(col(a))
        if data:
            bp = ax.boxplot(data, labels=labs, patch_artist=True)
            for patch, c_ in zip(bp["boxes"], cs): patch.set_facecolor(c_); patch.set_alpha(.7)
        ax.set_title(f"{t}, Tiefe {d} (je beste Konfig, 10 Seeds)", fontsize=9); ax.set_ylabel("log10 Testloss")
plt.tight_layout(); plt.savefig(f"{OUT}/g3_seedstreuung_tief.png", dpi=140); plt.close()

fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, k in zip(axs, ["c", "a", "gamma", "soft_s"]):
    v = sens[k]; ax.bar([str(x) for x in v.index], v.values, color="#01696f"); ax.axhline(0, color="k", lw=.6); ax.set_title(f"PreCap Tiefe 32/64: {k}")
axs[0].set_ylabel("mittl. log10-Abweichung (negativ = besser)")
plt.tight_layout(); plt.savefig(f"{OUT}/g4_precap_sensitivitaet.png", dpi=140); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
st = stab.copy()
for ax, m_, ttl in zip(axs, ["absmax_med", "grad_final_med", "gen_gap_med"], ["Aktivierungs-Maximum (Median, Ende)", "Gradienten-Norm (Median, Ende)", "Generalisierungslücke test-train (Median)"]):
    for (a, ar), s in st.groupby(["act", "arch"]):
        s = s.sort_values("depth"); ax.plot(s.depth, s[m_], marker="o", color=col(a), ls="-" if ar == "res_inv" else "--", label=f"{a} {ar}")
    ax.set_xscale("log", base=2); ax.set_xticks(depths); ax.set_xticklabels(depths); ax.set_yscale("symlog", linthresh=1e-4); ax.set_title(ttl, fontsize=9); ax.set_xlabel("Tiefe")
axs[2].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/g5_stabilitaet.png", dpi=140); plt.close()

# Bericht
def md(df, fmt=None):
    if df.empty: return "_keine Daten_\n"
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].map(lambda v: f"{v:.3g}")
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n"

tab = typ.pivot_table(index=["depth", "arch", "act"], columns="target", values="typ").reset_index()
top = pairs[pairs.gegner.isin(["silu", "gelu"])].sort_values(["depth", "target", "arch", "gegner"])
rep = [f"# v10 Auswertung ({'FINAL' if DONE else 'VORLAEUFIG'})", "",
       f"Stand: {pd.Timestamp.now():%Y-%m-%d %H:%M}. Runs: **{len(D)} von {EXP_N}**, Duplikate: {dups}, fehlende Konfigurationen: {len(missing)}, Ausfaelle (bad): {int(D.bad.sum())}.", "",
       "## Schritt 1 - Integritaet", md(integ), "## Schritt 2 - Typische Qualitaet (Median der Konfig-Mediane)", md(tab),
       "## Beste Konfiguration je Aufgabe/Tiefe", md(win[["target", "depth", "arch", "act", "c", "a", "gamma", "soft_s", "init_k", "lr", "med", "q25", "q75", "fails"]]),
       "## Schritt 3 - Stabilitaet", md(fail), md(stab),
       "## Schritt 4 (vorlaeufig) - beste PreCap-Konfig vs. beste Konfig der Gegner, Seed-gepaart", md(top[["target", "depth", "arch", "gegner", "precap_med", "gegner_med", "faktor", "seed_siege", "n_paare", "p_wilcoxon"]]),
       "Hinweis: 'beste Konfig' wird anhand derselben Seeds gewaehlt, p-Werte sind daher optimistisch. PreCap hat in v10 deutlich mehr Hyperparameter-Varianten als SiLU/GELU - gepoolte Medians ueber alle Zeilen sind verzerrt, deshalb hier Konfig-Mediane.", ""]
open(f"{OUT}/AUSWERTUNG_v10.md", "w").write("\n".join(rep))
print("OUT", OUT, "rows", len(D), "expected", EXP_N, "missing", len(missing), "dups", dups, "bad", int(D.bad.sum()))
