import glob, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
df = pd.concat([pd.read_csv(f) for f in glob.glob("experiments_v5*.csv")])
df["cfg"] = df.act + df.c.map(lambda x: "" if pd.isna(x) else f"_{x}")
df.loc[df.nan_or_inf, "test_loss"] = float("inf")
g = df.groupby(["target", "depth", "arch", "cfg", "init"]).agg(
    n=("seed", "count"), fail=("nan_or_inf", "sum"), med=("test_loss", "median"),
    q25=("test_loss", lambda s: s.quantile(.25)), q75=("test_loss", lambda s: s.quantile(.75))).reset_index()
g.to_csv("summary_v5_full.csv", index=False)
best = g.loc[g.groupby(["target", "depth", "arch", "cfg"]).med.idxmin()]
for t, s in best.groupby("target"):
    print(f"\n=== {t}: Median Test-Loss, beste Init je Aktivierung ===")
    print(s.pivot_table(index=["depth", "arch"], columns="cfg", values="med").round(4).to_string())
    print("--- beste Init ---")
    print(s.pivot_table(index=["depth", "arch"], columns="cfg", values="init", aggfunc="first").to_string())
