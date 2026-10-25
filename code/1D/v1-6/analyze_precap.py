import argparse
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="precap_results.csv")
a = ap.parse_args()

df = pd.read_csv(a.csv)
df["cfg"] = df.apply(lambda r: f"{r.act}_c{r.c}_g{r.gamma}_p{int(r.p)}_s{r.scale}_{r.mode}", axis=1)
df.loc[df.nan_or_inf == True, "test_loss"] = float("inf")

summary = df.groupby(["target", "depth", "init", "cfg"]).agg(
    n=("seed", "count"),
    fails=("nan_or_inf", "sum"),
    med_test=("test_loss", "median"),
    mean_test=("test_loss", "mean"),
    med_grad=("max_grad", "median"),
).reset_index()

summary.to_csv("precap_summary.csv", index=False)

for (target, depth, init), s in summary.groupby(["target", "depth", "init"]):
    print(f"\n=== target={target} depth={depth} init={init} ===")
    print(s.sort_values(["med_test", "fails", "med_grad"]).head(12).to_string(index=False))
