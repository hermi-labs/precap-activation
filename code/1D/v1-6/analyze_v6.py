import argparse, math
import pandas as pd

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 60)

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='?', default='experiments_v6.csv')
a = ap.parse_args()

df = pd.read_csv(a.csv)
df.loc[df['nan_or_inf'] == True, 'test_loss'] = float('inf')

def show(title, cols, n=20):
    s = df.groupby(cols).agg(
        runs=('seed','count'),
        fails=('nan_or_inf','sum'),
        med_test=('test_loss','median'),
        mean_test=('test_loss','mean'),
        med_100=('loss_100','median'),
        med_200=('loss_200','median'),
        med_400=('loss_400','median'),
        med_grad=('max_grad','median'),
    ).reset_index().sort_values(['med_test','fails','med_grad'])
    print(f"\n=== {title} ===")
    print(s.head(n).to_string(index=False))
    return s

show('overall', ['target','act','c','depth','init','arch','in_scale','gamma','p','soft_s'])
show('by activation', ['act'])
show('capexp only', ['target','act','c','depth','init','arch','in_scale'])
show('precap only', ['target','act','c','depth','init','arch','in_scale','gamma','p','soft_s'])
