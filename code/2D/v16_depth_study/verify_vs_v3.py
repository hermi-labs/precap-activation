import importlib.util,sys,torch,math
from pathlib import Path
R=Path(__file__).resolve().parent
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
v3=load('v3',R.parent/'v3_large_preregistered'/'run_large_2d_v3.py');v14=load('v14',R/'run_v16_depth.py')
torch.manual_seed(0);x=torch.empty(5000,2).uniform_(-1,1);res={}
for n in ['radial_ripple','ridge_mix','corner_kink','smooth_bowl','highfreq_grid','angled_kink']:
    res['task_'+n]=float((v3.fn(n,x[:,0],x[:,1])-v14.fn(n,x[:,0],x[:,1])).abs().max())
z=torch.linspace(-60,60,20001,dtype=torch.float64)
res['act_base']=float((v3.PreCap(1.,1.,.01,1.5).double()(z)-v14.PreCap(1.,1.,.01,1.5).double()(z)).abs().max())
res['act_base_vs_runner']=float((v14.act('precap_base').double()(z)-v3.make_act('precap').double()(z)).abs().max())
res['act_s225_formula']=float((v14.act('precap_s225').double()(z)-v3.PreCap(1.,1.,.01,2.25).double()(z)).abs().max())
for d in (16,32):
    torch.manual_seed(5);a=v3.Net(d,'precap');torch.manual_seed(5);b=v14.Net(d,'precap_base');xx=torch.randn(64,2)
    res[f'net_d{d}_out_diff']=float((a(xx)-b(xx)).abs().max())
    torch.manual_seed(5);a=v3.Net(d,'silu');torch.manual_seed(5);b=v14.Net(d,'silu');res[f'net_silu_d{d}_out_diff']=float((a(xx)-b(xx)).abs().max())
res['consts']=dict(v3=[v3.WIDTH,v3.BATCH,v3.LR,v3.INIT_K,v3.GAMMA],v14=[v14.WIDTH,v14.BATCH,v14.LR,v14.INIT_K,v14.GAMMA])
print(res)
