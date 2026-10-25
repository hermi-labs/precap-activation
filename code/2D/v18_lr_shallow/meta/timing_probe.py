
import importlib.util,pathlib,time,sys
s=importlib.util.spec_from_file_location('v18','/Users/aaronschmitt/PyCharmMiscProject/2D/v18_lr_shallow/run_v18_lr.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
depth=int(sys.argv[1]);b.OUTDIR=pathlib.Path('/tmp/v18time/d%d'%depth)
t=time.time();b.execute({'group':'v3','task':'ridge_mix','depth':depth,'scale':'inv_sqrt','lr':1e-3,'seed':9900,'act':'precap_base'},0);print(depth,round(time.time()-t,1))
