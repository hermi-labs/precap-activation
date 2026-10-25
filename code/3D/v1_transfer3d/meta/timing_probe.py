import importlib.util,sys,time,json
from pathlib import Path
s=importlib.util.spec_from_file_location('m',str(Path(__file__).resolve().parent.parent/'run_3d_v1.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=int(sys.argv[1]);m.STEPS=300;m.CHECKPOINTS=[0,300];m.OUTDIR=Path(__file__).resolve().parent/f'probe_tmp/p{i}'
acts=m.ACTS;res={}
for depth in [8,16]:
 for norm in ['none','layernorm']:
  c={'group':'3d','task':'gyroid','depth':depth,'scale':'inv_sqrt','norm':norm,'lr':3e-3,'seed':30000,'act':acts[i%5]}
  t=time.time();m.execute(c,0);res[f'd{depth}_{norm}']=round((time.time()-t)/300*3000,1)
print(json.dumps({'act':acts[i%5],'sek_pro_3000_schritte':res}))
