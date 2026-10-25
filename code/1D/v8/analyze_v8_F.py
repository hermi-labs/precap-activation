import csv, glob, math, statistics as st, collections
K=['target','act','c','in_scale','gamma','soft_s','depth','arch','init_k','lr','clip','warmup','seed']
D={}
for f in glob.glob('experiments_v8_F_shard*.csv'):
  for r in csv.DictReader(open(f)): D[tuple(r[k] for k in K)]=r
R=list(D.values()); print('unique',len(R),'expected 512')
def x(r):
 try: y=float(r['test_loss'])
 except: return math.inf
 return y if r['nan_or_inf']!='True' and math.isfinite(y) else math.inf
def summ(rs):
 v=[x(r) for r in rs]; fin=[z for z in v if math.isfinite(z)]
 return len(rs)-len(fin),len(rs),st.median(v), (st.pstdev(fin) if len(fin)>1 else float('nan'))
print('\n== Aggregate by stabilizer (all targets/depths/c/a/seeds)')
for k in ['init_k','clip','warmup']:
 g=collections.defaultdict(list)
 for r in R:g[r[k]].append(r)
 for a,rs in sorted(g.items(),key=lambda q:float(q[0])):
  b,n,m,s=summ(rs);print('%-8s=%-4s fail %3d/%3d  median %10.4g  sd %10.4g'%(k,a,b,n,m,s))
print('\n== Combination init x clip x warmup')
g=collections.defaultdict(list)
for r in R:g[(r['init_k'],r['clip'],r['warmup'])].append(r)
for k,rs in sorted(g.items(),key=lambda q:tuple(map(float,q[0]))):
 b,n,m,s=summ(rs);print('k=%s clip=%s warmup=%s fail %d/%d med %.4g sd %.4g'%(*k,b,n,m,s))
print('\n== By target/depth: best stabilizer combination, then all failure rates')
for t in ['sin3_quad','sin_mix']:
 for d in ['32','64']:
  z=[r for r in R if r['target']==t and r['depth']==d]
  h=collections.defaultdict(list)
  for r in z:h[(r['init_k'],r['clip'],r['warmup'])].append(r)
  out=[]
  for k,rs in h.items():
   b,n,m,s=summ(rs);out.append((m,b,n,k,s))
  print('\n',t,'d',d)
  for m,b,n,k,s in sorted(out): print('  k=%s clip=%s wu=%s: med %.4g fail %d/%d sd %.3g'%(*k,m,b,n,s))
print('\n== By c/a: failures across all stabilizers')
g=collections.defaultdict(list)
for r in R:g[(r['depth'],r['c'],r['in_scale'])].append(r)
for k,rs in sorted(g.items(),key=lambda q:tuple(map(float,q[0]))):
 b,n,m,s=summ(rs);print('d%s c%s a%s: fail %d/%d med %.4g'%(*k,b,n,m))
print('\n== Failure step and what prevented failures')
f=[r for r in R if x(r)==math.inf];print('failure steps',collections.Counter(r['fail_step'] for r in f))
for k in ['init_k','clip','warmup']:
 print(k,collections.Counter(r[k] for r in f))
