from pathlib import Path
import json, math, collections, numpy as np
import torch, torch.nn as nn
EXP=Path('/workspace/ayb/experiments/dfe-unified-p3'); OUT=EXP/'geometry_probe'; RUN=OUT/'pair_control_run'; RUN.mkdir(exist_ok=False)
rows=[json.loads(x) for x in (OUT/'geometry_features.jsonl').open() if x.strip()]
labels_raw=json.loads((EXP/'refined_labels.json').read_text()); labels={r['code']:float(r['pk']) for r in labels_raw}
assert len(rows)==5316 and all(r['code'] in labels and r['split'] in ('train','val','test') for r in rows)
sp=json.loads((EXP/'refined_split.json').read_text())['split']; split_map={r['code']:r['split'] for r in rows}
assert {k:len(v) for k,v in sp.items()}=={'train':4049,'val':444,'test':823}
vocab=sorted({k for r in rows if r['split']=='train' for k in r['pair_hist']}); vi={k:i for i,k in enumerate(vocab)}
def enc(rs, source=None):
 X=np.zeros((len(rs),len(vocab)),np.float32); y=np.array([labels[r['code']] for r in rs],np.float32)
 for i,r in enumerate(rs):
  rr=r if source is None else source[i]
  for k,v in rr['pair_hist'].items(): X[i,vi[k]]=math.log1p(v)
  X[i]/=max(1.,X[i].sum())
 return X,y
tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']; te=[r for r in rows if r['split']=='test']
Xtr,ytr=enc(tr); Xva,yva=enc(va); Xte,yte=enc(te)
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd; Xte=(Xte-mu)/sd
def pear(a,b):
 a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
class M(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def fit(Xt,yt,Xv,yv,Xe,ye):
 torch.manual_seed(20260915); m=M(Xt.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); xt=torch.tensor(Xt); yt=torch.tensor((yt-yt.mean())/(yt.std()+1e-6)); xv=torch.tensor(Xv); yv=torch.tensor(yv); xe=torch.tensor(Xe); best=(-9,None)
 for ep in range(250):
  m.train(); opt.zero_grad(); loss=((m(xt)-yt)**2).mean(); assert torch.isfinite(loss); loss.backward(); opt.step(); m.eval()
  with torch.no_grad(): p=m(xv).numpy(); pred=p*(ytr.std()+1e-6)+ytr.mean()
  q=pear(pred,yv)
  if q>best[0]: best=(q,{k:v.detach().clone() for k,v in m.state_dict().items()})
 m.load_state_dict(best[1]); m.eval()
 with torch.no_grad(): p=m(xe).numpy(); pred=p*(ytr.std()+1e-6)+ytr.mean()
 return {'val_pearson':best[0],'test_pearson':pear(pred,ye),'test_rmse':float(np.sqrt(np.mean((pred-ye)**2)))},pred
real,p_real=fit(Xtr,ytr,Xva,yva,Xte,yte)
# Match-preserving pair permutation: keep labels fixed, shuffle pocket/geometry records within each split.
rng=np.random.default_rng(20260915)
perms={n:rng.permutation(len(rs)) for n,rs in [('tr',tr),('va',va),('te',te)]}
Xtr_s,_=enc(tr,[tr[i] for i in perms['tr']]); Xva_s,_=enc(va,[va[i] for i in perms['va']]); Xte_s,_=enc(te,[te[i] for i in perms['te']])
Xtr_s=(Xtr_s-mu)/sd; Xva_s=(Xva_s-mu)/sd; Xte_s=(Xte_s-mu)/sd
shuf,p_shuf=fit(Xtr_s,ytr,Xva_s,yva,Xte_s,yte)
res={'protocol':'pair geometry histogram frozen probe','n':{'train':len(tr),'val':len(va),'test':len(te)},'feature_dim':len(vocab),'real':real,'within_split_pair_permutation':shuf,'delta_real_minus_shuffle':real['test_pearson']-shuf['test_pearson'],'fallback_rows_included':sum('geometry_recovery' in r for r in rows),'seed':20260915}
(RUN/'metrics.json').write_text(json.dumps(res,indent=2,allow_nan=False)); (RUN/'predictions.json').write_text(json.dumps({'real':p_real.tolist(),'shuffle':p_shuf.tolist()},allow_nan=False)); print(json.dumps(res,indent=2),flush=True); print('PAIR_CONTROL_DONE',flush=True)
