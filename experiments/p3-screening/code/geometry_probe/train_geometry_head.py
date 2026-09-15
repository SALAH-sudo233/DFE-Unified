from pathlib import Path
import json,collections,random,math,os,time
import numpy as np, torch
import torch.nn as nn
EXP=Path('/workspace/ayb/experiments/dfe-unified-p3'); OUT=EXP/'geometry_probe'; RUN=OUT/'geometry_head_run'; RUN.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (OUT/'geometry_features.jsonl').open() if x.strip()]
assert len(rows)==5316 and all(r['split']!='unknown' for r in rows)
labels={r['code']:float(r['label']['pk']) if isinstance(r.get('label'),dict) else None for r in rows}
rows=[r for r in rows if labels[r['code']] is not None]
# Build train vocabulary only; encode sparse pair histogram with count normalization.
vocab=sorted({k for r in rows if r['split']=='train' for k in r['pair_hist']}); vi={k:i for i,k in enumerate(vocab)}
def encode(rs, perm=None):
 X=np.zeros((len(rs),len(vocab)),np.float32); y=np.zeros(len(rs),np.float32); codes=[]
 for ii,r in enumerate(rs):
  rr=rs[perm[ii]] if perm is not None else r
  for k,v in rr['pair_hist'].items():
   if k in vi:X[ii,vi[k]]=math.log1p(v)
  X[ii]/=max(1.0,X[ii].sum()); y[ii]=labels[r['code']]; codes.append(r['code'])
 return X,y,codes
tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']; te=[r for r in rows if r['split']=='test']
Xtr,ytr,_=encode(tr); Xva,yva,_=encode(va); Xte,yte,tec=encode(te)
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd; Xte=(Xte-mu)/sd
y_mu=ytr.mean(); y_sd=ytr.std()+1e-6
def pear(a,b):
 a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
class M(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def run(Xtr,ytr,Xva,yva,Xte,yte,seed):
 torch.manual_seed(seed); m=M(Xtr.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); xt=torch.tensor(Xtr); yt=torch.tensor((ytr-y_mu)/y_sd); xv=torch.tensor(Xva); yv=torch.tensor(yva); xe=torch.tensor(Xte); best=(-9,None)
 for ep in range(250):
  m.train(); opt.zero_grad(); loss=((m(xt)-yt)**2).mean(); loss.backward(); opt.step(); m.eval()
  with torch.no_grad(): p=m(xv).numpy()*y_sd+y_mu
  q=pear(p,yva)
  if q>best[0]: best=(q,{k:v.detach().clone() for k,v in m.state_dict().items()})
 m.load_state_dict(best[1]); m.eval()
 with torch.no_grad(): p=m(xe).numpy()*y_sd+y_mu
 return {'val_pearson':best[0],'test_pearson':pear(p,yte),'test_rmse':float(np.sqrt(np.mean((p-yte)**2))),'n_train':len(ytr),'n_val':len(yva),'n_test':len(yte)},p
res,p=run(Xtr,ytr,Xva,yva,Xte,yte,20260915)
# Pair permutation preserves ligand/pocket marginal absent here; this control permutes geometry rows while labels remain fixed.
perm=np.random.default_rng(20260915).permutation(len(te)); Xte_perm,_,_=encode(te,perm=perm); Xte_perm=(Xte_perm-mu)/sd
res_perm,pperm=run(Xtr,ytr,Xva,yva,Xte_perm,yte,20260915)
result={'protocol':'PDBBind refined atom-pair element-distance histogram','feature_dim':len(vocab),'results':{'real':res,'pocket_geometry_permutation_test':res_perm},'fallback_recovery_included':True,'n_rows':len(rows),'splits':{'train':len(tr),'val':len(va),'test':len(te)},'seed':20260915,'note':'This is a frozen geometry probe, not generator joint fine-tuning and not a CrossDocked SOTA score.'}
(RUN/'metrics.json').write_text(json.dumps(result,indent=2)); torch.save({'model':M(Xtr.shape[1]).state_dict(),'vocab':vocab,'mu':mu,'sd':sd,'result':result},RUN/'geometry_head.pt'); print(json.dumps(result,indent=2),flush=True); print('GEOMETRY_HEAD_DONE',flush=True)
