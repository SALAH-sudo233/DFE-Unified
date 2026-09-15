from pathlib import Path
import json,collections,random,math,os,time
import numpy as np, torch
import torch.nn as nn
EXP=Path('/workspace/ayb/experiments/dfe-unified-p3'); OUT=EXP/'geometry_probe'; RUN=Path(os.environ['GEOMETRY_RUN']); RUN.mkdir(exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
rows=[json.loads(x) for x in (OUT/'geometry_features.jsonl').open() if x.strip()]
assert len(rows)==5316 and all(r['split']!='unknown' for r in rows)
labels_raw=json.loads((EXP/'refined_labels.json').read_text())
labels={r['code']:float(r['pk']) for r in labels_raw}
assert len(labels)==len(labels_raw)==5316
assert len({r['code'] for r in rows})==len(rows)==5316
assert set(labels)=={r['code'] for r in rows}
assert all(math.isfinite(v) for v in labels.values())
sp=json.loads((EXP/'refined_split.json').read_text())['split']
clusters=json.loads((EXP/'refined_clusters.json').read_text())['code2cluster']
sets={k:set(v) for k,v in sp.items()}
for a in sets:
 assert sets[a]=={r['code'] for r in rows if r['split']==a}
 for b in sets:
  if a!=b:
   assert not (sets[a]&sets[b])
   assert not ({clusters[c] for c in sets[a]}&{clusters[c] for c in sets[b]})
assert all(sum(r['pair_hist'].values())==r['n_lig']*r['n_pocket'] for r in rows)
assert all(math.isfinite(r['min_dist']) for r in rows)
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
assert (len(tr),len(va),len(te))==(4049,444,823)
assert len(vocab)>0
Xtr,ytr,trc=encode(tr); Xva,yva,vac=encode(va); Xte,yte,tec=encode(te)
print('PREFLIGHT',len(tr),len(va),len(te),'features',len(vocab),flush=True)
for x in (Xtr,Xva,Xte,ytr,yva,yte): assert np.isfinite(x).all()
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
 for ep in range(int(os.environ.get('GEOMETRY_EPOCHS','250'))):
  m.train(); opt.zero_grad(); loss=((m(xt)-yt)**2).mean(); assert torch.isfinite(loss), 'nonfinite loss'
  loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),100.,error_if_nonfinite=True); opt.step(); m.eval()
  with torch.no_grad(): p=m(xv).numpy()*y_sd+y_mu
  q=pear(p,yva)
  assert math.isfinite(q), 'nonfinite validation'
  if q>best[0]: best=(q,{k:v.detach().clone() for k,v in m.state_dict().items()}); best_ep=ep+1
  if (ep+1)%25==0: print('EPOCH',ep+1,'train_loss',float(loss.detach()),'val_pearson',q,flush=True)
 m.load_state_dict(best[1]); m.eval()
 with torch.no_grad(): p=m(xe).numpy()*y_sd+y_mu
 return {'val_pearson':best[0],'test_pearson':pear(p,yte),'test_rmse':float(np.sqrt(np.mean((p-yte)**2))),'n_train':len(ytr),'n_val':len(yva),'n_test':len(yte),'best_epoch':best_ep},p,best[1]
res,pred,state=run(Xtr,ytr,Xva,yva,Xte,yte,20260915)
assert all(math.isfinite(v) for v in res.values())
ck={'model':state,'vocab':vocab,'mu':mu,'sd':sd,'y_mu':float(y_mu),'y_sd':float(y_sd),'architecture':[len(vocab),128,64,1],'seed':20260915,'metrics':res}
torch.save(ck,RUN/'geometry_head.pt')
loaded=torch.load(RUN/'geometry_head.pt',map_location='cpu',weights_only=False)
reloaded=M(len(vocab)); reloaded.load_state_dict(loaded['model'],strict=True); reloaded.eval()
with torch.no_grad(): replay=reloaded(torch.tensor(Xte)).numpy()*loaded['y_sd']+loaded['y_mu']
assert np.array_equal(replay,pred), 'checkpoint prediction mismatch'
result={'protocol':'geometry histogram affinity regression; frozen generator; no sensitivity analysis','metrics':res,'splits':{'train':len(tr),'val':len(va),'test':len(te)},'feature_dim':len(vocab),'fallback_rows_included':sum('geometry_recovery' in r for r in rows),'checkpoint_reload_exact':True,'seed':20260915}
(RUN/'test_predictions.json').write_text(json.dumps([{'code':c,'target':float(y),'prediction':float(v)} for c,y,v in zip(tec,yte,pred)],allow_nan=False))
(RUN/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result),flush=True)
