from pathlib import Path
import json,os,collections,math,time
import numpy as np
from rdkit import Chem
import torch, torch.nn as nn

ROOT=Path('/workspace/ayb/data/pdbbind/v2020'); EXP=Path('/workspace/ayb/experiments/dfe-unified-p3'); OUT=EXP/'geometry_probe'/'causal_control_v2'; OUT.mkdir(parents=True,exist_ok=False)
# authoritative labels/split/cluster
sp=json.loads((EXP/'refined_split.json').read_text())['split']; split_map={str(c):n for n,cs in sp.items() for c in cs}
labels={r['code']:float(r['pk']) for r in json.loads((EXP/'refined_labels.json').read_text())}
clusters=json.loads((EXP/'refined_clusters.json').read_text())['code2cluster']
assert set(split_map)==set(labels)==set(clusters)
bins=np.array([0,2,3,4,5,6,8,10,np.inf],np.float32)
elems=['B','C','N','O','P','S','F','CL','BR','I','SE','OTHER']; ei={x:i for i,x in enumerate(elems)}
def atom_element(a):
 x=a.GetSymbol().upper(); return x if x in ei else 'OTHER'
def read_pocket(p):
 xyz=[]; es=[]
 for line in p.read_text(errors='ignore').splitlines():
  if line.startswith(('ATOM  ','HETATM')):
   try: xyz.append(tuple(map(float,(line[30:38],line[38:46],line[46:54])))); x=(line[76:78].strip() or line[12:16].strip()[0]).upper(); es.append(x if x in ei else 'OTHER')
   except: pass
 return np.asarray(xyz,np.float32),es
def load_lig(p):
 m=next((x for x in Chem.SDMolSupplier(str(p),sanitize=False,removeHs=False) if x is not None),None)
 if m is None: raise ValueError('unreadable_sdf')
 # declared fallback for aromatic failures
 for b in m.GetBonds():
  if b.GetIsAromatic() or b.GetBondType()==Chem.BondType.AROMATIC: b.SetIsAromatic(False); b.SetBondType(Chem.BondType.SINGLE)
 for a in m.GetAtoms(): a.SetIsAromatic(False)
 Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
 c=m.GetConformer(); xyz=np.asarray([[c.GetAtomPosition(i).x,c.GetAtomPosition(i).y,c.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())],np.float32); es=[atom_element(a) for a in m.GetAtoms()]
 return xyz,es
codes=[]; ligs={}; pockets={}; bad=[]
for code in sorted(labels):
 try:
  l,le=load_lig(ROOT/code/f'{code}_ligand.sdf'); p,pe=read_pocket(ROOT/code/f'{code}_pocket.pdb')
  if len(l)==0 or len(p)==0: raise ValueError('empty')
  codes.append(code); ligs[code]=(l,le); pockets[code]=(p,pe)
 except Exception as e: bad.append([code,str(e)])
assert len(codes)>=5200, len(codes)
# Use all valid rows; split exactness recorded.
LM=4*len(elems); PM=len(elems); PD=len(elems)*len(elems)*8
def feat(code, pocket_code=None):
 l,le=ligs[code]; p,pe=pockets[pocket_code or code]
 lm=np.zeros(LM,np.float32); pm=np.zeros(PM,np.float32)
 for x in le: lm[ei[x]]+=1
 for x in pe: pm[ei[x]]+=1
 d=np.sqrt(((l[:,None]-p[None,:])**2).sum(-1)); pair=np.zeros(PD,np.float32)
 for i,a in enumerate(le):
  for j,b in enumerate(pe): pair[(ei[a]*len(elems)+ei[b])*8+int(np.digitize(d[i,j],bins)-1)]+=1
 lm/=max(1,len(le)); pm/=max(1,len(pe)); pair/=max(1,pair.sum())
 return np.concatenate([lm,pm,pair])
# materialize per-code marginal and pair features once
X={};
for n,c in enumerate(codes,1):
 X[c]=feat(c)
 if n%500==0: print('EXTRACT',n,len(codes),flush=True)
# deterministic restricted within-split, cluster-disjoint permutation; preserves ligand rows and pocket marginal multiset.
rng=np.random.default_rng(20260915); perm={}
for split in ['train','val','test']:
 cs=[c for c in codes if split_map[c]==split]
 # deterministic cluster-disjoint derangement: order by cluster, cyclic shift by n//2
 order=sorted(cs,key=lambda c:(str(clusters[c]),c)); n=len(order); off=n//2
 assert off>0
 shifted=[order[(i+off)%n] for i in range(n)]
 pm={a:b for a,b in zip(order,shifted)}
 # verify; if any residual same-cluster/identity, repair by pairwise swap with a safe partner
 bad_keys=[a for a in order if pm[a]==a or clusters[pm[a]]==clusters[a]]
 for a in bad_keys:
  for b in order:
   if b==a: continue
   if clusters[pm[b]]!=clusters[a] and clusters[pm[a]]!=clusters[b] and pm[a]!=b and pm[b]!=a:
    pm[a],pm[b]=pm[b],pm[a]; break
 assert all(pm[a]!=a and clusters[pm[a]]!=clusters[a] for a in order), split
 perm.update(pm)
assert all(perm[c]!=c and split_map[c]==split_map[perm[c]] and clusters[c]!=clusters[perm[c]] for c in perm)
# Feature layout: ligand marginal, pocket marginal, pair. Controls retain explicit marginals.
def parts(x): return x[:LM],x[LM:LM+PM],x[LM+PM:]
def make(mode,cs):
 A=[];Y=[]
 for c in cs:
  l,p,q=parts(X[c]);
  if mode=='full': z=X[c]
  elif mode=='permuted':
   l2,p2,q2=parts(X[perm[c]]); z=np.concatenate([l,p2,q2])
  elif mode=='ligand_only': z=np.concatenate([l,np.zeros(PM+PD,np.float32)])
  elif mode=='pocket_only': z=np.concatenate([np.zeros(LM,np.float32),p,np.zeros(PD,np.float32)])
  else: raise ValueError(mode)
  A.append(z);Y.append(labels[c])
 return np.asarray(A,np.float32),np.asarray(Y,np.float32)
tr=[c for c in codes if split_map[c]=='train']; va=[c for c in codes if split_map[c]=='val']; te=[c for c in codes if split_map[c]=='test']; assert (len(tr),len(va),len(te))==(4049,444,823)
def pear(a,b):
 a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
class M(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def train(mode,seed=0):
 A,y=make(mode,tr); V,yv=make(mode,va); T,yt=make(mode,te); mu=A.mean(0); sd=A.std(0)+1e-6; A=(A-mu)/sd; V=(V-mu)/sd; T=(T-mu)/sd; ym=y.mean(); ys=y.std()+1e-6
 torch.manual_seed(seed); m=M(A.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); xt=torch.tensor(A); yy=torch.tensor((y-ym)/ys); xv=torch.tensor(V); xte=torch.tensor(T); best=(-99,None,0)
 for ep in range(200):
  m.train(); opt.zero_grad(); loss=((m(xt)-yy)**2).mean(); assert torch.isfinite(loss); loss.backward(); opt.step(); m.eval()
  with torch.no_grad(): pv=m(xv).numpy()*ys+ym
  q=pear(pv,yv); assert math.isfinite(q)
  if q>best[0]: best=(q,{k:v.detach().clone() for k,v in m.state_dict().items()},ep+1)
 m.load_state_dict(best[1]); m.eval();
 with torch.no_grad(): pred=m(xte).numpy()*ys+ym
 return {'mode':mode,'val_pearson':best[0],'test_pearson':pear(pred,yt),'test_rmse':float(np.sqrt(np.mean((pred-yt)**2))),'best_epoch':best[2],'n_train':len(y),'n_val':len(yv),'n_test':len(yt)},pred
results={}; preds={}
for mode in ['full','permuted','ligand_only','pocket_only']:
 r,p=train(mode,20260915); results[mode]=r; preds[mode]=p; print('RESULT',json.dumps(r),flush=True)
# pairwise deltas and permutation integrity
full=results['full']; permr=results['permuted']
result={'protocol':'causal pair-control v1; raw atom geometry; restricted within-split cluster-disjoint pocket permutation; no joint generator FT','feature_definition':{'ligand_marginal_dim':LM,'pocket_marginal_dim':PM,'pair_geometry_dim':PD,'total_dim':LM+PM+PD,'distance_bins':[0,2,3,4,5,6,8,10,'inf']},'results':results,'deltas':{'full_minus_permuted_test_pearson':full['test_pearson']-permr['test_pearson'],'full_minus_ligand_only_test_pearson':full['test_pearson']-results['ligand_only']['test_pearson'],'full_minus_pocket_only_test_pearson':full['test_pearson']-results['pocket_only']['test_pearson']},'permutation_checks':{'identity_count':sum(perm[c]==c for c in perm),'cross_split_count':sum(split_map[c]!=split_map[perm[c]] for c in perm),'same_cluster_count':sum(clusters[c]==clusters[perm[c]] for c in perm),'n_permuted':len(perm)},'valid_codes':len(codes),'parse_failures':bad,'fallback_all_included':True,'seed':20260915,'note':'The permutation control preserves ligand identity, split, and a restricted pocket marginal permutation; effect is descriptive intervention evidence, not a full causal claim.'}
assert result['permutation_checks']['identity_count']==result['permutation_checks']['cross_split_count']==result['permutation_checks']['same_cluster_count']==0
assert all(math.isfinite(float(v)) for r in results.values() for v in [r['test_pearson'],r['test_rmse']])
(OUT/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)); np.savez_compressed(OUT/'predictions.npz',**preds); (OUT/'permutation.json').write_text(json.dumps(perm,indent=2)); print('DONE',json.dumps(result),flush=True)
