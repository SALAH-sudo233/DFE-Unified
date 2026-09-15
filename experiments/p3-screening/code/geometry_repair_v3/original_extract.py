from pathlib import Path
import json, csv, hashlib, time, os, sys
import numpy as np
from rdkit import Chem

ROOT=Path('/workspace/ayb/data/pdbbind/v2020')
EXP=Path('/workspace/ayb/experiments/dfe-unified-p3')
OUT=EXP/'geometry_probe'
OUT.mkdir(parents=True,exist_ok=True)
# Load existing split/labels; inspect flexible key layouts.
split=json.loads((EXP/'refined_split.json').read_text())
labels=json.loads((EXP/'refined_labels.json').read_text())
clusters=json.loads((EXP/'refined_clusters.json').read_text())
def flatten(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if isinstance(v,list):
    for z in v: yield str(z),k
   elif isinstance(v,dict):
    yield from flatten(v)
forbidden=set()
# Create split map from common layouts.
split_map={}
if isinstance(split,dict):
 for k,v in split.items():
  if isinstance(v,list):
   for code in v: split_map[str(code)]=k
  elif isinstance(v,dict):
   for code,sp in v.items(): split_map[str(code)]=str(sp)
rows=[]; bad=[]
for d in sorted(ROOT.iterdir()):
 if not d.is_dir() or len(d.name)!=4: continue
 code=d.name; sdf=d/f'{code}_ligand.sdf'; pdb=d/f'{code}_pocket.pdb'
 if not sdf.exists() or not pdb.exists(): bad.append((code,'missing')); continue
 mol=Chem.MolFromMolFile(str(sdf),sanitize=True,removeHs=False)
 if mol is None or mol.GetNumAtoms()==0: bad.append((code,'ligand_parse')); continue
 conf=mol.GetConformer(); lig=np.array([[conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y,conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],dtype=np.float32)
 with open(pdb,errors='ignore') as f: lines=f.readlines()
 pocket=[]; elems=[]
 for line in lines:
  if line.startswith(('ATOM  ','HETATM')):
   try:
    x,y,z=map(float,(line[30:38],line[38:46],line[46:54])); el=line[76:78].strip() or line[12:16].strip()[0]
    pocket.append((x,y,z)); elems.append(el.upper())
   except: pass
 if not pocket: bad.append((code,'pocket_parse')); continue
 poc=np.array(pocket,dtype=np.float32)
 dist=np.sqrt(((lig[:,None,:]-poc[None,:,:])**2).sum(-1))
 # Stable fixed-length geometry summary: element-pair distance histograms, 8 bins.
 le=[a.GetSymbol().upper() for a in mol.GetAtoms()]
 bins=np.array([0,2,3,4,5,6,8,10,np.inf],dtype=np.float32)
 keys=[]; vals=[]
 for a,ae in enumerate(le):
  for b,pe in enumerate(elems):
   ix=np.digitize(dist[a],bins,right=False)-1
   for bi in range(8):
    n=int((ix==bi).sum())
    if n: keys.append(f'{ae}:{pe}:{bi}'); vals.append(n)
 rows.append({'code':code,'split':split_map.get(code,'unknown'),'n_lig':len(le),'n_pocket':len(elems),'min_dist':float(dist.min()),'pair_hist_keys':keys,'pair_hist_vals':vals,'label':labels.get(code) if isinstance(labels,dict) else None,'cluster':clusters.get(code) if isinstance(clusters,dict) else None})
# Write compact JSONL and manifest; no coordinates retained.
(OUT/'geometry_features.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
(OUT/'manifest.json').write_text(json.dumps({'n_valid':len(rows),'n_bad':len(bad),'bad':bad[:200],'root':str(ROOT),'bins':[0,2,3,4,5,6,8,10,'inf']},indent=2))
print('VALID',len(rows),'BAD',len(bad),'UNKNOWN_SPLIT',sum(r['split']=='unknown' for r in rows))
print('OUT',OUT)
