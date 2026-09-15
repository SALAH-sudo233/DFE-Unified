from pathlib import Path
import json,os,time,collections,tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import BondType
ROOT=Path('/workspace/ayb/data/pdbbind/v2020'); EXP=Path('/workspace/ayb/experiments/dfe-unified-p3'); OUT=EXP/'geometry_probe'; JSONL=OUT/'geometry_features.jsonl'; PROG=OUT/'progress.json'; MAN=OUT/'manifest.json'; LOCK=OUT/'recover.lock'
if LOCK.exists():
 try: os.kill(int(LOCK.read_text()),0); raise SystemExit('active recovery lock')
 except ProcessLookupError: pass
LOCK.write_text(str(os.getpid()))
try:
 sp=json.loads((EXP/'refined_split.json').read_text()); labels=json.loads((EXP/'refined_labels.json').read_text()); clusters=json.loads((EXP/'refined_clusters.json').read_text())
 split_map={str(c):n for n,cs in sp['split'].items() for c in cs}; label_map={str(r[0]):r for r in labels if isinstance(r,list) and r}; cmap=clusters['code2cluster']; bins=np.array([0,2,3,4,5,6,8,10,np.inf],np.float32)
 done=set();
 if JSONL.exists():
  for line in JSONL.open():
   try: done.add(json.loads(line)['code'])
   except: pass
 old=json.loads(PROG.read_text()) if PROG.exists() else {}; bad={x[0]:x[1] for x in old.get('bad',[]) if isinstance(x,list) and len(x)>1}; recovered=[]; still=[]
 for n,code in enumerate(sorted(list(bad)),1):
  if code in done: continue
  d=ROOT/code; sdf=d/f'{code}_ligand.sdf'; pdb=d/f'{code}_pocket.pdb'
  try:
   mols=Chem.SDMolSupplier(str(sdf),sanitize=False,removeHs=False)
   mol=next((m for m in mols if m is not None),None)
   if mol is None: raise ValueError('unreadable_sdf')
   for b in mol.GetBonds():
    if b.GetIsAromatic() or b.GetBondType()==BondType.AROMATIC: b.SetIsAromatic(False); b.SetBondType(BondType.SINGLE)
   for a in mol.GetAtoms(): a.SetIsAromatic(False)
   Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
   conf=mol.GetConformer(); lig=np.asarray([[conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y,conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],np.float32)
   poc=[]; elems=[]
   for line in pdb.read_text(errors='ignore').splitlines():
    if line.startswith(('ATOM  ','HETATM')):
     try: poc.append(tuple(map(float,(line[30:38],line[38:46],line[46:54])))); elems.append((line[76:78].strip() or line[12:16].strip()[0]).upper())
     except: pass
   if not poc: raise ValueError('pocket_parse')
   poc=np.asarray(poc,np.float32); dist=np.sqrt(((lig[:,None]-poc[None,:])**2).sum(-1)); le=[a.GetSymbol().upper() for a in mol.GetAtoms()]; h=collections.Counter()
   for i,ae in enumerate(le):
    for j,pe in enumerate(elems): h[f'{ae}:{pe}:{int(np.digitize(dist[i,j],bins)-1)}']+=1
   row={'code':code,'split':split_map.get(code,'unknown'),'cluster':cmap.get(code),'n_lig':len(le),'n_pocket':len(elems),'min_dist':round(float(dist.min()),5),'pair_hist':dict(h),'label':label_map.get(code),'geometry_recovery':'aromatic_to_single'}
   with JSONL.open('a') as f: f.write(json.dumps(row,separators=(',',':'))+'\n'); f.flush(); os.fsync(f.fileno())
   recovered.append(code); done.add(code)
  except Exception as e: still.append([code,str(e)[:160]])
  if n%25==0 or n==len(bad):
   state={'status':'recovering','updated':time.strftime('%F %T'),'done_valid':len(done),'recovered':len(recovered),'remaining_failed':len(still),'still_failed':still,'original_failed':len(bad)}; tmp=PROG.with_suffix('.recover.tmp'); tmp.write_text(json.dumps(state,separators=(',',':'))); os.replace(tmp,PROG); print('PROGRESS',n,len(bad),len(recovered),len(still),flush=True)
 # final manifest counts directly from JSONL
 counts=collections.Counter(); total=0
 for line in JSONL.open():
  r=json.loads(line); total+=1; counts[r.get('split','unknown')]+=1
 state={'status':'complete','total_dirs':len([d for d in ROOT.iterdir() if d.is_dir() and len(d.name)==4]),'n_valid':total,'n_bad':len(still),'unknown_split':counts.get('unknown',0),'split_counts':dict(counts),'recovered':recovered,'still_failed':still,'updated':time.strftime('%F %T')}; tmp=MAN.with_suffix('.recover.tmp'); tmp.write_text(json.dumps(state,separators=(',',':'))); os.replace(tmp,MAN); print('COMPLETE',json.dumps({k:state[k] for k in ['n_valid','n_bad','unknown_split','split_counts']}),flush=True)
finally:
 try: LOCK.unlink()
 except FileNotFoundError: pass
