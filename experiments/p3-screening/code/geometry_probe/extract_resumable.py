from pathlib import Path
import json, os, time, collections, hashlib, tempfile
import numpy as np
from rdkit import Chem

ROOT=Path('/workspace/ayb/data/pdbbind/v2020')
EXP=Path('/workspace/ayb/experiments/dfe-unified-p3')
OUT=EXP/'geometry_probe'
OUT.mkdir(parents=True,exist_ok=True)
LOCK=OUT/'extract.lock'
PID=OUT/'extract.pid'
JSONL=OUT/'geometry_features.jsonl'
MAN=OUT/'manifest.json'
PROGRESS=OUT/'progress.json'
# single-writer lock
if LOCK.exists():
    old=LOCK.read_text().strip()
    try:
        os.kill(int(old),0)
        raise SystemExit(f'active lock pid={old}')
    except ProcessLookupError: pass
    except ValueError: raise SystemExit(f'invalid lock {old}')
LOCK.write_text(str(os.getpid())); PID.write_text(str(os.getpid()))
try:
    sp=json.loads((EXP/'refined_split.json').read_text())
    labels=json.loads((EXP/'refined_labels.json').read_text())
    clusters=json.loads((EXP/'refined_clusters.json').read_text())
    split_map={str(code):name for name,codes in sp['split'].items() for code in codes}
    label_map={str(r[0]):r for r in labels if isinstance(r,list) and r}
    cluster_map=clusters['code2cluster']
    bins=np.array([0,2,3,4,5,6,8,10,np.inf],dtype=np.float32)
    done=set()
    if JSONL.exists():
        with JSONL.open() as f:
            for line in f:
                try:
                    r=json.loads(line)
                    if r.get('code'): done.add(r['code'])
                except json.JSONDecodeError: pass
    bad=[]
    if PROGRESS.exists():
        try: bad=json.loads(PROGRESS.read_text()).get('bad',[])
        except: pass
    bad_map={x[0]:x[1] for x in bad if isinstance(x,list) and len(x)>=2}
    dirs=[d for d in sorted(ROOT.iterdir()) if d.is_dir() and len(d.name)==4]
    total=len(dirs); start=time.time()
    for idx,d in enumerate(dirs,1):
        code=d.name
        if code in done or code in bad_map: continue
        sdf=d/f'{code}_ligand.sdf'; pdb=d/f'{code}_pocket.pdb'
        reason=None
        if not sdf.exists() or not pdb.exists(): reason='missing'
        else:
            mol=Chem.MolFromMolFile(str(sdf),sanitize=True,removeHs=False)
            if mol is None or not mol.GetNumAtoms(): reason='ligand_parse'
        if reason:
            bad_map[code]=reason
        else:
            try:
                conf=mol.GetConformer(); lig=np.asarray([[conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y,conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],np.float32)
                poc=[]; elems=[]
                for line in pdb.read_text(errors='ignore').splitlines():
                    if line.startswith(('ATOM  ','HETATM')):
                        try:
                            poc.append(tuple(map(float,(line[30:38],line[38:46],line[46:54])))); elems.append((line[76:78].strip() or line[12:16].strip()[0]).upper())
                        except: pass
                if not poc: raise ValueError('pocket_parse')
                poc=np.asarray(poc,np.float32); dist=np.sqrt(((lig[:,None]-poc[None,:])**2).sum(-1)); le=[a.GetSymbol().upper() for a in mol.GetAtoms()]
                hist=collections.Counter()
                for i,ae in enumerate(le):
                    for j,pe in enumerate(elems): hist[f'{ae}:{pe}:{int(np.digitize(dist[i,j],bins)-1)}']+=1
                row={'code':code,'split':split_map.get(code,'unknown'),'cluster':cluster_map.get(code),'n_lig':len(le),'n_pocket':len(elems),'min_dist':round(float(dist.min()),5),'pair_hist':dict(hist),'label':label_map.get(code)}
                with JSONL.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row,separators=(',',':'))+'\n'); f.flush(); os.fsync(f.fileno())
                done.add(code)
            except Exception as e: bad_map[code]=str(e)
        # checkpoint every item; atomic manifest/progress replacement
        state={'pid':os.getpid(),'updated':time.strftime('%F %T'),'total_dirs':total,'done_valid':len(done),'done_bad':len(bad_map),'remaining':total-len(done)-len(bad_map),'bad':[[k,v] for k,v in sorted(bad_map.items())]}
        tmp=PROGRESS.with_suffix('.tmp'); tmp.write_text(json.dumps(state,separators=(',',':'))); os.replace(tmp,PROGRESS)
        if idx%100==0 or idx==total:
            print('PROGRESS',idx,total,len(done),len(bad_map),flush=True)
    state={'status':'complete','total_dirs':total,'n_valid':len(done),'n_bad':len(bad_map),'unknown_split':sum(1 for line in JSONL.open() if json.loads(line).get('split')=='unknown'),'split_counts':dict(collections.Counter(json.loads(line).get('split') for line in JSONL.open())),'bad':[[k,v] for k,v in sorted(bad_map.items())],'updated':time.strftime('%F %T')}
    tmp=MAN.with_suffix('.tmp'); tmp.write_text(json.dumps(state,separators=(',',':'))); os.replace(tmp,MAN)
    print('COMPLETE',json.dumps({k:state[k] for k in ['total_dirs','n_valid','n_bad','unknown_split','split_counts']}),flush=True)
finally:
    for p in [LOCK,PID]:
        try: p.unlink()
        except FileNotFoundError: pass
