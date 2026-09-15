
import os,sys,json,glob,tempfile,subprocess,numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem
from posebusters import PoseBusters
from meeko import MoleculePreparation

ENVBIN='/workspace/ayb/miniconda3/envs/zatom310/bin'
ENVLIB='/workspace/ayb/miniconda3/envs/zatom310/lib'
VINA='/workspace/lsm/miniconda3/envs/AMP-vina/bin/vina'
os.environ['PATH']=ENVBIN+':'+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=ENVLIB+':'+os.environ.get('LD_LIBRARY_PATH','')

def obabel(args):
    return subprocess.run([os.path.join(ENVBIN,'obabel')]+args,capture_output=True,text=True,timeout=60,env=os.environ)

def prep_receptor(pocket_pdb):
    rq=tempfile.mktemp(suffix='.pdbqt')
    obabel([pocket_pdb,'-O',rq,'-xr'])
    return rq if os.path.exists(rq) and os.path.getsize(rq)>0 else None

def prep_ligand(mol):
    lp=tempfile.mktemp(suffix='.pdbqt')
    try:
        prep=MoleculePreparation(); prep.prepare(mol)
        try: s=prep.write_pdbqt_string()
        except Exception:
            from meeko import PDBQTWriterLegacy
            rr=PDBQTWriterLegacy.write_string(prep.setup); s=rr[0] if isinstance(rr,tuple) else rr
        open(lp,'w').write(s)
        return lp if os.path.getsize(lp)>0 else None
    except Exception:
        return None

def vina_dock(recq,lq,ctr,size=25):
    outq=lq.replace('.pdbqt','_d.pdbqt')
    cmd=[VINA,'--receptor',recq,'--ligand',lq,'--center_x',str(ctr[0]),'--center_y',str(ctr[1]),'--center_z',str(ctr[2]),
         '--size_x',str(size),'--size_y',str(size),'--size_z',str(size),'--exhaustiveness','8','--num_modes','9','--out',outq]
    try:
        res=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        out=res.stdout+'\n'+res.stderr
        scores=[]
        for line in out.split('\n'):
            parts=line.split()
            if len(parts)>=2 and parts[0].isdigit():
                try:
                    a=float(parts[1])
                    if -50<a<50: scores.append(a)
                except: pass
        try: os.unlink(outq)
        except: pass
        return scores[0] if scores else None
    except Exception:
        return None

def main():
    model=sys.argv[1]; pk=sys.argv[2]; pocket10=sys.argv[3]; ref_sdf=sys.argv[4]
    ROOT='/workspace/ayb/experiments/dfe-unified-cd100'
    pk_dir=f'{ROOT}/{model}/{pk}'
    if not os.path.exists(f'{pk_dir}/DONE'): print('nosample'); return
    # ref ligand center
    rm=Chem.MolFromMolFile(ref_sdf,sanitize=False)
    rc=rm.GetConformer(); rco=np.array([list(rc.GetAtomPosition(i)) for i in range(rm.GetNumAtoms())]); ctr=rco.mean(0).tolist()
    # gather generated mols
    sdfs=sorted(glob.glob(f'{pk_dir}/run/**/*.sdf',recursive=True))
    recq=prep_receptor(pocket10)
    pb=PoseBusters(config='mol')
    results=[]
    for s in sdfs:
        m=Chem.MolFromMolFile(s)
        if m is None:
            m2=Chem.MolFromMolFile(s,sanitize=False)
            results.append({'sdf':os.path.basename(s),'valid':False}); continue
        rec={'sdf':os.path.basename(s),'valid':True,'smiles':Chem.MolToSmiles(m)}
        try:
            rec['qed']=float(QED.qed(m)); rec['mw']=float(Descriptors.MolWt(m))
            rec['num_atoms']=int(m.GetNumAtoms()); rec['logp']=float(Descriptors.MolLogP(m))
            rec['num_frags_generated']=len(Chem.GetMolFrags(m))
        except Exception as e: rec['prop_err']=str(e)
        # PB direct on generated conformer
        try:
            res=pb.bust(mol_pred=m); row=res.iloc[0]
            rec['pb_pass']=bool(row.all())
            rec['pb_fails']=[c for c in res.columns if not bool(row[c])]
        except Exception as e:
            rec['pb_pass']=None; rec['pb_err']=str(e)
        # Vina dock
        if recq:
            mh=Chem.AddHs(m,addCoords=True)
            lq=prep_ligand(mh)
            if lq:
                rec['docking_score']=vina_dock(recq,lq,ctr)
                try: os.unlink(lq)
                except: pass
        results.append(rec)
    if recq:
        try: os.unlink(recq)
        except: pass
    out=f'{pk_dir}/eval_clean.json'
    json.dump({'model':model,'pocket':pk,'center':ctr,'n':len(results),'results':results},open(out,'w'))
    open(f'{pk_dir}/EVAL_CLEAN_DONE','w').write(f'{len(results)}\n')
    ok=[r for r in results if r.get('valid')]
    ds=[r['docking_score'] for r in ok if isinstance(r.get('docking_score'),(int,float))]
    pbp=sum(1 for r in ok if r.get('pb_pass') is True)
    print(f'{model}/{pk}: n={len(ok)} dock_mean={np.mean(ds):.3f} dock_best={min(ds):.3f} pb={pbp}/{len(ok)}' if ds else f'{model}/{pk}: no dock')

if __name__=='__main__': main()
