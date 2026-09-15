
import json, os, sys, subprocess, time, glob
from rdkit import Chem

ROOT='/workspace/ayb/experiments/dfe-unified-cd100'
ENVBIN='/workspace/ayb/miniconda3/envs/zatom310/bin'
ENVLIB='/workspace/ayb/miniconda3/envs/zatom310/lib'
PY=f'{ENVBIN}/python'
P2M='/workspace/ayb/Pocket2Mol'

def merge(pk_dir):
    sdfs=sorted(glob.glob(f"{pk_dir}/run/**/*.sdf",recursive=True))
    merged=f"{pk_dir}/merged.sdf"
    w=Chem.SDWriter(merged); n=0
    for s in sdfs:
        m=Chem.MolFromMolFile(s,sanitize=False)
        if m is not None:
            try: w.write(m); n+=1
            except: pass
    w.close()
    return merged,n

def eval_one(model, pk_name, pocket10):
    pk_dir=f"{ROOT}/{model}/{pk_name}"
    if not os.path.exists(f"{pk_dir}/DONE"): return 'nosample'
    if os.path.exists(f"{pk_dir}/EVAL_DONE"): return 'skip'
    merged,n=merge(pk_dir)
    if n==0: return 'empty'
    out=f"{pk_dir}/docking_results.json"
    env=dict(os.environ)
    env['PATH']=ENVBIN+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=ENVLIB+':'+env.get('LD_LIBRARY_PATH','')
    env['OMP_NUM_THREADS']='4'; env['MKL_NUM_THREADS']='4'
    cmd=[PY,'evaluate_docking_fixed.py','--sdf',merged,'--pocket_pdb',pocket10,'--output',out]
    rc=subprocess.call(cmd, cwd=P2M, env=env,
                       stdout=open(f"{pk_dir}/eval.log",'w'), stderr=subprocess.STDOUT)
    if os.path.exists(out):
        open(f"{pk_dir}/EVAL_DONE",'w').write(f'{n}\n')
        return f'ok:{n}'
    return f'FAIL:rc={rc}'

def main():
    shard=int(sys.argv[1]); nshard=int(sys.argv[2])
    man=json.load(open(f"{ROOT}/manifest.json"))
    prog=f"{ROOT}/evalworker_{shard}.log"
    def lg(s): open(prog,'a').write(f'[{time.strftime("%H:%M:%S")}] {s}\n')
    lg(f'START eval shard={shard}/{nshard}')
    # loop until all sampling done AND all evaluated
    for rounds in range(500):
        pending=0; did=0
        for model in ['530k','df500k']:
            for t in man['tasks']:
                if t['idx']%nshard!=shard: continue
                pk=t['pocket_dir']
                pk_dir=f"{ROOT}/{model}/{pk}"
                if os.path.exists(f"{pk_dir}/EVAL_DONE"): continue
                if not os.path.exists(f"{pk_dir}/DONE"):
                    pending+=1; continue
                res=eval_one(model, pk, t['pocket10'])
                lg(f'{model} {pk} -> {res}'); did+=1
        # exit condition: sampling fully done and nothing pending
        samp_done = (len(glob.glob(f"{ROOT}/530k/*/DONE"))>=100 and len(glob.glob(f"{ROOT}/df500k/*/DONE"))>=100)
        if samp_done and pending==0 and did==0:
            lg('ALL EVAL DONE'); break
        if did==0:
            time.sleep(180)  # wait for more samples
    open(f"{ROOT}/evalworker_{shard}.DONE",'w').write('done\n')

if __name__=='__main__': main()
