
import json,os,sys,subprocess,time,glob
ROOT='/workspace/ayb/experiments/dfe-unified-cd100'
PY='/workspace/ayb/miniconda3/envs/zatom310/bin/python'
P2M='/workspace/ayb/Pocket2Mol'
def main():
    shard=int(sys.argv[1]); nshard=int(sys.argv[2])
    man=json.load(open(f'{ROOT}/manifest.json'))
    prog=f'{ROOT}/cleaneval_{shard}.log'
    def lg(s): open(prog,'a').write(f'[{time.strftime("%H:%M:%S")}] {s}\n')
    lg(f'START cleaneval shard={shard}/{nshard}')
    for rounds in range(600):
        did=0; pending=0
        for model in ['530k','df500k']:
            for t in man['tasks']:
                if t['idx']%nshard!=shard: continue
                pk=t['pocket_dir']; pk_dir=f'{ROOT}/{model}/{pk}'
                if os.path.exists(f'{pk_dir}/EVAL_CLEAN_DONE'): continue
                if not os.path.exists(f'{pk_dir}/DONE'):
                    pending+=1; continue
                env=dict(os.environ); env['OMP_NUM_THREADS']='4'; env['MKL_NUM_THREADS']='4'
                rc=subprocess.call([PY,f'{ROOT}/clean_eval.py',model,pk,t['pocket10'],t['ref_sdf']],
                                   cwd=P2M,env=env,
                                   stdout=open(f'{pk_dir}/clean_eval.log','w'),stderr=subprocess.STDOUT)
                ok=os.path.exists(f'{pk_dir}/EVAL_CLEAN_DONE')
                lg(f'{model} {pk} -> {"ok" if ok else "FAIL:"+str(rc)}'); did+=1
        samp_done=(len(glob.glob(f'{ROOT}/530k/*/DONE'))>=100 and len(glob.glob(f'{ROOT}/df500k/*/DONE'))>=100)
        if samp_done and pending==0 and did==0:
            lg('ALL CLEAN EVAL DONE'); break
        if did==0: time.sleep(150)
    open(f'{ROOT}/cleaneval_{shard}.DONE','w').write('done\n')
if __name__=='__main__': main()
