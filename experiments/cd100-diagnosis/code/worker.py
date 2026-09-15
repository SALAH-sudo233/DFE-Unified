
import json, os, sys, subprocess, time, glob

ROOT='/workspace/ayb/experiments/dfe-unified-cd100'
PY='/workspace/ayb/miniconda3/envs/zatom310/bin/python'
P2M='/workspace/ayb/Pocket2Mol'
CKPT={
  '530k':'/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/530000.pt',
  'df500k':'/workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt',
}

def make_cfg(path, ckpt):
    open(path,'w').write(f"""model:
  checkpoint: {ckpt}
sample:
  seed: 2021
  num_samples: 100
  beam_size: 100
  max_steps: 100
  threshold:
    focal_threshold: 0.5
    pos_threshold: 0.25
    element_threshold: 0.3
    hasatom_threshold: 0.6
    bond_threshold: 0.4
""")

def run_one(model, task, gpu):
    pdir=task['pocket_dir']
    od=os.path.join(ROOT, model, pdir)
    done=os.path.join(od,'DONE')
    if os.path.exists(done):
        return 'skip'
    os.makedirs(od, exist_ok=True)
    cfg=os.path.join(od,'sample.yml'); make_cfg(cfg, CKPT[model])
    center=",".join(str(x) for x in task['center'])
    log=os.path.join(od,'sample.log')
    cmd=(f"cd {P2M} && CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 timeout 2400 {PY} sample_for_pdb_nodisk.py "
         f"--pdb_path {task['pocket10']} --center={center} --bbox_size 23.0 "
         f"--config {cfg} --device cuda --outdir {od}/run > {log} 2>&1")
    rc=subprocess.call(cmd, shell=True)
    nsdf=len(glob.glob(os.path.join(od,'run','**','*.sdf'), recursive=True))
    open(os.path.join(od,'result.json'),'w').write(json.dumps({'model':model,'pocket':pdir,'rc':rc,'n_sdf':nsdf,'gpu':gpu,'ts':time.time()}))
    if nsdf>0:
        open(done,'w').write(f'{nsdf}\n')
        # drop bulky samples_all.pt if any (safety)
        for f in glob.glob(os.path.join(od,'run','**','samples_all.pt'), recursive=True):
            try: os.remove(f)
            except: pass
        return f'ok:{nsdf}'
    return f'FAIL:rc={rc}'

def main():
    gpu=sys.argv[1]; shard=int(sys.argv[2]); nshard=int(sys.argv[3])
    manifest=json.load(open(os.path.join(ROOT,'manifest.json')))
    tasks=manifest['tasks']
    prog=os.path.join(ROOT,f'worker_gpu{gpu}.log')
    def lg(s):
        open(prog,'a').write(f'[{time.strftime("%H:%M:%S")}] {s}\n')
    lg(f'START gpu={gpu} shard={shard}/{nshard}')
    # models order: 530k first, then df500k baseline
    for model in ['530k','df500k']:
        for t in tasks:
            if t['idx'] % nshard != shard:
                continue
            r=run_one(model, t, gpu)
            lg(f'{model} idx{t["idx"]} {t["pocket_dir"]} -> {r}')
    lg('DONE ALL')
    open(os.path.join(ROOT,f'worker_gpu{gpu}.DONE'),'w').write('done\n')

if __name__=='__main__':
    main()
