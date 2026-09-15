import json, os, subprocess, glob
ROOT='/workspace/ayb/experiments/dfe-unified-cd100'
PY='/workspace/ayb/miniconda3/envs/zatom310/bin/python'
P2M='/workspace/ayb/Pocket2Mol'
CKPT={'530k':'/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/530000.pt','df500k':'/workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt'}
PILOT=json.load(open(ROOT+'/diagnosis_pilot_v2.json'))

def write_cfg(path, ckpt, seed, n):
    text = f'''model:
  checkpoint: {ckpt}
sample:
  seed: {seed}
  num_samples: {n}
  beam_size: 100
  max_steps: 100
  threshold:
    focal_threshold: 0.5
    pos_threshold: 0.25
    element_threshold: 0.3
    hasatom_threshold: 0.6
    bond_threshold: 0.4
'''
    open(path,'w').write(text)

for model in ['530k','df500k']:
    for e in PILOT['entries']:
        for seed in PILOT['seeds']:
            od=f"{ROOT}/diagnosis_pilot_v2/{model}/idx{e['manifest_idx']}/seed{seed}"
            done=od+'/DONE'
            if os.path.exists(done): continue
            os.makedirs(od,exist_ok=True)
            c=od+'/sample.yml'; write_cfg(c,CKPT[model],seed,PILOT['num_samples'])
            center=','.join(map(str,e['center'])); log=od+'/sample.log'
            cmd=(f"cd {P2M} && CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
                 f"timeout 1800 {PY} sample_for_pdb_nodisk.py --pdb_path {e['pocket10']} --center={center} "
                 f"--bbox_size 23.0 --config {c} --device cuda --outdir {od}/run > {log} 2>&1")
            rc=subprocess.call(cmd,shell=True)
            n=len(glob.glob(od+'/run/**/*.sdf',recursive=True))
            json.dump({'model':model,'idx':e['manifest_idx'],'seed':seed,'rc':rc,'n_sdf':n},open(od+'/result.json','w'))
            if n: open(done,'w').write(str(n)+'\n')
print('PILOT_DONE')
