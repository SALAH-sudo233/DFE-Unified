import json,os,subprocess,glob
ROOT='/workspace/ayb/experiments/dfe-unified-cd100'; PY='/workspace/ayb/miniconda3/envs/zatom310/bin/python'
x=json.load(open(ROOT+'/diagnosis_pilot_v2.json'))
for model in ['530k','df500k']:
 for e in x['entries']:
  for seed in x['seeds']:
   rel=f'diagnosis_pilot_v2/{model}/idx{e["manifest_idx"]}/seed{seed}'
   od=f'{ROOT}/{rel}'
   if not os.path.exists(od+'/DONE') or os.path.exists(od+'/EVAL_CLEAN_DONE'): continue
   sdfs=glob.glob(od+'/run/**/*.sdf',recursive=True)
   if not sdfs: continue
   with open(od+'/eval.log','w') as log:
    r=subprocess.run([PY,f'{ROOT}/clean_eval.py',model,rel,e['pocket10'],e['ref_sdf']],cwd='/workspace/ayb/Pocket2Mol',stdout=log,stderr=subprocess.STDOUT,timeout=1800)
   open(od+'/eval_process.rc','w').write(str(r.returncode))
print('CLEAN_PILOT_DONE')
