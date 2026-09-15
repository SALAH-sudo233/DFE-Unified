import torch,json,math,os
from collections import defaultdict
paths={'500k':'/workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt','528k':'/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/528000.pt','530k':'/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/530000.pt'}
out='/workspace/ayb/experiments/dfe-unified-trackA/diagnostics_530k_param_drift'
os.makedirs(out,exist_ok=True)
ck={}
for tag,p in paths.items():
 d=torch.load(p,map_location='cpu',weights_only=False);ck[tag]=d['model'];print(tag,d.get('iteration'),len(d['model']))
rows=[]
for name,w0 in ck['500k'].items():
 a=w0.detach().float(); group='encoder.'+name.split('.')[2] if name.startswith('encoder.interactions.') else name.split('.')[0]
 r={'name':name,'group':group,'numel':a.numel(),'base_norm':float(torch.linalg.vector_norm(a))}
 for tag in ['528k','530k']:
  diff=ck[tag][name].detach().float()-a;r[tag+'_diff_l2']=float(torch.linalg.vector_norm(diff));r[tag+'_rel']=float(torch.linalg.vector_norm(diff)/(torch.linalg.vector_norm(a)+1e-12));r[tag+'_max_abs']=float(diff.abs().max())
 rows.append(r)
for tag in ['528k','530k']:
 print('===',tag,'top relative changes===')
 for r in sorted(rows,key=lambda x:x[tag+'_rel'],reverse=True)[:25]:print(r['group'],r['name'],'rel=%.6g max=%.6g'%(r[tag+'_rel'],r[tag+'_max_abs']))
 print('=== group aggregate===');groups=defaultdict(lambda:[0.,0.,0])
 for r in rows:g=groups[r['group']];g[0]+=r[tag+'_diff_l2']**2;g[1]+=r['base_norm']**2;g[2]+=r['numel']
 for k,v in sorted(groups.items(),key=lambda kv:kv[1][0],reverse=True):print(k,'diff_l2=%.6g rel_group=%.6g numel=%d'%(math.sqrt(v[0]),math.sqrt(v[0])/(math.sqrt(v[1])+1e-12),v[2]))
json.dump({'paths':paths,'rows':rows},open(out+'/param_drift.json','w'),indent=2)
