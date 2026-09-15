from pathlib import Path
import subprocess,hashlib,json
b=Path('/workspace/ayb/Pocket2Mol')
print(subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
for n in ['sample.py','configs/sample.yml','utils/datasets/__init__.py']:
 p=b/n; lines=p.read_text().splitlines();print('FILE',n)
 if n=='sample.py': lines=lines[185:]
 print('\n'.join(lines))
for p in [b/'ckpt',b/'data']:
 print('DIR',str(p),[(x.name,x.stat().st_size) for x in p.iterdir() if x.is_file()])
import torch
s=torch.load(b/'data/split_by_name.pt',weights_only=False)
print('SPLIT', {k:len(v) for k,v in s.items()},'firsttest',s['test'][:3])
for p in [b/'ckpt/pretrained_Pocket2Mol.pt',b/'logs/checkpoints/500000.pt',Path('/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/530000.pt')]:
 print('CKPT',str(p),'exists',p.exists())
 if p.exists():
  c=torch.load(p,map_location='cpu',weights_only=False);print('keys',list(c),'modelconf',c['config'].model,'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
