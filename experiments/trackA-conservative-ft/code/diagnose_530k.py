import paramiko, os, json, re, statistics, math
HOST='21.tcp.cpolar.top';PORT=10431;USER='ayb';PW=os.environ['AIDD_PW']
BASE='/workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt'
FT='/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/530000.pt'
LOGB='/workspace/ayb/Pocket2Mol/logs/train_df_2026_08_21__16_00_50/log.txt'
LOGF='/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/train.log'
PY='/workspace/ayb/miniconda3/envs/zatom310/bin/python'
remote=r'''
import torch,json,re,statistics,math,sys,os
base,ft,logb,logf=sys.argv[1:]
def ck(path):
 d=torch.load(path,map_location='cpu',weights_only=False)
 out={'path':path,'top_keys':list(d.keys()) if isinstance(d,dict) else None}
 if isinstance(d,dict):
  for k in ['iteration','iter','step','config']:
   if k in d:
    v=d[k]
    out[k]=v if k!='config' else str(v)
  for k in ['optimizer','optimizer_state_dict','scheduler','scheduler_state_dict']:
   if k in d:
    v=d[k]
    if k.startswith('optimizer') and isinstance(v,dict):
     out[k+'_keys']=list(v.keys())
     st=v.get('state_dict',v)
     out[k+'_param_groups']=st.get('param_groups') if isinstance(st,dict) else None
    else: out[k+'_repr']=str(v)[:1000]
 return out
def logs(path):
 text=open(path,errors='replace').read()
 val=[]; train=[]
 pat=re.compile(r'\[(Train|Validate)\]\s+Iter\s+(\d+)\s+\|\s+Loss\s+([-+0-9.eE]+)\s+\|\s+Loss\(Fron\)\s+([-+0-9.eE]+)\s+\|\s+Loss\(Pos\)\s+([-+0-9.eE]+)\s+\|\s+Loss\(Cls\)\s+([-+0-9.eE]+)\s+\|\s+Loss\(Edge\)\s+([-+0-9.eE]+)\s+\|\s+Loss\(Real\)\s+([-+0-9.eE]+)\s+\|\s+Loss\(Fake\)\s+([-+0-9.eE]+)(?:\s+\|\s+Loss\(Surf\)\s+([-+0-9.eE]+))?')
 for m in pat.finditer(text):
  typ=m.group(1); vals=[float(m.group(i)) for i in range(3,10)]; vals.append(float(m.group(10)) if m.group(10) else None)
  row={'iter':int(m.group(2)),'loss':vals[0],'fron':vals[1],'pos':vals[2],'cls':vals[3],'edge':vals[4],'real':vals[5],'fake':vals[6],'surf':vals[7]}
  (val if typ=='Validate' else train).append(row)
 return {'n_train':len(train),'n_val':len(val),'train_first':train[:3],'train_last':train[-3:],'val':val}
print(json.dumps({'baseline_ckpt':ck(base),'ft_ckpt':ck(ft),'baseline_log':logs(logb),'ft_log':logs(logf)},ensure_ascii=False))
'''
import base64
b64=base64.b64encode(remote.encode()).decode()
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(HOST,port=PORT,username=USER,password=PW,timeout=40,banner_timeout=40)
cmd=f"echo {b64}|base64 -d >/tmp/diag_ck.py && {PY} /tmp/diag_ck.py {BASE} {FT} {LOGB} {LOGF}"
si,so,se=c.exec_command(cmd,timeout=180);out=so.read().decode(errors='replace');err=se.read().decode(errors='replace');c.close()
print(out)
# stderr intentionally omitted from machine-readable output
