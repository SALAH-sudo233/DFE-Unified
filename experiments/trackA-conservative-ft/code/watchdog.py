"""Bounded monitoring of an already running sampler; never starts GPU work."""
import os, signal, time, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/workspace/ayb/experiments/dfe-unified-trackA/eval_530k_1a9q')
PID=2676287
EXPECTED=str(ROOT/'sample_530k.yml')
P=Path(f'/proc/{PID}')
def identity():
 try:
  cmd=(P/'cmdline').read_bytes().split(b'\0')
  if b'sample_for_pdb_nodisk.py' not in cmd or EXPECTED.encode() not in cmd:return None
  return (P/'stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None

def save(name,data):
 p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def snapshot(state):
 t=(ROOT/'sample.log').read_text(errors='replace')
 pools=[s for s in t.splitlines() if '[Pool]' in s]
 sdfs=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'run').rglob('*.sdf'))
 d={'time_utc':datetime.now(timezone.utc).isoformat(),'pid':PID,'state':state,'last_pool':pools[-1] if pools else None,'sdf_count':len(sdfs),'sdf_paths':sdfs,'traceback': 'Traceback (most recent call last)' in t,'exit_code':None,'note':'Attached monitor cannot recover sampler exit code; SDF count is not requested-attempt success rate.'}
 save('watch_status.json',d);return d

start=identity()
# Fixed 90-minute budget from this attempt's config creation, not per poll.
deadline=(ROOT/'sample_530k.yml').stat().st_mtime+5400
save('watch_contract.json',{'pid':PID,'proc_start_ticks':start,'deadline_epoch':deadline,'timeout_action':'SIGINT, then SIGTERM after 60s if same process','model':'530000.pt','scope':EXPECTED})
while start is not None and identity()==start:
 snapshot('running')
 if time.time()>=deadline:
  os.kill(PID,signal.SIGINT)
  for _ in range(12):
   if identity()!=start:break
   time.sleep(5)
  if identity()==start:os.kill(PID,signal.SIGTERM)
  snapshot('budget_exhausted');break
 time.sleep(30)
else:
 snapshot('process_exited_unverified')
