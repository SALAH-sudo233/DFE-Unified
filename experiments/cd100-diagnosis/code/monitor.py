
import time, glob, os, subprocess
ROOT='/workspace/ayb/experiments/dfe-unified-cd100'
def sh(c):
    return subprocess.run(c,shell=True,capture_output=True,text=True).stdout
log=os.path.join(ROOT,'monitor.log')
for _ in range(200):  # ~200*20min coverage
    d530=len(glob.glob(f"{ROOT}/530k/*/DONE"))
    d500=len(glob.glob(f"{ROOT}/df500k/*/DONE"))
    alive=sh("ps -u ayb -o cmd | grep 'cd100/worker.py' | grep -v grep | wc -l").strip()
    fails=sh(f"grep -h FAIL {ROOT}/worker_gpu*.log 2>/dev/null | wc -l").strip()
    la=sh("cat /proc/loadavg").strip().split()[0]
    disk=sh("df -h /workspace | tail -1").split()[3]
    ts=time.strftime('%m-%d %H:%M')
    line=f"[{ts}] 530k={d530}/100 df500k={d500}/100 workers={alive} fails={fails} load={la} freedisk={disk}\n"
    open(log,'a').write(line)
    # stop when all done and no workers
    if d530>=100 and d500>=100:
        open(log,'a').write(f"[{ts}] ALL SAMPLING DONE\n"); break
    time.sleep(1200)
