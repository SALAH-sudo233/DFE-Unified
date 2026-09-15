import json,glob,os
for f in glob.glob("/workspace/ayb/Pocket2Mol/outputs/multi_pocket_eval*.json")+glob.glob("/workspace/ayb/Pocket2Mol/**/fair_comparison*.json",recursive=True):
    print("FILE",f)
    try:
        d=json.load(open(f))
    except Exception as e:
        print(" err",e); continue
    items=d.items() if isinstance(d,dict) else list(enumerate(d))
    for k,v in items:
        if isinstance(v,dict):
            print(" ",k,"model=",v.get("model"),"dock=",v.get("docking_mean"),"qed=",v.get("qed_mean"),"pb%=",v.get("posebuster_pass_pct"),"mw=",v.get("mw_mean"),"n=",v.get("n_total"))
