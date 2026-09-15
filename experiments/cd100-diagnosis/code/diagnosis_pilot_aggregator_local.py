#!/usr/bin/env python3
"""Aggregate CrossDocked diagnosis pilot outputs without modifying inputs."""
import argparse,csv,json,math,random,statistics
from pathlib import Path

MODELS=("530k","df500k"); SEEDS=(2020,2021,2022)
BINS=((8,11),(12,14),(15,17),(18,20),(21,99))

def finite(x): return isinstance(x,(int,float)) and math.isfinite(float(x))
def last_nonempty(p):
    try: lines=p.read_text(errors='replace').splitlines()
    except FileNotFoundError:return ''
    return next((x.strip() for x in reversed(lines) if x.strip()),'')
def load_entry(d):
    j=d/'eval_clean.json'; done=d/'EVAL_CLEAN_DONE'; log=d/'eval.log'
    if not j.exists() or not done.exists(): return None,'missing_eval_or_done'
    last=last_nonempty(log)
    if last.lower().endswith('nosample'): return None,'eval_log_nosample'
    try: obj=json.loads(j.read_text())
    except Exception as e: return None,'bad_json:'+str(e)
    rows=obj.get('results') if isinstance(obj,dict) else obj
    if not isinstance(rows,list): return None,'missing_results'
    if isinstance(obj,dict) and 'n' in obj and obj['n'] != len(rows):
        raise AssertionError(f'{d}: JSON n={obj["n"]} != len(results)={len(rows)}')
    valid=[]
    for r in rows:
        if not isinstance(r,dict) or not finite(r.get('docking_score')): continue
        score=float(r['docking_score']); ha=r.get('num_atoms')
        if not finite(ha) or float(ha)<=0: continue
        valid.append(r)
    if not valid:return None,'no_valid_docking_score'
    return valid,None

def mean(xs): return statistics.mean(xs) if xs else None
def median(xs): return statistics.median(xs) if xs else None
def metric(rows):
    scores=[float(r['docking_score']) for r in rows]; ha=[float(r['num_atoms']) for r in rows]
    q=[float(r['qed']) for r in rows if finite(r.get('qed'))]
    le=[-scores[i]/ha[i] for i in range(len(rows))]
    pb=[bool(r['pb_pass']) for r in rows if 'pb_pass' in r]
    smiles=[str(r['smiles']) for r in rows if r.get('smiles') is not None]
    return {'n':len(rows),'vina_mean':mean(scores),'vina_median':median(scores),'vina_best':min(scores) if scores else None,
      'pb_pass_rate':mean([int(x) for x in pb]) if pb else None,'pb_n':len(pb),
      'qed':mean(q),'heavy_atoms':mean(ha),'ligand_efficiency':mean(le),
      'unique_ratio':len(set(smiles))/len(smiles) if smiles else None}
def bootstrap(xs,B=20000,seed=20260914):
    xs=[float(x) for x in xs if finite(x)]
    if not xs:return [None,None,None]
    rng=random.Random(seed); vals=[]
    for _ in range(B): vals.append(mean([xs[rng.randrange(len(xs))] for _ in xs]))
    vals.sort(); return [vals[int(.025*len(vals))],median(vals),vals[int(.975*len(vals))-1]]
def fmt(x): return '' if x is None else f'{x:.6g}'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text()); tasks=manifest.get('tasks',[])
    # Pilot scope is defined by directories present under diagnosis_pilot_v2, not full 100-pocket manifest.
    pilot_idxs=set()
    for m in MODELS:
        md=args.root/m
        if md.exists():
            for p in md.glob('idx*'):
                if p.is_dir() and p.name[3:].isdigit(): pilot_idxs.add(int(p.name[3:]))
    if len(pilot_idxs) != 12:
        raise AssertionError(f'pilot manifest_idx count={len(pilot_idxs)}, expected 12: {sorted(pilot_idxs)}')
    # Validate every pilot index exists in the authoritative manifest.
    manifest_idxs={int(t['idx']) for t in tasks}
    if not pilot_idxs <= manifest_idxs:
        raise AssertionError(f'pilot idx absent from manifest: {sorted(pilot_idxs-manifest_idxs)}')
    expected={(m,idx,s) for m in MODELS for idx in pilot_idxs for s in SEEDS}
    found=[]; missing=[]; rejected=[]
    for m,idx,s in sorted(expected):
        d=args.root/m/f'idx{idx}'/f'seed{s}'; rows,reason=load_entry(d)
        if rows is None: rejected.append({'model':m,'manifest_idx':idx,'seed':s,'reason':reason}); continue
        found.append((m,idx,s,metric(rows),rows))
    args.outdir.mkdir(parents=True,exist_ok=True)
    keys=['model','manifest_idx','seed','n','vina_mean','vina_median','vina_best','pb_pass_rate','pb_n','qed','heavy_atoms','ligand_efficiency','unique_ratio']
    with (args.outdir/'detail.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for m,i,s,x,_ in found:w.writerow({'model':m,'manifest_idx':i,'seed':s,**x})
    # model/seed/pocket macro means
    macro=[]
    metric_names=['vina_mean','vina_median','vina_best','pb_pass_rate','qed','heavy_atoms','ligand_efficiency','unique_ratio']
    for m in MODELS:
      for s in SEEDS:
        es=[x for mm,i,ss,x,r in found if mm==m and ss==s]
        row={'model':m,'seed':s,'n_pockets':len(es)}
        for k in metric_names: row[k]=mean([e[k] for e in es if finite(e.get(k))])
        macro.append(row)
    with (args.outdir/'macro_seed.csv').open('w',newline='',encoding='utf-8') as f:
      ks=['model','seed','n_pockets']+metric_names;w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(macro)
    # paired deltas, only exact model/seed/idx matches
    pair=[]
    for i in sorted({i for _,i,_,_,_ in found}):
      for s in SEEDS:
        a=next((x for m,ii,ss,x,_ in found if m=='530k' and ii==i and ss==s),None)
        b=next((x for m,ii,ss,x,_ in found if m=='df500k' and ii==i and ss==s),None)
        if a and b: pair.append({'manifest_idx':i,'seed':s,**{k:(a[k]-b[k] if finite(a.get(k)) and finite(b.get(k)) else None) for k in metric_names}})
    with (args.outdir/'paired_delta.csv').open('w',newline='',encoding='utf-8') as f:
      ks=['manifest_idx','seed']+metric_names;w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(pair)
    # strata
    strata=[]
    for m,i,s,x,rows in found:
      for lo,hi in BINS:
       rr=[r for r in rows if lo<=float(r['num_atoms'])<=hi]; y=metric(rr) if rr else {'n':0}
       strata.append({'model':m,'manifest_idx':i,'seed':s,'ha_bin':f'[{lo},{hi}]',**y})
    with (args.outdir/'heavy_atom_strata.csv').open('w',newline='',encoding='utf-8') as f:
      ks=['model','manifest_idx','seed','ha_bin']+list(metric([]).keys());w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(strata)
    seed_summary=[]
    for m in MODELS:
      for k in metric_names:
       vals=[r[k] for r in macro if r['model']==m and finite(r.get(k))]
       seed_summary.append({'model':m,'metric':k,'seed_mean':mean(vals),'seed_sd':statistics.stdev(vals) if len(vals)>1 else None,'bootstrap95':bootstrap(vals)})
    delta_summary=[]
    for k in metric_names:
      vals=[r[k] for r in pair if finite(r.get(k))]
      delta_summary.append({'metric':k,'n_pairs':len(vals),'paired_delta_mean':mean(vals),'paired_delta_sd':statistics.stdev(vals) if len(vals)>1 else None,'bootstrap95':bootstrap(vals)})
    summary={'expected_runs':len(expected),'accepted_runs':len(found),'rejected_runs':len(rejected),'missing_or_rejected':rejected,
      'manifest_tasks':len(tasks),'accepted_pockets':sorted({i for _,i,_,_,_ in found}), 'seed_summary':seed_summary,'paired_delta':delta_summary}
    (args.outdir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# CrossDocked diagnosis pilot v2 汇总','',f"- 期望 run: **{len(expected)}**;有效 run: **{len(found)}**;拒绝/缺失: **{len(rejected)}**。",'- 筛选: `eval_clean.json`、`EVAL_CLEAN_DONE`、`eval.log` 非 `nosample` 结尾、逐分子有效 `docking_score`。','- 唯一键: `(model, manifest_idx, seed)`；不使用 `pocket_dir`。','', '## 有效性与缺失', '```json',json.dumps(rejected,indent=2,ensure_ascii=False),'```','', '## 输出文件','- `detail.csv`: 每个 model/idx/seed 的明细','- `macro_seed.csv`: 12 pocket macro-average（按有效 entry）','- `paired_delta.csv`: 530k − df500k 配对差','- `heavy_atom_strata.csv`: heavy-atom 分层','- `summary.json`: seed 均值/SD/bootstrap CI 与配对差']
    (args.outdir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if rejected: print('REJECTED_OR_MISSING',len(rejected),json.dumps(rejected,ensure_ascii=False))
    print('ACCEPTED',len(found),'EXPECTED',len(expected),'OUTDIR',args.outdir)
if __name__=='__main__':main()
