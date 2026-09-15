#!/usr/bin/env python3
"""Size-matched comparison: I3 vs stock DF-abs, controlling for heavy-atom count.
Question: after removing the size shift, does I3 have a REAL per-molecule geometric
advantage (Vina/PB) or is the whole picture just 'I3 makes smaller molecules'?
Metrics: within heavy-atom bins, compare Vina Dock, LE, PB pass, QED, count."""
import json, os, statistics as st
POCK=['1fm9','1a1e','1a9q','3ebp','1sgu']
HERE=os.path.dirname(os.path.abspath(__file__))

def load(sub):
    recs=[]
    for pk in POCK:
        p=os.path.join(HERE, sub, pk+'.json') if sub else os.path.join(HERE, pk+'.json')
        for r in json.load(open(p,encoding='utf-8')):
            if not r.get('valid'): continue
            v=r.get('docking_score'); n=r.get('num_atoms',0) or 0
            if v is None or n<=0: continue
            recs.append({'pk':pk,'n':n,'vina':v,'le':-v/n,
                         'pb':1 if (r.get('posebuster') or {}).get('passed') else 0,
                         'qed':r.get('qed',0)})
    return recs

i3=load('')          # p4_pool/*.json = i3
stock=load('stock')  # p4_pool/stock/*.json = stock DF-abs
print(f"I3 valid={len(i3)}  stock valid={len(stock)}")

def agg(recs):
    return (len(recs), st.mean([r['vina'] for r in recs]), st.mean([r['le'] for r in recs]),
            sum(r['pb'] for r in recs)/len(recs), st.mean([r['qed'] for r in recs]),
            st.mean([r['n'] for r in recs]))

print("\n=== RAW (unmatched) ===")
for name,recs in [('I3',i3),('stock',stock)]:
    n,v,le,pb,q,ha=agg(recs)
    print(f"{name:6} N={n:4} HA={ha:5.1f} Vina={v:6.2f} LE={le:.3f} PB={pb*100:4.1f}% QED={q:.3f}")

# heavy-atom bins
bins=[(6,10),(11,13),(14,16),(17,19),(20,24)]
print("\n=== SIZE-MATCHED (per heavy-atom bin) ===")
print(f"{'HA bin':8} {'grp':6} {'N':>4} {'Vina':>7} {'LE':>6} {'PB%':>6} {'QED':>6}")
overlap_deltas_vina=[]; overlap_deltas_le=[]; overlap_deltas_pb=[]
for lo,hi in bins:
    i3b=[r for r in i3 if lo<=r['n']<=hi]
    stb=[r for r in stock if lo<=r['n']<=hi]
    for name,b in [('I3',i3b),('stock',stb)]:
        if b:
            n,v,le,pb,q,ha=agg(b)
            print(f"{lo:2}-{hi:<3}   {name:6} {n:4} {v:7.2f} {le:6.3f} {pb*100:5.1f} {q:6.3f}")
        else:
            print(f"{lo:2}-{hi:<3}   {name:6} {0:4}   (empty)")
    # if both non-empty, record within-bin delta (I3 - stock), weight by min count
    if i3b and stb:
        w=min(len(i3b),len(stb))
        overlap_deltas_vina.append((st.mean([r['vina'] for r in i3b])-st.mean([r['vina'] for r in stb]), w))
        overlap_deltas_le.append((st.mean([r['le'] for r in i3b])-st.mean([r['le'] for r in stb]), w))
        overlap_deltas_pb.append(((sum(r['pb'] for r in i3b)/len(i3b))-(sum(r['pb'] for r in stb)/len(stb)), w))
    print()

def wmean(pairs):
    if not pairs: return None
    W=sum(w for _,w in pairs)
    return sum(d*w for d,w in pairs)/W

print("=== SIZE-CONTROLLED delta (I3 - stock), count-weighted over overlapping bins ===")
print(f"overlapping bins: {len(overlap_deltas_vina)}  total weight: {sum(w for _,w in overlap_deltas_vina)}")
dv=wmean(overlap_deltas_vina); dle=wmean(overlap_deltas_le); dpb=wmean(overlap_deltas_pb)
print(f"delta Vina = {dv:+.3f}  (negative => I3 better at SAME size)")
print(f"delta LE   = {dle:+.4f} (positive => I3 better per-atom at SAME size)")
print(f"delta PB   = {dpb*100:+.1f}pp (positive => I3 more valid poses at SAME size)")
