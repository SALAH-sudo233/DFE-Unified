#!/usr/bin/env python3
"""30% sequence-identity single-linkage clustering for PDBBind refined (no external tools).
Strategy: k-mer (k=5) sets per sequence; candidate pairs via inverted index on shared k-mers;
for each candidate pair, compute a fast identity estimate = |A cap B| / min(|A|,|B|) (k-mer
containment, a standard proxy). Union-find single-linkage at containment >= threshold ~ maps
to ~30% identity. This is a conservative leakage-control clustering, not an alignment score.
"""
import json,os,sys,time
from collections import defaultdict
OUT="/workspace/ayb/experiments/dfe-unified-p3"
K=5
# containment threshold tuned to be conservative for ~30% identity leakage control
THRESH=0.30
seqs={}
for line in open(OUT+"/refined_seqs.fasta"):
    line=line.strip()
    if line.startswith(">"): cur=line[1:]; seqs[cur]=""
    else: seqs[cur]+=line
codes=list(seqs)
print("seqs",len(codes),flush=True)
# build k-mer sets
kmer={}
for c in codes:
    s=seqs[c]
    ks=set(s[i:i+K] for i in range(len(s)-K+1)) if len(s)>=K else {s}
    kmer[c]=ks
# inverted index kmer -> codes
inv=defaultdict(list)
for c in codes:
    for km in kmer[c]: inv[km].append(c)
print("kmers",len(inv),flush=True)
# candidate pairs: co-occur in same kmer bucket; count shared kmers
pair_shared=defaultdict(int)
t0=time.time()
for km,cl in inv.items():
    if len(cl)<2: continue
    # cap bucket size to avoid O(n^2) blowup on ubiquitous kmers
    if len(cl)>400: continue
    for i in range(len(cl)):
        ci=cl[i]
        for j in range(i+1,len(cl)):
            a,b=ci,cl[j]
            if a>b: a,b=b,a
            pair_shared[(a,b)]+=1
print("candidate pairs",len(pair_shared),"in",round(time.time()-t0,1),"s",flush=True)
# union-find
parent={c:c for c in codes}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(x,y):
    rx,ry=find(x),find(y)
    if rx!=ry: parent[rx]=ry
edges=0
for (a,b),sh in pair_shared.items():
    cont=sh/min(len(kmer[a]),len(kmer[b]))
    if cont>=THRESH:
        union(a,b); edges+=1
print("linkage edges",edges,flush=True)
clusters=defaultdict(list)
for c in codes: clusters[find(c)].append(c)
cl_list=sorted(clusters.values(),key=len,reverse=True)
print("clusters",len(cl_list),"largest",len(cl_list[0]),flush=True)
# assign cluster id per code
code2cl={}
for i,cl in enumerate(cl_list):
    for c in cl: code2cl[c]=i
json.dump({"threshold":THRESH,"k":K,"n_clusters":len(cl_list),"code2cluster":code2cl,
           "cluster_sizes":[len(c) for c in cl_list]},open(OUT+"/refined_clusters.json","w"))
# size histogram
from collections import Counter
sz=Counter(len(c) for c in cl_list)
print("size hist (size:count) top:",dict(sorted(sz.items())[:10]),flush=True)
print("singletons",sz[1],"/",len(cl_list),flush=True)
print("DONE",flush=True)
