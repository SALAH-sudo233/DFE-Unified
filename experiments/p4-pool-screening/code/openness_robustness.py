#!/usr/bin/env python3
"""Robustness recomputation for openness vs generation-quality (n=21 PDBBind pockets).
Pure stdlib: Pearson/Spearman/Kendall, drop-outlier, bootstrap percentile CI,
jackknife influence, partial correlation controlling MW. No numpy/scipy needed."""
import math, random

# name, openness, vina_mean, vina_best, qed, mw, diversity
DATA = [
("1fm9",0.000,-8.33,-11.35,0.681,242.6,0.923),
("1ui0",0.029,-6.09,-8.41,0.486,238.8,0.852),
("2yi0",0.039,-7.36,-9.80,0.577,319.5,0.862),
("3i4b",0.045,-8.79,-11.06,0.550,302.3,0.869),
("5u14",0.076,-7.08,-9.08,0.350,304.2,0.937),
("4afg",0.097,-8.13,-10.67,0.687,260.4,0.838),
("1a9q",0.113,-7.92,-9.65,0.544,283.9,0.890),
("1sgu",0.136,-8.22,-11.51,0.546,381.6,0.949),
("1a94",0.144,-6.75,-8.75,0.619,282.5,0.868),
("3ddg",0.145,-6.66,-8.93,0.452,256.1,0.867),
("5jxq",0.147,-7.38,-9.71,0.508,283.8,0.897),
("3fv3",0.154,-7.08,-9.40,0.498,333.5,0.896),
("1a30",0.181,-7.08,-8.60,0.677,252.3,0.824),
("6fag",0.199,-6.99,-8.89,0.612,329.0,0.855),
("10gs",0.237,-6.87,-8.63,0.667,268.9,0.811),
("1g30",0.251,-7.23,-9.68,0.691,263.2,0.861),
("1a4k",0.252,-7.06,-8.67,0.657,217.7,0.861),
("1zgi",0.276,-7.63,-10.74,0.646,300.3,0.870),
("1ezq",0.292,-7.00,-8.35,0.640,236.1,0.847),
("3ebp",0.376,-8.37,-11.90,0.460,270.7,0.819),
("1a1e",0.522,-5.36,-6.70,0.603,265.6,0.902),
]
names=[d[0] for d in DATA]
cols=list(zip(*[d[1:] for d in DATA]))
op,vm,vb,qed,mw,div = [list(c) for c in cols]
METRICS={"Vina_mean":vm,"Vina_best":vb,"QED":qed,"MW":mw,"Diversity":div}

def mean(x): return sum(x)/len(x)
def pearson(x,y):
    n=len(x); mx,my=mean(x),mean(y)
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    d=math.sqrt(sxx*syy)
    return sxy/d if d>0 else float('nan')
def rankdata(x):
    # average ranks for ties, 1-based
    idx=sorted(range(len(x)),key=lambda i:x[i])
    ranks=[0.0]*len(x); i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[idx[j+1]]==x[idx[i]]: j+=1
        avg=(i+j)/2.0+1
        for k in range(i,j+1): ranks[idx[k]]=avg
        i=j+1
    return ranks
def spearman(x,y): return pearson(rankdata(x),rankdata(y))
def kendall_tau(x,y):
    n=len(x); c=d=0
    for i in range(n):
        for j in range(i+1,n):
            s=(x[i]-x[j])*(y[i]-y[j])
            if s>0: c+=1
            elif s<0: d+=1
    # tau-b would need tie correction; data essentially tie-free
    return (c-d)/(0.5*n*(n-1))
def pear_pt(r,n):  # two-sided p via t-approx
    if abs(r)>=1: return 0.0
    t=r*math.sqrt((n-2)/(1-r*r))
    # incomplete-beta t p-value, stdlib approximation via series-free method
    df=n-2
    x=df/(df+t*t)
    return betai(df/2,0.5,x)
def betai(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    lbeta=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    front=math.exp(math.log(x)*a+math.log(1-x)*b-lbeta)/a
    # continued fraction (Lentz)
    def cf(a,b,x):
        tiny=1e-30; c=1.0; d=1.0-(a+b)*x/(a+1); 
        if abs(d)<tiny: d=tiny
        d=1/d; h=d
        for m in range(1,200):
            m2=2*m
            aa=m*(b-m)*x/((a+m2-1)*(a+m2))
            d=1+aa*d; c=1+aa/c
            if abs(d)<tiny:d=tiny
            if abs(c)<tiny:c=tiny
            d=1/d; h*=d*c
            aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+m2+1))
            d=1+aa*d; c=1+aa/c
            if abs(d)<tiny:d=tiny
            if abs(c)<tiny:c=tiny
            d=1/d; de=d*c; h*=de
            if abs(de-1)<1e-10: break
        return h
    if x<(a+1)/(a+b+2): return front*cf(a,b,x)
    else: return 1-front*cf(b,a,1-x)  # approx symmetry (adequate here)

def resid(a,b):
    mb=mean(b); ma=mean(a)
    sbb=sum((v-mb)**2 for v in b)
    beta=sum((a[i]-ma)*(b[i]-mb) for i in range(len(a)))/sbb
    alpha=ma-beta*mb
    return [a[i]-(alpha+beta*b[i]) for i in range(len(a))]
def partial(x,y,z): return pearson(resid(x,z),resid(y,z))

def boot_ci(x,y,fn,B=20000,seed=0):
    rng=random.Random(seed); n=len(x); out=[]
    for _ in range(B):
        idx=[rng.randrange(n) for _ in range(n)]
        xs=[x[i] for i in idx]; ys=[y[i] for i in idx]
        try:
            v=fn(xs,ys)
            if v==v: out.append(v)
        except ZeroDivisionError: pass
    out.sort()
    lo=out[int(0.025*len(out))]; hi=out[int(0.975*len(out))]; md=out[len(out)//2]
    return lo,md,hi

n=len(op)
print(f"n={n}\n")
print("=== (1) Full-sample correlations (openness vs metric) ===")
print(f"{'metric':10s} {'Pearson':>9s} {'p':>7s} {'Spearman':>9s} {'Kendall':>8s}")
for k,v in METRICS.items():
    r=pearson(op,v); p=pear_pt(r,n); rho=spearman(op,v); tau=kendall_tau(op,v)
    print(f"{k:10s} {r:+9.3f} {p:7.3f} {rho:+9.3f} {tau:+8.3f}")
print(f"\nPearson crit (df=19, a=.05, two-sided) = 0.433 ; Spearman crit ~0.435")

print("\n=== (2) Drop outlier 1a1e (n=20) ===")
mask=[nm!="1a1e" for nm in names]
def sub(v): return [v[i] for i in range(n) if mask[i]]
opd=sub(op)
print(f"{'metric':10s} {'Pear full':>10s} {'Pear -1a1e':>11s} {'Spear full':>11s} {'Spear -1a1e':>12s}")
for k,v in METRICS.items():
    print(f"{k:10s} {pearson(op,v):+10.3f} {pearson(opd,sub(v)):+11.3f} {spearman(op,v):+11.3f} {spearman(opd,sub(v)):+12.3f}")

print("\n=== (3) Bootstrap 95% CI, percentile (B=20000) ===")
for k,v in METRICS.items():
    lo,md,hi=boot_ci(op,v,pearson)
    flag="CROSSES 0" if lo<0<hi else "excludes 0"
    print(f"{k:10s} Pearson median={md:+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]  {flag}")
print("  Spearman CI:")
for k,v in METRICS.items():
    lo,md,hi=boot_ci(op,v,spearman)
    flag="CROSSES 0" if lo<0<hi else "excludes 0"
    print(f"{k:10s} Spearman median={md:+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]  {flag}")

print("\n=== (4) Jackknife influence on Pearson(openness,Vina_mean) ===")
base=pearson(op,vm)
infl=[]
for i in range(n):
    m=[j for j in range(n) if j!=i]
    infl.append((names[i], pearson([op[j] for j in m],[vm[j] for j in m])-base))
infl.sort(key=lambda t:abs(t[1]),reverse=True)
print(f"base r={base:+.3f}")
for nm,dch in infl[:6]:
    print(f"  drop {nm}: dr={dch:+.3f}  -> r={base+dch:+.3f}")

print("\n=== (5) Collinearity & partial correlations (MW as confounder/mediator) ===")
print(f"openness vs MW      Pearson {pearson(op,mw):+.3f}")
print(f"MW vs Vina_mean     Pearson {pearson(mw,vm):+.3f}")
print(f"MW vs QED           Pearson {pearson(mw,qed):+.3f}")
print(f"partial(open,Vina_mean | MW) = {partial(op,vm,mw):+.3f}   (raw {pearson(op,vm):+.3f})")
print(f"partial(open,QED      | MW) = {partial(op,qed,mw):+.3f}   (raw {pearson(op,qed):+.3f})")
print(f"partial(open,Vina_mean | MW) dropping 1a1e = {partial(opd,sub(vm),sub(mw)):+.3f}")

print("\n=== (6) Tertile groups (closed / mid / open by openness) ===")
order=sorted(range(n),key=lambda i:op[i])
g=[order[0:7],order[7:14],order[14:21]]
lab=["closed","mid","open"]
for gi,grp in enumerate(g):
    print(f"{lab[gi]:7s} open={mean([op[i] for i in grp]):.3f} "
          f"Vina={mean([vm[i] for i in grp]):+.2f} QED={mean([qed[i] for i in grp]):.3f} "
          f"MW={mean([mw[i] for i in grp]):.1f} Div={mean([div[i] for i in grp]):.3f}")
