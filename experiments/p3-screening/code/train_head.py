import os, sys, json, time, math
import torch, torch.nn as nn
import numpy as np

OUTDIR = "/workspace/ayb/experiments/dfe-unified-p3"
dev = "cuda"
torch.manual_seed(20260908); np.random.seed(20260908)

emb = torch.load(OUTDIR + "/embeddings_shared.pt", map_location="cpu")
labels = json.load(open(OUTDIR + "/refined_labels.json"))
split = json.load(open(OUTDIR + "/refined_split.json"))
pk2lab = {d["code"]: float(d["pk"]) for d in labels}
eval_pockets = split["eval_pockets"]

def build(codes):
    L, P, Y, C = [], [], [], []
    for c in codes:
        if c in emb and c in pk2lab:
            L.append(emb[c]["lig"]); P.append(emb[c]["prot"]); Y.append(pk2lab[c]); C.append(c)
    L = torch.stack(L).float(); P = torch.stack(P).float(); Y = torch.tensor(Y).float()
    return L, P, Y, C

trL, trP, trY, trC = build(split["split"]["train"])
vaL, vaP, vaY, vaC = build(split["split"]["val"])
teL, teP, teY, teC = build(split["split"]["test"])
print(f"usable train={len(trC)} val={len(vaC)} test={len(teC)}", flush=True)

# normalization from train
def make_feats(L, P, mode):
    if mode == "full": X = torch.cat([L, P], 1)
    elif mode == "ligand_only": X = L
    elif mode == "pocket_only": X = P
    return X

class ScreeningHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum() / (a.norm()*b.norm() + 1e-9))

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(torch.tensor(ra, dtype=torch.float), torch.tensor(rb, dtype=torch.float))

def train_eval(Xtr, Ytr, Xva, Yva, Xte, Yte, tag, seed=0, save_ckpt=None):
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    ymu = Ytr.mean(); ysd = Ytr.std() + 1e-6
    Xtr_ = ((Xtr - mu)/sd).to(dev); Xva_ = ((Xva - mu)/sd).to(dev); Xte_ = ((Xte - mu)/sd).to(dev)
    Ytr_ = ((Ytr - ymu)/ysd).to(dev)
    model = ScreeningHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best_val = -2; best_state = None; best_ep = 0
    n = Xtr_.shape[0]; bs = 256
    for ep in range(300):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            pred = model(Xtr_[idx])
            loss = lossf(pred, Ytr_[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_).cpu()*ysd + ymu
        vp = pearson(pv, Yva)
        if vp > best_val:
            best_val = vp; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = (model(Xte_).cpu()*ysd + ymu)
    tp = pearson(pt, Yte); ts = spearman(pt.numpy(), Yte.numpy())
    rmse = float(((pt - Yte)**2).mean().sqrt())
    res = {"tag": tag, "test_pearson": tp, "test_spearman": ts, "test_rmse": rmse,
           "val_pearson": best_val, "best_epoch": best_ep, "n_test": len(Yte)}
    if save_ckpt:
        torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "ymu": float(ymu), "ysd": float(ysd),
                    "feat_dim": Xtr.shape[1], "mode": tag}, save_ckpt)
        res["ckpt"] = save_ckpt
        res["test_pred"] = pt.numpy().tolist()
    return res, pt

results = {}
# --- FULL (primary) ---
Xtr = make_feats(trL, trP, "full"); Xva = make_feats(vaL, vaP, "full"); Xte = make_feats(teL, teP, "full")
res_full, te_pred_full = train_eval(Xtr, trY, Xva, vaY, Xte, teY, "full", seed=0,
                                     save_ckpt=OUTDIR + "/screeninghead_full.pt")
results["full"] = {k: res_full[k] for k in ["test_pearson","test_spearman","test_rmse","val_pearson","best_epoch"]}
print("FULL", results["full"], flush=True)

# --- ligand_only ---
Xtr = make_feats(trL, trP, "ligand_only"); Xva = make_feats(vaL, vaP, "ligand_only"); Xte = make_feats(teL, teP, "ligand_only")
r, _ = train_eval(Xtr, trY, Xva, vaY, Xte, teY, "ligand_only", seed=0)
results["ligand_only"] = {k: r[k] for k in ["test_pearson","test_spearman","test_rmse"]}
print("LIG", results["ligand_only"], flush=True)

# --- pocket_only ---
Xtr = make_feats(trL, trP, "pocket_only"); Xva = make_feats(vaL, vaP, "pocket_only"); Xte = make_feats(teL, teP, "pocket_only")
r, _ = train_eval(Xtr, trY, Xva, vaY, Xte, teY, "pocket_only", seed=0)
results["pocket_only"] = {k: r[k] for k in ["test_pearson","test_spearman","test_rmse"]}
print("POCKET", results["pocket_only"], flush=True)

# --- label_permutation (shuffle train+val labels) ---
gtr = torch.Generator().manual_seed(123); gva = torch.Generator().manual_seed(456)
trY_p = trY[torch.randperm(len(trY), generator=gtr)]
vaY_p = vaY[torch.randperm(len(vaY), generator=gva)]
Xtr = make_feats(trL, trP, "full"); Xva = make_feats(vaL, vaP, "full"); Xte = make_feats(teL, teP, "full")
r, _ = train_eval(Xtr, trY_p, Xva, vaY_p, Xte, teY, "label_permutation", seed=0)
results["label_permutation"] = {k: r[k] for k in ["test_pearson","test_spearman","test_rmse"]}
print("LABELPERM", results["label_permutation"], flush=True)

# --- pocket_shuffle (break lig-pocket pairing in ALL splits) ---
gp = torch.Generator().manual_seed(789)
trP_s = trP[torch.randperm(len(trP), generator=torch.Generator().manual_seed(1))]
vaP_s = vaP[torch.randperm(len(vaP), generator=torch.Generator().manual_seed(2))]
teP_s = teP[torch.randperm(len(teP), generator=torch.Generator().manual_seed(3))]
Xtr = torch.cat([trL, trP_s],1); Xva = torch.cat([vaL, vaP_s],1); Xte = torch.cat([teL, teP_s],1)
r, _ = train_eval(Xtr, trY, Xva, vaY, Xte, teY, "pocket_shuffle", seed=0)
results["pocket_shuffle"] = {k: r[k] for k in ["test_pearson","test_spearman","test_rmse"]}
print("POCKETSHUF", results["pocket_shuffle"], flush=True)

# --- eval-pocket enrichment using FULL model test predictions ---
# pooled ranking metrics over test set treating "actives" as top pk quartile
def ef_bedroc(scores, y):
    # actives = top 25% by true pk within this set
    n = len(y)
    thr = np.quantile(y, 0.75)
    active = (y >= thr).astype(int)
    order = np.argsort(-scores)  # rank by predicted score desc
    active_sorted = active[order]
    n_active = active.sum(); 
    out = {"n": int(n), "n_active": int(n_active)}
    for frac, name in [(0.01,"EF1pct"),(0.05,"EF5pct")]:
        k = max(1, int(round(n*frac)))
        hits = active_sorted[:k].sum()
        ef = (hits/k) / (n_active/n) if n_active>0 else float('nan')
        out[name] = float(ef)
    # BEDROC alpha=20
    alpha = 20.0
    ranks = np.where(active_sorted==1)[0] + 1
    if n_active>0 and n_active<n:
        ra = n_active/n
        s = np.sum(np.exp(-alpha*ranks/n))
        rie = s / ((n_active/n)*(1-math.exp(-alpha))/(math.exp(alpha/n)-1))
        bedroc = rie*ra*math.sinh(alpha/2)/(math.cosh(alpha/2)-math.cosh(alpha/2-alpha*ra)) + 1/(1-math.exp(alpha*(1-ra)))
        out["BEDROC20"] = float(bedroc)
    return out

enr = {}
te_scores = te_pred_full.numpy(); te_y = teY.numpy()
teC_arr = np.array(teC)
# per eval-pocket: too few per single pocket; report pooled over all test + per-pocket presence
enr["pooled_test"] = ef_bedroc(te_scores, te_y)
present = [p for p in eval_pockets if p in teC]
enr["eval_pockets_in_test"] = present
enr["note"] = ("Each of the 5 eval pockets is a single complex in test (one ligand each), "
               "so per-target virtual-screening EF/BEDROC over a ligand library is not computable "
               "from PDBBind-refined labels alone. Reported pooled EF/BEDROC over the 823-complex test "
               "set using top-25% pk as actives, ranked by ScreeningHead predicted pk.")
results_out = {
    "controls": {
        "full": {"pearson": results["full"]["test_pearson"], "spearman": results["full"]["test_spearman"]},
        "ligand_only": {"pearson": results["ligand_only"]["test_pearson"], "spearman": results["ligand_only"]["test_spearman"]},
        "pocket_only": {"pearson": results["pocket_only"]["test_pearson"], "spearman": results["pocket_only"]["test_spearman"]},
        "label_permutation": {"pearson": results["label_permutation"]["test_pearson"], "spearman": results["label_permutation"]["test_spearman"]},
        "pocket_shuffle": {"pearson": results["pocket_shuffle"]["test_pearson"], "spearman": results["pocket_shuffle"]["test_spearman"]},
    },
    "detail": results,
    "eval_pocket_enrichment": enr,
    "test_pearson": results["full"]["test_pearson"],
    "test_spearman": results["full"]["test_spearman"],
    "test_rmse": results["full"]["test_rmse"],
    "n_usable": {"train": len(trC), "val": len(vaC), "test": len(teC)},
    "seed": 20260908,
}
json.dump(results_out, open(OUTDIR + "/screeninghead_metrics.json", "w"), indent=2)
print(json.dumps(results_out["controls"], indent=2), flush=True)
print("TRAIN_DONE_MARKER", flush=True)
