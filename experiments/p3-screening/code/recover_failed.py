import os, sys, warnings, torch, tempfile, json, time
warnings.filterwarnings("ignore")
CO = "/workspace/ayb/checkouts/dfe-unified-sci1-aed7317"
sys.path.insert(0, CO); os.chdir(CO)
from rdkit import Chem
from rdkit.Chem import BondType
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.transforms import (FeaturizeProteinAtom, FeaturizeLigandAtom, AtomComposer,
    LigandCountNeighbors, RefineData)
from utils.data import ProteinLigandData, torchify_dict
from models.maskfill import MaskFillModelVN

OUTDIR = "/workspace/ayb/experiments/dfe-unified-p3"
DATA = "/workspace/ayb/data/pdbbind/v2020/"

def fallback_sdf(path):
    """Kekulize failed: force aromatic bonds -> SINGLE (degraded featurization) and write SDF."""
    m = next(iter(Chem.SDMolSupplier(path, removeHs=False, sanitize=False)))
    if m is None:
        raise ValueError("unreadable")
    for b in m.GetBonds():
        if b.GetBondType() == BondType.AROMATIC or b.GetIsAromatic():
            b.SetBondType(BondType.SINGLE); b.SetIsAromatic(False)
    for a in m.GetAtoms():
        a.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
    except Exception:
        pass
    tf = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w")
    w = Chem.SDWriter(tf.name); w.SetKekulize(False); w.write(m); w.close()
    return tf.name

ckpt = "/workspace/ayb/experiments/dfe-unified-arm-a/armb_i3_run/checkpoints/25000.pt"
ck = torch.load(ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]
pf = FeaturizeProteinAtom(); lf = FeaturizeLigandAtom(); refine = RefineData(); count = LigandCountNeighbors()
model = MaskFillModelVN(cfg.model, num_classes=7, num_bond_types=3,
    protein_atom_feature_dim=pf.feature_dim, ligand_atom_feature_dim=lf.feature_dim)
model.load_state_dict(ck["model"]); model.eval(); model.set_science_vector_origin("zero")
dev = "cuda"; model = model.to(dev); knn = cfg.model.encoder.knn

def embed_from_sdf(pk, sdfpath):
    base = DATA + pk
    pocket = PDBProtein(base + "/" + pk + "_pocket.pdb").to_dict_atom()
    ligand = parse_sdf_file(sdfpath)
    # remap any leftover aromatic bond_type index to single-index so bond one-hot < num_bond_types
    bt = ligand["bond_type"]
    if len(bt):
        maxbt = int(bt.max())
        # aromatic index is the highest in the BOND_TYPES enum path; clamp anything >=3 to single(idx for SINGLE)
        import numpy as np
        # SINGLE index per parse_sdf_file BOND_TYPES ordering
        from rdkit.Chem.rdchem import BondType as BT
        BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
        single_idx = BOND_TYPES[BT.SINGLE]
        arom_idx = BOND_TYPES[BT.AROMATIC]
        bt = bt.copy(); bt[bt == arom_idx] = single_idx
        # also clamp any index >=3
        bt[bt >= 3] = single_idx
        ligand["bond_type"] = bt
    data = ProteinLigandData.from_protein_ligand_dicts(protein_dict=torchify_dict(pocket), ligand_dict=torchify_dict(ligand))
    data = refine(data); data = count(data); data = pf(data); data = lf(data)
    data.ligand_context_pos = data.ligand_pos
    data.ligand_context_feature_full = data.ligand_atom_feature_full
    data.ligand_context_bond_index = data.ligand_bond_index
    data.ligand_context_bond_type = data.ligand_bond_type
    comp = AtomComposer(pf.feature_dim, lf.feature_dim, knn=knn)
    data = comp(data)
    with torch.no_grad():
        cf = data.compose_feature.float().to(dev); cp = data.compose_pos.to(dev)
        il = data.idx_ligand_ctx_in_compose.to(dev); ip = data.idx_protein_in_compose.to(dev)
        h = model._embed_compose(cf, cp, il, ip)
        ei = data.compose_knn_edge_index.to(dev); ef = data.compose_knn_edge_feature.float().to(dev)
        out = model.encoder(node_attr=h, pos=cp, edge_index=ei, edge_feature=ef)
        sca = out[0]
        return sca[il].mean(0).cpu(), sca[ip].mean(0).cpu()

failed = json.load(open(OUTDIR + "/embed_failed.json"))
emb = torch.load(OUTDIR + "/embeddings_shared.pt", map_location="cpu")
recovered = 0; still_failed = []
for d in failed:
    pk = d["code"]
    try:
        sdf = fallback_sdf(DATA + pk + "/" + pk + "_ligand.sdf")
        lig, prot = embed_from_sdf(pk, sdf); os.unlink(sdf)
        emb[pk] = {"lig": lig, "prot": prot, "fallback": True}
        recovered += 1
    except Exception as e:
        still_failed.append({"code": pk, "err": str(e)[:200]})

torch.save(emb, OUTDIR + "/embeddings_shared.pt")
json.dump(still_failed, open(OUTDIR + "/embed_failed.json", "w"), indent=2)
print(f"RECOVERED={recovered} STILL_FAILED={len(still_failed)} TOTAL_EMB={len(emb)}", flush=True)
print("RECOVER_DONE_MARKER", flush=True)
