import os, sys, warnings, torch, tempfile, json, time
warnings.filterwarnings("ignore")
CO = "/workspace/ayb/checkouts/dfe-unified-sci1-aed7317"
sys.path.insert(0, CO); os.chdir(CO)
from rdkit import Chem
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.transforms import (FeaturizeProteinAtom, FeaturizeLigandAtom, AtomComposer,
    LigandCountNeighbors, RefineData)
from utils.data import ProteinLigandData, torchify_dict
from models.maskfill import MaskFillModelVN

OUTDIR = "/workspace/ayb/experiments/dfe-unified-p3"
DATA = "/workspace/ayb/data/pdbbind/v2020/"

def kekulized_sdf(path):
    m = next(iter(Chem.SDMolSupplier(path, removeHs=False, sanitize=True)))
    if m is None:
        m = next(iter(Chem.SDMolSupplier(path, removeHs=False, sanitize=False)))
        Chem.SanitizeMol(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    tf = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w")
    w = Chem.SDWriter(tf.name); w.write(m); w.close()
    return tf.name

ckpt = "/workspace/ayb/experiments/dfe-unified-arm-a/armb_i3_run/checkpoints/25000.pt"
ck = torch.load(ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]
pf = FeaturizeProteinAtom(); lf = FeaturizeLigandAtom(); refine = RefineData(); count = LigandCountNeighbors()
model = MaskFillModelVN(cfg.model, num_classes=7, num_bond_types=3,
    protein_atom_feature_dim=pf.feature_dim, ligand_atom_feature_dim=lf.feature_dim)
model.load_state_dict(ck["model"]); model.eval(); model.set_science_vector_origin("zero")
dev = "cuda"; model = model.to(dev)
knn = cfg.model.encoder.knn

def embed(pk):
    base = DATA + pk
    pocket = PDBProtein(base + "/" + pk + "_pocket.pdb").to_dict_atom()
    kek = kekulized_sdf(base + "/" + pk + "_ligand.sdf")
    ligand = parse_sdf_file(kek); os.unlink(kek)
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

labels = json.load(open(OUTDIR + "/refined_labels.json"))
codes = [d["code"] for d in labels]
print("total", len(codes), flush=True)

emb = {}
failed = []
t0 = time.time()
for i, pk in enumerate(codes):
    try:
        lig, prot = embed(pk)
        emb[pk] = {"lig": lig, "prot": prot}
    except Exception as e:
        failed.append({"code": pk, "err": str(e)[:200]})
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(codes)} ok={len(emb)} fail={len(failed)} elapsed={time.time()-t0:.0f}s", flush=True)

torch.save(emb, OUTDIR + "/embeddings_shared.pt")
json.dump(failed, open(OUTDIR + "/embed_failed.json", "w"), indent=2)
print(f"DONE embedded={len(emb)} failed={len(failed)} elapsed={time.time()-t0:.0f}s", flush=True)
print("EMB_DONE_MARKER", flush=True)
