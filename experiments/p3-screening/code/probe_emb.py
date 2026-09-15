
import os,sys,warnings,torch,numpy as np
warnings.filterwarnings("ignore")
CO="/workspace/ayb/checkouts/dfe-unified-sci1-aed7317"
sys.path.insert(0,CO); os.chdir(CO)
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom, AtomComposer
from utils.data import ProteinLigandData, torchify_dict
from models.maskfill import MaskFillModelVN
from models.common import embed_compose
import yaml
from easydict import EasyDict

ckpt="/workspace/ayb/experiments/dfe-unified-arm-a/armb_i3_run/checkpoints/25000.pt"
ck=torch.load(ckpt,map_location="cpu",weights_only=False)
cfg=ck["config"] if "config" in ck else None
print("ckpt keys:",list(ck.keys())[:8])
print("has config:",cfg is not None)

pf=FeaturizeProteinAtom(); lf=FeaturizeLigandAtom()
print("protein_dim",pf.feature_dim,"ligand_dim",lf.feature_dim)

# build ONE complex: 1a1e
base="/workspace/ayb/data/pdbbind/v2020/1a1e"
pocket=PDBProtein(base+"/1a1e_pocket.pdb").to_dict_atom()
ligand=parse_sdf_file(base+"/1a1e_ligand.sdf")
data=ProteinLigandData.from_protein_ligand_dicts(protein_dict=torchify_dict(pocket),ligand_dict=torchify_dict(ligand))
data=pf(data); data=lf(data)
# full ligand as context (no masking)
data.ligand_context_pos=data.ligand_pos
data.ligand_context_feature_full=data.ligand_atom_feature_full
data.ligand_context_bond_index=data.ligand_bond_index
data.ligand_context_bond_type=data.ligand_bond_type
comp=AtomComposer(pf.feature_dim, lf.feature_dim, knn=16)
data=comp(data)
print("compose_feature",tuple(data.compose_feature.shape),"pos",tuple(data.compose_pos.shape))
print("n_ligand_ctx",len(data.idx_ligand_ctx_in_compose),"n_protein",len(data.idx_protein_in_compose))

# build model
model=MaskFillModelVN(cfg.model, num_classes=7, num_bond_types=3,
    protein_atom_feature_dim=pf.feature_dim, ligand_atom_feature_dim=lf.feature_dim)
model.load_state_dict(ck["model"]); model.eval()
model.set_science_vector_origin("zero")
dev="cuda"; model=model.to(dev)
with torch.no_grad():
    cf=data.compose_feature.float().to(dev); cp=data.compose_pos.to(dev)
    il=data.idx_ligand_ctx_in_compose.to(dev); ip=data.idx_protein_in_compose.to(dev)
    h=model._embed_compose(cf,cp,il,ip)
    ei=data.compose_knn_edge_index.to(dev); ef=data.compose_knn_edge_feature.float().to(dev)
    out=model.encoder(node_attr=h,pos=cp,edge_index=ei,edge_feature=ef)
    sca=out[0]  # (N,256)
    lig_emb=sca[il].mean(0)   # complex-level shared rep from ligand atoms
    prot_emb=sca[ip].mean(0)
    print("encoder scalar out",tuple(sca.shape))
    print("ligand-pooled emb dim",lig_emb.shape[0],"norm",round(lig_emb.norm().item(),3))
    print("protein-pooled emb dim",prot_emb.shape[0],"norm",round(prot_emb.norm().item(),3))
    print("EMB_OK")
