#!/usr/bin/env python3
"""Select 30 pockets from PDBBind v2020 for evaluation.

Includes the original 10 standard pockets + 20 randomly selected new ones.
Outputs PDBID and ligand center coordinates for each pocket.
"""
import os
import sys
import random
import json
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PDBBIND = '/workspace/ayb/data/pdbbind/v2020'

# Original 10 standard pockets
ORIGINAL_10 = ['10gs', '1a1e', '1a30', '1a4k', '1a94', '1a9q', '2yi0', '4afg', '4kzu', '4zl4']

def get_ligand_center(pdb_id):
    """Get ligand center from mol2 or sdf file."""
    mol2_path = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_ligand.mol2')
    sdf_path = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_ligand.sdf')

    mol = None
    if os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
    if mol is None and os.path.exists(sdf_path):
        mol = Chem.MolFromMolFile(sdf_path, sanitize=False, removeHs=False)

    if mol is None:
        return None

    try:
        conf = mol.GetConformer(0)
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        center = pos.mean(axis=0)
        return center.tolist()
    except Exception:
        return None

def has_required_files(pdb_id):
    """Check if pocket has required files for evaluation."""
    protein_pdb = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_protein.pdb')
    pocket_pdb = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_pocket.pdb')
    pocket_dir = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_prot')
    ligand_mol2 = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_ligand.mol2')
    ligand_sdf = os.path.join(PDBBIND, pdb_id, f'{pdb_id}_ligand.sdf')

    has_protein = os.path.exists(protein_pdb)
    has_ligand = os.path.exists(ligand_mol2) or os.path.exists(ligand_sdf)
    has_pocket = os.path.exists(pocket_pdb) or os.path.isdir(pocket_dir)

    return has_protein and has_ligand and has_pocket

def main():
    random.seed(42)

    # Get all valid PDB IDs
    all_dirs = sorted(os.listdir(PDBBIND))
    all_pdb_ids = [d for d in all_dirs if len(d) == 4 and os.path.isdir(os.path.join(PDBBIND, d))]
    print(f'Total PDB directories: {len(all_pdb_ids)}')

    # Filter valid pockets
    valid_pockets = []
    for pdb_id in all_pdb_ids:
        if pdb_id in ORIGINAL_10:
            continue
        if has_required_files(pdb_id):
            valid_pockets.append(pdb_id)

    print(f'Valid candidate pockets (excluding original 10): {len(valid_pockets)}')

    # Randomly select 20 new pockets
    new_20 = random.sample(valid_pockets, 20)
    new_20.sort()

    # Combine all 30 pockets
    all_30 = ORIGINAL_10 + new_20
    print(f'\nSelected 30 pockets:')

    results = {}
    for pdb_id in all_30:
        center = get_ligand_center(pdb_id)
        if center is None:
            print(f'  {pdb_id}: FAILED to get center')
            continue
        results[pdb_id] = center
        center_str = f'{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}'
        print(f'  {pdb_id}: {center_str}')

    # Save results
    output_path = '/workspace/ayb/Pocket2Mol/pocket_centers_30.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {output_path}')
    print(f'Total pockets with centers: {len(results)}')

    # Also print bash format for eval script
    print('\n=== BASH FORMAT ===')
    for pdb_id in all_30:
        if pdb_id in results:
            center = results[pdb_id]
            center_str = f'{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}'
            print(f'CENTER[{pdb_id}]="{center_str}"')

if __name__ == '__main__':
    main()
