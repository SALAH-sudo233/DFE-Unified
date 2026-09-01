#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed PoseBuster evaluation: properly converts Vina PDBQT output to SDF
using obabel, preserving bond information for PoseBuster checks.

Root cause: Chem.MolFromPDBFile() doesn't parse PDBQT BRANCH/ROOT tree
structure, resulting in lost bonds and disconnected fragments.
This caused the all_atoms_connected check to fail for ~95% of molecules.
"""
import argparse
import json
import os
import sys
import tempfile
import subprocess
import glob
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# PoseBusters
from posebusters import PoseBusters as PoseBuster

# Meeko for ligand preparation
from meeko import MoleculePreparation

VINA_BINARY = '/workspace/lsm/miniconda3/envs/AMP-vina/bin/vina'

def prepare_receptor(pdb_path, output_pdbqt=None):
    """Prepare receptor PDBQT from PDB file using OpenBabel."""
    if output_pdbqt is None:
        output_pdbqt = pdb_path.replace('.pdb', '_receptor.pdbqt')
    if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 0:
        return output_pdbqt
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInFormat('pdb')
        conv.SetOutFormat('pdbqt')
        mol = ob.OBMol()
        conv.AddOption('r', ob.OBConversion.INOPTIONS)
        if not conv.ReadFile(mol, pdb_path):
            conv2 = ob.OBConversion()
            conv2.SetInFormat('pdb')
            conv2.SetOutFormat('pdbqt')
            mol2 = ob.OBMol()
            conv2.ReadFile(mol2, pdb_path)
            conv2.WriteFile(mol2, output_pdbqt)
        else:
            conv.WriteFile(mol, output_pdbqt)
        if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 0:
            # Strip flexible ligand tags
            with open(output_pdbqt, 'r') as f:
                lines = f.readlines()
            fixed_lines = [l for l in lines if not l.strip().startswith(
                ('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF'))]
            with open(output_pdbqt, 'w') as f:
                f.writelines(fixed_lines)
            return output_pdbqt
        return None
    except Exception as e:
        print(f"  Receptor prep error: {e}")
        return None


def prepare_ligand(mol, output_pdbqt=None):
    """Prepare ligand PDBQT from RDKit molecule using Meeko."""
    if output_pdbqt is None:
        fd, output_pdbqt = tempfile.mkstemp(suffix='.pdbqt')
        os.close(fd)
    try:
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        try:
            pdbqt_string = prep.write_pdbqt_string()
        except Exception:
            from meeko import PDBQTWriterLegacy
            setup = setups[0] if isinstance(setups, list) else prep.setup
            result = PDBQTWriterLegacy.write_string(setup)
            pdbqt_string = result[0] if isinstance(result, tuple) else result
        with open(output_pdbqt, 'w') as f:
            f.write(pdbqt_string)
        if os.path.getsize(output_pdbqt) > 0:
            return output_pdbqt
        return None
    except Exception as e:
        print(f"  Ligand prep error: {e}")
        return None


def pdbqt_to_mol_fixed(pdbqt_path, original_mol=None):
    """FIX: Properly convert docked PDBQT to RDKit Mol with bond information.
    
    Uses obabel to convert PDBQT -> SDF, preserving bond connectivity.
    Falls back to using original_mol bonds with docked coordinates if obabel fails.
    """
    # Method 1: Use obabel to convert PDBQT to SDF
    sdf_path = pdbqt_path.replace('.pdbqt', '_fixed.sdf')
    try:
        result = subprocess.run(
            ['obabel', pdbqt_path, '-O', sdf_path, '-h', '-xl'],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            suppl = Chem.SDMolSupplier(sdf_path)
            mols = [m for m in suppl if m is not None]
            if mols:
                # Use first model (best docking score)
                docked_mol = mols[0]
                # Clean up temp file
                try:
                    os.unlink(sdf_path)
                except:
                    pass
                return docked_mol
    except Exception as e:
        print(f"  obabel conversion failed: {e}")
    
    # Method 2: Use meeko PDBQTMolecule
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        pdbqt_mol = PDBQTMolecule(pdbqt_path)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        if rdkit_mols:
            return rdkit_mols[0]
    except Exception as e:
        print(f"  meeko PDBQT parsing failed: {e}")
    
    # Method 3: Fallback - use original mol bonds with docked coordinates
    if original_mol is not None:
        try:
            docked_pdb = Chem.MolFromPDBFile(pdbqt_path, removeHs=True, sanitize=False)
            if docked_pdb is None:
                return None
            # Get docked coordinates
            docked_conf = docked_pdb.GetConformer()
            docked_coords = np.array([list(docked_conf.GetAtomPosition(i)) 
                                       for i in range(docked_pdb.GetNumAtoms())])
            # Create new mol with original bonds but docked coordinates
            new_mol = Chem.RWMol(original_mol)
            conf = Chem.Conformer(new_mol.GetNumAtoms())
            for i in range(min(len(docked_coords), new_mol.GetNumAtoms())):
                conf.SetAtomPosition(i, docked_coords[i])
            new_mol.AddConformer(conf, assignId=True)
            return new_mol
        except Exception as e:
            print(f"  Fallback coordinate mapping failed: {e}")
    
    return None


def compute_pocket_box(pocket_pdb, padding=4.0):
    """Compute docking box from pocket PDB coordinates."""
    mol = Chem.MolFromPDBFile(pocket_pdb, removeHs=False, sanitize=False)
    if mol is None:
        return None
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    center = coords.mean(axis=0)
    sizes = coords.max(axis=0) - coords.min(axis=0) + padding
    return {
        'center': [float(center[0]), float(center[1]), float(center[2])],
        'size': [float(max(s, 15.0)) for s in sizes]
    }


def run_vina_docking(receptor_pdbqt, ligand_pdbqt, box_center, box_size,
                     exhaustiveness=8, num_modes=9, energy_range=3):
    """Run Vina docking via command-line binary."""
    out_pdbqt = ligand_pdbqt.replace('.pdbqt', '_docked.pdbqt')
    cmd = [
        VINA_BINARY,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--center_x', str(box_center[0]),
        '--center_y', str(box_center[1]),
        '--center_z', str(box_center[2]),
        '--size_x', str(box_size[0]),
        '--size_y', str(box_size[1]),
        '--size_z', str(box_size[2]),
        '--exhaustiveness', str(exhaustiveness),
        '--num_modes', str(num_modes),
        '--energy_range', str(energy_range),
        '--out', out_pdbqt,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + '\n' + result.stderr
        scores = []
        for line in output.split('\n'):
            if 'affinity' in line and 'kcal' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'affinity:':
                        try:
                            score = float(parts[i+1])
                            scores.append(score)
                        except (ValueError, IndexError):
                            pass
                        break
            elif line.strip() and line.strip()[0].isdigit() and len(line.split()) >= 2:
                parts = line.split()
                try:
                    mode_num = int(parts[0])
                    affinity = float(parts[1])
                    if -50 < affinity < 50:
                        scores.append(affinity)
                except (ValueError, IndexError):
                    pass
        if not scores and result.returncode != 0:
            return {'error': f'Vina failed (exit {result.returncode}): {output[-500:]}'}
        best_affinity = scores[0] if scores else None
        return {
            'docking_score': best_affinity,
            'all_scores': scores[:5],
            'docked_pose': out_pdbqt if os.path.exists(out_pdbqt) else None,
        }
    except subprocess.TimeoutExpired:
        return {'error': 'Vina docking timed out (120s)'}
    except Exception as e:
        return {'error': f'Vina error: {e}'}


def run_posebusters(protein_pdb, docked_mol, ligand_mol=None):
    """Run PoseBusters check on docked pose."""
    try:
        if ligand_mol is not None:
            pb = PoseBuster(config='redock')
            results = pb.bust(mol_pred=docked_mol, mol_true=ligand_mol, mol_cond=protein_pdb)
        else:
            pb = PoseBuster(config='dock')
            results = pb.bust(mol_pred=docked_mol, mol_cond=protein_pdb)
        if hasattr(results, 'all'):
            passed = bool(results.all().all())
        elif hasattr(results, 'values'):
            passed = bool(results.values.all())
        else:
            passed = False
        details = {}
        if hasattr(results, 'columns'):
            for col in results.columns:
                try:
                    val = results[col].iloc[0] if hasattr(results[col], 'iloc') else results[col]
                    details[col] = bool(val) if not isinstance(val, bool) else val
                except Exception:
                    details[col] = str(results[col])
        return {'passed': passed, 'details': details}
    except Exception as e:
        return {'error': str(e), 'passed': None}


def evaluate_molecule(mol, receptor_pdbqt, pocket_pdb, box_info, posebuster_protein=None):
    """Full evaluation for a single molecule."""
    result = {
        'smiles': Chem.MolToSmiles(mol) if mol else None,
        'valid': mol is not None,
    }
    if mol is None:
        return result

    mol_with_h = Chem.AddHs(mol)
    if mol_with_h.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_with_h)
        except Exception:
            pass

    try:
        result['qed'] = float(QED.qed(mol))
        result['mw'] = float(Descriptors.MolWt(mol))
        result['logp'] = float(Descriptors.MolLogP(mol))
        result['num_hba'] = int(Descriptors.NumHAcceptors(mol))
        result['num_hbd'] = int(Descriptors.NumHDonors(mol))
        result['lipinski_violations'] = int(
            result['mw'] > 500 or result['logp'] > 5 or
            result['num_hba'] > 10 or result['num_hbd'] > 5)
        result['num_rings'] = int(mol.GetRingInfo().NumRings())
        result['num_atoms'] = int(mol.GetNumAtoms())
    except Exception as e:
        result['property_error'] = str(e)

    try:
        lig_pdbqt = prepare_ligand(mol_with_h)
        if lig_pdbqt and os.path.exists(lig_pdbqt):
            dock = run_vina_docking(
                receptor_pdbqt, lig_pdbqt,
                box_info['center'], box_info['size'])
            result.update(dock)

            # FIXED: PoseBusters check with proper PDBQT -> Mol conversion
            if dock.get('docked_pose') and posebuster_protein:
                # FIX: Use obabel to convert PDBQT to SDF (preserves bonds)
                docked_mol = pdbqt_to_mol_fixed(dock['docked_pose'], original_mol=mol)
                if docked_mol:
                    # Verify connectivity
                    frags = Chem.GetMolFrags(docked_mol, asMols=False, sanitizeFrags=False)
                    pb_result = run_posebusters(posebuster_protein, docked_mol)
                    pb_result['num_fragments'] = len(frags)
                    result['posebuster'] = pb_result
                else:
                    result['posebuster'] = {'passed': None, 'error': 'Failed to convert PDBQT to Mol'}
        else:
            result['docking_error'] = 'Failed to prepare ligand PDBQT'
    except Exception as e:
        result['docking_error'] = str(e)
    finally:
        if 'lig_pdbqt' in dir() and lig_pdbqt and os.path.exists(lig_pdbqt):
            try:
                os.unlink(lig_pdbqt)
            except:
                pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sdf', required=True, help='Path to merged SDF file')
    ap.add_argument('--pocket_pdb', required=True, help='Path to pocket PDB')
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument('--exhaustiveness', type=int, default=8)
    ap.add_argument('--num_modes', type=int, default=9)
    args = ap.parse_args()

    print(f"Loading molecules from {args.sdf}")
    suppl = Chem.SDMolSupplier(args.sdf)
    mols = [m for m in suppl if m is not None]
    print(f"Loaded {len(mols)} valid molecules")

    # Prepare receptor
    receptor_pdbqt = prepare_receptor(args.pocket_pdb)
    if not receptor_pdbqt:
        print("ERROR: Failed to prepare receptor")
        sys.exit(1)
    print(f"Receptor ready: {receptor_pdbqt}")

    # Compute pocket box
    box_info = compute_pocket_box(args.pocket_pdb)
    if not box_info:
        print("ERROR: Failed to compute pocket box")
        sys.exit(1)
    print(f"Box center: {box_info['center']}, size: {box_info['size']}")

    # Evaluate each molecule
    results = []
    for i, mol in enumerate(mols):
        print(f"  [{i+1}/{len(mols)}] Evaluating {Chem.MolToSmiles(mol)[:50]}...", flush=True)
        result = evaluate_molecule(
            mol, receptor_pdbqt, args.pocket_pdb, box_info,
            posebuster_protein=args.pocket_pdb
        )
        result['mol_index'] = i
        results.append(result)
        
        # Print summary
        dock_score = result.get('docking_score', 'N/A')
        pb_passed = (result.get('posebuster') or {}).get('passed', 'N/A')
        pb_frags = (result.get('posebuster') or {}).get('num_fragments', 'N/A')
        print(f"    dock={dock_score}, PB_passed={pb_passed}, frags={pb_frags}", flush=True)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    valid = [r for r in results if r.get('valid')]
    pb_passed = [r for r in valid if (r.get('posebuster') or {}).get('passed') is True]
    good_dock = [r for r in valid if r.get('docking_score') is not None and r['docking_score'] < -7.0]
    good_dock_pb = [r for r in good_dock if (r.get('posebuster') or {}).get('passed') is True]
    
    print(f"\n=== SUMMARY ===")
    print(f"Total: {len(results)}")
    print(f"Valid: {len(valid)}")
    print(f"PoseBuster passed: {len(pb_passed)} ({len(pb_passed)/max(len(valid),1)*100:.1f}%)")
    print(f"Docking < -7.0: {len(good_dock)}")
    print(f"PB pass among docking < -7.0: {len(good_dock_pb)} / {len(good_dock)}")


if __name__ == '__main__':
    main()
