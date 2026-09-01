#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge individual SDF files produced by sample_for_pdb.py into a single SDF.

sample_for_pdb.py writes one SDF per generated molecule under
<outdir>/<config>_<pdb>_<timestamp>/SDF/*.sdf .  evaluate_docking.py expects a
single SDF (or a SMILES file), so this helper concatenates them.

Uses RDKit's SDMolSupplier + SDWriter to guarantee correct SDF format,
avoiding the text-concatenation issues that can cause RDKit parse errors.

Usage:
    python convert_samples_to_sdf.py --input <dir> --output <merged.sdf>
        # --input may be the timestamped log dir (contains SDF/) or a parent
        #   directory; the script locates the SDF/ subdir automatically.
"""
import argparse
import glob
import os
import sys

from rdkit import Chem


def find_sdf_dir(base):
    """Return the SDF/ directory under `base`, or None."""
    direct = os.path.join(base, 'SDF')
    if os.path.isdir(direct):
        return direct
    # search one level down for */SDF and pick the newest
    candidates = []
    for sub in glob.glob(os.path.join(base, '*')):
        d = os.path.join(sub, 'SDF')
        if os.path.isdir(d):
            candidates.append(d)
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Sampling output dir (contains SDF/).')
    ap.add_argument('--output', required=True, help='Path of merged SDF file.')
    args = ap.parse_args()

    sdf_dir = find_sdf_dir(args.input)
    if sdf_dir is None:
        sys.stderr.write('ERROR: no SDF/ directory found under %s\n' % args.input)
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(sdf_dir, '*.sdf')))
    if not files:
        sys.stderr.write('ERROR: no *.sdf files in %s\n' % sdf_dir)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    n_loaded = 0
    n_written = 0
    with Chem.SDWriter(args.output) as writer:
        for f in files:
            suppl = Chem.SDMolSupplier(f)
            for mol in suppl:
                if mol is None:
                    continue
                n_loaded += 1
                # Use the original filename stem as molecule name for traceability
                stem = os.path.splitext(os.path.basename(f))[0]
                mol.SetProp('_Name', stem)
                writer.write(mol)
                n_written += 1

    print('Merged %d / %d molecules -> %s' % (n_written, n_loaded, args.output))

    # Verify the output
    verify = Chem.SDMolSupplier(args.output)
    verify_count = sum(1 for m in verify if m is not None)
    print('Verification: %d molecules in output' % verify_count)


if __name__ == '__main__':
    main()