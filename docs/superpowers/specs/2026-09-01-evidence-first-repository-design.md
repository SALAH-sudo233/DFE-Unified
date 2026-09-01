# DFE-Unified Evidence-First Repository Design

## Purpose

Create a private, clean-history GitHub repository at `SALAH-sudo233/DFE-Unified` that preserves the currently verifiable Pocket2Mol direction-field implementation, its 500,000-iteration checkpoint, and the completed portion of its multi-pocket evaluation without carrying forward credentials, temporary server operations, or unsupported scientific claims.

## Authority And Scope

- Upstream authority: `PengXingang/Pocket2Mol` commit `836a0c4ce487297ad24bc54ac2ebd163de13242c`.
- Implementation authority: the files actually used by `/workspace/ayb/Pocket2Mol` for the DF 500K run.
- Valid model artifact: `logs/checkpoints/500000.pt`, SHA-256 `34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0`.
- Valid experimental scope: 21 completed pockets from the interrupted 30-pocket run, containing 2,331 evaluated molecule records.
- Dataset policy: do not redistribute CrossDocked or PDBBind data; publish provenance, expected paths, sizes, and hashes only.
- Repository visibility: private.

## Repository Contents

- Preserve the upstream Pocket2Mol source tree and MIT license so the modified implementation remains runnable in its native module layout.
- Include the actual DF integration, training/resume, sampling, conversion, docking, metric, and partial multi-pocket scripts used by the run.
- Store the 500K checkpoint through Git LFS.
- Store per-pocket `merged_all.sdf`, `SMILES.txt`, and `docking_results.json` for the 21 completed pockets.
- Add machine-readable manifests that bind code, model, data split, logs, and results by SHA-256.
- Add documentation that distinguishes observed facts, limitations, and invalidated experiments.

## Exclusions

- No SSH, Paramiko, password, token, host-tunnel, heartbeat, cleanup, or remote administration scripts.
- No encoded or plaintext credentials.
- No incomplete ADF/BIF checkpoint. Its zero-gradient optimizer evidence is documented as an invalidated experiment.
- No claim that the partial evaluation is a completed 30-pocket benchmark.
- No claim of 100% generation validity. The retained validity metric is explicitly described as post-reconstruction and post-SDF filtering.
- No claim of E(3)/SE(3) equivariance or SOTA superiority.
- No server Git history or unreviewed dirty-tree snapshot.

## Verification Gates

1. Validate the model and source hashes against the remote server.
2. Confirm exactly 21 result directories and 2,331 result records.
3. Validate every JSON document and parse every SDF with RDKit when available.
4. Run focused source and artifact tests.
5. Scan tracked files and Git history for credentials and forbidden operational files.
6. Verify Git LFS tracks the checkpoint.
7. Push to the private repository and verify its visibility, default branch, commit, and LFS object.

## Delivery Boundary

Creating and pushing the private repository is authorized. Publishing a release, making the repository public, running new experiments, or changing the server is outside scope.
