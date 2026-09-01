# DFE-Unified Evidence-First Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and push a private, clean-history `DFE-Unified` repository containing the verifiable Pocket2Mol-DF 500K implementation, checkpoint, and partial 21-pocket evidence set.

**Architecture:** Start from the pinned upstream Pocket2Mol source, overlay only the remote files that can be tied to the 500K run, and add evidence manifests and limitations documentation. Large model content uses Git LFS; external datasets remain references rather than redistributed artifacts.

**Tech Stack:** Git, Git LFS, GitHub CLI, Python 3.10+, PyTorch, RDKit, YAML, JSON, SHA-256.

## Global Constraints

- The GitHub repository is `SALAH-sudo233/DFE-Unified` and remains private.
- Remote server access is read-only.
- No credentials or remote administration scripts may enter the repository.
- The published experimental scope is exactly the completed 21-pocket partial run.
- The ADF/BIF 400K experiment is documented as invalidated and its checkpoint is excluded.
- No release publication, public visibility change, or new training is authorized.

---

### Task 1: Establish The Clean Source Baseline

**Files:**
- Create: repository root from upstream Pocket2Mol commit `836a0c4ce487297ad24bc54ac2ebd163de13242c`
- Create: `UPSTREAM.md`

**Interfaces:**
- Consumes: pinned upstream commit and MIT license.
- Produces: a clean source tree to receive the verified overlay.

- [ ] Initialize Git with branch `main` and configure Git LFS for `*.pt`.
- [ ] Materialize the pinned upstream source without importing its Git history.
- [ ] Record upstream URL, commit, license, and retrieval date in `UPSTREAM.md`.
- [ ] Commit the baseline and design documents.

### Task 2: Import The Verified DF Implementation

**Files:**
- Modify: native Pocket2Mol source and configuration files used by the remote run.
- Create: `evidence/code-manifest.json`
- Create: `tests/test_repository_evidence.py`

**Interfaces:**
- Consumes: read-only files under `/workspace/ayb/Pocket2Mol`.
- Produces: source files bound to remote SHA-256 values.

- [ ] Download only the allowlisted model, training, sampling, evaluation, and configuration files.
- [ ] Write a failing manifest test for missing files or hash mismatches.
- [ ] Generate the code manifest with relative path, byte size, SHA-256, and remote modification time.
- [ ] Run the manifest test and require all allowlisted hashes to match.
- [ ] Commit the verified implementation overlay.

### Task 3: Import Model And Experimental Evidence

**Files:**
- Create: `artifacts/checkpoints/df-500k.pt`
- Create: `artifacts/MANIFEST.json`
- Create: `results/df-500k-21-pocket/per-pocket/*`
- Create: `results/df-500k-21-pocket/summary.json`
- Create: `results/df-500k-21-pocket/provenance.json`

**Interfaces:**
- Consumes: remote checkpoint, training log boundary, aggregate JSON, and 21 per-pocket directories.
- Produces: hash-bound model and raw result evidence.

- [ ] Download the 500K checkpoint and verify SHA-256 before staging.
- [ ] Download the 21 completed result directories and aggregate result JSON.
- [ ] Write tests requiring 21 pockets, 2,331 JSON records, and a loadable iteration-500000 checkpoint.
- [ ] Generate artifact and result manifests with hashes and counts.
- [ ] Run JSON, count, hash, checkpoint, and optional RDKit SDF validation.
- [ ] Commit the model and result evidence with checkpoint content stored by Git LFS.

### Task 4: Document Provenance And Limitations

**Files:**
- Create: `README.md`
- Create: `data/README.md`
- Create: `data/crossdocked-manifest.json`
- Create: `docs/architecture.md`
- Create: `docs/methodology-limitations.md`
- Create: `docs/reproduction.md`
- Create: `evidence/evidence-index.md`
- Create: `evidence/invalidated-experiments/adf-bif-400k.md`

**Interfaces:**
- Consumes: verified manifests and observed optimizer/result evidence.
- Produces: claims whose language does not exceed the evidence.

- [ ] Describe the implemented DF as a learned heuristic feature projection, not a proven physical or equivariant field.
- [ ] Explain post-filter validity, incomplete pocket coverage, PoseBusters pass rate, and lack of matched baseline.
- [ ] Record CrossDocked target sizes and split hash without redistributing data.
- [ ] Record the ADF/BIF zero-gradient optimizer-state finding and exclude that model from valid artifacts.
- [ ] Link every completion claim to a manifest or raw result path.
- [ ] Commit documentation and evidence indices.

### Task 5: Security And Reproducibility Verification

**Files:**
- Create: `scripts/verify_repository.py`
- Create: `SECURITY.md`
- Modify: `.gitignore`
- Modify: `.gitattributes`

**Interfaces:**
- Consumes: the complete proposed repository.
- Produces: a machine-checkable pass/fail verification report.

- [ ] Scan tracked content for password, token, private-key, Paramiko, tunnel host, and credential-encoding patterns.
- [ ] Reject forbidden operational filenames and files over the GitHub non-LFS size limit.
- [ ] Verify manifests, JSON parsing, result counts, checkpoint hash, and LFS tracking.
- [ ] Run the full test suite and verification script with zero failures.
- [ ] Inspect `git status`, `git diff --check`, and tracked-file inventory.
- [ ] Commit the final verification tooling.

### Task 6: Create And Push The Private GitHub Repository

**Files:**
- No new source files.

**Interfaces:**
- Consumes: verified local `main` branch.
- Produces: private `SALAH-sudo233/DFE-Unified` on GitHub.

- [ ] Create the empty private GitHub repository without generated README or license files.
- [ ] Add the GitHub remote and push `main` including Git LFS objects.
- [ ] Verify repository visibility is private and the default branch is `main`.
- [ ] Verify the remote HEAD equals the local HEAD and the checkpoint is represented as an LFS pointer in Git.
- [ ] Report the repository URL, commit, included evidence, exclusions, and verification results.
