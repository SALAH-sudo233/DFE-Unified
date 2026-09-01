# DF 500K Phase 0 Operator Runbook

This runbook executes the frozen, non-training Phase 0 protocol. It does not
modify or resume the DF 500K checkpoint and does not reuse the historical
`/workspace/ayb/Pocket2Mol` checkout as source.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Command completed and its declared gate passed |
| `1` | Infrastructure, dependency, input or artifact-integrity failure |
| `2` | Complete scientific audit with a failed hypothesis or gate |
| `130` | Scheduler stopped after SIGINT; workers were allowed to checkpoint |

## 1. Read-only preflight

Connect with `ssh aidd`. Record hostname, `nvidia-smi` inventory and free
memory, free disk, Python/PyTorch/CUDA/RDKit versions, project-user processes,
and existence/readability/hash of the expected PDBBind inputs and DF checkpoint.
Do not create directories in this step. Confirm that the proposed run root does
not exist.

The frozen first run root is:

```text
/workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1
```

If it exists, stop and choose a new explicit version. Never delete, clean,
overwrite or reuse an experiment root.

## 2. Isolated source checkout

Fetch the private GitHub repository using the server's existing authorized Git
mechanism. Check out the exact pushed implementation commit in a new directory,
run `git lfs pull`, verify the checkpoint SHA-256, and require `git status
--porcelain` to be empty. Do not copy source from the historical checkout.

```bash
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

## 3. Freeze manifest

```bash
python scripts/build_phase0_manifest.py \
  --config configs/diagnostics/phase0_df500k.yaml \
  --pdbbind-root /workspace/ayb/data/pdbbind/v2020 \
  --centers pocket_centers_30.json \
  --output-root /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1 \
  --require-clean
```

Manifest creation is create-only. It validates all 30 pockets, centers,
sampling policy and checkpoint before creating the run root.

## 4. Openness and SE(3)

```bash
python scripts/compute_pocket_openness.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json

python scripts/run_se3_audit.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --device cuda:0 --rotations 100 --translations 10
```

Exit `2` from the SE(3) command is a complete scientific failure report, not an
infrastructure crash. Preserve `se3-audit.json` in either case.

## 5. Dry-run and smoke

Pass only GPUs verified as free in the preflight. Valid examples are `none`,
`cuda:0`, or `cuda:0,cuda:2`. The scheduler never claims unspecified devices.

```bash
python scripts/run_phase0_jobs.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --stage smoke --devices cuda:0 --dry-run
```

Inspect the exact 54-job list. First run a one-attempt D0 parity check against
the retained sampler with the same seed and write the create-only
`d0-parity.json`. Then launch smoke without `--dry-run`.

```bash
python scripts/run_phase0_jobs.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --stage smoke --devices cuda:0

python scripts/summarize_phase0.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json

python scripts/analyze_phase0.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --stage smoke
```

Do not start main unless `gate-smoke.json` says `pass`.

## 6. Main, summary and analysis

```bash
python scripts/run_phase0_jobs.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --stage main --devices cuda:0,cuda:1,cuda:2,cuda:3

python scripts/summarize_phase0.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json

python scripts/analyze_phase0.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --stage main
```

With zero available GPUs, use `--devices none` to list pending work and continue
only CPU validation, summaries or analysis of already terminal jobs.

## Resume and storage

Every attempt is declared before model initialization. JSONL evidence is
append-only and fsync-backed; job checkpoints are atomically replaced. Use the
same command after interruption. The scheduler adds `--resume` only for an
existing nonterminal ledger, closes interrupted attempts explicitly, skips
terminal jobs, and never automatically retries failed attempts.

Budget storage before smoke for ledgers, traces, serialized candidates and SDF;
measure actual bytes per attempt after the first D0 job and extrapolate to 540
smoke attempts and at most 16,200 main attempts. Keep Vina outputs outside the
online-overhead timing boundary.

## Final hash verification

```bash
python scripts/summarize_phase0.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --verify-only
python scripts/analyze_phase0.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --stage main --verify-only
sha256sum /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/*.json
git status --short
```

Record only content-free evidence in Git: implementation commit, manifest hash,
job/attempt counts, gate status, runtime and GPU model. Do not commit raw
licensed datasets, credentials or absolute-path run artifacts.
