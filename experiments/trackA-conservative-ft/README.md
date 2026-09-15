# Track A — 530k regression diagnosis + conservative fine-tune

## Context

The 530k continuation of DF-500k **regressed** on CrossDocked (see
`../cd100-diagnosis/` pilot and `../../docs/experiments/530k-regression-diagnosis-report.md`).
Root cause: the resume inherited the 500k optimizer/scheduler state (effective
LR 1e-5, full-model update). This directory holds the diagnosis and the
conservative fine-tune designed to replace that resume semantics.

## Code (`code/`)

- `diagnose_530k.py`, `param_drift.py`, `comparable_recon.py` — parameter-level
  drift analysis across 500k/528k/530k (drift concentrated in encoder
  interaction/message path, not frontier/position head).
- `train_conservative.py`, `train_A.py`, `train_B.py`, `train_A_fixed.py`,
  `train_B_fixed.py` — conservative fine-tune runners. All start from
  `500000.pt` (model only), **reset the optimizer**, use fixed short-horizon LR,
  and write to isolated logdirs. A = output heads only; B = heads + position.
  `_fixed` variants correct a freeze-set / trainable-parameter-naming bug found
  in dry-run.
- `watchdog.py` — sampling/eval watchdog used for the 1a9q eval.

## Configs (`configs/`)

`train*.yml` (formal + dry-run) and `conservative_finetune_configs.yml` (the
reviewed A–E design). Priority from the review: **E > A > B > C > D**.

## Records (`records/`)

- `head_drift_500_528_530.json`, `param_drift.json` — drift analysis outputs.
- `eval_530k_1a9q_docking_results.json`, `eval_530k_1a9q_watch_contract.json`.

## Status

Conservative fine-tune framework is ready and dry-run-gated. The Plan A formal
5000-step checkpoint (registered, not committed) has **not** yet passed the
paired acceptance gate (Vina not worse than baseline by >0.10; LE not down;
direct PB not down >2pp). Not accepted as an A/B candidate yet.
