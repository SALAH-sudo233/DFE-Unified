# CrossDocked-100 diagnosis pilot (Track A)

Paired evaluation orchestration used to diagnose the 530k regression on
CrossDocked, and the reusable harness for future generation-quality pilots.

## Code (`code/`)

- `dfe_pilot_launcher.py`, `dfe_pilot_clean_launcher*.py` — schedule paired
  sampling runs (model × pocket × seed) with create-only run roots.
- `worker.py`, `evalworker.py`, `cleaneval_worker.py`, `clean_eval.py` —
  sampling + docking/PoseBusters/QED evaluation workers.
- `diagnosis_pilot_aggregator.py` (+ `_local.py`) — strict aggregation:
  accepts a run only if all expected entries are present, records the
  missing/rejected list, computes per-seed mean/SD and bootstrap 95% CI.
- `monitor.py`, `read_abl.py` — progress monitor and ablation reader helpers.

## Records (`records/`)

- `diagnosis_pilot_v2_summary.json` — **72/72 accepted** (12 pockets × 2 models
  × 3 seeds), 0 rejected. Paired delta (530k − DF-500k): Vina mean
  +0.19 kcal/mol [+0.03, +0.35], direct PB pass −0.055, heavy atoms −0.44.
  → 530k regressed; stop reusing it.
- `manifest.json` — the 100-task CrossDocked manifest for this pilot.

## Note

This pilot's absolute numbers are not the user's earlier 93-pocket preliminary
result and are not a CrossDocked-100 SOTA claim; the sample scope differs. Used
only for the paired 530k-vs-500k regression decision.
