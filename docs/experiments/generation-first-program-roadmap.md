# Generation-First Unified Experimental Program

## Authority

The authoritative research contract is
`docs/superpowers/specs/2026-09-01-generation-first-unified-experiment-design.md`.
This roadmap orders its independently gated subprojects. It does not authorize
skipping a gate or turning a diagnostic observation into a performance claim.

## Program sequence

| Stage | Fixed input | Deliverable | Long training allowed? |
| --- | --- | --- | --- |
| P0: DF 500K diagnosis | Existing hash-pinned DF 500K | Openness manifest, SE(3) audit, interventions, attempt ledger, failure localization | No |
| P1: DF-v2 selection | P0 failure location and error spectrum | At most two SE(3)-safe candidates after 20–50K short runs | Short runs only |
| P2: BIF validation | Versioned biochemical feature schema | Gradient/optimizer proof, tiny overfit, contact and element/bond probes | Probe training only |
| P3: ScreeningHead | Leak-free protein/scaffold split | Calibrated affinity/ranking model and counterfactual controls | Yes, independent branch |
| P4: Unified-v1 | Frozen DF candidate pools and validated ranker | Constraint-first post-generation selection with <=15% online overhead | No generator training |
| P5: Unified-v2 | All prior gates passed | Step reranking, bounded rejection, then one-way stop-gradient BIF injection | Only gated candidates |

P0 is the only stage whose implementation can be fully specified before new
measurements. P1 architecture is selected from the observed P0 failure mode.
P2 depends on the audited availability of biochemical labels and atom typing.
P3 depends on the chosen affinity dataset and its connected-component split.
P4 and P5 depend on the actual P2/P3 gates. Each later stage therefore receives
its own design delta and implementation plan after its predecessor is accepted.

## Experiment identifiers

| ID | Meaning |
| --- | --- |
| `P0-MANIFEST-v1` | Frozen 30-pocket inputs, hashes, centers, covariates and checkpoint |
| `P0-OPENNESS-v1` | 2,048-ray enclosure/openness calculation and label audit |
| `P0-SE3-v1` | Raw DF through model-head SE(3) error audit |
| `P0-INT-SMOKE-v1` | Six-pocket, one-seed intervention smoke |
| `P0-INT-MAIN-v1` | Thirty-pocket, three-seed intervention run |
| `P0-TRACE-v1` | Autoregressive stage trace and first-failure taxonomy |
| `P0-STATS-v1` | Pocket-clustered effect and openness-interaction analysis |

The preregistered seeds are `20260901`, `20260902`, and `20260903`. Smoke uses
only `20260901`. The first six smoke pockets are selected after openness is
computed as the two lowest, two nearest the median, and two highest openness
values; selection uses no model outcome.

## Resource schedule

The scheduler treats GPUs as independent workers and never requires distributed
training for P0. A sampling job is one `pocket x seed x intervention` process.
Smoke jobs request 10 attempts; main jobs request 20 attempts.

| Available GPU | Scheduler action |
| ---: | --- |
| 0 | Build manifests, hashes, openness, CPU tests, summaries and statistics |
| 1 | Execute one SE(3), trace or sampling job at a time |
| 2 | Pair baseline/intervention jobs or run two seeds of the same arm |
| 3 | Run all three seeds for one arm concurrently |
| 4 | Run three seeds plus one smoke/baseline worker |

The expanded matrix has nine arms: D0, D1, D2, D3, D4 and D5 gates at 0.25,
0.5, 1.0 and 1.5. `P0-INT-SMOKE-v1` therefore contains 54 jobs and at most 540
attempts. `P0-INT-MAIN-v1` contains 90 jobs per retained arm and at most 810
jobs/16,200 attempts when all nine arms pass smoke. Main is not launched until
the smoke report proves deterministic manifests, complete attempt accounting,
normal-mode parity and recoverable outputs.

## Gate reviews

Every gate report has four machine-readable states: `pass`, `fail`, `blocked`,
or `inconclusive`. `blocked` is reserved for unavailable inputs or infrastructure;
`inconclusive` is an experimental result and must not be silently retried with
new thresholds. Threshold, seed, pocket or endpoint changes create a new
versioned experiment ID.

P0 completion creates the design decision for P1:

- projected scalar SE(3) failure selects `DF-invariant` and `DF-vector` probes;
- SE(3) passes but low-confidence open-pocket failure selects multiscale
  concentration/gating probes;
- no stable openness interaction pauses the open/closed narrative and treats DF
  as a general intervention;
- failure only after reconstruction pauses model changes and prioritizes the
  molecule construction/evaluation pipeline.

No stage overwrites `artifacts/checkpoints/df-500k.pt` or the retained
`results/df-500k-21-pocket` evidence.
