# SCI-1 SE(3) Audit and SCI-2A DF Feature Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scientific audit layer that consumes Issue #2 execution artifacts to locate SE(3) failures in the complete DF path and attribute the current frozen checkpoint's behavior to its eight raw DF features without changing generation policy.

**Architecture:** Issue #2 owns immutable manifests, hashes, run roots, ledgers, scheduler/resume, D0 parity, observer/trace plumbing, and openness calculation. This slice adds only science-facing adapters and analysis: a transformation-law evaluator, a frozen feature-intervention matrix, and reports that reference (rather than rewrite) Issue #2 artifacts. The production model path remains unchanged when the science adapter is disabled.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, JSON/JSONL, pandas, unittest, SHA-256.

## Global Constraints

- Baseline is `origin/main` commit `c1de0af483d89339007b81dd4d302a667bf1ccdb`.
- Use experiment IDs `SCI-1-SE3-v1` and `SCI-2A-FEATURE-v1`; outputs must live under a new versioned run root and never overwrite `artifacts/checkpoints/df-500k.pt` or `results/df-500k-21-pocket`.
- Consume Issue #2 artifacts and interfaces; do not create a second manifest builder, run-root allocator, attempt ledger, scheduler, resume protocol, D0 parity implementation, observer transport, or openness calculator.
- SCI-2A is frozen-checkpoint inference attribution. It is not a trained no-DF baseline and must not be reported as causal training evidence.
- Every report retains `attempts`, `generated`, `reconstructed`, `valid`, `dockable`, and `checked` counts. Missing downstream evaluation is missing data, never an imputed score.
- Analytical float64 target is `1e-8`; model float32 normalized target is `1e-4`. A failed scientific hypothesis still emits a complete report and exit status distinct from infrastructure failure.
- Before any GPU or long run, validate finite gradients/outputs, checkpoint optimizer-state expectations, and tiny deterministic fixtures.

## Files and Interfaces

- Create `dfe/science/__init__.py`: public package surface.
- Create `dfe/science/artifact_contract.py`: read-only validation of Issue #2 manifest and artifact hashes.
- Create `dfe/science/se3_audit.py`: transformation laws, error summaries, and first-divergence bookkeeping.
- Create `scripts/run_sci1_se3_audit.py`: CLI consuming a completed Issue #2 manifest and observer traces, writing `se3-audit.json` create-only.
- Create `dfe/science/feature_interventions.py`: immutable SCI-2A intervention definitions and raw 8D transformations.
- Create `scripts/run_sci2a_feature_interventions.py`: CLI that invokes the existing sampler/model with the intervention hook and writes per-stage aggregate JSON/CSV.
- Create `tests/science/test_artifact_contract.py`, `tests/science/test_se3_audit.py`, and `tests/science/test_feature_interventions.py`.
- Modify `models/df_module.py` only if needed to expose a side-effect-free `raw_features(...)` helper; preserve `forward(...)` numerics for the default path.
- Modify `models/maskfill.py` only to consume the existing Issue #2 diagnostic hook; no new scheduler, ledger, or sampling policy.

### Task 1: Add the Issue #2 artifact contract adapter

**Files:**
- Create: `dfe/science/artifact_contract.py`
- Create: `dfe/science/__init__.py`
- Test: `tests/science/test_artifact_contract.py`

**Interfaces:**
- `load_science_manifest(path: Path) -> ScienceManifest`
- `ScienceManifest.require(experiment_id: str, checkpoint_sha256: str) -> None`
- `verify_artifact_hash(path: Path, expected_sha256: str) -> None`
- `assert_create_only_output(path: Path) -> None`

- [ ] **Step 1: Write failing tests**

```python
def test_manifest_rejects_wrong_issue2_experiment(self):
    manifest = load_science_manifest(self.manifest_path)
    with self.assertRaises(ValueError):
        manifest.require("SCI-1-SE3-v1", "wrong")

def test_output_contract_is_create_only(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "se3-audit.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            assert_create_only_output(path)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.science.test_artifact_contract -v`
Expected: import failure because `dfe.science.artifact_contract` is absent.

- [ ] **Step 3: Implement the adapter**

Parse JSON with `encoding="utf-8"`; require `manifest_schema`, `manifest_sha256`, checkpoint hash, source commit, and an explicit reference to the Issue #2 run root. Hash files with chunked SHA-256. Reject absolute output paths, mixed schema versions, and any manifest whose checkpoint hash differs from the pinned 500K hash. `assert_create_only_output` must fail before opening an existing path.

- [ ] **Step 4: Run focused and repository tests**

Run: `python -m unittest tests.science.test_artifact_contract -v`
Expected: all contract tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dfe/science tests/science/test_artifact_contract.py
git commit -m "feat: validate science inputs against phase0 artifacts"
```

### Task 2: Implement transformation-law and first-failure analysis

**Files:**
- Create: `dfe/science/se3_audit.py`
- Test: `tests/science/test_se3_audit.py`

**Interfaces:**
- `compare_invariant(reference: Tensor, transformed: Tensor) -> ErrorSummary`
- `compare_vector(reference: Tensor, transformed: Tensor, rotation: Tensor) -> ErrorSummary`
- `compare_position(reference: Tensor, transformed: Tensor, rotation: Tensor, translation: Tensor) -> ErrorSummary`
- `first_discrete_divergence(reference: Sequence[int], transformed: Sequence[int]) -> int | None`
- `summarize_errors(errors: Sequence[float], tolerance: float) -> dict[str, Any]`

- [ ] **Step 1: Write failing numerical tests**

```python
def test_rotation_and_translation_laws(self):
    x = torch.tensor([[1., 2., 3.]])
    r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    t = torch.tensor([4., -2., 1.])
    self.assertLess(compare_vector(x, x @ r.T, r).max_error, 1e-8)
    self.assertLess(compare_position(x, x @ r.T + t, r, t).max_error, 1e-8)

def test_first_divergence_is_stable(self):
    self.assertEqual(first_discrete_divergence([1, 2, 3], [1, 9, 3]), 1)
    self.assertIsNone(first_discrete_divergence([1, 2], [1, 2]))
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m unittest tests.science.test_se3_audit -v`
Expected: import failure because the audit module is absent.

- [ ] **Step 3: Implement stable error summaries**

Map scalar/logit/sigma/pi events to invariant comparison; raw direction, VN/GVP vectors, and relative position means to rotation comparison; absolute positions to `x @ R.T + t`. Normalize float32 errors by `max(reference.abs().max(), 1.0)`. Return `max`, `median`, `p95`, `count`, `tolerance`, and `passed`; reject non-finite tensors. Keep first discrete mismatch while continuing comparisons on shared prefixes.

- [ ] **Step 4: Run tests and commit**

Run: `python -m unittest tests.science.test_se3_audit -v`
Expected: all numerical and failure-bookkeeping tests PASS.

```bash
git add dfe/science/se3_audit.py tests/science/test_se3_audit.py
git commit -m "feat: add stage-level se3 transformation analysis"
```

### Task 3: Add the SCI-1 trace consumer CLI

**Files:**
- Create: `scripts/run_sci1_se3_audit.py`
- Modify: `dfe/science/artifact_contract.py`
- Test: `tests/science/test_se3_audit.py`

**Interfaces:**
- CLI: `python scripts/run_sci1_se3_audit.py --manifest <issue2-manifest> --trace <trace-root> --output <se3-audit.json> --rotations 100 --translations 10`
- `run_audit(manifest: Path, trace_root: Path, output: Path, rotations: int, translations: int, device: str) -> dict`

- [ ] **Step 1: Add CLI fixture tests**

Create a synthetic manifest with 20 states and observer events for `df.raw`, `df.hidden`, `df.projected`, encoder scalar/vector, frontier, position, element, bond, and termination. Assert the report includes every declared event, first failure, all error quantiles, transform counts, and input/checkpoint hashes.

- [ ] **Step 2: Implement trace consumption**

Load only traces declared by the manifest; pair transformed states by `(pocket_id, partial_state_id, transform_id)` and use shared random seeds. Require at least 20 states, exactly the requested transform counts, and finite tensors. Exit `0` when all laws pass, `2` when scientific thresholds fail with a complete report, and `1` for missing/corrupt inputs. Write output through `assert_create_only_output` and include a `status` of `pass` or `scientific_fail`.

- [ ] **Step 3: Verify**

Run: `python -m unittest tests.science.test_se3_audit -v && python scripts/run_sci1_se3_audit.py --help`
Expected: tests PASS and help exits `0` without touching output files.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sci1_se3_audit.py dfe/science tests/science/test_se3_audit.py
git commit -m "feat: add sci1 se3 audit report consumer"
```

### Task 4: Define frozen SCI-2A feature interventions

**Files:**
- Create: `dfe/science/feature_interventions.py`
- Modify: `models/df_module.py`
- Modify: `models/maskfill.py`
- Test: `tests/science/test_feature_interventions.py`

**Interfaces:**
- `INTERVENTIONS: tuple[FeatureIntervention, ...]`
- `FeatureIntervention.apply(raw_features: Tensor, *, seed: int) -> Tensor`
- `FeatureIntervention.provenance() -> dict[str, Any]`
- `AnalyticalDirectionField.raw_features(...) -> Tensor`
- `MaskFillModelVN.set_science_intervention(intervention: FeatureIntervention | None) -> None`

- [ ] **Step 1: Write intervention tests**

Cover full 8D identity; direction-zero columns `1:4` zeroed; deterministic direction randomization with a seed; one-at-a-time removal of `distance_sq` (column 6), `inverse_distance` (column 7), charge proxy (column 4), and hydrophobic proxy (column 5); and all non-distance attributes zeroed. Assert input tensors are not mutated and repeated seeds are identical.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m unittest tests.science.test_feature_interventions -v`
Expected: import or attribute failures because the intervention API is absent.

- [ ] **Step 3: Implement side-effect-free raw feature access**

Refactor the existing feature construction into `raw_features` and have `forward` call it followed by the unchanged `field_proj`. Keep the default `MaskFillModelVN` path numerically identical. Install the intervention only behind an explicit science hook; when unset, execute the legacy call path.

- [ ] **Step 4: Verify model parity and intervention behavior**

Run: `python -m unittest tests.science.test_feature_interventions -v`
Expected: identity intervention equals legacy raw features bit-for-bit within dtype; all other interventions alter only declared columns.

- [ ] **Step 5: Commit**

```bash
git add dfe/science/feature_interventions.py models/df_module.py models/maskfill.py tests/science/test_feature_interventions.py
git commit -m "feat: add frozen df feature attribution interventions"
```

### Task 5: Add the SCI-2A sampler/report wrapper and local gates

**Files:**
- Create: `scripts/run_sci2a_feature_interventions.py`
- Create: `tests/science/test_sci2a_cli.py`
- Modify: `scripts/verify_repository.py`

**Interfaces:**
- CLI: `python scripts/run_sci2a_feature_interventions.py --manifest <issue2-manifest> --checkpoint <df-500k.pt> --output-root <new-root> --devices cpu|cuda:N --attempts 10|20`
- `run_interventions(...) -> dict[str, Any]`

- [ ] **Step 1: Write deterministic wrapper tests**

Use a fake model and sampler to assert every intervention uses the same pocket order, seed, sampling policy, and checkpoint hash; output rows include stage metrics and all six denominator fields; rerunning against an existing output root fails without rewriting files.

- [ ] **Step 2: Implement wrapper by consuming Issue #2**

Resolve jobs and attempt IDs from the Issue #2 manifest/ledger APIs. Register only SCI-2A intervention metadata and call the existing sampler. Do not add retries, alter frontier thresholds, or create a second ledger. Emit per-stage `df.raw`, `df.hidden`, `df.projected`, frontier, position, element/bond, termination, reconstruction, docking, and pose summaries with first-failure codes supplied by Issue #2.

- [ ] **Step 3: Add repository verification hooks**

Verify the two science experiment IDs, plan/spec paths, importability of science modules, and absence of tracked absolute server paths. Keep verification independent of external checkpoints and datasets.

- [ ] **Step 4: Run complete local gate**

Run:

```bash
python -m unittest tests.science -v
python -m unittest discover -s tests -v
python -m py_compile dfe/science/*.py scripts/run_sci*.py
python scripts/verify_repository.py
git diff --check
```

Expected: all tests PASS, compilation and repository verification exit `0`, and no historical evidence files change.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sci2a_feature_interventions.py tests/science/test_sci2a_cli.py scripts/verify_repository.py
git commit -m "feat: run auditable sci2a feature interventions"
```

## Handoff and stop rule

After this plan is reviewed and implemented, stop before SCI-2B short training. Review `se3-audit.json` and the SCI-2A report first. Only a subsequent approved design delta may add invariant/vector DF candidates, width/depth sweeps, fixed-candidate DF+BIF ranking, or openness interaction analysis.

## Stage gate and retry policy

Every stage emits one of three non-ambiguous statuses through
`dfe.science.gates.evaluate_gate`:

- `pass`: evidence is complete and all preregistered thresholds pass; the next
  stage may start.
- `scientific_fail`: evidence is complete but a scientific threshold fails;
  stop the stage, preserve its report, perform literature/code review and a
  targeted fix or revised design, then create a new versioned run before retry.
- `blocked`: required input, dependency, remote host, authentication, or
  execution infrastructure is unavailable; repair that condition first and do
  not interpret the result scientifically.

The helper `retry_action(gate)` returns the required next action. A stage may
not advance on `scientific_fail` or `blocked`, and retries must not overwrite
the failed run root or alter seeds, denominators, thresholds, or checkpoint
hashes.

Current execution status: local implementation gates pass; real AIDD
execution is `blocked` pending GitHub push/network recovery and valid `ssh
aidd` authentication. No scientific conclusion is drawn from this block.

Plan complete and saved to `docs/superpowers/plans/2026-09-02-sci1-se3-and-sci2a.md`. Two execution options:

1. **Subagent-Driven (recommended):** dispatch a fresh subagent per task with review checkpoints.
2. **Inline Execution:** execute the tasks in this session with batch checkpoints.
