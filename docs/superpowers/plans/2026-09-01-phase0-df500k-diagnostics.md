# DF 500K Phase 0 Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, non-training Phase 0 diagnostic system for the existing DF 500K checkpoint that quantifies pocket openness, audits SE(3), performs inference interventions, records every generation attempt, and localizes open-pocket failures.

**Architecture:** Add a focused `dfe.diagnostics` package for immutable run contracts, geometry, SE(3), interventions, tracing, ledgers and statistics. Existing Pocket2Mol behavior remains the production path; model and sampler receive default-off diagnostic interfaces whose `normal` mode must be numerically identical. Scripts construct a frozen input manifest, execute CPU/GPU stages as restartable jobs, and generate gate reports from append-only artifacts.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, pandas, PyYAML, RDKit, PyTorch Geometric, Git LFS, unittest, JSON/JSONL, SHA-256.

## Global Constraints

- The model anchor is `artifacts/checkpoints/df-500k.pt`, iteration `500000`, SHA-256 `34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0`.
- Never modify, replace, or resume training from the anchor during Phase 0.
- The existing 21-pocket evidence remains immutable and is not merged into a new run directory.
- `normal` diagnostics must preserve current model tensors and candidate decisions within exact equality where deterministic and `rtol=1e-7, atol=1e-8` otherwise.
- A generation attempt enters the denominator before model initialization; failures are terminal records, not dropped rows.
- Smoke jobs contain exactly 10 attempts; main jobs contain exactly 20 attempts.
- Main seeds are exactly `20260901`, `20260902`, and `20260903`; smoke uses `20260901`.
- Openness uses exactly 2,048 Fibonacci directions, 12 Angstrom maximum ray length, and Bondi-style van der Waals radii recorded in the manifest.
- Primary float32 SE(3) normalized-error gate is `<1e-4`; analytical float64 DF gate is `<1e-8`.
- P0 performs no training and no scientific threshold changes after observing model outcomes.
- AIDD writes only to a new confirmed checkout and `/workspace/ayb/experiments/dfe-unified-phase0`; the existing `/workspace/ayb/Pocket2Mol` is read-only evidence.
- No SSH credentials, server operation helpers, raw licensed datasets, or absolute workstation paths enter Git.

---

### Task 1: Define Immutable Run Contracts and Phase 0 Configuration

**Files:**
- Create: `dfe/__init__.py`
- Create: `dfe/diagnostics/__init__.py`
- Create: `dfe/diagnostics/contracts.py`
- Create: `configs/diagnostics/phase0_df500k.yaml`
- Create: `tests/diagnostics/test_contracts.py`

**Interfaces:**
- Consumes: repository commit, checkpoint path/hash, pocket input paths and experiment configuration.
- Produces: `RunManifest`, `PocketRecord`, `JobSpec`, `AttemptRecord`, `load_phase0_config(path)`, `write_new_manifest(path, manifest)`, and canonical JSON hashing.

- [ ] **Step 1: Write failing contract tests**

```python
# tests/diagnostics/test_contracts.py
import json
import tempfile
import unittest
from pathlib import Path

from dfe.diagnostics.contracts import (
    AttemptRecord,
    canonical_sha256,
    write_new_manifest,
)


class ContractTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}),
                         canonical_sha256({"b": 2, "a": 1}))

    def test_manifest_is_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_new_manifest(path, {"experiment_id": "P0-MANIFEST-v1"})
            with self.assertRaises(FileExistsError):
                write_new_manifest(path, {"experiment_id": "changed"})

    def test_attempt_requires_all_attempt_denominator_fields(self):
        record = AttemptRecord.new("run", "10gs", 20260901, "D0", 0)
        payload = record.to_dict()
        self.assertEqual(payload["status"], "requested")
        self.assertEqual(payload["attempt_id"], "run:10gs:20260901:D0:0000")
        self.assertIn("sampling_status", payload)
        self.assertIn("reconstruction_status", payload)
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `python -m unittest tests.diagnostics.test_contracts -v`

Expected: `ModuleNotFoundError: No module named 'dfe'`.

- [ ] **Step 3: Implement deterministic contracts**

`contracts.py` must use frozen dataclasses and the following public surface:

```python
def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")

def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

def write_new_manifest(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value))

@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    run_id: str
    pocket_id: str
    seed: int
    intervention: str
    sample_index: int
    status: str = "requested"
    sampling_status: str = "pending"
    reconstruction_status: str = "pending"
    evaluation_status: str = "pending"
    error_code: str | None = None

    @classmethod
    def new(cls, run_id, pocket_id, seed, intervention, sample_index):
        attempt_id = f"{run_id}:{pocket_id}:{seed}:{intervention}:{sample_index:04d}"
        return cls(attempt_id, run_id, pocket_id, seed, intervention, sample_index)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

Add frozen `PocketRecord` and `JobSpec` dataclasses. Reject unknown config keys,
relative-to-CWD ambiguity, duplicate pocket IDs, duplicate job keys, attempt
counts other than 10 for smoke or 20 for main, seeds outside the frozen list,
and checkpoint hash mismatch.

The YAML must freeze experiment IDs, seeds, `smoke_attempts: 10`,
`main_attempts: 20`, checkpoint path/hash, openness constants, interventions
`D0`–`D5`, SE(3) counts/tolerances and output schema version `phase0.v1`.

- [ ] **Step 4: Run tests and repository verification**

Run: `python -m unittest tests.diagnostics.test_contracts -v`

Expected: 3 tests pass.

Run: `python scripts/verify_repository.py`

Expected: repository verification passes; generated experiment outputs remain ignored.

- [ ] **Step 5: Commit**

```bash
git add dfe configs/diagnostics tests/diagnostics/test_contracts.py
git commit -m "feat: define immutable phase0 run contracts"
```

### Task 2: Build a Hash-Bound 30-Pocket Manifest

**Files:**
- Create: `dfe/diagnostics/io.py`
- Create: `scripts/build_phase0_manifest.py`
- Create: `tests/diagnostics/test_manifest_builder.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `pocket_centers_30.json`, PDBBind root, DF checkpoint, Git state and phase configuration.
- Produces: create-only `run-manifest.json`, `pockets.jsonl`, and `jobs.jsonl`; `sha256_file(path)` and `resolve_pdbbind_inputs(root, pocket_id)`.

- [ ] **Step 1: Write failing manifest-builder tests**

Use a temporary PDBBind fixture containing `10gs_protein.pdb`,
`10gs_pocket.pdb`, and `10gs_ligand.sdf`. Assert that every resolved input has
relative logical name, byte size and SHA-256; assert missing protein, ligand or
pocket raises `InputManifestError`; assert a dirty Git tree is recorded and
causes `--require-clean` to fail.

```python
def test_missing_ligand_is_rejected(self):
    with self.assertRaisesRegex(InputManifestError, "ligand"):
        resolve_pdbbind_inputs(self.root, "10gs")
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.diagnostics.test_manifest_builder -v`

Expected: import failure for `dfe.diagnostics.io`.

- [ ] **Step 3: Implement manifest construction**

`scripts/build_phase0_manifest.py` must accept only explicit arguments:

```text
--config configs/diagnostics/phase0_df500k.yaml
--pdbbind-root /workspace/ayb/data/pdbbind/v2020
--centers pocket_centers_30.json
--output-root /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1
--require-clean
```

It must verify all 30 centers and inputs before creating the output directory,
hash protein/pocket/ligand/checkpoint/config/source commit, record environment
versions without credentials, and create job keys for smoke and main stages.
It must not inspect existing model outcomes when choosing smoke pockets.

Add these ignore rules without ignoring versioned configs:

```gitignore
runs/
diagnostic-runs/
*.jsonl.tmp
```

- [ ] **Step 4: Verify manifest idempotence and failure behavior**

Run the unit test twice against a temporary fixture. The first create succeeds;
the second exits nonzero with `FileExistsError`. Delete only the temporary test
directory through the test framework, never an experiment directory.

- [ ] **Step 5: Commit**

```bash
git add .gitignore dfe/diagnostics/io.py scripts/build_phase0_manifest.py tests/diagnostics/test_manifest_builder.py
git commit -m "feat: freeze phase0 pocket and job manifests"
```

### Task 3: Implement Rotation-Invariant Pocket Openness

**Files:**
- Create: `dfe/diagnostics/openness.py`
- Create: `scripts/compute_pocket_openness.py`
- Create: `tests/diagnostics/test_openness.py`

**Interfaces:**
- Consumes: heavy-atom coordinates, elements, center, 2,048 directions, 12 Angstrom cutoff and a versioned VDW-radius mapping.
- Produces: `fibonacci_directions(n) -> np.ndarray`, `ray_sphere_blocked(...) -> np.ndarray`, `compute_openness(...) -> OpennessResult`, and `openness.jsonl`.

- [ ] **Step 1: Write geometric RED tests**

```python
def test_openness_is_rotation_and_translation_invariant(self):
    result = compute_openness(self.coords, self.elements, self.center)
    moved = compute_openness(self.coords @ self.rotation.T + self.shift,
                             self.elements,
                             self.center @ self.rotation.T + self.shift)
    self.assertAlmostEqual(result.openness, moved.openness, places=12)

def test_empty_environment_is_fully_open(self):
    result = compute_openness(np.empty((0, 3)), [], np.zeros(3))
    self.assertEqual(result.blocked_rays, 0)
    self.assertEqual(result.openness, 1.0)
```

Add a synthetic enclosing shell that blocks more than 95% of rays and a
one-sided half-shell that is more open than the enclosing shell.

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.diagnostics.test_openness -v`

Expected: import failure for `dfe.diagnostics.openness`.

- [ ] **Step 3: Implement vectorized ray-sphere intersections**

Use ray origin `c`, unit direction `d`, atom center `p`, radius `r`, and solve
`||c + t d - p||^2 <= r^2` for `0 <= t <= 12`. A ray is blocked when any atom
has discriminant `b^2 - q >= 0` and the interval intersects the cutoff. Process
atoms in chunks so 2,048 rays do not allocate an unbounded matrix.

The result must include blocked count, enclosure, openness, nearest distance,
atom density, radius table version, direction count and cutoff. Protein parsing
must exclude hydrogen and record unknown-element fallback counts.

- [ ] **Step 4: Generate the smoke strata without model outcomes**

`compute_pocket_openness.py` reads `pockets.jsonl`, writes a create-only
`openness.jsonl`, sorts all 30 records by `(openness, pocket_id)`, and writes
`smoke-pockets.json` from ranks `0, 1, 14, 15, 28, 29`. These are the two
lowest, two median-nearest and two highest pockets with deterministic,
non-overlapping selection. It also freezes the display tertiles and D4 pairing:
within each tertile, sort by `(openness, pocket_id)` and pair each pocket with
the next record cyclically. Pairing uses no model outcome.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest tests.diagnostics.test_openness -v
git add dfe/diagnostics/openness.py scripts/compute_pocket_openness.py tests/diagnostics/test_openness.py
git commit -m "feat: quantify pocket openness geometrically"
```

### Task 4: Add SE(3) Transform and Error Primitives

**Files:**
- Create: `dfe/diagnostics/se3.py`
- Create: `tests/diagnostics/test_se3.py`

**Interfaces:**
- Consumes: seed, positions/tensors, transform law and reference/transformed outputs.
- Produces: `sample_so3(seed, count)`, `apply_points`, `apply_vectors`, `normalized_error`, `compare_invariant`, and `compare_equivariant`.

- [ ] **Step 1: Write RED tests for SO(3) and comparisons**

Assert `R.T @ R = I`, `det(R)=1`, pairwise distances are unchanged, vector
comparisons pass after inverse rotation, invariant comparison rejects an xyz
component, and zero-norm references use a stable denominator.

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.diagnostics.test_se3 -v`

Expected: import failure for `dfe.diagnostics.se3`.

- [ ] **Step 3: Implement deterministic Haar SO(3)**

Generate normalized random quaternions with `numpy.random.Generator(PCG64(seed))`
and convert to rotation matrices. Define normalized max error as:

```python
def normalized_error(actual, expected, eps=1e-12):
    delta = np.linalg.norm(actual - expected, axis=-1)
    scale = np.maximum(np.linalg.norm(expected, axis=-1), eps)
    return ErrorStats(float(np.max(delta / scale)),
                      float(np.median(delta / scale)),
                      float(np.quantile(delta / scale, 0.95)))
```

Also report absolute error because normalized errors at true zero need context.

- [ ] **Step 4: Run tests and commit**

```bash
python -m unittest tests.diagnostics.test_se3 -v
git add dfe/diagnostics/se3.py tests/diagnostics/test_se3.py
git commit -m "test: add reusable se3 audit primitives"
```

### Task 5: Separate Raw DF Features and Add Default-Off Interventions

**Files:**
- Modify: `models/df_module.py`
- Modify: `models/maskfill.py`
- Create: `dfe/diagnostics/interventions.py`
- Create: `tests/diagnostics/test_df_interventions.py`

**Interfaces:**
- Consumes: current DF inputs and a non-persistent `DFIntervention` object.
- Produces: `AnalyticalDirectionField.raw_features(...)`, unchanged `forward(...)`, `MaskFillModelVN.set_diagnostics(intervention, observer)`, and interventions D0–D5.

- [ ] **Step 1: Capture normal-path behavior before implementation**

Create deterministic synthetic inputs, instantiate the current DF with seed 7,
and store expected raw-independent `forward` output in the test process before
using the new API. The regression assertion is:

```python
legacy = module(query, pocket, types, mask).detach().clone()
raw = module.raw_features(query, pocket, types, mask)
current = module.project_features(raw)
torch.testing.assert_close(current, legacy, rtol=0, atol=0)
```

The test initially fails because `raw_features` does not exist.

- [ ] **Step 2: Implement raw/project separation without state-dict changes**

Move only the existing feature computation into `raw_features`; add
`project_features(raw)` returning `self.field_proj(raw)`; make `forward` call
the two methods. Do not rename or re-register `field_proj`, buffers or weights.
Verify the 500K checkpoint still loads with no missing/unexpected keys.

- [ ] **Step 3: Implement typed interventions**

```python
@dataclass(frozen=True)
class DFIntervention:
    name: str = "D0"
    gate: float = 1.0
    zero_direction: bool = False
    shuffle_seed: int | None = None
    alternate_raw: torch.Tensor | None = None

    def apply_raw(self, raw: torch.Tensor) -> torch.Tensor:
        value = raw.clone()
        if self.zero_direction:
            value[..., 1:4] = 0
        if self.shuffle_seed is not None:
            generator = torch.Generator(device=value.device).manual_seed(self.shuffle_seed)
            value = value[torch.randperm(value.shape[0], generator=generator, device=value.device)]
        if self.alternate_raw is not None:
            value = self.alternate_raw.to(value)
        return value

    def apply_projected(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.gate
```

D1 uses gate 0; D2 sets `zero_direction`; D3 uses a job-derived seed; D4 receives
center-aligned raw features computed from the manifest-paired wrong pocket; D5
uses gates `0.25`, `0.5`, `1.0`, `1.5`. Reject other names or gate values.

- [ ] **Step 4: Add non-persistent model diagnostics**

`set_diagnostics` stores intervention and observer using `object.__setattr__` or
plain attributes that are not parameters/buffers. `compute_df_features_all`
applies raw intervention before `df_module.project_features`, applies projected
gate after `df_proj`, and emits copies to the observer. With diagnostics unset,
execute the legacy `self.df_module(...) -> self.df_proj(...)` path exactly.

- [ ] **Step 5: Verify checkpoint and normal parity**

Run:

```bash
python -m unittest tests.diagnostics.test_df_interventions -v
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

Expected: D0 equals legacy; D1 output is zero; D2 preserves columns 0 and 4–7;
D3 is deterministic; checkpoint iteration/hash/state tensor count remain fixed.

- [ ] **Step 6: Commit**

```bash
git add models/df_module.py models/maskfill.py dfe/diagnostics/interventions.py tests/diagnostics/test_df_interventions.py
git commit -m "feat: add noninvasive df diagnostic interventions"
```

### Task 6: Instrument Model Stages and Run the SE(3) Audit

**Files:**
- Create: `dfe/diagnostics/observer.py`
- Create: `dfe/diagnostics/model_audit.py`
- Create: `scripts/run_se3_audit.py`
- Create: `tests/diagnostics/test_observer.py`
- Create: `tests/diagnostics/test_model_audit.py`
- Modify: `models/maskfill.py`

**Interfaces:**
- Consumes: frozen checkpoint, real pocket/partial-ligand states and SE(3) transforms.
- Produces: named tensor events and create-only `se3-audit.json` with first failure, max/median/p95 error per output law.

- [ ] **Step 1: Write observer RED tests**

Test that an unset observer performs no allocation visible to callers; a
`TensorObserver` detaches/clones to CPU; duplicate `(step,event)` keys fail; and
mutating observed tensors does not mutate model output.

- [ ] **Step 2: Add stable event points**

Emit the following names without changing return tuples:

```text
df.raw, df.hidden, df.projected
encoder.scalar, encoder.vector
frontier.logits, frontier.indices
position.relative_mu, position.absolute_mu, position.sigma, position.pi
element.logits, element.probability
bond.logits, bond.probability
termination.has_frontier
```

Observer calls must be behind `if self._diagnostic_observer is not None` and
must not call random functions.

- [ ] **Step 3: Build transformation-law comparisons**

`model_audit.py` maps scalar/logit/sigma/pi events to invariant comparison and
vector/relative_mu to rotation comparison. Absolute positions compare against
`x @ R.T + translation`. Discrete indices must match exactly; on mismatch the
audit records first divergence but continues comparing pre-divergence tensors.

- [ ] **Step 4: Run analytical and real-model audits**

`run_se3_audit.py` accepts manifest, device, `--rotations 100`,
`--translations 10`, and output. It first runs float64 analytical raw-feature
tests, then float32 model tests on at least 20 manifest states. It exits `0`
only if all declared laws meet their tolerance; a scientifically expected DF
projection failure exits `2` and writes a complete report rather than raising.
Infrastructure/input failure exits `1`.

- [ ] **Step 5: Verify tests and commit**

```bash
python -m unittest tests.diagnostics.test_observer tests.diagnostics.test_model_audit -v
python -m unittest discover -s tests -v
git add dfe/diagnostics models/maskfill.py scripts/run_se3_audit.py tests/diagnostics
git commit -m "feat: audit df500k transformation laws by model stage"
```

### Task 7: Add an Append-Only Attempt Ledger and Autoregressive Trace

**Files:**
- Create: `dfe/diagnostics/ledger.py`
- Create: `dfe/diagnostics/trace.py`
- Create: `tests/diagnostics/test_ledger.py`
- Create: `tests/diagnostics/test_trace.py`
- Create: `sample_diagnostic.py`

**Interfaces:**
- Consumes: one `JobSpec`, current sampler functions, model observer events and reconstruction outcomes.
- Produces: append-only `attempts.jsonl`, `events.jsonl`, per-job checkpoint and raw candidate outputs.

- [ ] **Step 1: Write ledger state-machine tests**

Allowed attempt transitions are:

```text
requested -> initialized -> sampling -> generated -> reconstructed -> evaluated
requested/initialized/sampling/generated -> failed
```

Assert illegal backwards transitions fail, every append is flushed and
`os.fsync`-ed, duplicate terminal records fail, and replay reconstructs current
state after a truncated final line while reporting that truncation.

- [ ] **Step 2: Write trace schema tests**

Every event requires `run_id/job_id/attempt_id/pocket_id/seed/intervention/step`,
event name, tensor summary or decision payload, monotonic timestamp and schema
version. Test finite numeric summaries and JSON encoding of empty frontier/bond
sets.

- [ ] **Step 3: Implement ledger and trace writers**

Use exclusive job directories and append-only files. Write a job checkpoint via
temporary file, `fsync`, then atomic `os.replace`; never rewrite JSONL evidence.
On resume, replay the ledger and continue only non-terminal attempts. Preserve
the raw traceback in a job-local log but put only stable error codes/messages in
JSONL.

- [ ] **Step 4: Implement the diagnostic sampler wrapper**

`sample_diagnostic.py` imports the existing transform, `get_init`, `get_next`
and reconstruction functions. It adds no new sampling policy. Before calling
initialization it appends 20 requested attempts. It records initialization,
each autoregressive step, queue pruning, finish, duplicate, disconnected,
reconstruction exception and max-step exhaustion.

CLI:

```text
python sample_diagnostic.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --job-id main:10gs:20260901:D0 --device cuda:0
```

Reject output directories not declared by the manifest. D4 loads the paired
wrong-pocket input from the manifest, center-aligns coordinates, computes raw
features for current query points and never changes the actual pocket input.
Smoke job IDs are selected from `smoke-pockets.json` and are invoked only by
`run_phase0_jobs.py`; do not substitute a pocket before openness is frozen.

- [ ] **Step 5: Prove normal candidate parity on a tiny deterministic fixture**

Monkeypatch model outputs with deterministic tensors and compare legacy outer
sampling decisions against `sample_diagnostic` D0: same queue indices, stop
reason, reconstructed SMILES and attempt ordering. Then run one real pocket with
one attempt on AIDD and compare D0 with the existing script using the same seed.

- [ ] **Step 6: Commit**

```bash
python -m unittest tests.diagnostics.test_ledger tests.diagnostics.test_trace -v
git add dfe/diagnostics/ledger.py dfe/diagnostics/trace.py sample_diagnostic.py tests/diagnostics
git commit -m "feat: record complete diagnostic generation attempts"
```

### Task 8: Implement Restartable 0–4 GPU Job Scheduling

**Files:**
- Create: `dfe/diagnostics/scheduler.py`
- Create: `scripts/run_phase0_jobs.py`
- Create: `tests/diagnostics/test_scheduler.py`

**Interfaces:**
- Consumes: frozen jobs, stage, available device IDs and terminal ledger state.
- Produces: deterministic job queue, one subprocess per GPU, `scheduler-events.jsonl`, and resumable stage completion.

- [ ] **Step 1: Write scheduler RED tests**

Use fake subprocess commands to prove: zero GPUs lists pending work without
launching; one GPU runs serially; four GPUs never exceed four live jobs;
completed jobs are skipped; failed jobs are not automatically retried; SIGINT
stops dispatch and lets workers checkpoint; D0/D1 paired ordering is stable.

- [ ] **Step 2: Implement explicit device scheduling**

The script takes `--devices cpu`, `--devices cuda:0`, or a comma-separated list.
It must not infer or claim ownership of all GPUs. For every worker pass both
`--device cuda:N` and a process-local `CUDA_VISIBLE_DEVICES` mapping consistently.
Use no distributed process group.

Stages are `smoke` and `main`. `main` refuses to start unless
`gate-smoke.json` has status `pass`. A `--dry-run` prints exact commands and job
counts without creating outputs.

- [ ] **Step 3: Run tests and commit**

```bash
python -m unittest tests.diagnostics.test_scheduler -v
git add dfe/diagnostics/scheduler.py scripts/run_phase0_jobs.py tests/diagnostics/test_scheduler.py
git commit -m "feat: schedule restartable phase0 gpu jobs"
```

### Task 9: Recompute End-to-End Metrics from the Attempt Denominator

**Files:**
- Create: `dfe/diagnostics/metrics.py`
- Create: `scripts/summarize_phase0.py`
- Create: `tests/diagnostics/test_metrics.py`

**Interfaces:**
- Consumes: attempt/event ledgers, candidates, docking/PoseBusters results and openness records.
- Produces: `per-attempt.parquet`, `per-pocket-seed.csv`, `per-pocket.csv`, `failure-taxonomy.csv`, and gate-ready summary JSON.

- [ ] **Step 1: Write denominator regression tests**

Construct 10 attempts: 2 sampling failures, 1 reconstruction failure, 1
disconnected molecule and 6 valid molecules. Assert end-to-end validity is
`0.6`, not `1.0`; dockable and PoseBusters rates each retain explicit numerator
and denominator; missing docking is not imputed; duplicates stay in validity
but are excluded from uniqueness numerator as defined.

- [ ] **Step 2: Implement stable failure taxonomy**

Use mutually exclusive first-failure codes:

```text
init_no_frontier, init_threshold_exhausted, early_no_frontier,
max_steps, queue_empty, reconstruction_error, disconnected,
sanitize_error, sdf_write_error, docking_error, posebusters_error, success
```

Keep later-stage errors as additional flags. Compute first-atom success,
early-stop, pocket-containment, DF/position alignment, first clash step,
heavy-atom count, bonds, rings, QED, diversity, docking and pose checks.

- [ ] **Step 3: Generate all aggregation levels**

Aggregate attempt -> pocket/seed/intervention -> pocket/intervention. Always
write `attempt_count`, each stage's computable count, numerator, denominator and
rate. Refuse mixed schema versions, manifest hashes, checkpoint hashes or
metric-definition versions.

- [ ] **Step 4: Run tests and commit**

```bash
python -m unittest tests.diagnostics.test_metrics -v
git add dfe/diagnostics/metrics.py scripts/summarize_phase0.py tests/diagnostics/test_metrics.py
git commit -m "feat: summarize phase0 with end-to-end denominators"
```

### Task 10: Implement Pocket-Clustered Statistics and Gate Reports

**Files:**
- Create: `dfe/diagnostics/statistics.py`
- Create: `scripts/analyze_phase0.py`
- Create: `tests/diagnostics/test_statistics.py`

**Interfaces:**
- Consumes: per-pocket/seed data and the frozen primary endpoint list.
- Produces: paired effects, 10,000-draw pocket bootstrap CIs, openness-interaction regression, BH-FDR values and `gate-smoke.json`/`gate-phase0.json`.

- [ ] **Step 1: Write synthetic statistical tests**

Generate repeated samples from 30 synthetic pockets with a known negative
`D2 x openness` interaction. Assert the coefficient sign is recovered, all rows
from a resampled pocket stay together, identical paired values yield zero
effect, FDR values are monotone in sorted p-values, and fixed seed reproduces
the exact CI.

- [ ] **Step 2: Implement cluster bootstrap and regression**

Use pockets as the outer resampling unit and preserve all three seeds. Fit
ordinary least squares to pocket-seed aggregates with columns:

```text
metric = intercept + intervention + openness
       + intervention:openness + pocket_atom_count + reference_ligand_heavy_atoms
```

Derive uncertainty from 10,000 pocket bootstrap fits, not naive molecule rows.
For proportions, analyze both rate differences and numerator/denominator
bootstrap. Report effect sizes even when significance is absent.

- [ ] **Step 3: Encode gates without changing thresholds**

Smoke passes only when all expected jobs are terminal, 10 attempts exist per
job, D0 normal parity passes, no non-finite trace values occur, resume replay is
clean and all output hashes validate. Phase 0 passes on artifact completeness;
SE(3) or openness hypotheses may validly be `fail`/`inconclusive` and still
complete Phase 0, provided the report identifies the failure and does not claim
the hypothesis passed.

- [ ] **Step 4: Run tests and commit**

```bash
python -m unittest tests.diagnostics.test_statistics -v
git add dfe/diagnostics/statistics.py scripts/analyze_phase0.py tests/diagnostics/test_statistics.py
git commit -m "feat: add pocket-clustered phase0 analysis gates"
```

### Task 11: Add End-to-End Local Verification and Operator Documentation

**Files:**
- Create: `tests/diagnostics/test_phase0_pipeline.py`
- Create: `docs/experiments/phase0-operator-runbook.md`
- Modify: `scripts/verify_repository.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all Phase 0 modules and a synthetic two-pocket fixture.
- Produces: one-command local validation and an exact, credential-free operator runbook.

- [ ] **Step 1: Write an end-to-end synthetic pipeline test**

Build two synthetic pockets, construct a manifest, compute openness, execute a
fake D0/D2 sampler producing success and failure attempts, summarize and analyze.
Assert all artifact hashes link to the same manifest and rerunning a completed
stage does not rewrite evidence.

- [ ] **Step 2: Extend repository verification**

Add checks that committed experiment configs parse, plan/spec paths exist,
diagnostic output directories are ignored, and no run artifact containing
absolute AIDD dataset paths is tracked. Do not require external datasets for
repository verification.

- [ ] **Step 3: Write the operator runbook**

Document commands in this order: read-only preflight; isolated checkout;
environment capture; manifest; openness; SE(3); dry-run scheduler; smoke;
smoke gate; main; summary; analysis; hash export. Include exit-code meanings,
resume behavior and disk estimates. Do not include SSH credentials or local
password-bearing helper scripts.

- [ ] **Step 4: Run the complete local gate**

```bash
python -m unittest discover -s tests -v
python -m py_compile dfe/diagnostics/*.py scripts/*.py sample_diagnostic.py
python scripts/verify_repository.py
git diff --check
git status --short
```

Expected: all tests pass, compilation and verification exit 0, and only the
intended Phase 0 source/doc changes are present before commit.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/experiments scripts/verify_repository.py tests/diagnostics/test_phase0_pipeline.py
git commit -m "docs: add phase0 diagnostic operator workflow"
```

### Task 12: Execute the AIDD Read-Only Preflight and Isolated Smoke

**Files:**
- No tracked source files are modified during server execution.
- Create on AIDD after target confirmation: `/workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/...`

**Interfaces:**
- Consumes: pushed implementation commit, existing external PDBBind/CrossDocked inputs and 500K checkpoint.
- Produces: content-hash inventory, SE(3) report, smoke ledgers and smoke gate; no existing server file mutation.

- [ ] **Step 1: Perform read-only preflight**

Using `ssh aidd`, record without secrets: hostname alias, GPU inventory and free
memory, free disk, Python/CUDA/tool versions, existence/readability and hashes of
the expected dataset/pocket/checkpoint inputs, active processes owned by the
project user, and whether the proposed target already exists. Do not create the
target in this step.

- [ ] **Step 2: Confirm the exact mutation target**

The execution report must name the source checkout and run root. If
`/workspace/ayb/experiments/dfe-unified-phase0` already contains unrelated
state, choose a new explicit versioned child; never clean or reuse it.

- [ ] **Step 3: Materialize the pushed commit in an isolated checkout**

Fetch the private repository through the server's existing authorized Git
mechanism. Check out the exact implementation commit in a new directory. Run
`git lfs pull`, verify the checkpoint hash, and require a clean tree. Do not copy
the dirty `/workspace/ayb/Pocket2Mol` tree as source.

- [ ] **Step 4: Run CPU and one-GPU gates**

```bash
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/build_phase0_manifest.py --config configs/diagnostics/phase0_df500k.yaml --pdbbind-root /workspace/ayb/data/pdbbind/v2020 --centers pocket_centers_30.json --output-root /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1 --require-clean
python scripts/compute_pocket_openness.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json
python scripts/run_se3_audit.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --device cuda:0 --rotations 100 --translations 10
python scripts/run_phase0_jobs.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --stage smoke --devices cuda:0 --dry-run
```

Inspect the exact dry-run job list, then run without `--dry-run`. Generate
summary, analysis and `gate-smoke.json`.

- [ ] **Step 5: Apply smoke Go/No-Go**

If smoke fails, stop; preserve outputs and fix the implementation through a new
commit and new run ID. If smoke passes, record content-free evidence: commit,
manifest hash, job counts, attempt counts, gate status, runtime and GPU model.
Do not start main automatically in the same command.

### Task 13: Execute Main Interventions and Close Phase 0

**Files:**
- No tracked source files are modified by model execution.
- Create: versioned result manifests under the isolated AIDD run root.
- Later import only approved aggregate/non-sensitive evidence through a separate evidence plan.

**Interfaces:**
- Consumes: passing smoke gate and unchanged run manifest.
- Produces: complete D0–D5 ledgers, traces, summaries, statistics and Phase 0 architecture-decision report.

- [ ] **Step 1: Freeze the retained full arms from the preregistered matrix**

D0, D1 and D2 always enter main. D3, D4 and D5 enter only if their smoke jobs
have complete accounting and finite outputs; an experimentally null result is
not grounds for exclusion. Record included/excluded arms and infrastructure
reason before main outcomes are read.

- [ ] **Step 2: Schedule main for available 0–4 GPUs**

With zero GPUs, continue only validation/summaries. With one to four GPUs, pass
the explicit currently available device list. Never seize a GPU with an active
unrelated process. Resume jobs only through ledger state, not output filename
guessing.

- [ ] **Step 3: Verify completeness before statistics**

Require exactly 20 requested attempts for every declared job, one terminal
state per attempt, no mixed hashes/schema, all raw output hashes present, and
all three seeds for every main arm. An incomplete pocket remains visible; do not
drop it to make a paired table.

- [ ] **Step 4: Generate and review `gate-phase0.json`**

The report must answer separately: whether openness is reproducible; where SE(3)
first fails; whether the checkpoint depends on DF; which autoregressive stage
interacts with openness; whether findings survive pocket clustering/covariates;
and which DF-v2 branch is permitted by the design contract.

- [ ] **Step 5: Write the next design delta, not implementation code**

Create `docs/superpowers/specs/2026-09-01-df-v2-selection-design.md` from the actual
Phase 0 result. Present it for approval before creating the P1 implementation
plan. If Phase 0 points to reconstruction/evaluation rather than DF, design that
pipeline fix instead of forcing a DF-v2 model.

## Final Verification Before Declaring Phase 0 Complete

Run fresh on the exact implementation commit and exact AIDD run manifest:

```bash
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/summarize_phase0.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --stage main --verify-only
python scripts/analyze_phase0.py --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json --verify-only
git status --short
```

Completion requires zero test failures, repository verification exit 0, both
run verifiers exit 0, a clean source checkout, a terminal `gate-phase0.json`,
and a documented P1/no-go decision. Scientific hypotheses may fail; missing or
silently excluded evidence may not.
