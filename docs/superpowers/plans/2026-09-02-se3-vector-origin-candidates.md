# SE(3) Vector-Origin Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add checkpoint-compatible `absolute`, `centered`, and `zero` initial vector-origin modes, then run gated single-pocket and full SCI-1 comparisons without changing the production default or Issue #2 infrastructure.

**Architecture:** A pure science helper derives the coordinate tensor used only by `AtomEmbedding`; `MaskFillModelVN` applies it behind a non-persistent explicit hook while retaining original coordinates for graph geometry, DF, and downstream positions. The existing SCI-1 runner gains preflight/full stages and records the arm, topology checks, strict-load status, and transform categories in create-only reports.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, unittest/pytest, JSON, Git, GitHub CLI, AIDD CUDA environment.

## Global Constraints

- The governing design is `docs/superpowers/specs/2026-09-02-se3-vector-origin-candidates-design.md`.
- The work belongs to GitHub Issue #1 and must not modify Issue #2 manifests, scheduler, ledger, observer transport, openness analysis, probe jobs, or running Phase 0 processes.
- `absolute` remains the production/default behavior; `centered` and `zero` are explicit non-persistent science modes.
- `centered` subtracts `mean(compose_pos[idx_protein], dim=0)` from all vector-embedding positions. It does not recenter encoder positions, kNN inputs, DF inputs, or generated coordinates.
- `zero` supplies `zeros_like(compose_pos)` only to `AtomEmbedding`.
- Every arm must strict-load checkpoint SHA-256 `34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0`.
- V1 uses one real pocket and deterministic identity, pure rotation, pure translation, and rigid transforms with fixed topology. Both `encoder.scalar` and `encoder.vector` must have normalized maximum error `< 1e-4` in every category.
- V2 retains 20 real pockets, 100 rotations, 10 translations, analytical tolerance `1e-8`, model tolerance `1e-4`, unchanged seeds, and all existing event-law gates.
- Outputs are create-only. Every retry uses a new commit and versioned run root; failed evidence is preserved.
- Do not overwrite checkpoints, `results/df-500k-21-pocket`, any prior SCI-1 root, or `/workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v5`.
- SCI-2A remains blocked until at least one candidate passes V2. Do not start SCI-2B.

---

## File Map

- Create `dfe/science/vector_origin.py`: enum validation and pure vector-embedding position transformation.
- Create `tests/science/test_vector_origin.py`: mathematical laws, mutation safety, and invalid-input tests.
- Modify `models/maskfill.py`: non-persistent mode setter and one shared compose-embedding path for inference and loss.
- Modify `scripts/run_sci1_se3_audit.py`: arm/stage CLI, preflight transforms, topology evidence, and report metadata.
- Modify `tests/science/test_sci1_cli.py`: CLI defaults, mode/stage validation, and create-only failure behavior.
- Create `tests/science/test_vector_origin_model_hook.py`: behavioral model-hook integration without checkpoint files.
- Modify `scripts/verify_repository.py`: require the new science helper, tests, spec, and plan.
- Modify `docs/superpowers/plans/2026-09-02-se3-vector-origin-candidates.md`: mark completed execution steps as evidence is produced.

### Task 1: Implement the pure vector-origin transformation

**Files:**
- Create: `dfe/science/vector_origin.py`
- Create: `tests/science/test_vector_origin.py`

**Interfaces:**
- Produces: `VECTOR_ORIGIN_MODES: tuple[str, ...]`
- Produces: `normalize_vector_origin_mode(mode: str | None) -> str`
- Produces: `vector_embedding_positions(compose_pos: Tensor, idx_protein: Tensor, mode: str | None) -> Tensor`

- [ ] **Step 1: Write the failing mathematical and validation tests**

```python
class VectorOriginTests(unittest.TestCase):
    def setUp(self):
        self.pos = torch.tensor([[2.0, 0.0, 0.0], [4.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        self.idx_protein = torch.tensor([1, 2])

    def test_none_and_absolute_preserve_values_without_aliasing(self):
        for mode in (None, "absolute"):
            actual = vector_embedding_positions(self.pos, self.idx_protein, mode)
            self.assertTrue(torch.equal(actual, self.pos))
            self.assertNotEqual(actual.data_ptr(), self.pos.data_ptr())

    def test_centered_uses_protein_centroid_for_all_atoms(self):
        expected = self.pos - self.pos[self.idx_protein].mean(dim=0, keepdim=True)
        actual = vector_embedding_positions(self.pos, self.idx_protein, "centered")
        self.assertTrue(torch.equal(actual, expected))

    def test_centered_obeys_rigid_transform_law(self):
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = torch.tensor([7.0, -3.0, 2.0])
        moved = self.pos @ rotation.T + translation
        reference = vector_embedding_positions(self.pos, self.idx_protein, "centered")
        actual = vector_embedding_positions(moved, self.idx_protein, "centered")
        self.assertTrue(torch.allclose(actual, reference @ rotation.T, atol=1e-6, rtol=0.0))

    def test_zero_returns_independent_zeros(self):
        actual = vector_embedding_positions(self.pos, self.idx_protein, "zero")
        self.assertTrue(torch.equal(actual, torch.zeros_like(self.pos)))
        self.assertNotEqual(actual.data_ptr(), self.pos.data_ptr())

    def test_centered_rejects_empty_protein_selection(self):
        with self.assertRaisesRegex(ValueError, "protein"):
            vector_embedding_positions(self.pos, torch.empty(0, dtype=torch.long), "centered")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vector-origin"):
            normalize_vector_origin_mode("learned")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.science.test_vector_origin -v`

Expected: FAIL because `dfe.science.vector_origin` does not exist.

- [ ] **Step 3: Implement the minimal pure helper**

```python
from __future__ import annotations

import torch
from torch import Tensor

VECTOR_ORIGIN_MODES = ("absolute", "centered", "zero")


def normalize_vector_origin_mode(mode: str | None) -> str:
    normalized = "absolute" if mode is None else str(mode).lower()
    if normalized not in VECTOR_ORIGIN_MODES:
        choices = ", ".join(VECTOR_ORIGIN_MODES)
        raise ValueError(f"invalid vector-origin mode {mode!r}; expected one of: {choices}")
    return normalized


def vector_embedding_positions(
    compose_pos: Tensor,
    idx_protein: Tensor,
    mode: str | None,
) -> Tensor:
    normalized = normalize_vector_origin_mode(mode)
    if compose_pos.ndim != 2 or compose_pos.shape[-1] != 3:
        raise ValueError("compose_pos must have shape [N, 3]")
    if normalized == "absolute":
        return compose_pos.clone()
    if normalized == "zero":
        return torch.zeros_like(compose_pos)
    if idx_protein.numel() == 0:
        raise ValueError("centered vector-origin mode requires protein atoms")
    origin = compose_pos[idx_protein].mean(dim=0, keepdim=True)
    return compose_pos - origin
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest tests.science.test_vector_origin -v`

Expected: all six tests PASS.

- [ ] **Step 5: Commit the pure transformation**

```bash
git add dfe/science/vector_origin.py tests/science/test_vector_origin.py
git commit -m "feat: add vector-origin science candidates"
```

### Task 2: Integrate the non-persistent model hook

**Files:**
- Modify: `models/maskfill.py`
- Create: `tests/science/test_vector_origin_model_hook.py`

**Interfaces:**
- Consumes: `vector_embedding_positions(compose_pos, idx_protein, mode)` from Task 1.
- Produces: `MaskFillModelVN.set_science_vector_origin(mode: str | None = None) -> None`
- Produces: `MaskFillModelVN._embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein)`

- [ ] **Step 1: Write failing behavioral hook tests**

Use an uninitialized lightweight module shell and mock the existing
`models.maskfill.embed_compose` function so the test verifies the exact tensor
sent to AtomEmbedding without constructing the full CUDA model:

```python
class VectorOriginModelHookTests(unittest.TestCase):
    def setUp(self):
        self.model = MaskFillModelVN.__new__(MaskFillModelVN)
        torch.nn.Module.__init__(self.model)
        self.model.emb_dim = [4, 2]
        self.model.ligand_atom_emb = object()
        self.model.protein_atom_emb = object()
        object.__setattr__(self.model, "_science_vector_origin", "absolute")
        self.feature = torch.randn(3, 5)
        self.pos = torch.tensor([[2.0, 0.0, 0.0], [4.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        self.idx_ligand = torch.tensor([0])
        self.idx_protein = torch.tensor([1, 2])

    def test_default_hook_passes_absolute_embedding_positions(self):
        with patch("models.maskfill.embed_compose", return_value="embedded") as mocked:
            self.assertEqual(self.model._embed_compose(self.feature, self.pos, self.idx_ligand, self.idx_protein), "embedded")
        self.assertTrue(torch.equal(mocked.call_args.args[1], self.pos))

    def test_centered_hook_changes_only_embedding_positions(self):
        self.model.set_science_vector_origin("centered")
        with patch("models.maskfill.embed_compose", return_value="embedded") as mocked:
            self.model._embed_compose(self.feature, self.pos, self.idx_ligand, self.idx_protein)
        expected = self.pos - self.pos[self.idx_protein].mean(dim=0, keepdim=True)
        self.assertTrue(torch.equal(mocked.call_args.args[1], expected))
        self.assertTrue(torch.equal(self.pos, self.pos.clone()))

    def test_reset_restores_absolute_mode(self):
        self.model.set_science_vector_origin("zero")
        self.model.set_science_vector_origin()
        self.assertEqual(self.model._science_vector_origin, "absolute")
```

- [ ] **Step 2: Run the hook tests and verify RED**

Run: `python -m unittest tests.science.test_vector_origin_model_hook -v`

Expected: FAIL because `set_science_vector_origin` and `_embed_compose` are absent.

- [ ] **Step 3: Add the model hook and replace both direct embedding calls**

Add in `__init__`:

```python
object.__setattr__(self, "_science_vector_origin", "absolute")
```

Add methods:

```python
def set_science_vector_origin(self, mode=None):
    object.__setattr__(
        self,
        "_science_vector_origin",
        normalize_vector_origin_mode(mode),
    )

def _embed_compose(self, compose_feature, compose_pos, idx_ligand, idx_protein):
    embedding_pos = vector_embedding_positions(
        compose_pos,
        idx_protein,
        self._science_vector_origin,
    )
    return embed_compose(
        compose_feature,
        embedding_pos,
        idx_ligand,
        idx_protein,
        self.ligand_atom_emb,
        self.protein_atom_emb,
        self.emb_dim,
    )
```

Replace the direct `embed_compose(...)` calls in `sample_focal` and `get_loss`
with `self._embed_compose(...)`. Do not change the `compose_pos` passed to
`self.encoder`, `compute_df_features_all`, `pos_predictor`, or field queries.

- [ ] **Step 4: Run model-hook, equivariance, and science-hook tests**

Run:

```bash
python -m unittest tests.science.test_vector_origin_model_hook tests.science.test_vector_origin tests.science.test_vn_equivariance tests.science.test_science_hooks -v
```

Expected: all tests PASS and the existing SCI-2A hook remains unchanged.

- [ ] **Step 5: Commit model integration**

```bash
git add models/maskfill.py tests/science/test_vector_origin_model_hook.py
git commit -m "feat: apply science vector origin at atom embedding"
```

### Task 3: Add preflight/full audit stages and evidence fields

**Files:**
- Modify: `scripts/run_sci1_se3_audit.py`
- Modify: `tests/science/test_sci1_cli.py`

**Interfaces:**
- Consumes: `MaskFillModelVN.set_science_vector_origin(mode)` from Task 2.
- Produces CLI options: `--stage {preflight,full}` and `--vector-origin-mode {absolute,centered,zero}`.
- Produces: `_preflight_transforms(rotation: np.ndarray, translation: np.ndarray) -> tuple[dict[str, object], ...]`.
- Produces report fields: `stage`, `vector_origin_mode`, `checkpoint_sha256`, `checkpoint_strict_load`, `topology_match`, and per-record `transform_category`.

- [ ] **Step 1: Extend CLI tests before implementation**

```python
def test_help_lists_vector_origin_and_stage(self):
    result = subprocess.run(
        [sys.executable, "scripts/run_sci1_se3_audit.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    self.assertEqual(result.returncode, 0)
    self.assertIn("--vector-origin-mode", result.stdout)
    self.assertIn("--stage", result.stdout)

def test_invalid_vector_origin_is_rejected_without_output(self):
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "report.json"
        result = subprocess.run(
            [sys.executable, "scripts/run_sci1_se3_audit.py", "--manifest", "missing.json",
             "--vector-origin-mode", "learned", "--output", str(output)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
```

Add a direct test asserting `_preflight_transforms` returns categories in this
exact order: `identity`, `rotation`, `translation`, `rigid`, with identity
`R=I,t=0`, rotation `t=0`, translation `R=I`, and rigid using both inputs.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `python -m unittest tests.science.test_sci1_cli -v`

Expected: FAIL because the two CLI flags and transform helper are absent.

- [ ] **Step 3: Implement parser and deterministic transform suite**

Add parser definitions:

```python
parser.add_argument("--stage", choices=("preflight", "full"), default="full")
parser.add_argument(
    "--vector-origin-mode",
    choices=("absolute", "centered", "zero"),
    default="absolute",
)
```

Define the preflight suite:

```python
def _preflight_transforms(rotation, translation):
    identity = np.eye(3, dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    return (
        {"category": "identity", "rotation": identity, "translation": zero},
        {"category": "rotation", "rotation": rotation, "translation": zero},
        {"category": "translation", "rotation": identity, "translation": translation},
        {"category": "rigid", "rotation": rotation, "translation": translation},
    )
```

For `preflight`, select exactly the first manifest pocket and the first frozen
rotation and translation. For `full`, preserve the existing 20-pocket and
100-rotation loop unchanged.

- [ ] **Step 4: Route the arm into the model and record topology**

After strict checkpoint loading, call:

```python
model.set_science_vector_origin(vector_origin_mode)
```

Refactor `_run_model_state` to return both the observer and a detached CPU copy
of `compose_knn_edge_index`. Compare every transformed edge tensor to the
reference with `torch.equal`; include `topology_match` in each record. A
topology mismatch yields a complete scientific-failure record with first
failure `topology.edge_index`, rather than silently comparing different graphs.

Set the arm back to `absolute` after each model audit, including exception
cleanup, and leave the SCI-2A diagnostics hook untouched.

- [ ] **Step 5: Implement V1 gate evaluation and report metadata**

For preflight, inspect the named `encoder.scalar` and `encoder.vector` events in
all four records. Set `model.passed=True` only when topology matches, strict load
is true, and both events have `normalized_max < 1e-4` in every category. Retain
all other event audits for diagnosis but do not allow them to weaken the V1
encoder gate definition.

For full, preserve the existing all-event `AuditReport.passed` rule. Include the
manifest checkpoint hash directly as `checkpoint_sha256` and write `stage` and
`vector_origin_mode` at report top level.

- [ ] **Step 6: Run focused CLI and audit tests**

Run:

```bash
python -m unittest tests.science.test_sci1_cli tests.science.test_se3_audit tests.science.test_vector_origin_model_hook -v
python scripts/run_sci1_se3_audit.py --help
```

Expected: tests PASS; help exits `0`; existing missing-manifest behavior still
writes one create-only `infrastructure_failure` report when arguments are valid.

- [ ] **Step 7: Commit the runner**

```bash
git add scripts/run_sci1_se3_audit.py tests/science/test_sci1_cli.py
git commit -m "feat: gate vector-origin se3 preflight"
```

### Task 4: Enforce repository V0 and run local verification

**Files:**
- Modify: `scripts/verify_repository.py`
- Test: `tests/test_verify_repository.py`

**Interfaces:**
- Consumes the new helper, tests, design, and plan paths.
- Produces no runtime API; extends repository evidence validation only.

- [ ] **Step 1: Add required-file expectations to the verifier test**

Extend the existing required-path fixture/assertion to include:

```python
Path("dfe/science/vector_origin.py")
Path("tests/science/test_vector_origin.py")
Path("tests/science/test_vector_origin_model_hook.py")
Path("docs/superpowers/specs/2026-09-02-se3-vector-origin-candidates-design.md")
Path("docs/superpowers/plans/2026-09-02-se3-vector-origin-candidates.md")
```

- [ ] **Step 2: Run the verifier test and verify RED**

Run: `python -m unittest tests.test_verify_repository -v`

Expected: the new assertion fails until `verify_repository.py` contains the
same required paths.

- [ ] **Step 3: Add the paths to repository verification**

Append the five paths to the existing science required-file collection. Do not
change checkpoint-size/hash logic, Phase 0 contracts, or absolute-path checks.

- [ ] **Step 4: Run the complete local V0 gate**

Run:

```bash
python -m unittest tests.science -v
python -m unittest discover -s tests -v
python -m py_compile dfe/science/vector_origin.py scripts/run_sci1_se3_audit.py models/maskfill.py
python scripts/verify_repository.py
git diff --check
```

Expected: all tests PASS, compilation exits `0`, repository verification prints
`Repository verification passed.`, and `git diff --check` emits no output.

- [ ] **Step 5: Commit V0 verification**

```bash
git add scripts/verify_repository.py tests/test_verify_repository.py docs/superpowers/plans/2026-09-02-se3-vector-origin-candidates.md
git commit -m "test: verify vector-origin science contract"
```

### Task 5: Push the candidate and execute AIDD V0/V1

**Files:**
- No tracked source files unless a V0/V1 failure requires one targeted fix.
- Create-only remote reports under `/workspace/ayb/experiments/dfe-unified-sci1/`.

**Interfaces:**
- Consumes the Phase 0 manifest at `/workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json`.
- Produces one preflight report per arm under a unique commit-qualified root.

- [ ] **Step 1: Push the reviewed implementation branch**

Run locally:

```bash
git push origin sci1-se3-df-ablation
```

Expected: the remote branch advances to the V0-verified commit. Retry ordinary
network failures without changing commits or credentials.

- [ ] **Step 2: Transfer a bundle and create an isolated AIDD checkout**

Create a bundle containing the branch, transfer it through the existing SSH
alias, then connect with `ssh aidd`. Enter the password only at the interactive
prompt; never place it in a command, file, shell history, report, or Issue.

On AIDD derive `sci_commit=$(git rev-parse --short=7 sci1-se3-df-ablation)` from
the transferred bundle and create:

```bash
GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false worktree add "/workspace/ayb/checkouts/dfe-unified-sci1-${sci_commit}" sci1-se3-df-ablation
```

Verify the resolved checkout is below `/workspace/ayb/checkouts` before adding
or moving files. Do not reuse `/workspace/ayb/checkouts/dfe-unified-sci1-7dec4af`.

- [ ] **Step 3: Run focused and checkpoint strict-load V0 on AIDD**

Use `/workspace/ayb/miniconda3/envs/zatom310/bin/python` to run the new science
tests and load the real checkpoint through the SCI-1 runtime for each of
`absolute`, `centered`, and `zero`. Confirm every load reports no missing or
unexpected keys and the checkpoint hash equals the pinned SHA-256.

Expected: V0 passes for all arms. A failure is `blocked` for missing dependency
or input, otherwise it is a code-contract failure that must be fixed and
reverified locally before a new commit and checkout are created.

- [ ] **Step 4: Create three unique V1 roots and run preflight**

For each arm, create a distinct root named:

```text
/workspace/ayb/experiments/dfe-unified-sci1/sci1-vector-origin-<commit>-<arm>-v1
```

Run from the isolated checkout:

```bash
/workspace/ayb/miniconda3/envs/zatom310/bin/python scripts/run_sci1_se3_audit.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --device cuda:0 \
  --stage preflight \
  --vector-origin-mode <arm> \
  --rotations 100 \
  --translations 10 \
  --output /workspace/ayb/experiments/dfe-unified-sci1/sci1-vector-origin-<commit>-<arm>-v1/se3-preflight.json
```

Here `<commit>` is replaced by the exact seven-character commit derived above,
and `<arm>` is executed once each as `absolute`, `centered`, and `zero`; these
are operator substitutions, not additional protocol choices.

- [ ] **Step 5: Evaluate V1 without weakening the gate**

For every report verify:

- `status`, `stage`, `vector_origin_mode`, and `checkpoint_sha256` are present;
- `checkpoint_strict_load` and all four `topology_match` values are true;
- identity, rotation, translation, and rigid categories are all present;
- both encoder events have normalized maximum error `< 1e-4` in every category.

Retain the expected `absolute` translation failure as control evidence. Mark
`centered` and `zero` independently pass or scientific-fail. For each failure,
record the first divergent event and errors, review that exact mechanism, make
only one justified variable change, and restart Tasks 1-5 with a new commit and
run-root version. Never reuse or delete the failed root.

### Task 6: Run full V2 for passing candidates and update Issue #1

**Files:**
- No tracked source files unless a V2 failure produces an approved targeted correction.
- Create-only full reports under `/workspace/ayb/experiments/dfe-unified-sci1/`.

**Interfaces:**
- Consumes only candidate arms whose V1 reports pass.
- Produces full SCI-1 reports, SHA-256 evidence, and a non-sensitive Issue #1 progress comment.

- [ ] **Step 1: Allocate one unique V2 root per V1-passing arm**

Name each root:

```text
/workspace/ayb/experiments/dfe-unified-sci1/sci1-vector-origin-<commit>-<arm>-full-v1
```

Do not create a V2 root for a failed arm.

- [ ] **Step 2: Execute the frozen full audit**

```bash
/workspace/ayb/miniconda3/envs/zatom310/bin/python scripts/run_sci1_se3_audit.py \
  --manifest /workspace/ayb/experiments/dfe-unified-phase0/phase0-df500k-v1/run-manifest.json \
  --device cuda:0 \
  --stage full \
  --vector-origin-mode <passing-arm> \
  --rotations 100 \
  --translations 10 \
  --output /workspace/ayb/experiments/dfe-unified-sci1/sci1-vector-origin-<commit>-<passing-arm>-full-v1/se3-audit.json
```

Expected: 20 pockets and 2,000 model comparisons; analytical and every model
event law pass at the frozen tolerances. Exit code `2` is preserved as a
scientific failure, not retried as infrastructure.

- [ ] **Step 3: Hash and summarize evidence without exposing server paths**

For each report compute SHA-256 and extract only: source commit, arm, stage,
checkpoint hash, pocket/comparison counts, status, first failure, topology
status, and encoder scalar/vector maximum error summaries by transform category.
Do not include passwords, tokens, full absolute input paths, or environment
contents.

- [ ] **Step 4: Update GitHub Issue #1**

Post a progress comment using the authenticated `SALAH-sudo233` GitHub CLI
session. Include V0 status, three V1 outcomes, each immutable report hash, V2
outcomes for passing candidates, and the enforced next action. Explicitly state
that SCI-2A remains blocked unless V2 passed and that an equivariance pass does
not establish docking score, pose, affinity, or production quality.

- [ ] **Step 5: Apply the terminal gate**

If at least one candidate passes V2, report SCI-1 vector-origin completion and
stop for review before SCI-2A. If all candidates scientifically fail, preserve
the evidence, perform targeted literature/code diagnosis, propose one new
mechanistic candidate, and obtain design approval before changing code. If an
infrastructure condition fails, repair it and resume the same immutable
protocol; do not classify it as a scientific result.
