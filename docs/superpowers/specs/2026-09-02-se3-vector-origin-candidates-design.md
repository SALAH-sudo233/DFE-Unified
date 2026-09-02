# SE(3) Vector-Origin Candidate Experiment Design

## 1. Purpose

SCI-1 has established that the analytical direction field and its 8D-to-hidden
MLP projection satisfy the frozen SE(3) audit, while the frozen Pocket2Mol
encoder fails primarily under translation. Pure rotation is nearly compliant,
but translating all coordinates changes the encoder scalar and vector outputs
far beyond the `1e-4` float32 tolerance.

This experiment isolates one remaining mechanism: Pocket2Mol supplies absolute
atom coordinates as the initial vector feature before the encoder injects
relative edge geometry. The experiment compares three parameter-free choices
for that initial vector feature while preserving the frozen checkpoint, graph,
DF path, sampling policy, and production default.

The work belongs to Issue #1. It must not create or modify Issue #2 manifests,
schedulers, ledgers, trace transport, pocket-openness analysis, probe jobs, or
running Phase 0 processes.

## 2. Alternatives Considered

### A. Explicit science-only model mode (selected)

Add a non-persistent vector-origin mode to the model and expose it through the
SCI-1 audit CLI. The mode changes only the coordinate tensor supplied to the
protein and ligand `AtomEmbedding` modules. Encoder positions, kNN edges, edge
features, DF inputs, and all downstream coordinate calculations remain
unchanged.

This option gives a controlled, checkpoint-compatible comparison and cannot
silently change normal inference because `absolute` remains the default.

### B. Recenter the complete input structure before composition

Translate every coordinate into a pocket-centered frame before building the
graph, then restore the origin on generated coordinates. This could yield a
translation-invariant end-to-end sampler, but it changes a larger production
surface and mixes vector-feature diagnosis with coordinate I/O behavior. It is
not appropriate until the smaller intervention proves the mechanism.

### C. Replace the initial vector representation and retrain

Introduce a learned equivariant origin, local frames, or a new vector encoder
and train a new checkpoint. This may ultimately improve generation, but it
cannot identify whether the frozen checkpoint already works with a corrected
input convention. It is deferred until the frozen three-arm comparison is
complete.

## 3. Experiment Arms

The public science mode is an exact enum with three values:

- `absolute`: supply `compose_pos` to `AtomEmbedding`. This is the current
  behavior and the frozen control arm.
- `centered`: compute one reference origin from the protein atoms in the
  composed state, `origin = mean(compose_pos[idx_protein], dim=0)`, and supply
  `compose_pos - origin` to both protein and ligand `AtomEmbedding` modules.
- `zero`: supply `zeros_like(compose_pos)` to both `AtomEmbedding` modules.

The protein centroid is selected instead of the full compose centroid because
it is stable as ligand context atoms are added. Under a rigid transform
`x' = x R^T + t`, the origin follows `o' = o R^T + t`, so centered vectors obey
`(x' - o') = (x - o) R^T`. The zero arm removes the initial absolute-coordinate
channel entirely; relative encoder edge vectors remain available.

All arms use the same protein atoms, ordering, features, edge set, edge order,
DF raw features, DF projection, weights, thresholds, and random seeds. The
intervention must not alter or register parameters or buffers, so the pinned
checkpoint loads with `strict=True` in every arm.

## 4. Code Boundaries

### Model hook

`MaskFillModelVN` owns a non-persistent science-only setting named for the
vector-origin mode. A small helper derives the vector-embedding positions from
`compose_pos` and `idx_protein`, validates the enum, rejects an empty protein
selection for `centered`, and returns a new tensor without mutating inputs.

Both inference and loss paths call one shared compose-embedding helper so the
setting has one implementation. With no explicit science setting, the helper
executes the existing `absolute` path. The encoder still receives the original
`compose_pos`; only `AtomEmbedding.vector_input` changes.

The existing SCI-2A raw-DF intervention hook remains independent. The SCI-1
origin mode must not read, replace, or compose with SCI-2A interventions.

### Audit runner

`scripts/run_sci1_se3_audit.py` accepts the selected origin mode and records it
in every report. It supports two explicit stages:

- `preflight`: one manifest pocket, fixed graph topology, and four deterministic
  transform categories: identity, pure rotation, pure translation, and rotation
  plus translation.
- `full`: the frozen 20-pocket, 100-rotation, 10-translation SCI-1 protocol.

Preflight and full reports use create-only output paths. Separate arms or
retries require separate versioned run roots; an existing report is never
overwritten.

### Report contract

Each report includes the source commit, checkpoint path hash, manifest hash,
origin mode, stage, device, strict-load result, pocket and comparison counts,
tolerances, per-transform-category event audits, first failure, and gate
status. Scientific threshold failures must still produce a complete report and
exit with the existing scientific-failure code. Missing inputs, invalid modes,
checkpoint mismatch, non-finite outputs, or dependency errors are
infrastructure failures and must not be interpreted scientifically.

## 5. Data Flow

For each state, the composer builds the unchanged `compose_pos`, features, and
kNN graph. The model derives a second tensor used only for vector embedding:

```text
compose_pos + idx_protein
        |
        +-- absolute -> compose_pos
        +-- centered -> compose_pos - protein_centroid
        +-- zero     -> zeros_like(compose_pos)
        |
        v
AtomEmbedding vector input
        |
        v
encoder node_attr

compose_pos -------------------------> encoder pos / relative edges / DF
```

The observer captures the same named events as the existing SCI-1 audit. Event
laws remain unchanged: scalar outputs are invariant, vector and relative
position outputs are rotation-equivariant, absolute positions transform as
points, and discrete choices match exactly.

## 6. Stage Gates

### Gate V0: local contract and checkpoint compatibility

Acceptance requires:

- invalid mode and empty-protein cases fail explicitly;
- `absolute` is numerically identical to the current default path;
- centered inputs translate invariantly and rotate equivariantly in unit tests;
- zero inputs remain zero under all rigid transforms;
- all three arms strict-load the pinned checkpoint with no missing or unexpected
  keys;
- focused tests, the complete test suite, repository verification, compilation,
  and `git diff --check` pass.

Failure blocks remote execution. Fixes must be scoped to the failed contract.

### Gate V1: single-pocket fixed-topology preflight

Run `absolute`, `centered`, and `zero` against the same first manifest pocket
and the same deterministic identity, pure-rotation, pure-translation, and rigid
transforms. The graph edge set and edge order must match the reference state.

For a candidate arm to pass, both `encoder.scalar` and `encoder.vector` must
have normalized maximum error strictly below `1e-4` in every transform
category, and the checkpoint must have loaded strictly. Identity must be exact
within float32 arithmetic. The `absolute` arm is retained as a control and is
expected to document the known translation failure; its failure does not block
passing candidates.

An arm that fails V1 cannot enter the full audit. Preserve its report, identify
the first divergent event, review the relevant code and literature, make at
most one mechanistically justified variable change, commit it, and retry in a
new versioned run root without changing tolerances, seeds, or transforms.

### Gate V2: full frozen SCI-1

Only V1-passing candidate arms may run V2. Each candidate uses 20 real pockets,
100 SO(3) rotations, 10 translations, analytical float64 tolerance `1e-8`,
model float32 tolerance `1e-4`, and checkpoint SHA-256
`34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0`.

The existing full SCI-1 model gate remains authoritative: every declared event
law and every comparison must pass, not only the encoder events. A failed arm
emits `scientific_failure`, retains all evidence, and follows the same
single-variable retry rule. Thresholds, denominators, transform counts, pocket
count, checkpoint, and random seeds are immutable.

SCI-2A remains blocked until at least one candidate passes V2. Passing V2 does
not by itself authorize a production-default change or training.

## 7. Interpretation Rules

- If `centered` passes and `zero` fails, the frozen checkpoint uses useful
  pocket-relative initial vector information.
- If `zero` passes and `centered` fails, absolute-coordinate initialization is
  harmful and relative edge geometry is sufficient for equivariance.
- If both pass, equivariance alone does not select a production candidate;
  generation quality and downstream parity require a separately approved
  experiment.
- If neither passes, the remaining failure is downstream of the initial vector
  origin or arises from another coupled non-equivariant operation. The reports
  must guide a new, separately reviewed candidate rather than weakening V1 or
  V2.
- The result addresses representation symmetry only. It does not establish
  docking-score accuracy, pose correctness, affinity ranking quality, or the
  user-reported Pearson correlation as a repository-verified fact.

## 8. Non-Goals

This change does not alter the production default, recenter generated molecule
coordinates, retrain any model, start SCI-2B, evaluate open versus closed
pockets, modify BIF/DF fusion, change the 8D DF MLP, run docking, or claim a new
Pocket2Mol baseline. It also does not stop, reuse, or overwrite any existing
AIDD process, checkpoint, result directory, or Phase 0 run root.

## 9. Acceptance Criteria

The implementation is complete only when V0 passes locally and on AIDD, all
three V1 reports exist under unique create-only roots, the control behavior is
documented, and every V1-passing candidate has either completed V2 or has a
preserved infrastructure-failure report with the exact blocker. Scientific
completion requires at least one V2 pass. Until then, the stage remains active
and the next action is targeted investigation and a versioned retry, not
advancement to SCI-2A.
