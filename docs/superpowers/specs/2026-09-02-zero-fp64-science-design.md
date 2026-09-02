# SCI-1 `zero-fp64-science` Design

## Purpose

The three frozen SCI-1 vector-origin arms all failed the V1 encoder gate in
float32, while an identical CPU float64 diagnostic reduced the downstream
vector error to approximately machine precision. This candidate isolates
inference precision as the next mechanism without changing the vector-origin
intervention, checkpoint, graph, transforms, or production defaults.

## Single Variable

Use `vector_origin_mode=zero` and run the complete model forward in float64.
The checkpoint is loaded into the original float32 model with `strict=True`
first, then the model is converted to float64. No weights are retrained or
modified on disk. `float32` remains the CLI default and the production path.

## Contract

- Supported model dtypes are exactly `float32` and `float64`.
- Analytical DF remains float64 with tolerance `1e-8`.
- Model event-law tolerance remains `<1e-4`; no threshold is relaxed for this
  candidate.
- Coordinates, composed features, and observer tensors use the selected model
  dtype; graph indices and discrete edge features remain unchanged.
- Strict checkpoint loading must occur before dtype conversion and must report
  no missing or unexpected keys.
- Reports record `model_dtype`, stage, origin mode, and strict-load status.
- This is a science-only inference precision intervention, not a trained
  baseline or a production recommendation.

## Gates

V0 requires helper tests, CLI/source-contract tests, local full tests,
repository verification, compilation, and strict loading for all three origin
modes in float32 and float64 where dependencies are available.

V1 runs only the `zero` arm on one real pocket with identity, pure rotation,
pure translation, and rigid transforms. Every topology check must match and
both encoder scalar/vector errors must be below `1e-4`.

Only a V1 pass authorizes the frozen full V2: 20 pockets, 100 rotations, 10
translations, unchanged seeds and event-law gates. A V1 failure is preserved
as scientific evidence and triggers diagnosis before any new candidate.

## Non-goals

Do not alter Issue #2, checkpoint files, generated results, DF/BIF fusion,
8-dimensional direction projection, docking, posture, affinity claims, or the
desktop meeting document. Do not call a precision-only inference run a newly
trained Pocket2Mol baseline.
