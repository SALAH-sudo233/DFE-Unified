# `zero-fp64-science` Implementation Plan

- [x] Add `dfe/science/model_precision.py` with strict `float32`/`float64`
  normalization and torch dtype mapping.
- [x] Add RED tests for helper behavior, CLI exposure/defaults, report dtype,
  strict-load ordering, and repository source authorities.
- [x] Thread `--model-dtype` through both SCI-1 and shared SE(3) audit runners.
  Load checkpoint strictly before converting model dtype; convert model inputs
  and positions to the selected dtype while preserving graph/discrete tensors.
- [x] Run focused RED/GREEN tests, then local science/full repository gates,
  compilation, diff checks, and repository verification.
- [ ] Commit the candidate and create a fresh AIDD checkout and create-only V0
  root. Run strict-load V0 for all origin modes in both dtypes where possible.
- [ ] Run a fresh `zero` V1 preflight on AIDD. If and only if it passes, run
  the complete frozen V2 in a new root.
- [ ] Hash and summarize reports, update Issue #1 without secrets or absolute
  server paths, and keep SCI-2A blocked unless V2 passes.
