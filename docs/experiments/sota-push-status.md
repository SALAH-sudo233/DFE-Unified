# SOTA push status

- DF-500k is not currently established as a strict CrossDocked-100 SOTA win.
- Strict paired 530k continuation regressed against DF-500k baseline.
- Size-matched I3 evidence showed a geometry advantage in some bins, but not an end-to-end SOTA claim.
- Atom-level geometry data: 5316/5316 records, splits 4049/444/823, no unknown split.
- Geometry affinity probe completed: test Pearson 0.3605, RMSE 2.2105; frozen probe, not generator fine-tuning.
- Pair-control v1 failed on unseen feature keys; v2 failed on a path/schema bug; v3 is rerunning with fixed vocabulary and ndarray correlation conversion.
- No BIF+DF joint fine-tune is authorized by evidence yet. It requires a passing real-pair vs within-split permutation control first.
- All fallback aromatic-to-single rows are included per user instruction; the count is 676 and is recorded in manifests.
