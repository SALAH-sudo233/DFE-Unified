# Methodology and limitations

## What can be asserted

- The retained checkpoint loads as iteration 500,000 and contains 392 model
  state tensors.
- Its SHA-256 is fixed in `artifacts/MANIFEST.json`.
- The retained partial run contains 21 pocket summaries and 2,331 evaluated
  records.
- Across the 21 pocket summaries, the macro mean docking score is approximately
  -7.304, macro mean QED is 0.579, and macro mean molecular weight is 280.614.
- PoseBusters marked 986 of the 2,331 evaluated records as passed. A total of
  1,388 records have docking scores below -7.0.

## What cannot be asserted

The experiment does not support a completed 30-pocket result, a matched
DF-versus-baseline effect estimate, statistical significance, generalization to
new target families, or a state-of-the-art claim. The selected pockets and the
nine missing pockets may introduce completion bias.

The name `AnalyticalDirectionField` should not be read as validation that the
values model a physical field. Partial-charge and hydrophobicity constants are
fixed heuristics, and no calibration evidence is retained. Although the
Pocket2Mol baseline uses equivariant components, this extension projects raw
direction components through an ordinary MLP into scalar channels. No proof or
empirical rotation test establishes E(3) or SE(3) equivariance for this path.

## Validity denominator

The retained evaluator starts from molecules successfully reconstructed and
loaded from SDF. Failures before that point, including invalid generated graphs,
reconstruction failures, and molecules that could not be written or read, are
absent. Therefore per-pocket `success_rate` and `valid_molecules` fields are not
end-to-end generation validity and must not be reported as such.

## Docking and PoseBusters

Docking scores are computational estimates, not binding measurements. A score
threshold is not evidence of biological activity. PoseBusters is an additional
geometry and chemistry screen; the observed 42.3% pass rate shows that many
evaluated structures fail at least one configured check. Neither measure
substitutes for experimental validation.

## Training provenance gaps

The exact complete package environment could not be captured reliably from the
source server. The interpreter path and logs indicate Python 3.10, while the
upstream `env_cuda113.yml` documents an older reference environment. Neither is
an exact lockfile for the retained run. Randomness, GPU kernels, docking tool
versions, external data, and scheduler history may prevent byte-identical
reproduction.
