# Invalidated ADF/BIF 400K experiment

An experimental checkpoint described as an ADF+BIF 400K model was inspected but
is deliberately excluded from `artifacts/` and from valid model claims.

The checkpoint optimizer state showed:

| Parameter group | Parameters | Parameters with optimizer state |
| --- | ---: | ---: |
| backbone | 381 | 369 |
| ADF | 18 | 0 |
| BIF | 24 | 0 |
| affinity | 10 | 0 |

In Adam-family training, the absence of optimizer state for every parameter in
a group is evidence that those parameters did not receive optimizer updates in
the retained training state. The configuration also set affinity loss weight
and adaptive gate regularization to zero, while the available evaluation loss
was a placeholder zero. These observations do not prove why the wiring failed,
but they invalidate the claim that the retained ADF/BIF parameters were trained
as intended.

The checkpoint must not be used for performance comparison. A future rerun
requires tests that show nonzero gradients, changing parameter values, optimizer
state creation for every intended group, non-placeholder validation metrics,
and an ablation against the same baseline and data split.
