# Environment provenance

The retained evidence indicates a Python 3.10 training process on the AIDD
research server. Training logs and checkpoint serialization are consistent with
PyTorch, but a complete package inventory query did not finish within the
read-only diagnostic window. Exact versions of PyTorch, PyTorch Geometric,
RDKit, CUDA, the GPU driver, docking tools, and PoseBusters are therefore not
asserted.

The root `env_cuda113.yml` comes from the pinned Pocket2Mol upstream source. It
documents Python 3.8, PyTorch 1.10.1, and CUDA 11.3-era dependencies and is
useful as historical installation guidance only. It is not an environment lock
for the DF 500K artifact.

Future experiments should capture at minimum:

- an immutable container image digest or complete conda/pip lock;
- Python, PyTorch, PyG, RDKit, CUDA, driver, compiler, docking, and PoseBusters
  versions;
- GPU type, random seeds, deterministic settings, and command lines;
- input dataset and split hashes;
- source commit, clean/dirty state, configuration hash, and checkpoint hash.
