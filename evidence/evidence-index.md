# Evidence index

| Evidence | Location | Interpretation |
| --- | --- | --- |
| Upstream source | `UPSTREAM.md` | Pinned Pocket2Mol baseline and license |
| Retrieved file ledger | `download-records.json` | Allowlisted server paths, sizes, times, and hashes |
| Code manifest | `code-manifest.json` | Hash binding for the observed DF implementation |
| Model manifest | `../artifacts/MANIFEST.json` | DF 500K checkpoint and training-log identity |
| Partial-run provenance | `../results/df-500k-21-pocket/provenance.json` | Requested/completed pockets, record counts, aggregate observations |
| Per-pocket raw records | `../results/df-500k-21-pocket/per-pocket/` | SDF, SMILES, and docking/PoseBusters records |
| Dataset observation | `../data/crossdocked-manifest.json` | External input sizes and split hash without redistribution |
| Invalidated ADF/BIF experiment | `invalidated-experiments/adf-bif-400k.md` | Why the experiment is excluded from valid model evidence |

Manifests are generated deterministically by `scripts/generate_manifests.py`.
`scripts/verify_repository.py` performs the repository-level integrity and
security checks. Remote modification timestamps are provenance observations,
not cryptographic proof of when an experiment ran; the SHA-256 values bind the
retained bytes.
