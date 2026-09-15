# P3 — Unified shared-representation screening (C-line)

Can a small affinity head trained on the **frozen generator's own encoder
representation** predict binding affinity? This is the paper's main sell-point
and its thinnest evidence.

## Code (`code/`)

- `cluster_refined.py` — 30% identity single-linkage clustering (pure-Python
  k-mer containment; no mmseqs/cd-hit/internet on the server) → 1324 clusters.
- `extract_all.py`, `probe_emb*.py` — shared-representation extractor over the
  frozen i3 encoder. `probe_emb3.py` is the verified version (kekulize ligands;
  transform order RefineData → LigandCountNeighbors → FeaturizeProteinAtom →
  FeaturizeLigandAtom → AtomComposer(knn); `set_science_vector_origin('zero')`).
- `recover_failed.py` — recovers complexes that failed kekulization on first pass.
- `train_head.py` — ScreeningHead training + full control suite.
- `geometry_probe/` — raw-atom-geometry alternative to encoder embeddings:
  ligand/pocket element marginals + cross-molecule element-pair distance
  histogram; `causal_control_v*.py` run full / permuted / ligand-only /
  pocket-only arms; `train_geometry_head*.py`, `train_pair_control*.py` iterate
  the head and controls.
- `geometry_repair_v3/` — TDD regression (`test_original.py` + captured RED)
  pinning the original feature-extraction behavior before repair.

## Records (`records/`)

- `screeninghead_metrics.json` — **headline**: full 0.648 vs ligand_only 0.628 /
  pocket_only 0.631 / pocket_shuffle 0.611 / label_permutation 0.117; pooled
  EF1% ≈ 2.0, BEDROC20 ≈ 0.40. Weak pair-specificity.
- `refined_split.json` — no-leak split (seed 20260908, 4049/444/823).
- `geometry_probe/causal_control_v2_metrics.json` — **NO-GO**: full 0.320 <
  ligand_only 0.510; permutation integrity all-zero.
- `geometry_probe/*_metrics.json`, `manifest.json`, `progress.json`,
  `geometry_head_v2_provenance.json` — per-run control metrics + input hash binding.

## Not committed (registered)

`embeddings_shared.pt`, `screeninghead_full.pt`, `refined_seqs.fasta`,
`refined_labels.json`, `refined_clusters.json`, `geometry_features.jsonl`,
`causal_control_v2/predictions.npz|permutation.json` — see
`../../evidence/large-artifacts-registry.json`.
