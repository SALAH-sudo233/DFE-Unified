# P4a — rule-BIF re-rank on frozen candidate pool (C-line)

End-to-end selection/enrichment framework run on the frozen 590-molecule I3
candidate pool. Vina is used only as a blind oracle.

## Code (`code/`)

- `unified_v1_rank.py` — eligible gate → BIF rank (raw / ligand-efficiency) →
  selection metrics (LE + Vina) → anti-gaming controls (heavy-atom, scaffold
  diversity, QED) → per-pocket bootstrap CI.
- `size_matched.py` — size-matched comparison to expose heavy-atom confounds.
- `openness_robustness.py` — robustness sweep helper.

## Records (`records/`)

- `unified_v1_report.json` — raw-Vina ranking (looks enriched: top-1 Δ≈−0.78,
  3/5 beat CI) — **but a size artifact** (BIF top-10 heavy +1.5~+5.5 vs pool).
- `unified_v1_report_le.json` — size-neutral ligand-efficiency oracle:
  enrichment **vanishes** (raw rank Δ≈−0.07, 0/5; debiased Δ≈0).

## Conclusion

Rule descriptors do not predict ligand efficiency; rule-BIF cannot produce
trustworthy enrichment. Real enrichment evidence must come from a learned head
(see `../p3-screening/`). The framework itself is reusable and hooks a learned
head directly.
