# Conservative fine-tuning chain — EXIT decision (2026-09-16)

## Decision
Close the entire conservative fine-tuning line. DF-500k `500000.pt` is declared the best generation checkpoint. No further FT (A/B/E) or BIF+DF joint fine-tuning will be pursued under the current representation.

## Evidence chain that led here
1. **530k continuation** (30k steps, lr 1e-4, full model): degraded vs 500k. 12-pocket x 3-seed pilot: 530k Vina -6.055 vs baseline -6.247 (delta +0.192 worse), PB 63.1% vs 68.6%, only 12/36 paired runs improved.
2. **Checkpoint trajectory 500k/528k/530k**: main drift in encoder interaction message path, not a single output head.
3. **BIF geometry signal probe (causal control v2)**: NO-GO. full test Pearson 0.320 < ligand_only 0.510 (reversed); full-permuted +0.096 attributable to pocket marginal, not pair-specific interaction. Real atomic geometry gives no usable pairwise signal beyond ligand marginal. Independent confirmation of earlier P3 finding.
4. **Plan A (freeze encoder+DF, train output heads only, 5k, lr 1e-5)**: converged in supervised loss (val 1.51). Directional check idx6/59/89 seed2020 vs baseline:
   - idx6  Vina -5.658 vs -6.182 (worse 0.52), PB 19/30 vs 25/36
   - idx59 Vina -6.006 vs -5.817 (better 0.19), PB 22/31 vs 20/31
   - idx89 Vina -4.084 vs -4.258 (worse 0.17), PB 31/34 vs 33/39
   - Net Vina ~ -0.17 kcal/mol (worse), 1 of 3 improved. Fails the go/no-go gate (paired Vina not worse than baseline by >0.10).

## Conclusion
Conservative fine-tuning from 500k does not reliably beat baseline on raw Vina; the same small-degradation direction as 530k reappears. Combined with the BIF geometry NO-GO, there is no evidence-supported path to improve generation quality by further fine-tuning under the current architecture/representation.

## Locked position
- Best checkpoint: `/workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt` (DF-500k).
- Protected/preserved: 500000.pt (best), 528000.pt, 530000.pt (degradation evidence), armb 25000.pt (I3).
- No new GPU spend on A/B/E/BIF joint FT.
- Redirect any future work to: matched-protocol reporting of DF-500k / I3 vs baseline (size-matched Vina, LE, direct PB), not new fine-tuning.

## Not claimed
DF-500k is best among the checkpoints we tested on this protocol; this is not a claim of CrossDocked SOTA. Directional check is single-seed / 3-pocket, sufficient only to reject the line given consistent direction with the fuller 12-pocket 530k pilot.
