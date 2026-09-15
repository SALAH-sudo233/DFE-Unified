#!/usr/bin/env python3
"""Unified-v1 (P4a) rule-BIF constraint-first re-ranking on a FROZEN candidate pool.

Contract:
- Candidate pool = the 590 i3 molecules already generated (5 pockets). NEVER regenerate.
- docking_score (Vina) is a BLIND offline oracle: used ONLY to score selection quality,
  NEVER as a ranking feature (that would be leakage / cheating).
- Rule-BIF has two separated stages, recorded independently:
    (1) eligible gate: validity + PoseBusters pass  (+ optional min diversity)
    (2) interaction-compatibility rank score: descriptor-level proxy (no learned params).
- Report top-k selection vs random-rank baseline with bootstrap 95% CI.
- Anti-gaming controls: report heavy atoms, scaffold diversity, QED of selected top-k,
  so enrichment cannot be an artifact of smaller / single-scaffold / low-diversity picks.
"""
import json, os, math, random, statistics as st
from collections import defaultdict

random.seed(20260908)
POCKETS = ['1fm9', '1a1e', '1a9q', '3ebp', '1sgu']
HERE = os.path.dirname(os.path.abspath(__file__))


def load_pool():
    pool = {}
    for pk in POCKETS:
        with open(os.path.join(HERE, pk + '.json'), encoding='utf-8') as f:
            pool[pk] = json.load(f)
    return pool


# ---- rule-BIF stage 1: eligible gate ----
def is_eligible(rec):
    if not rec.get('valid', False):
        return False
    pb = rec.get('posebuster') or {}
    if not pb.get('passed', False):
        return False
    return True


# ---- rule-BIF stage 2: interaction-compatibility rank score (descriptor proxy) ----
# Higher = better predicted pocket compatibility. NO docking_score used.
# Two variants:
#   'raw'  : rewards absolute interaction bulk (H-bonds, rings) + drug-likeness, with a
#            tiny-fragment penalty. Correlated with molecule size -> size-biased.
#   'le'   : SIZE-DEBIASED. Interaction terms are normalized PER HEAVY ATOM (interaction
#            density), no size bonus/penalty. Aligned with ligand-efficiency oracle.
def bif_rank_score(rec, mode='raw'):
    hba = rec.get('num_hba', 0) or 0
    hbd = rec.get('num_hbd', 0) or 0
    rings = rec.get('num_rings', 0) or 0
    logp = rec.get('logp', 0.0) or 0.0
    qed = rec.get('qed', 0.0) or 0.0
    n = rec.get('num_atoms', 0) or 0
    hbond = hba + hbd
    hydro = -abs(logp - 2.5)
    if mode == 'raw':
        ringterm = rings
        size_pen = -max(0, 10 - n) * 0.3
        return 1.0 * hbond + 0.6 * ringterm + 0.4 * hydro + 1.5 * qed + size_pen
    # 'le': interaction density per heavy atom, size-neutral
    if n <= 0:
        return -1e9
    hbond_density = hbond / n
    ring_density = rings / n
    return 6.0 * hbond_density + 4.0 * ring_density + 0.4 * hydro + 1.5 * qed


# ---- Bemis-Murcko-ish scaffold key without RDKit: ring-count+atom-count bucket ----
def scaffold_key(rec):
    return (rec.get('num_rings', 0), rec.get('num_atoms', 0) // 3)


def lig_eff(rec):
    """Ligand efficiency = -docking_score / heavy_atoms. Higher = better. Size-neutral oracle."""
    v = rec.get('docking_score')
    n = rec.get('num_atoms', 0) or 0
    if v is None or n <= 0:
        return None
    return -v / n


def topk_stats(recs, k):
    sel = recs[:k]
    vinas = [r['docking_score'] for r in sel if r.get('docking_score') is not None]
    les = [lig_eff(r) for r in sel if lig_eff(r) is not None]
    pb = [1 for r in sel if (r.get('posebuster') or {}).get('passed')]
    heavy = [r.get('num_atoms', 0) for r in sel]
    qed = [r.get('qed', 0) for r in sel]
    scafs = {scaffold_key(r) for r in sel}
    return {
        'vina_mean': round(st.mean(vinas), 3) if vinas else None,
        'le_mean': round(st.mean(les), 4) if les else None,
        'le_best': round(max(les), 4) if les else None,
        'pb_rate': round(len(pb) / len(sel), 3) if sel else None,
        'heavy_mean': round(st.mean(heavy), 2) if heavy else None,
        'qed_mean': round(st.mean(qed), 3) if qed else None,
        'scaffold_diversity': round(len(scafs) / len(sel), 3) if sel else None,
    }


def random_baseline(recs, k, metric='le', n_boot=2000):
    """Bootstrap random-rank baseline for the chosen oracle metric.
    metric='le' -> mean ligand efficiency (higher better);
    metric='vina' -> mean docking score (lower better)."""
    means = []
    for _ in range(n_boot):
        sample = random.sample(recs, min(k, len(recs)))
        if metric == 'le':
            vals = [lig_eff(r) for r in sample if lig_eff(r) is not None]
        else:
            vals = [r['docking_score'] for r in sample if r.get('docking_score') is not None]
        if vals:
            means.append(st.mean(vals))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return round(st.mean(means), 4), round(lo, 4), round(hi, 4)


def run_mode(pool, rank_mode):
    """rank_mode in {'raw','le'}. Oracle = ligand efficiency (size-neutral, higher better)."""
    report = {'rank_mode': rank_mode, 'oracle': 'ligand_efficiency', 'pockets': {}, 'ks': [1, 5, 10]}
    agg = defaultdict(list)
    for pk in POCKETS:
        recs = pool[pk]
        eligible = [r for r in recs if is_eligible(r)]
        ranked = sorted(eligible, key=lambda r: bif_rank_score(r, rank_mode), reverse=True)
        pj = {'n_total': len(recs), 'n_eligible': len(eligible), 'topk': {}, 'random': {}}
        for k in report['ks']:
            pj['topk'][k] = topk_stats(ranked, k)
            rm, lo, hi = random_baseline(eligible, k, metric='le')
            pj['random'][k] = {'le_mean': rm, 'ci95': [lo, hi]}
            # LE: higher is better -> enrichment means BIF le_mean ABOVE random CI upper bound
            bif_le = pj['topk'][k]['le_mean']
            pj['topk'][k]['beats_random_ci'] = (bif_le is not None and bif_le > hi)
            agg[k].append((bif_le, rm))
        report['pockets'][pk] = pj
    report['macro'] = {}
    for k in report['ks']:
        bif = [b for b, _ in agg[k] if b is not None]
        rnd = [r for _, r in agg[k] if r is not None]
        report['macro'][k] = {
            'bif_le_mean': round(st.mean(bif), 4) if bif else None,
            'random_le_mean': round(st.mean(rnd), 4) if rnd else None,
            'delta': round(st.mean(bif) - st.mean(rnd), 4) if bif and rnd else None,
            'pockets_beating_random_ci': sum(
                1 for pk in POCKETS if report['pockets'][pk]['topk'][k]['beats_random_ci']
            ),
        }
    return report


def main():
    pool = load_pool()
    full = {}
    for mode in ('raw', 'le'):
        rep = run_mode(pool, mode)
        full[mode] = rep
        print(f"\n===== rank_mode={mode}  (oracle=ligand_efficiency, higher=better) =====")
        print(json.dumps(rep['macro'], indent=2))
    print('\nper-pocket eligible:',
          {pk: full['le']['pockets'][pk]['n_eligible'] for pk in POCKETS})
    out = os.path.join(HERE, 'unified_v1_report_le.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2)
    print('report ->', out)


if __name__ == '__main__':
    main()
