"""Sampling guards shared by the diagnostic and retained sampler paths."""

from __future__ import annotations


def _all_below(values, threshold: float) -> bool:
    reduced = (values < threshold).all()
    if hasattr(reduced, "item"):
        reduced = reduced.item()
    return bool(reduced)


def relax_initialization_thresholds(
    threshold,
    *,
    pdf_pos,
    p_focal,
    element_prob,
) -> bool:
    """Relax the first threshold that excludes every candidate.

    Returns false when the empty candidate set is not caused by any threshold,
    so callers can terminate instead of retrying the same state forever.
    """

    if _all_below(pdf_pos, threshold.pos_threshold):
        threshold.pos_threshold /= 2
        return True
    if _all_below(p_focal, threshold.focal_threshold):
        threshold.focal_threshold /= 2
        return True
    if _all_below(element_prob, threshold.element_threshold):
        threshold.element_threshold /= 2
        return True
    return False
