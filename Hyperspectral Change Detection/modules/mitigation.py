import numpy as np


def apply_readout_mitigation(counts, calibration_matrix=None):
    """
    Simple mitigation: filters out low-probability noise based on shot threshold.
    In a real NISQ setup, this would use a CalibrationMatrix.
    """
    if not counts:
        return {}

    total_shots = sum(counts.values())
    threshold = 0.005 * total_shots  # Filter results appearing in < 1% of shots
    mitigated_counts = {k: v for k, v in counts.items() if v > threshold}
    return mitigated_counts


def confidence_score(counts):
    """Calculates a confidence ratio between the top hit and the mean noise."""
    if not counts:
        return 0.0

    sorted_v = sorted(counts.values(), reverse=True)
    top_hit = sorted_v[0]
    avg_noise = np.mean(sorted_v[1:]) if len(sorted_v) > 1 else 1.0
    return top_hit / avg_noise