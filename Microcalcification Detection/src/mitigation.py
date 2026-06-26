import numpy as np


def apply_cem_layer(raw_counts, shots):
    """
    HQC-Vision CEM Layer:
    - Removes low-probability noise states
    - Applies statistical majority voting
    - Returns (top_state, confidence_score)

    FIX: Original noise_threshold = 0.01 * shots = ~10 counts.
    With 12 qubits (4096 possible states) and 1024 shots, the average
    count per state is only 0.25 — even the top Grover-amplified state
    may only have 3-8 counts. Threshold is now adaptive: 
    max(1, mean_count * 0.5) so it removes genuine noise while keeping
    the amplified signal.
    """
    if not raw_counts:
        return None, 0.0

    counts_array = np.array(list(raw_counts.values()))
    mean_count   = counts_array.mean()

    # Adaptive noise floor: half the mean, minimum 1
    noise_threshold = max(1, mean_count * 0.5)

    filtered_counts = {k: v for k, v in raw_counts.items()
                       if v > noise_threshold}

    if not filtered_counts:
        # Fallback: just take the single highest state
        top_state = max(raw_counts, key=raw_counts.get)
        top_votes = raw_counts[top_state]
        confidence = (top_votes / shots) * 100
        return top_state, confidence

    # Statistical majority voting
    sorted_results = sorted(filtered_counts.items(),
                            key=lambda item: item[1], reverse=True)
    top_state, top_votes = sorted_results[0]

    # Confidence = ratio of top state vs total filtered signal
    total_signal  = sum(filtered_counts.values())
    confidence    = (top_votes / total_signal) * 100

    return top_state, confidence