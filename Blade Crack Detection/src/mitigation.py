import numpy as np
from typing import Optional


class QuantumErrorMitigator:
    """
    Implements classical post-selection and confidence filtering
    to mitigate 'shot noise' from the Grover measurement.
    """

    @staticmethod
    def filter_detections(counts: dict, threshold: float = 0.15) -> list:
        """
        Filters quantum measurement results by probability threshold.

        Raised default threshold from 0.10 → 0.15:
        - For a 400-state space (20x20), random chance is 1/400 = 0.0025
        - 0.10 was letting in weak noise-amplified states
        - 0.15 keeps only states where Grover clearly amplified the amplitude
        """
        total_shots = sum(counts.values())
        mitigated = [
            int(state, 2) for state, val in counts.items()
            if (val / total_shots) > threshold
        ]
        return mitigated

    @staticmethod
    def consensus_voting(results: list) -> Optional[int]:
        """Returns the most frequent coordinate across multiple quantum runs."""
        if not results:
            return None
        return max(set(results), key=results.count)