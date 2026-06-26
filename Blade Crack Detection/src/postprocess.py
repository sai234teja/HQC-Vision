import numpy as np
from typing import List  # FIX: List was used in type hints but never imported


class PostProcessor:
    @staticmethod
    def aggregate_results(detections: List[dict], proximity_threshold: int = 5) -> List[dict]:
        """
        Merges detections that are too close to each other (duplicates).
        """
        if not detections:
            return []

        # FIX: operate on a copy so the original list is not mutated by pop/remove
        detections = list(detections)
        merged = []

        while detections:
            base = detections.pop(0)
            neighbors = [d for d in detections if
                         np.linalg.norm(np.array([base['y'], base['x']]) -
                                        np.array([d['y'], d['x']])) < proximity_threshold]

            if neighbors:
                avg_y = int(np.mean([base['y']] + [n['y'] for n in neighbors]))
                avg_x = int(np.mean([base['x']] + [n['x'] for n in neighbors]))
                avg_conf = np.mean([base['confidence']] + [n['confidence'] for n in neighbors])
                merged.append({'y': avg_y, 'x': avg_x, 'confidence': avg_conf})

                for n in neighbors:
                    detections.remove(n)
            else:
                merged.append(base)

        return merged

    @staticmethod
    def calculate_confidence(counts: dict, total_shots: int) -> float:
        """Calculates the probability peak of the measured state."""
        max_val = max(counts.values())
        return max_val / total_shots