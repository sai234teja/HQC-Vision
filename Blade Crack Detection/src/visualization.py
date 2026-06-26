import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np


class Visualizer:
    @staticmethod
    def plot_results(original: np.ndarray, edges: np.ndarray,
                     detections: list, save_path: str):

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1 — Original
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Surface")
        axes[0].axis('on')

        # Panel 2 — Edge map
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title("Edge Map (Quantum Input)")
        axes[1].axis('on')

        # Panel 3 — Detections overlaid
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        axes[2].imshow(overlay)
        axes[2].set_title(f"Grover Detected Cracks ({len(detections)} points)")

        if detections:
            xs = [d['x'] for d in detections]
            ys = [d['y'] for d in detections]
            confs = [d.get('confidence', 1.0) for d in detections]

            # Scatter with confidence-based color: brighter = higher confidence
            scatter = axes[2].scatter(
                xs, ys,
                c=confs,
                cmap='autumn_r',   # red (high conf) → yellow (lower conf)
                s=30,              # smaller marker size (was circle r=2 → ~12px area)
                edgecolors='white',
                linewidths=0.4,
                vmin=0, vmax=0.3,
                zorder=5
            )
            plt.colorbar(scatter, ax=axes[2], label='Confidence', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] Visualization saved → {save_path}")