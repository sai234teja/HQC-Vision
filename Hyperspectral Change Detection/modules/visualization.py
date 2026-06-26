import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config


def plot_results(t1, t2, diff, candidate_mask, final_map):
    """
    Generates and saves a multi-panel results figure.

    Panels:
        1. T1 image (pre-change)
        2. T2 image (post-change)
        3. Spectral difference map (normalized)
        4. Candidate mask (threshold-based)
        5. Quantum-enhanced change map (Grover output)

    Parameters:
        t1 (np.ndarray): Preprocessed T1 image (H, W, C).
        t2 (np.ndarray): Preprocessed T2 image (H, W, C).
        diff (np.ndarray): Normalized 2D difference map.
        candidate_mask (np.ndarray): Boolean 2D mask of candidate pixels.
        final_map (np.ndarray): 2D float map from quantum postprocessing.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Quantum-Enhanced Change Detection Results", fontsize=14)

    # Panel 1 — T1
    axes[0].imshow(np.clip(t1, 0, 1))
    axes[0].set_title("T1 (Before)")
    axes[0].axis("off")

    # Panel 2 — T2
    axes[1].imshow(np.clip(t2, 0, 1))
    axes[1].set_title("T2 (After)")
    axes[1].axis("off")

    # Panel 3 — Difference map
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Difference Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4 — Candidate mask
    axes[3].imshow(candidate_mask, cmap="gray")
    axes[3].set_title(f"Candidate Mask\n(top {100 - Config.THRESHOLD_PERCENTILE}%)")
    axes[3].axis("off")

    # Panel 5 — Quantum change map
    im2 = axes[4].imshow(final_map, cmap="plasma")
    axes[4].set_title("Quantum Change Map\n(Grover output)")
    axes[4].axis("off")
    plt.colorbar(im2, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(Config.OUTPUT_DIR, "change_detection_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {out_path}")