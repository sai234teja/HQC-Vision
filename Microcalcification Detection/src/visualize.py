import matplotlib.pyplot as plt
import cv2
import os

def save_detection_plot(original_roi, mask_roi, pred_coord, output_path):
    """
    Visualizes the Quantum Search result against Ground Truth.

    NOTE: Previously saved as 'vizualize.py' (typo). Correct filename is
    'visualize.py'. Also renamed from 'save_results' → 'save_detection_plot'
    to match the actual function signature.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original ROI
    axes[0].imshow(original_roi, cmap='gray')
    axes[0].set_title("Original ROI (64x64)")

    # Ground Truth
    axes[1].imshow(mask_roi, cmap='jet')
    axes[1].set_title("Ground Truth Mask")

    # Quantum Detection Overlay
    display_img = cv2.cvtColor(original_roi, cv2.COLOR_GRAY2RGB)
    if pred_coord:
        x, y = pred_coord
        cv2.circle(display_img, (x, y), 3, (0, 255, 0), -1)

    axes[2].imshow(display_img)
    axes[2].set_title("Grover Detection (Green Dot)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [VIZ] Saved → {output_path}")


def plot_complexity_comparison(n_values, classical_times, quantum_times, save_path):
    """Plots timing benchmarks O(N) vs O(sqrt(N))."""
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, classical_times, label="Classical O(N)", marker='o')
    plt.plot(n_values, quantum_times, label="Quantum O(√N)", marker='s')
    plt.xlabel("Search Space Size (Pixels)")
    plt.ylabel("Inference Time (ms)")
    plt.title("Search Complexity: Classical vs Grover")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()