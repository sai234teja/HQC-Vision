import cv2
import numpy as np
import os
from config import Config
from src.preprocessing import ImagePreprocessor
from src.grover_detector import GroverCrackDetector
from src.visualization import Visualizer
from src.utils import generate_synthetic_crack
from src.mitigation import QuantumErrorMitigator
from src.postprocess import PostProcessor


def main():
    # 1. Load or Generate Data
    image_path = "data/blade_sample.png"
    image = cv2.imread(image_path, 0)
    if image is None:
        print("[INFO] No image found at data/blade_sample.png — using synthetic crack.")
        image = generate_synthetic_crack(128, 128)

    # 2. Preprocess
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.enhance(image)
    edges = preprocessor.detect_edges(enhanced)

    print(f"[INFO] Image size     : {image.shape}")
    print(f"[INFO] Total edge px  : {np.sum(edges > 0)}")

    # 3. Patch Processing & Quantum Search
    detector = GroverCrackDetector(Config.PATCH_SIZE)
    total_shots = 2048
    all_detections = []
    patches_processed = 0
    patches_skipped = 0

    for y in range(0, edges.shape[0] - Config.PATCH_SIZE + 1, Config.STRIDE):
        for x in range(0, edges.shape[1] - Config.PATCH_SIZE + 1, Config.STRIDE):
            patch = edges[y:y + Config.PATCH_SIZE, x:x + Config.PATCH_SIZE]

            # Edge density guard — skip near-empty patches
            edge_density = np.sum(patch > 0) / (Config.PATCH_SIZE ** 2)
            if edge_density < 0.02:
                patches_skipped += 1
                continue

            patches_processed += 1
            counts = detector.run(patch)
            if counts is None:
                continue

            # Adaptive threshold: denser patches need a higher bar to reduce
            # false positives; sparser crack patches use a lower threshold
            # so genuine weak Grover amplifications are not discarded.
            adaptive_threshold = max(0.04, min(0.10, edge_density * 0.8))
            filtered = QuantumErrorMitigator.filter_detections(counts, threshold=adaptive_threshold)
            if not filtered:
                continue

            top_state = max(counts, key=counts.get)
            idx = int(top_state, 2)
            py, px = divmod(idx, Config.PATCH_SIZE)

            confidence = counts[top_state] / total_shots
            all_detections.append({
                'x': x + px,
                'y': y + py,
                'confidence': confidence
            })

    # 4. Post-process: merge nearby duplicate detections
    # proximity_threshold=8 accounts for the overlapping patches from stride=10
    merged_detections = PostProcessor.aggregate_results(all_detections, proximity_threshold=8)

    # 5. Visualization
    os.makedirs("outputs", exist_ok=True)
    Visualizer.plot_results(image, edges, merged_detections, "outputs/result.png")

    print(f"[INFO] Patches skipped (low density) : {patches_skipped}")
    print(f"[INFO] Patches sent to quantum circuit: {patches_processed}")
    print(f"Analysis complete.")
    print(f"  Raw detections  : {len(all_detections)}")
    print(f"  After merging   : {len(merged_detections)}")
    print(f"  Output saved to : outputs/result.png")


if __name__ == "__main__":
    main()