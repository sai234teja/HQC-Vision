import argparse
import os
import logging
import numpy as np

from modules import (loader, preprocessing, pca_reduce, difference_map,
                     quantum_encoder, grover_runner, postprocess, visualization,
                     mitigation)
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum-Enhanced Hyperspectral Change Detection")
    parser.add_argument("--t1", required=True, help="Path to T1 (before) image")
    parser.add_argument("--t2", required=True, help="Path to T2 (after) image")
    parser.add_argument("--shots", type=int, default=Config.SHOTS, help="Number of quantum shots")
    return parser.parse_args()


def main():
    args = parse_args()
    Config.T1_PATH = args.t1
    Config.T2_PATH = args.t2
    Config.SHOTS = args.shots

    # Set seed explicitly (not at import time)
    Config.set_seed()

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(filename=Config.LOG_FILE, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Pipeline started.")

    # 1. Load
    logging.info("Step 1: Loading images.")
    t1, t2 = loader.HyperspectralLoader.load_data(Config.T1_PATH, Config.T2_PATH)

    # 2. Preprocess
    logging.info("Step 2: Preprocessing.")
    t1_p, t2_p = preprocessing.preprocess_cubes(t1, t2)

    # 3. PCA
    logging.info("Step 3: PCA reduction.")
    t1_pca, t2_pca = pca_reduce.apply_pca(t1_p, t2_p, Config.N_COMPONENTS)

    # 4. Difference Map
    logging.info("Step 4: Computing difference map.")
    diff = difference_map.compute_difference(t1_pca, t2_pca)
    mask = difference_map.get_candidate_mask(diff, Config.THRESHOLD_PERCENTILE)

    # 5. Quantum Encoding
    logging.info("Step 5: Encoding candidates for quantum search.")
    n_qubits, mapping = quantum_encoder.encode_candidates(mask, Config.MAX_QUBITS)

    # Derive targets from the top-scoring candidates in the diff map
    # rather than hardcoding indices
    candidate_coords = [mapping[i] for i in range(len(mapping))]
    candidate_scores = [diff[coord] for coord in candidate_coords]
    n_targets = max(1, len(mapping) // 10)  # Top 10% of candidates
    target_indices = list(np.argsort(candidate_scores)[::-1][:n_targets])
    logging.info(f"Targeting {len(target_indices)} state(s) across {n_qubits} qubit(s).")

    # 6. Quantum Execution
    logging.info("Step 6: Running Grover search.")
    counts = grover_runner.run_quantum_search(n_qubits, target_indices, shots=Config.SHOTS)

    # 7. Readout Mitigation
    logging.info("Step 7: Applying readout mitigation.")
    counts = mitigation.apply_readout_mitigation(counts)
    confidence = mitigation.confidence_score(counts)
    logging.info(f"Confidence score: {confidence:.2f}")

    # 8. Post-process & Visualize
    logging.info("Step 8: Postprocessing and visualization.")
    final_map = postprocess.counts_to_image(counts, mapping, diff.shape)
    final_map_refined = postprocess.refine_map(final_map > 0.5)
    visualization.plot_results(t1_p, t2_p, diff, mask, final_map)

    print(f"Process complete. Results saved in '{Config.OUTPUT_DIR}'. Confidence: {confidence:.2f}")
    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()