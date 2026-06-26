import os
import yaml
import numpy as np
from qiskit import transpile
from qiskit_aer import Aer
from src.loader import DataLoader
from src.preprocess import preprocess_roi
from src.quantum_encoder import QuantumImageEncoder
from src.oracle import HQC_Oracle
from src.grover import build_hqc_core
from src.visualize import save_detection_plot
from src.roi import ROIExtractor
from src.mitigation import apply_cem_layer
from src.metrics import compute_metrics
from src.utils import decode_to_bounding_box

def main():
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)

    # 1. Load Data via CSV manifest
    loader = DataLoader(cfg['paths']['data_dir'], split='train')
    if len(loader) == 0:
        print("ERROR: No matched pairs found.")
        return

    roi_extractor = ROIExtractor(
        size=cfg['params']['roi_size'],
        stride=cfg['params']['stride']
    )
    backend = Aer.get_backend('qasm_simulator')
    shots   = 1024

    all_y_true, all_y_pred, all_y_probs = [], [], []

    for i in range(len(loader)):
        image, mask, meta = loader.load_pair(i)
        patient_id = meta['patient_id']

        # 2. ROI Extraction
        roi, roi_origin = roi_extractor.get_lesion_center_roi(image, mask)
        if roi is None:
            print(f"[SKIP] No lesion in mask for {patient_id}")
            continue

        roi_size = cfg['params']['roi_size']
        roi_mask = mask[
            roi_origin[1]:roi_origin[1] + roi_size,
            roi_origin[0]:roi_origin[0] + roi_size
        ]

        # 3. Preprocess
        enhanced, binary = preprocess_roi(roi)

        # 4. Quantum Encoding (Hadamard uniform superposition)
        encoder    = QuantumImageEncoder(num_qubits=12)
        encoder_qc = encoder.encode()

        # 5. Candidate Screening
        flat_enhanced = enhanced.flatten()
        threshold = np.percentile(flat_enhanced, cfg['params']['threshold_percentile'])
        targets   = np.where(flat_enhanced > threshold)[0].tolist()

        if not targets:
            print(f"[SKIP] No targets above threshold for {patient_id}")
            continue

        # 6. Oracle & Grover circuit (unmeasured)
        oracle    = HQC_Oracle(num_qubits=12, marked_indices=targets[:5])
        oracle_qc = oracle.construct()
        circuit   = build_hqc_core(encoder_qc, oracle_qc,
                                   iterations=cfg['params']['grover_iterations'])

        # 7. Transpile entire circuit to Aer basis gates, add measurements
        circuit = transpile(circuit, backend=backend, optimization_level=1)
        circuit.measure_all()

        # 8. Execute
        job        = backend.run(circuit, shots=shots)
        raw_counts = job.result().get_counts()

        # 9. CEM Noise Mitigation
        top_state, confidence = apply_cem_layer(raw_counts, shots=shots)
        if top_state is None:
            print(f"[WARN] CEM returned no signal for {patient_id}")
            continue

        # 10. Decode to coordinates
        pred_coord, bbox = decode_to_bounding_box(
            top_state, roi_origin, roi_size=roi_size
        )

        # 11. Pixel-level metrics
        pred_map = np.zeros_like(roi_mask, dtype=np.uint8)
        px = int(np.clip(pred_coord[0] - roi_origin[0], 0, roi_size - 1))
        py = int(np.clip(pred_coord[1] - roi_origin[1], 0, roi_size - 1))
        pred_map[py, px] = 255

        y_true_flat  = (roi_mask.flatten() > 127).astype(int)
        y_pred_flat  = (pred_map.flatten() > 127).astype(int)
        y_probs_flat = np.zeros(len(y_true_flat))
        y_probs_flat[py * roi_size + px] = confidence / 100.0

        all_y_true.extend(y_true_flat.tolist())
        all_y_pred.extend(y_pred_flat.tolist())
        all_y_probs.extend(y_probs_flat.tolist())

        # 12. Visualization
        out_path = os.path.join(
            cfg['paths']['output_dir'],
            f"result_{patient_id}_{i}.png"
        )
        save_detection_plot(roi, roi_mask, (px, py), out_path)
        print(f"[{i+1}/{len(loader)}] {patient_id} | "
              f"conf={confidence:.1f}% | bbox={bbox} | {meta['pathology']}")

    # 13. Final Metrics
    if all_y_true:
        metrics = compute_metrics(
            np.array(all_y_true),
            np.array(all_y_pred),
            np.array(all_y_probs)
        )
        print("\n=== Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")

if __name__ == "__main__":
    main()