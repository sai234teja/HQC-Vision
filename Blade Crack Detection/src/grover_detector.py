from qiskit_aer import AerSimulator          # FIX: qiskit.Aer removed in Qiskit 1.x; use qiskit_aer
from qiskit import QuantumCircuit, transpile  # FIX: removed deprecated `execute`
from qiskit.circuit.library import GroverOperator
from qiskit_aer.primitives import Sampler     # FIX: use Sampler primitive instead of deprecated execute()
from src.oracle import CrackOracle
import numpy as np


class GroverCrackDetector:
    def __init__(self, patch_size: int):
        # Ensure we have enough qubits to cover the entire patch area
        # For 20x20 = 400 pixels, we need 9 qubits (2^9 = 512)
        self.n_qubits = int(np.ceil(np.log2(patch_size * patch_size)))
        self.backend = AerSimulator()  # FIX: AerSimulator() replaces Aer.get_backend('qasm_simulator')

    def run(self, edge_map: np.ndarray):
        """
        Runs the Grover search on a 2D patch.
        """
        # 1. Flatten the map to 1D indices for the Oracle
        flat_map = edge_map.flatten()
        target_indices = np.where(flat_map > 0)[0].tolist()

        # Guard clause for empty patches
        if not target_indices:
            return None

        # 2. Build the Oracle
        # If there are too many 'marked' pixels, Grover loses efficiency.
        # We cap it or use a random subset to maintain the search speedup.
        max_targets = min(len(target_indices), 10)
        target_indices = target_indices[:max_targets]

        oracle_circ = CrackOracle(self.n_qubits, target_indices).construct()

        # 3. Create Grover Operator
        grover_op = GroverOperator(oracle_circ)

        # 4. Determine Optimal Iterations
        # pi/4 * sqrt(N/M)
        N_states = 2 ** self.n_qubits
        M_targets = len(target_indices)
        num_iterations = max(1, int(np.pi / 4 * np.sqrt(N_states / M_targets)))

        # 5. Build Final Circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))  # Initial superposition

        for _ in range(num_iterations):
            qc.compose(grover_op, inplace=True)

        qc.measure_all()

        # 6. Transpile and Execute
        # FIX: replaced deprecated execute() with transpile + backend.run()
        optimized_qc = transpile(qc, self.backend, optimization_level=3)
        result = self.backend.run(optimized_qc, shots=2048).result()
        return result.get_counts()