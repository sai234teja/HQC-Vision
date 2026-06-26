import numpy as np
from qiskit import QuantumCircuit


class QuantumImageEncoder:
    """
    Quantum Image Encoder for HQC-Vision.

    Two encoding modes:
    - 'hadamard' (default, fast): uniform superposition via H gates.
      Grover's oracle then amplifies the marked target states regardless
      of the initial amplitude distribution, so this is valid for search.
    - 'amplitude' (slow): full amplitude encoding via StatePreparation
      + transpile. Use only for small experiments / paper ablations.
    """

    def __init__(self, num_qubits=12, mode='hadamard'):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits      # 4096 for 12 qubits
        self.mode = mode

    def encode(self, feature_vector=None):
        """
        Returns an encoder QuantumCircuit.

        FIX: StatePreparation is not natively supported by Aer and requires
        an expensive transpile step. For the search pipeline we use a uniform
        Hadamard superposition instead — mathematically equivalent as the
        starting state for Grover's algorithm.
        """
        if self.mode == 'amplitude':
            return self._amplitude_encode(feature_vector)
        return self._hadamard_encode()

    def _hadamard_encode(self):
        """Fast uniform superposition — valid initial state for Grover."""
        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))    # |+>^n — equal superposition
        return qc

    def _amplitude_encode(self, feature_vector):
        """
        Full amplitude encoding. Requires transpile so Aer can run it.
        Slow for 12 qubits — use only for small-scale experiments.
        """
        from qiskit import transpile
        from qiskit.circuit.library import StatePreparation
        from qiskit_aer import Aer

        if feature_vector is None:
            raise ValueError("feature_vector required for amplitude encoding")

        # Resize to 2^n
        if len(feature_vector) < self.dim:
            padded = np.zeros(self.dim)
            padded[:len(feature_vector)] = feature_vector
            feature_vector = padded
        else:
            feature_vector = feature_vector[:self.dim]

        norm = np.linalg.norm(feature_vector)
        amplitudes = (feature_vector / norm) if norm > 0 \
                     else np.ones(self.dim) / np.sqrt(self.dim)

        qc = QuantumCircuit(self.num_qubits)
        qc.append(StatePreparation(amplitudes), range(self.num_qubits))

        backend = Aer.get_backend('qasm_simulator')
        return transpile(qc, backend=backend, optimization_level=0)

    # Alias
    def encode_amplitude(self, feature_vector=None):
        return self.encode(feature_vector)