import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from modules.oracle import phase_oracle


def create_diffuser(n_qubits):
    """
    Creates the Grover Diffuser (inversion about the mean).
    Uses H · MCX · H to implement the multi-controlled Z gate,
    since qc.mcz() does not exist in Qiskit.
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    # Multi-controlled Z via H · MCX · H on the last qubit
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_quantum_search(n_qubits, targets, shots=1024):
    """
    Assembles and runs the Grover circuit on the Aer qasm_simulator.

    Parameters:
        n_qubits (int): Number of qubits (search space = 2^n_qubits).
        targets (list[int]): Integer indices of the marked states.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement counts keyed by bitstring.
    """
    if not targets:
        return {}

    # 1. Initialize backend
    backend = Aer.get_backend("qasm_simulator")

    # 2. Build the circuit
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))  # Uniform superposition

    # Optimal number of Grover iterations: floor(pi/4 * sqrt(N/M))
    n_states = 2 ** n_qubits
    n_targets = len(targets)
    iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(n_states / n_targets))))

    # Get composable sub-circuits
    oracle = phase_oracle(n_qubits, targets)
    diffuser = create_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure_all()

    # 3. Compile and run
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()

    return result.get_counts()