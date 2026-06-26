from qiskit import QuantumCircuit
import numpy as np


def phase_oracle(n_qubits, target_indices):
    """
    Builds a phase-kickback oracle sub-circuit that marks the target states
    with a -1 phase. Used as a composable component by grover_runner.py.

    Parameters:
        n_qubits (int): Number of qubits in the search space.
        target_indices (list[int]): Integer indices of the states to mark.

    Returns:
        QuantumCircuit (no measurements, suitable for compose())
    """
    qc = QuantumCircuit(n_qubits)

    for target in target_indices:
        binary_target = format(target, f'0{n_qubits}b')

        # Flip qubits where the target bit is '0' so the all-|1> state
        # corresponds to this target
        for i, bit in enumerate(reversed(binary_target)):
            if bit == '0':
                qc.x(i)

        # Multi-controlled Z via: H · MCX · H on the last qubit
        # (qc.mcz does not exist in Qiskit; this is the correct equivalent)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

        # Unflip the conditioning qubits
        for i, bit in enumerate(reversed(binary_target)):
            if bit == '0':
                qc.x(i)

    return qc


def build_grover_circuit(n_qubits, target_indices):
    """
    Builds a self-contained Grover circuit including superposition,
    oracle, diffuser, and measurements.

    Note: grover_runner.py now owns the main execution path. This function
    is kept for standalone testing / reference.

    Parameters:
        n_qubits (int): Number of qubits.
        target_indices (list[int]): States to mark.

    Returns:
        QuantumCircuit (with measurements)
    """
    if not target_indices:
        raise ValueError("target_indices must not be empty.")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))  # Superposition

    n_states = 2 ** n_qubits
    iterations = max(1, int(np.floor(
        (np.pi / 4) * np.sqrt(n_states / len(target_indices))
    )))

    oracle = phase_oracle(n_qubits, target_indices)

    for _ in range(iterations):
        # Oracle
        qc.compose(oracle, inplace=True)

        # Diffuser (inversion about the mean)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc