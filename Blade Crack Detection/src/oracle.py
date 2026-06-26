from qiskit import QuantumCircuit
# FIX: ZGate was imported but never used directly; removed unused import


class CrackOracle:
    """
    Advanced Quantum Oracle for Line-like Defect Marking.
    Implements a phase-flip for indices identified as potential cracks.
    """
    def __init__(self, n_qubits: int, target_indices: list):  # FIX: list[int] requires Python 3.9+
        self.n_qubits = n_qubits
        self.target_indices = target_indices

    def construct(self) -> QuantumCircuit:
        """
        Constructs the oracle circuit.
        Uses H-MCX-H pattern for a correct multi-controlled phase flip.
        """
        qc = QuantumCircuit(self.n_qubits, name="CrackOracle")

        if not self.target_indices:
            return qc

        for index in self.target_indices:
            # 1. Flip zeros to ones for the target binary string
            binary = format(index, f'0{self.n_qubits}b')
            # Reverse because Qiskit uses little-endian bit ordering
            binary = binary[::-1]

            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

            # 2. Multi-Controlled Phase Flip via H-MCX-H on the target qubit
            # FIX: the original code applied mcx(...) then qc.z(...) independently,
            # which does NOT implement a phase flip. The correct pattern is:
            #   H  →  MCX  →  H  on the last qubit, with all others as controls.
            # This maps |1...1⟩ → -|1...1⟩ (phase kickback), matching Grover's oracle spec.
            controls = list(range(self.n_qubits - 1))
            target = self.n_qubits - 1
            qc.h(target)
            qc.mcx(controls, target)
            qc.h(target)

            # 3. Uncompute X gates to restore basis
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

        return qc