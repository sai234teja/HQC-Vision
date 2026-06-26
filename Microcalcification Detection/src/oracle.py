from qiskit import QuantumCircuit

class HQC_Oracle:
    """
    Phase-flip oracle for Grover's algorithm.

    FIX summary:
    - main.py used class name 'CalcificationOracle' → added as alias at bottom.
    - main.py passed kwarg 'target_indices' → accepted as alias for 'marked_indices'.
    - main.py called '.construct_circuit()' → added as alias for '.construct()'.
    """

    def __init__(self, num_qubits, marked_indices=None, target_indices=None):
        self.num_qubits = num_qubits
        # FIX: accept either parameter name
        self.marked_indices = marked_indices if marked_indices is not None else (target_indices or [])

    def construct(self):
        """
        Modified Oracle: Marks the target pattern indices identified
        by the classical feature pre-screening.
        """
        oracle_qc = QuantumCircuit(self.num_qubits)

        for index in self.marked_indices:
            binary = format(index, f'0{self.num_qubits}b')

            # 1. Flip zeros to ones to enable Multi-Controlled X gate
            for i, bit in enumerate(reversed(binary)):
                if bit == '0':
                    oracle_qc.x(i)

            # 2. Phase Flip (the marking step)
            oracle_qc.h(self.num_qubits - 1)
            oracle_qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            oracle_qc.h(self.num_qubits - 1)

            # 3. Restore basis
            for i, bit in enumerate(reversed(binary)):
                if bit == '0':
                    oracle_qc.x(i)

        return oracle_qc

    # FIX: alias used by main.py
    def construct_circuit(self):
        return self.construct()


# FIX: alias used by main.py
CalcificationOracle = HQC_Oracle