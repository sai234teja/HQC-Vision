from qiskit.circuit.library import GroverOperator


def build_hqc_core(encoder_qc, oracle_qc, iterations=1):
    """
    Assembles the Quantum Processing Core.
    Returns an unmeasured circuit — transpile + measure_all happens in main.py.
    
    FIX: Removed .decompose(reps=5) which crashes in Qiskit 2.x.
    Decomposition is handled entirely by transpile() in main.py.
    """
    grover_op = GroverOperator(oracle_qc)

    hqc_circuit = encoder_qc.copy()

    for _ in range(iterations):
        hqc_circuit.compose(grover_op, inplace=True)

    return hqc_circuit