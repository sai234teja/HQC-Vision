import numpy as np

def encode_candidates(candidate_mask, max_qubits=13):
    """Maps 2D coordinates to binary strings for Grover."""
    coords = np.argwhere(candidate_mask)
    num_candidates = len(coords)
    
    # Grover needs N = 2^n states
    n_qubits = int(np.ceil(np.log2(num_candidates))) if num_candidates > 0 else 1
    n_qubits = min(n_qubits, max_qubits)
    
    # For simulation, we limit the search space
    search_space_size = 2**n_qubits
    selected_coords = coords[:search_space_size]
    
    mapping = {i: tuple(coord) for i, coord in enumerate(selected_coords)}
    return n_qubits, mapping