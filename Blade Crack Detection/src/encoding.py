import numpy as np
from typing import Tuple  # FIX: Tuple was used but never imported


class QuantumEncoder:
    """
    Encodes 2D patch coordinates into a 1D Hilbert space.
    Basis: |index> where index = y * patch_width + x
    """
    def __init__(self, patch_size: int):
        self.patch_size = patch_size
        self.total_states = patch_size ** 2
        self.n_qubits = int(np.ceil(np.log2(self.total_states)))

    def map_to_index(self, y: int, x: int) -> int:
        """Classical coordinate to quantum index."""
        return y * self.patch_size + x

    def map_to_coordinate(self, index: int) -> Tuple[int, int]:  # FIX: now Tuple is properly imported
        """Quantum index back to classical patch-relative coordinate."""
        y = index // self.patch_size
        x = index % self.patch_size
        return y, x