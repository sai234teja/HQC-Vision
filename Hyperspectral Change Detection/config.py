import numpy as np

class Config:
    # Data Params
    T1_PATH = None  # Placeholder for CLI
    T2_PATH = None
    IMAGE_SHAPE = (128, 128)  # Downsample for quantum simulation speed
    CHANNELS = 3  # RGB channels after loading

    # PCA Params
    N_COMPONENTS = 3

    # Quantum Params
    MAX_QUBITS = 13  # Constraints for simulator/HW
    SHOTS = 8192
    THRESHOLD_PERCENTILE = 85  # Top 5% of changes go to Quantum Oracle

    # Paths
    OUTPUT_DIR = "outputs"
    LOG_FILE = "outputs/pipeline.log"

    # Seed
    SEED = 42

    @staticmethod
    def set_seed():
        """Call explicitly in main() to avoid import-time side effects."""
        np.random.seed(Config.SEED)