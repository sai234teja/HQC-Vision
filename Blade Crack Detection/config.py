import os

class Config:

    PATCH_SIZE = 20
    STRIDE = 10       # Reduced from 20 → 10: overlapping patches give denser crack coverage

    CANNY_LOW = 40
    CANNY_HIGH = 120

    TOTAL_QUBITS = 9
    SHOTS = 4096
    SIMULATOR = "qasm_simulator"

    # Paths
    INPUT_DIR = "data/NEU/Scratches"
    OUTPUT_DIR = "outputs/NEU_Results"

    @classmethod
    def setup(cls):
        os.makedirs(cls.INPUT_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

Config.setup()