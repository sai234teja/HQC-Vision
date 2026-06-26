import time
import cv2
import numpy as np


def generate_synthetic_crack(height: int, width: int) -> np.ndarray:
    """
    FIX: This function is imported in main.py but was completely missing from utils.py.
    Generates a grayscale image with a synthetic diagonal crack for testing.
    """
    image = np.ones((height, width), dtype=np.uint8) * 200  # light gray background
    # Draw a thin diagonal line to simulate a crack
    cv2.line(image, (width // 4, height // 4), (3 * width // 4, 3 * height // 4),
             color=30, thickness=1)
    # Add slight Gaussian noise for realism
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def benchmark_speedup(quantum_time: float, patch_size: int):
    """
    Calculates theoretical speedup metrics.
    N = Total pixels in patch
    Classical search ~ O(N)
    Quantum search ~ O(√N)
    """
    N = patch_size ** 2
    classical_complexity = N
    quantum_complexity = np.sqrt(N)

    print(f"\n--- Performance Metrics ---")
    print(f"Search Space (N): {N} pixels")
    print(f"Theoretical Classical Ops: {classical_complexity}")
    print(f"Theoretical Quantum Ops: {int(quantum_complexity)}")
    print(f"Actual Execution Time: {quantum_time:.4f}s")


def hough_comparison(image: np.ndarray):
    """Classical standard for line/crack detection."""
    start = time.time()
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=10,
                             minLineLength=5, maxLineGap=2)
    end = time.time()
    return end - start, lines