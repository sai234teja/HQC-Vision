import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    @staticmethod
    def enhance(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Stronger denoising — larger kernel + non-zero sigma suppresses
        # background Gaussian noise that was generating false edges
        denoised = cv2.GaussianBlur(image, (7, 7), 1.5)  # was (5,5), 0

        # Contrast Enhancement (CLAHE) — unchanged, works well
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    @staticmethod
    def detect_edges(image: np.ndarray, method: str = 'canny') -> np.ndarray:
        if method == 'canny':
            # Raised thresholds to suppress weak noise edges
            # low=80  (was 50): ignores faint noise-driven gradients
            # high=200 (was 150): only strong, confident edges pass
            return cv2.Canny(image, 80, 200)
        return cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)