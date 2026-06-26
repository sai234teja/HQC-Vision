import cv2
import numpy as np
from config import Config

class HyperspectralLoader:
    @staticmethod
    def load_data(path1, path2):
        """
        Loads two images from disk, converts BGR -> RGB, resizes to
        Config.IMAGE_SHAPE, and normalizes to [0, 1] float32.
        """
        t1 = cv2.imread(path1)
        t2 = cv2.imread(path2)

        if t1 is None:
            raise FileNotFoundError(f"Could not load image: {path1}")
        if t2 is None:
            raise FileNotFoundError(f"Could not load image: {path2}")

        # Convert BGR (OpenCV default) to RGB
        t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2RGB)
        t2 = cv2.cvtColor(t2, cv2.COLOR_BGR2RGB)

        # Resize to target shape
        t1 = cv2.resize(t1, Config.IMAGE_SHAPE)
        t2 = cv2.resize(t2, Config.IMAGE_SHAPE)

        # Normalize to [0, 1]
        return t1.astype(np.float32) / 255.0, t2.astype(np.float32) / 255.0