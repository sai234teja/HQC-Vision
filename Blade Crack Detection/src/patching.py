import numpy as np
from typing import List, Tuple


class PatchExtractor:
    def __init__(self, patch_size: int, stride: int):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """
        Slices the image into patches.
        Returns: List of (patch_array, top_y, left_x)
        """
        patches = []
        h, w = image.shape

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y: y + self.patch_size, x: x + self.patch_size]
                patches.append((patch, y, x))

        return patches