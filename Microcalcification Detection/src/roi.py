import numpy as np

class ROIExtractor:
    def __init__(self, size=64, stride=32):
        self.size = size
        self.stride = stride

    def extract_patches(self, image):
        """Generates 64x64 patches and their top-left coordinates."""
        h, w = image.shape
        patches = []
        coords = []

        for y in range(0, h - self.size + 1, self.stride):
            for x in range(0, w - self.size + 1, self.stride):
                patch = image[y:y+self.size, x:x+self.size]
                patches.append(patch)
                coords.append((x, y))
                
        return patches, coords

    def get_lesion_center_roi(self, image, mask):
        """Finds the ground truth lesion and returns a centered ROI."""
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return None, None
            
        center_y, center_x = coords.mean(axis=0).astype(int)
        
        # Calculate bounds
        y1 = max(0, center_y - self.size // 2)
        x1 = max(0, center_x - self.size // 2)
        
        # Adjust if out of bounds
        y1 = min(y1, image.shape[0] - self.size)
        x1 = min(x1, image.shape[1] - self.size)
        
        roi = image[y1:y1+self.size, x1:x1+self.size]
        return roi, (x1, y1)