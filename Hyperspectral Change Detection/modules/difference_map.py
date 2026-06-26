import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_difference(t1, t2):
    """Computes spectral Euclidean distance."""
    diff = np.sqrt(np.sum((t1 - t2)**2, axis=2))
    # Normalize diff to 0-1
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    return diff

def get_candidate_mask(diff, percentile=95):
    threshold = np.percentile(diff, percentile)
    return diff >= threshold