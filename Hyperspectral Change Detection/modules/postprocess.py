import numpy as np
from scipy import ndimage


def counts_to_image(counts, mapping, image_shape):
    """
    Converts quantum measurement counts back to a 2D spatial map.

    Parameters:
        counts (dict): Bitstring -> count from Grover measurement.
        mapping (dict): Integer index -> (row, col) coordinate.
        image_shape (tuple): Shape of the output 2D array (H, W).

    Returns:
        np.ndarray: Float array of shape image_shape with intensity in [0, 1].
    """
    final_map = np.zeros(image_shape)
    if not counts:
        return final_map

    max_count = max(counts.values())
    for state_str, count in counts.items():
        # Convert bitstring back to integer index
        idx = int(state_str, 2)
        if idx in mapping:
            coord = mapping[idx]
            # Intensity proportional to measurement frequency
            final_map[coord] = count / max_count

    return final_map


def refine_map(binary_map):
    """
    Applies morphological operations to clean noise from a binary change map.
    - Opening removes small isolated white pixels (noise).
    - Closing fills small holes inside change regions.
    """
    refined = ndimage.binary_opening(binary_map, structure=np.ones((2, 2)))
    refined = ndimage.binary_closing(refined, structure=np.ones((2, 2)))
    return refined