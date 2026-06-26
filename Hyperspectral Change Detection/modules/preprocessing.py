import cv2
import numpy as np


def preprocess_cubes(t1_rgb, t2_rgb):
    """
    Main entry point called by main.py.
    Pipeline:
      1. Per-channel histogram equalization (handles local contrast)
      2. Gaussian blur (removes sensor noise)
      3. Global mean-std normalization of T2 to match T1's statistics
         (removes broad lighting/seasonal shifts between acquisitions)
    Returns float32 arrays of shape (H, W, C) in [0, 1].
    """
    t1_out = _process_single(t1_rgb)
    t2_out = _process_single(t2_rgb)

    # Match T2's global illumination statistics to T1 channel-by-channel
    t2_out = _match_illumination(t1_out, t2_out)

    return t1_out, t2_out


def _process_single(img_rgb):
    """
    Per-channel histogram equalization + Gaussian blur.
    Input:  float32 (H, W, C) in [0, 1]
    Output: float32 (H, W, C) in [0, 1]
    """
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    channels = []
    for c in range(img_uint8.shape[2]):
        ch = img_uint8[:, :, c]
        ch = cv2.equalizeHist(ch)
        ch = cv2.GaussianBlur(ch, (5, 5), 0)
        channels.append(ch)
    result = np.stack(channels, axis=2)
    return result.astype(np.float32) / 255.0


def _match_illumination(reference, target):
    """
    Shifts each channel of `target` so its mean and std match `reference`.
    This corrects for global illumination/seasonal differences between
    T1 and T2 without distorting local change signals.

    Input/Output: float32 (H, W, C) in [0, 1]
    """
    out = np.empty_like(target)
    for c in range(reference.shape[2]):
        ref_ch = reference[:, :, c]
        tgt_ch = target[:, :, c]

        ref_mean, ref_std = ref_ch.mean(), ref_ch.std() + 1e-8
        tgt_mean, tgt_std = tgt_ch.mean(), tgt_ch.std() + 1e-8

        # Scale and shift target channel to match reference statistics
        normalized = (tgt_ch - tgt_mean) / tgt_std * ref_std + ref_mean
        out[:, :, c] = np.clip(normalized, 0.0, 1.0)

    return out


def to_grayscale_and_norm(t1_rgb, t2_rgb):
    """
    Legacy helper: converts RGB images to grayscale with equalization.
    Returns 2D float32 arrays — not suitable for PCA directly.
    """
    t1_gray = cv2.cvtColor((t1_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    t2_gray = cv2.cvtColor((t2_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    t1_eq = cv2.equalizeHist(t1_gray)
    t2_eq = cv2.equalizeHist(t2_gray)

    t1_final = cv2.GaussianBlur(t1_eq, (5, 5), 0)
    t2_final = cv2.GaussianBlur(t2_eq, (5, 5), 0)

    return t1_final.astype(np.float32) / 255.0, t2_final.astype(np.float32) / 255.0