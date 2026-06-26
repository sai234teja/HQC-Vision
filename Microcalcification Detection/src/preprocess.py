import cv2
import numpy as np
from skimage import exposure

def preprocess_roi(roi):
    # CLAHE Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(roi)
    
    # Denoising
    denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
    
    # Adaptive Thresholding for candidate identification
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological Opening to remove tiny noise
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return enhanced, opening