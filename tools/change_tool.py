import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_change_detection(arr1, arr2):
    """
    arr1, arr2: numpy arrays from your GeoTIFFs
    Returns: a binary mask (0 for no change, 1 for change)
    """
    # Hackathon Fallback: Absolute difference thresholding
    diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
    mean_diff = np.mean(diff, axis=0) # Collapse bands
    
    threshold = np.percentile(mean_diff, 90) # Top 10% of changes
    binary_mask = (mean_diff > threshold).astype(np.uint8)
    
    changed_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    change_pct = round((changed_pixels / total_pixels) * 100, 2)
    
    return binary_mask, change_pct
