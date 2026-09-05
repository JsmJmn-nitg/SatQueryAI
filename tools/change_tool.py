import numpy as np

def run_change_detection(arr1, arr2):
    diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
    mean_diff = np.mean(diff, axis=0) 
    
    threshold = np.percentile(mean_diff, 92) # Top 8% difference = change
    binary_mask = (mean_diff > threshold).astype(np.uint8)
    
    changed_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    change_pct = round((changed_pixels / total_pixels) * 100, 2)
    
    stats = {
        'changed_pixels': int(changed_pixels),
        'change_percentage': change_pct,
        'method': 'Absolute Difference (92nd percentile)'
    }
    
    return binary_mask, change_pct, stats
