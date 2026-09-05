import numpy as np

def run_fusion(optical_arr, sar_arr):
    """
    Fuses optical (multispectral) and SAR data to detect water bodies.
    
    optical_arr: assumed to have bands [Blue, Green, Red, NIR, ...] (Sentinel-2 style)
    sar_arr: SAR backscatter intensity
    
    Returns: fused water mask, stats dict
    """
    # Ensure we have enough bands
    if optical_arr.shape[0] < 4:
        # Fallback: if we don't have NIR, use simple blue channel threshold
        print("Warning: Optical image has fewer than 4 bands. Using simplified detection.")
        blue = optical_arr[0].astype(float)
        optical_water_mask = (blue > np.percentile(blue, 70)).astype(np.uint8)
    else:
        # Proper NDWI: (Green - NIR) / (Green + NIR)
        # Assuming band order: [Blue=0, Green=1, Red=2, NIR=3]
        green = optical_arr[1].astype(float)
        nir = optical_arr[3].astype(float)
        ndwi = (green - nir) / (green + nir + 1e-6)
        optical_water_mask = (ndwi > 0.1).astype(np.uint8)
    
    # SAR Water Mask
    # Water acts like a mirror to radar, scattering it away. So water is very DARK in SAR.
    # We threshold the low intensities.
    sar_band = sar_arr[0].astype(float)
    sar_water_mask = (sar_band < np.percentile(sar_band, 15)).astype(np.uint8)
    
    # Resize masks if they don't match (shouldn't happen if images are co-registered)
    if optical_water_mask.shape != sar_water_mask.shape:
        import cv2
        sar_water_mask = cv2.resize(sar_water_mask, 
                                     (optical_water_mask.shape[1], optical_water_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    
    # Fuse them (Union: if either thinks it's water, mark it as water)
    fused_mask = np.logical_or(optical_water_mask, sar_water_mask).astype(np.uint8)
    
    stats = {
        "fusion_method": "NDWI (Optical) OR SAR_Low_Backscatter (Union)",
        "optical_water_pixels": int(np.sum(optical_water_mask)),
        "sar_water_pixels": int(np.sum(sar_water_mask)),
        "fused_water_pixels": int(np.sum(fused_mask)),
        "note": "Union operator captures water detected by either sensor"
    }
    
    return fused_mask, stats
