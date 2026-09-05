import numpy as np

def run_fusion(optical_arr, sar_arr):
    if optical_arr.shape[0] < 4:
        blue = optical_arr[0].astype(float)
        optical_water_mask = (blue > np.percentile(blue, 70)).astype(np.uint8)
    else:
        green = optical_arr[1].astype(float)
        nir = optical_arr[3].astype(float)
        ndwi = (green - nir) / (green + nir + 1e-6)
        optical_water_mask = (ndwi > 0.1).astype(np.uint8)
    
    sar_band = sar_arr[0].astype(float)
    sar_water_mask = (sar_band < np.percentile(sar_band, 15)).astype(np.uint8)
    
    if optical_water_mask.shape != sar_water_mask.shape:
        import cv2
        sar_water_mask = cv2.resize(sar_water_mask, (optical_water_mask.shape[1], optical_water_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    fused_mask = np.logical_or(optical_water_mask, sar_water_mask).astype(np.uint8)
    
    stats = {
        "fusion_method": "NDWI (Optical) OR SAR_Low_Backscatter (Union)",
        "fused_water_pixels": int(np.sum(fused_mask))
    }
    
    return fused_mask, stats
