import numpy as np

def run_fusion(optical_arr, sar_arr):
    # Assuming optical_arr is Multispectral (e.g., Green is band 1, NIR is band 3)
    # 1. Optical Water Mask (NDWI)
    # Formula: (Green - NIR) / (Green + NIR)
    
    # *Note: Add a small epsilon (1e-6) to avoid divide-by-zero errors*
    green = optical_arr[1].astype(float)
    nir = optical_arr[3].astype(float)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    optical_water_mask = (ndwi > 0.1).astype(np.uint8)
    
    # 2. SAR Water Mask
    # Water acts like a mirror to radar, scattering it away. So water is very DARK in SAR.
    # We threshold the low intensities.
    sar_band = sar_arr[0].astype(float)
    sar_water_mask = (sar_band < np.percentile(sar_band, 15)).astype(np.uint8)
    
    # 3. Fuse them (Union: if either thinks it's water, mark it as water)
    fused_mask = np.logical_or(optical_water_mask, sar_water_mask).astype(np.uint8)
    
    return fused_mask, {"fusion_method": "NDWI OR SAR_Low_Backscatter"}
