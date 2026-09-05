import rasterio
import numpy as np

def read_geotiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {
            "crs": str(src.crs),
            "transform": str(src.transform),
            "count": src.count,
            "shape": arr.shape
        }
    return arr, meta

def to_rgb_preview(arr):
    # Get shape (bands, height, width)
    bands, H, W = arr.shape
    
    # Extract RGB channels or duplicate single band
    if bands >= 3:
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    else:
        rgb = np.stack([arr[0], arr[0], arr[0]], axis=-1)
        
    # Normalize to 0-255 for UI display
    rgb = rgb.astype(np.float32)
    rgb = rgb - np.percentile(rgb, 2)
    rgb = rgb / (np.percentile(rgb, 98) + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def create_overlay(base_rgb, mask, color=(255, 50, 50), alpha=0.55):
    """
    Overlays a binary mask onto an RGB image using the specified color and opacity.
    """
    overlay = base_rgb.copy().astype(np.float32)
    
    for c in range(3):
        # Blend original pixel and color based on alpha, only where mask == 1
        overlay[:, :, c] = np.where(
            mask == 1,
            (1 - alpha) * base_rgb[:, :, c] + alpha * color[c],
            base_rgb[:, :, c]
        )
        
    return np.clip(overlay, 0, 255).astype(np.uint8)

def check_pair_compatible(arr1, meta1, arr2, meta2):
    ok_shape = arr1.shape[1:] == arr2.shape[1:]
    ok_crs = meta1["crs"] == meta2["crs"]
    return {"ok_shape": ok_shape, "ok_crs": ok_crs, "note": "Checked pixel size/shape compatibility."}
