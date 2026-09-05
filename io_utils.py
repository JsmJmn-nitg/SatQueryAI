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
        
    # Normalize to 0-255 for Gradio UI display
    rgb = rgb.astype(np.float32)
    rgb = rgb - np.percentile(rgb, 2)
    rgb = rgb / (np.percentile(rgb, 98) + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def check_pair_compatible(arr1, meta1, arr2, meta2):
    # Ensure Image 1 and Image 2 have the same dimensions for Change Detection
    ok_shape = arr1.shape[1:] == arr2.shape[1:]
    ok_crs = meta1["crs"] == meta2["crs"]
    return {"ok_shape": ok_shape, "ok_crs": ok_crs, "note": "Checked pixel size/shape compatibility."}
