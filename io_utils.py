import rasterio
import numpy as np
import tempfile
import cv2

def read_geotiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {"crs": str(src.crs), "transform": str(src.transform), "count": src.count, "shape": arr.shape}
    return arr, meta

def to_rgb_preview(arr):
    bands, H, W = arr.shape
    if bands >= 3:
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    else:
        rgb = np.stack([arr[0], arr[0], arr[0]], axis=-1)
        
    rgb = rgb.astype(np.float32)
    rgb = rgb - np.percentile(rgb, 2)
    rgb = rgb / (np.percentile(rgb, 98) + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

def save_temp_png(rgb_arr):
    """Saves numpy array to a temp PNG so models can process it easily."""
    path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
    return path

def check_pair_compatible(arr1, meta1, arr2, meta2):
    ok_shape = arr1.shape[1:] == arr2.shape[1:]
    ok_crs = meta1["crs"] == meta2["crs"]
    return {"ok_shape": ok_shape, "ok_crs": ok_crs, "note": "Checked pixel size/shape compatibility."}

def create_evidence_overlay(base_rgb_arr, mask, color=(255, 0, 0), opacity=0.5):
    if mask.shape != base_rgb_arr.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (base_rgb_arr.shape[1], base_rgb_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = base_rgb_arr.copy().astype(np.float32)
    colored_mask = np.zeros_like(base_rgb_arr, dtype=np.float32)
    colored_mask[mask == 1] = color
    
    mask_bool = mask == 1
    overlay[mask_bool] = (base_rgb_arr[mask_bool].astype(np.float32) * (1 - opacity) + colored_mask[mask_bool] * opacity)
    return np.clip(overlay, 0, 255).astype(np.uint8)
