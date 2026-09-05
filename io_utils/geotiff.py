import rasterio
import numpy as np

def read_geotiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {"crs": str(src.crs), "transform": str(src.transform), "count": src.count, "shape": arr.shape}
    return arr, meta

def to_rgb_preview(arr):
    # (Insert the array-to-RGB code from Page 4 here)
    pass
