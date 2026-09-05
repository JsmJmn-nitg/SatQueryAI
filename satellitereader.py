#build the satellite 
def read_satellite_image(path):

    with rasterio.open(path) as src:

        data = src.read()

        metadata = {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "dtype": str(src.dtypes[0])
        }

    return data, metadata
  
  def create_preview(data):

    bands, height, width = data.shape

    if bands >= 3:

        rgb = np.stack(
            [data[0], data[1], data[2]],
            axis=-1
        )

    else:

        rgb = np.stack(
            [data[0], data[0], data[0]],
            axis=-1
        )

    rgb = rgb.astype(np.float32)

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    rgb = (rgb - low) / (high - low + 1e-8)

    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)
