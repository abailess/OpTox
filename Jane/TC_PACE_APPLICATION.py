
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import xarray as xr
import pandas as pd
import numpy as np

# === Paths ===
nc_path = r"/Volumes/rsp/Agriculture/28566197/images/reprojected/PACE_OCI.20250719T105436.L2.OC_AOP.V3_0.NRT.nc_reprojected.nc"
csv_path = r"/Volumes/rsp/Agriculture/28566197/output/tasseled_cap_PACE.csv"
shapefile_path = r'/Volumes/rsp/Agriculture/28566197/images/roi/balthicsea.shp'
DEST = r'/Volumes/rsp/Agriculture/28566197/images/PACE_OCI.20250719T105436.L2.OC_AOP.V3_0.NRT'

# === Load NetCDF and TC Coefficients ===
ds = xr.open_dataset(nc_path)
tcc_ref = pd.read_csv(csv_path)

# === Build Rrs band list ===
rrs_band_names = [str(wl) for wl in tcc_ref['Wavelength']]
rrs_bands = ['Rrs_' + wl for wl in rrs_band_names]
available_bands = [b for b in rrs_bands if b in ds.data_vars]

# === Stack bands into (lat, lon, band) ===
stacked = xr.concat([ds[band] for band in available_bands], dim="band").transpose("lat", "lon", "band")
stacked = stacked.where(stacked != -9999, np.nan)

# === Mask valid pixels ===
valid_mask = ~np.isnan(stacked).any(dim="band")
pixel_data = stacked.values[valid_mask.values]  # shape: (N, bands)

# === Apply TC Coefficients ===
brightness_coeffs = tcc_ref['Brightness'].values
greenness_coeffs = tcc_ref['Greenness'].values
wetness_coeffs = tcc_ref['Wetness'].values

brightness = np.dot(pixel_data, brightness_coeffs)
greenness = np.dot(pixel_data, greenness_coeffs)
wetness = np.dot(pixel_data, wetness_coeffs)

# === Create full 2D images ===
b_img = np.full(valid_mask.shape, np.nan)
g_img = np.full(valid_mask.shape, np.nan)
w_img = np.full(valid_mask.shape, np.nan)

# ==== AWV ========
numerator = np.sum(pixel_data, axis=1)
denominator = np.sum(pixel_data / np.array([int(i[4:]) for i in available_bands]), axis=1)
avw = numerator / denominator
avw_img = np.full(valid_mask.shape, np.nan)

valid_indices = np.argwhere(valid_mask.values)
for idx, (i, j) in enumerate(valid_indices):
    b_img[i, j] = brightness[idx]
    g_img[i, j] = greenness[idx]
    w_img[i, j] = wetness[idx]
    avw_img[i, j] = avw[idx]

# =========== RGB =======================
red_b = ds['Rrs_665'].values
green_b = ds['Rrs_560'].values
blue_b = ds['Rrs_490'].values

# === Optionally: print stats ===
# print("Brightness:", np.nanmin(b_img), "to", np.nanmax(b_img))
# print("Greenness:", np.nanmin(g_img), "to", np.nanmax(g_img))
# print("Wetness:", np.nanmin(w_img), "to", np.nanmax(w_img))

################### Export as TIFF:
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Get lat/lon for georeferencing
lat = ds['lat'].values
lon = ds['lon'].values

# Assumes 2D lat/lon or creates mesh
if lat.ndim == 1 and lon.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat)
else:
    lon2d, lat2d = lon, lat

# Define GeoTIFF spatial metadata
height, width = b_img.shape
transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), width, height)
crs = CRS.from_epsg(4326)  # You can adjust if using a different CRS

# Output folder
os.makedirs(DEST, exist_ok=True)

# Helper to write one band
def write_geotiff(filename, data, transform, crs):
    data = np.nan_to_num(data, nan=-9999)
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data.astype(rasterio.float32), 1)

# Write each component
write_geotiff(os.path.join(DEST, "Brightness.tif"), b_img, transform, crs)
write_geotiff(os.path.join(DEST, "Greenness.tif"), g_img, transform, crs)
write_geotiff(os.path.join(DEST, "Wetness.tif"), w_img, transform, crs)
write_geotiff(os.path.join(DEST, "AWV.tif"), avw_img, transform, crs)

############## Plot the images based on ROI:
# Gets lat/long:
lat = ds['lat'].values
lon = ds['lon'].values

if lat.ndim == 1 and lon.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat)
else:
    lon2d, lat2d = lon, lat

# Loop through each component image (brightness, greenness, wetness)
for name, comp_image in zip(["Brightness", "Greenness", "Wetness", "AWV"], [b_img, g_img, w_img, avw_img]):
    assert comp_image.shape == lat2d.shape == lon2d.shape

    # Load ROI
    gdf = gpd.read_file(shapefile_path)

    # Create transform for masking
    height, width = comp_image.shape
    transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), width, height)

    # Mask to ROI
    mask = geometry_mask(
        geometries=gdf.geometry,
        out_shape=(height, width),
        transform=transform,
        invert=True
    )

    masked_image = np.where(mask, comp_image, np.nan)

    if name == "AWV":
        masked_image = np.where((masked_image > 400) |
                                (masked_image < 700), masked_image, np.nan)

    masked_plot = np.ma.masked_invalid(masked_image)

    # Set color map and normalization (continuous)
    cmap = cm.get_cmap('jet')  # You can try: 'plasma', 'inferno', 'jet', etc.

    if name == "AWV":
        norm = Normalize(vmin=400, vmax=700)  # or 750 if your data extends slightly higher
    else:
        norm = Normalize(vmin=np.nanmin(masked_plot), vmax=np.nanmax(masked_plot))

    # Plot
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_facecolor("black")

    c = ax.pcolormesh(lon2d, lat2d, masked_plot, cmap=cmap, norm=norm, shading='auto')
    cb = fig.colorbar(c, ax=ax, orientation='vertical', shrink=0.75, pad=0.03)
    cb.set_label(f"{name}", fontsize=8)

    # Zoom to ROI
    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    # Save
    plt.savefig(f"{DEST}/{name}.png", dpi=600, bbox_inches='tight')
    plt.show()

# === Reconstruct full image arrays ===
shape = valid_mask.shape

# === Helper: Percentile normalization ===
import numpy as np

def normalize_tc_component(tc_array, lower_percentile=5, upper_percentile=95):
    # Remove NaNs for percentile calculation
    valid = tc_array[~np.isnan(tc_array)]
    if valid.size == 0:
        return np.zeros_like(tc_array)

    min_val = np.percentile(valid, lower_percentile)
    max_val = np.percentile(valid, upper_percentile)

    # Avoid divide-by-zero if flat
    if max_val == min_val:
        return np.zeros_like(tc_array)

    clipped = np.clip(tc_array, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)

    # Retain NaNs in original positions
    normalized[np.isnan(tc_array)] = np.nan
    return normalized

# === Normalize each component ===
b_norm = normalize_tc_component(b_img)
g_norm = normalize_tc_component(g_img)
w_norm =normalize_tc_component(w_img)

# === Stack into RGB ===
rgb_image = np.dstack([b_norm, g_norm, w_norm])

# ======== True color ======
# red_norm_true = normalize_tc_component(red_b, lower_percentile=1, upper_percentile=99)
# green_norm_true = normalize_tc_component(green_b, lower_percentile=1, upper_percentile=99)
# blue_norm_true =normalize_tc_component(blue_b, lower_percentile=1, upper_percentile=99)

def clip_and_normalize(arr, min_val=0, max_val=0.05):
    arr = np.clip(arr, min_val, max_val)
    norm = (arr - min_val) / (max_val - min_val)
    norm[np.isnan(arr)] = np.nan
    return norm

red_norm_true   = clip_and_normalize(red_b)
green_norm_true= clip_and_normalize(green_b)
blue_norm_true = clip_and_normalize(blue_b)

rdb_true = np.dstack([red_norm_true, green_norm_true, blue_norm_true])
rdb_true = np.clip(rdb_true, 0, 1)
rdb_true = np.power(rdb_true, 1/2)  # simulate human perception

# === Get lat/lon ===
lat = ds['lat'].values
lon = ds['lon'].values
if lat.ndim == 1 and lon.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat)
else:
    lon2d, lat2d = lon, lat

# === Load ROI shapefile and create mask ===
gdf = gpd.read_file(shapefile_path)
height, width = shape
transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), width, height)
mask = geometry_mask(
    geometries=gdf.geometry,
    out_shape=(height, width),
    transform=transform,
    invert=True
)

# === Apply mask to RGB ===
rgb_masked = np.where(mask[:, :, None], rgb_image, np.nan)
rgb_true_masked = np.where(mask[:, :, None], rdb_true, np.nan)

# === Plot RGB ===
for name, img in zip(['RGB_TC', 'RGB'], [rgb_masked, rgb_true_masked]):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(5.1, 3))
    ax.set_facecolor("black")

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(np.clip(img, 0, 1), extent=extent, origin='upper')

    # Zoom to ROI
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # ax.set_title("Tasseled Cap RGB Composite\nR=Brightness, G=Greenness, B=Wetness", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{DEST}/{name}.png", dpi=600, bbox_inches='tight')
    plt.show()
