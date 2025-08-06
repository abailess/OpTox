import earthaccess
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
from matplotlib import colors
from scipy.integrate import trapezoid

auth = earthaccess.login()
# are we authenticated?
if not auth.authenticated:
    # ask for credentials and persist them in a .netrc file
    auth.login(strategy="interactive", persist=True)

bbox = (-82.85257, 23.847017, -81.68223, 28.215572)

import xarray as xr
results = earthaccess.search_data(
    short_name = "PACE_OCI_L3M_RRS",
    temporal = ("2024-08-15", "2025-01-15"),
    granule_name="*.DAY.*.4km.*"
)
fileset = earthaccess.open(results);

results2 = earthaccess.search_data(
    short_name = "PACE_OCI_L3M_AVW",
    temporal = ("2024-08-15", "2024-08-16"),
    granule_name="*.DAY.*.4km.*"
)
fileset2 = earthaccess.open(results2);

def time_from_attr(ds):
    """Set the time attribute as a dataset variable
    Args:
        ds: a dataset corresponding to one or multiple Level-2 granules
    Returns:
        the dataset with a scalar "time" coordinate
    """
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["date"] = ((), np.datetime64(datetime, "ns"))
    ds = ds.set_coords("date")
    return ds

# Load relevant datasets
dataset = xr.open_mfdataset(fileset, combine="nested", concat_dim="date", preprocess=time_from_attr)
dataset2 = xr.open_mfdataset(fileset2, combine="nested", concat_dim="date", preprocess=time_from_attr)

# Assign core variables
latitude = dataset["lat"]
longitude = dataset["lon"]
Rrs = dataset["Rrs"]
wavelengths = dataset["wavelength"]
avw = dataset2["avw"]

