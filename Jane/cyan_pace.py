# -- Python - jane - pace-hackweek - aug 4, 2025

import os
import numpy as np
import pandas as pd
from xarray.backends.api import open_datatree
import matplotlib.pyplot as plt

import earthaccess


class ExtractPace:
    def __init__(self, start, end, target_lat, target_lon, dest, plot=None):
        self.start = start
        self.end = end
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.dest = dest
        self.plot = plot

    def run(self):

        # Authentication:
        auth = earthaccess.login(persist=True)

        # Search by PACE Rrs:
        tspan = (self.start, self.end)

        buffer_deg = 0.05 # ~5-km

        # Create bounding box (lon_min, lat_min, lon_max, lat_max)
        bbox = (
            self.target_lon - buffer_deg,
            self.target_lat  - buffer_deg,
            self.target_lon + buffer_deg,
            self.target_lat + buffer_deg
        )

        results = earthaccess.search_data(
            short_name="PACE_OCI_L2_AOP",
            temporal=tspan,
            bounding_box=bbox,
            cloud_cover=(0, 100),
        )

        if not results:
            raise ValueError("No granules found for this search.")

        # Load the images:
        paths = earthaccess.open(results)
        datatree = open_datatree(paths[0])

        # Access the variables -> Rrs and lat/long:
        id_image = str(paths[0]).split(',')[1].split('/')[-1][:-1]
        lat = datatree["/navigation_data"]["latitude"]
        lon = datatree["/navigation_data"]["longitude"]
        rhos = datatree["/geophysical_data"]["Rrs"]
        urhos = datatree["/geophysical_data"]["Rrs_unc"]
        wavelengths = datatree["/sensor_band_parameters"]["wavelength_3d"]

        # Find the pixel correspondent to lat/long:
        def find_nearest_pixel(lat_array, lon_array, target_lat, target_lon):
            dist = np.hypot(lat_array - target_lat, lon_array - target_lon)
            flat_index = np.argmin(dist.values)  # Ensure it's a NumPy array
            return np.unravel_index(flat_index, dist.shape)

        yidx, xidx = find_nearest_pixel(lat, lon, self.target_lat, self.target_lon)

        # Extract the spectra:
        rhos_pixel = rhos.isel(number_of_lines=yidx, pixels_per_line=xidx)
        urhos_pixel = urhos.isel(number_of_lines=yidx, pixels_per_line=xidx)

        df = pd.DataFrame({'date': self.start, 'lat': self.target_lat, 'lon': self.target_lon,
                           'wavelength': wavelengths.values, 'rrs': rhos_pixel.values, 'u_rrs': urhos_pixel.values})
        df.to_csv(os.path.join(self.dest, id_image + str(self.target_lat) + '_' + str(self.target_lon) + '.csv'))

        if self.plot:
            path_plot = os.path.join(self.dest, 'plots')
            os.makedirs(path_plot, exist_ok=True)
            plt.figure(figsize=(5, 5))
            plt.plot(wavelength_values, rhos_pixel_values, marker='o', linestyle='-')
            plt.title(f"Rrs\nLat: {self.target_lat}, Lon: {self.target_lon}", fontsize=12)
            plt.xlabel("Wavelength (nm)", fontsize=10)
            plt.ylabel("Reflectance (sr⁻¹)", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(path_plot, id_image + str(self.target_lat) + '_' + str(self.target_lon) + '.png'))
            # plt.show()



# Example to use it!
# import time
#
# start = "2024-04-17"
# end ="2024-04-17"
# target_lat = -45.9341
# target_lon = -65.2864
#
# start_time = time.time()
# a = ExtractPace(start, end, target_lat, target_lon,
#                 dest="/Users/rsp/Documents/PACEHACKWEEK/project/dataset/roi", plot=True)
#
# a.run()
# end_time = time.time()
# print(end_time - start_time, ' seconds')
