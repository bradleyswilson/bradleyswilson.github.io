---
layout: post
title:  "Wet Bulb Globe Temperature forecasts with Dynamical.org"
date:   2025-05-26 12:00:00 -0500
categories: jekyll update
---

Ever since seeing the announcment for non-demo releases from [dynamical.org][dynamical-hp] a few months ago, I've been eager to experiment with the data. Dynamical's mission is to:
`advance humanity’s ability to access, understand, and act on accurate weather and climate data"`

In practice, this means hosting open-source Zarr-formatted datasets with standardized structure to make weather data easier to work with. Go read more about the [project][dynamical-about]]!

After looking through the variable list in the NOAA GEFS forecast, I decided that a Wet Bulb Globe Temperatures (WBGT) conversion would be a perfect experiment since all the vairables required by the "gold standard" Liljegren approximation were availble. WBGT is a heat stress metric that takes into account temperature, humidity, wind speed, and solar radiation to provide a more robust estimate of human heat stress compared to more common metrics like the heat index. 

For this simple project, my goal was to grab data for a seven day forecast for Arkansas (my home state) from the GEFS data and produce an quick animation showing the WBGT values over time. To calculate WBGT, I used a python implementation of Liljegren's original C scripts by [Kong and Huber (2022)][konghuber], available on Github [here][pywbgt]. This repo also includes a helpful notebook with a worked CMIP6 example that I've modified to work with the GEFS data. 

{% highlight python %}
import xarray as xr
import numpy as np
from datetime import date
import dask.array as da
import geopandas as gpd
from coszenith import coszda, cosza
from WBGT import WBGT_Liljegren, fdir

today = date.today()         
gdf = gpd.read_file('tlgpkg_2024_a_05_ar.gpkg')  
lon_min, lat_min, lon_max, lat_max = gdf.total_bounds

variables = [
    "temperature_2m",  
    "relative_humidity_2m", 
    "wind_u_10m",
    "wind_v_10m",
    "pressure_surface",
    "downward_short_wave_radiation_flux_surface"
  ]

ds = xr.open_zarr("https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=b.steven.wilson@gmail.com")

subset = (
    ds[variables]
    .sel(init_time=f"{today}T00", method='nearest')
    .sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    .isel(lead_time=slice(0, 168)) # 7 days, 3 hour intervals
    .mean('ensemble_member')
  ).chunk({'lead_time':8})

subset['sfcwind'] = np.sqrt(subset['wind_u_10m'] ** 2 + subset['wind_v_10m'] ** 2) # combine wind speed

# get cosine zenith angles and daylight cosine zenith angles
date = xr.DataArray(subset.valid_time.values,
                    dims=('lead_time'), 
                    coords={'lead_time': subset.lead_time}
                    ).chunk({'lead_time': 8})

cza = da.map_blocks(cosza_wrapper, date.data, lat, lon, 3,
                    chunks=(8, lat.shape[0], lat.shape[1]),
                    new_axis=[1,2],
                    dtype=np.float64)
czda = da.map_blocks(coszda_wrapper, date.data, lat, lon, 3,
                     chunks=(8, lat.shape[0], lat.shape[1]),
                     new_axis=[1,2],
                     dtype=np.float64)
                    
cza=xr.DataArray(cza, dims=subset.dims, coords=subset.coords) 
czda=xr.DataArray(czda, dims=subset.dims, coords=subset.coords)
czda=xr.where(czda<=0,-0.5,czda) # avoid divide by zero

# calculate ratio of direct solar radiation
f = xr.apply_ufunc(
    fdir_wrapper,
    cza, czda, subset['downward_short_wave_radiation_flux_surface'], date,
    dask="parallelized", 
    output_dtypes=[float]
  )

f=xr.where(cza<=np.cos(85/180*np.pi),0,f) 
f=xr.where(f>0.9,0.9,f)
f=xr.where(f<0,0,f)
f=xr.where(subset['downward_short_wave_radiation_flux_surface']<=0,0,f)

# Calculate WBGT using Liljegren's original formulation
wbgt_liljegren = xr.apply_ufunc(WBGT_Liljegren_wrapper,
    subset['temperature_2m'] + 273.15,
    subset['relative_humidity_2m'],
    subset['pressure_surface'],
    subset['sfcwind'],
    subset['downward_short_wave_radiation_flux_surface'],
    f,
    czda,
    True,
    dask="parallelized",
    output_dtypes=[float])

subset["wbgt"] = wbgt_liljegren - 273.15
{% endhighlight %}

[dynamical-hp]: https://dynamical.org/
[dynamical-about]: https://dynamical.org/about/
[pywbgt]: https://github.com/QINQINKONG/PyWBGT
[konghuber]: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EF002334
