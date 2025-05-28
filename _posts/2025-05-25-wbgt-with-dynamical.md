---
layout: post
title:  "Wet Bulb Globe Temperature forecasts with Dynamical.org"
date:   2025-05-26 12:00:00 -0500
categories: jekyll update
---

Ever since seeing the announcement for non-demo releases from [dynamical.org][dynamical-hp] a few months ago, I've been eager to experiment with the data. Dynamical's mission is to:

> advance humanity’s ability to access, understand, and act on accurate weather and climate data

In practice, this means hosting open-source Zarr-formatted datasets with standardized structure to make weather data easier to work with. Go read more about the [project][dynamical-about]! 

As someone who works extensively with climate data but hasn't dealt specifically with this type of operational forecast data, this was the perfect jumping in point. After a quick look through the variable list in the NOAA GEFS forecast, I decided that a Wet Bulb Globe Temperatures (WBGT) conversion would be a perfect experiment since all the vairables required by the "gold standard" Liljegren approximation were availble. WBGT is a heat stress metric that considers temperature, humidity, wind speed, and solar radiation to provide a more robust estimate of human heat stress compared to more common metrics like the heat index. 

For this simple project, my goal was to grab data for a seven-day forecast for Arkansas (my home state) from the GEFS data and estimate the WBGT values over time. To calculate WBGT, I used a python implementation of Liljegren's original C scripts by [Kong and Huber (2022)][konghuber], available on Github [here][pywbgt]. This repo also includes a helpful notebook with a worked CMIP6 example that I've modified here to work with the GEFS data hosted on dynamical. I've stripped the dask setup from the original tutorial, but a proper dask implementation would be desirable for scaling a larger subset of data. 


{% highlight python %}
iimport xarray as xr
import numpy as np
from datetime import date
import geopandas as gpd
from coszenith import coszda, cosza
from WBGT import WBGT_Liljegren, fdir

today = date.today()                  
gdf = gpd.read_file('tlgpkg_2024_a_05_ar.gpkg', layer='county')  
lon_min, lat_min, lon_max, lat_max = gdf.total_bounds

variables = [
    "temperature_2m",  # Air temperature at 2 m
    "relative_humidity_2m",  # Specific humidity at 2 m
    "wind_u_10m",
    "wind_v_10m",
    "pressure_surface",
    "downward_short_wave_radiation_flux_surface"
]

ds = xr.open_zarr("https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?")

subset = (
    ds[variables]
    .sel(init_time=f"{today}T00", method='nearest')
    .sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    .isel(
        lead_time=slice(0, 56), # grab 7 days worth
        ensemble_member=0)
)

subset['sfcwind'] = np.sqrt(subset['wind_u_10m'] ** 2 + subset['wind_v_10m'] ** 2)
subset['temperature_2m'] = subset['temperature_2m'] + 273.15 # K to C

# get lat/grid in radians
lon,lat=np.meshgrid(subset.longitude,subset.latitude)
lat=lat*np.pi/180
lon=lon*np.pi/180

# get cosine zenith angles and daylight cosize zenith angles
date = xr.DataArray(subset.valid_time.values, dims=('lead_time'), 
                    coords={'lead_time': subset.lead_time})

cza = cosza(date.data, lat, lon, 3)
czda = coszda(date.data, lat, lon, 3)
                    
cza=xr.DataArray(cza,dims=subset.dims, coords=subset.coords) 
czda=xr.DataArray(czda,dims=subset.dims, coords=subset.coords)
czda=xr.where(czda<=0,-0.5,czda)

# fraction of total solar radiation
f = fdir(cza.values,czda.values, subset['downward_short_wave_radiation_flux_surface'].values, date.data)

f=xr.where(cza<=np.cos(85/180*np.pi),0,f) 
f=xr.where(f>0.9,0.9,f)
f=xr.where(f<0,0,f)
f=xr.where(subset['downward_short_wave_radiation_flux_surface']<=0,0,f)

# calculate WBGT
wbgt_liljegren = WBGT_Liljegren(
    subset['temperature_2m'].values,
    subset['relative_humidity_2m'].values,
    subset['pressure_surface'].values,
    subset['sfcwind'].values,
    subset['downward_short_wave_radiation_flux_surface'].values,
    f.values,
    czda.values,
    True)

wbgt = xr.DataArray(wbgt_liljegren- 273.15,dims=subset.dims, coords=subset.coords)
{% endhighlight %}

Simple enough! This left me with a xarray data array that can be plotted or animated as desired. Huge kudos to the dynamical team, this is an awesome resource to be able to access so easily. 

![WBGT](posts/Images/wbgt.png)


[dynamical-hp]: https://dynamical.org/
[dynamical-about]: https://dynamical.org/about/
[pywbgt]: https://github.com/QINQINKONG/PyWBGT
[konghuber]: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EF002334
