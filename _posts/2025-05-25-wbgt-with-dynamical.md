---
layout: post
title:  "Wet Bulb Globe Temperature Forecasts with Dynamical.org"
date:   2025-05-26 12:00:00 -0500
categories: jekyll update
---



{% highlight python %}
variables = [
    "temperature_2m",  # Air temperature at 2 m
    "relative_humidity_2m",  # Specific humidity at 2 m
    "wind_u_10m",
    "wind_v_10m",
    "pressure_surface",
    "downward_short_wave_radiation_flux_surface"
]

ds = xr.open_zarr("https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=b.steven.wilson@gmail.com")

subset = (ds[variables]
 .sel(init_time=f"{today}T00",
      method='nearest')
 .sel(latitude=slice(lat_max, lat_min),
      longitude=slice(lon_min, lon_max))
 ).mean('ensemble_member')
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[dynamical-hp]: https://dynamical.org/
