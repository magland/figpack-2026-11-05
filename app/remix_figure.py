import json
import urllib.request
import numpy as np
import figpack.views as vv

# Figure for the REMIX slide: the weather in New York City in October 2026,
# hourly, from the Open-Meteo historical weather API (ERA5 reanalysis at the
# grid point nearest Central Park). Several panels with their own colors,
# labels and legends, in the default style, give a remix plenty to change.
url = (
    "https://archive-api.open-meteo.com/v1/archive?latitude=40.78&longitude=-73.97"
    "&start_date=2026-10-01&end_date=2026-10-31&timezone=America%2FNew_York"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m,surface_pressure"
)
with urllib.request.urlopen(url) as response:
    hourly = json.load(response)["hourly"]
data = {k: np.array(v, dtype=np.float32) for k, v in hourly.items() if k != "time"}

# Only the bottom panel shows the time axis labels and the toolbar
upper = {"hide_nav_toolbar": True, "hide_time_axis_labels": True}

# Time in seconds since 1 October, 00:00 local time
t = np.arange(len(hourly["time"]), dtype=np.float32) * 3600
days = t.reshape(-1, 24)[:, 0]

temp = data["temperature_2m"].reshape(-1, 24)
temperature = vv.TimeseriesGraph(y_label="Temperature (°C)", legend_opts={"location": "northeast"}, **upper)
temperature.add_line_series(name="hourly", t=t, y=data["temperature_2m"], color="#d62728", width=1.5)
temperature.add_marker_series(name="daily high", t=days + 3600 * temp.argmax(axis=1), y=temp.max(axis=1), color="#8c1515", radius=4)
temperature.add_marker_series(name="daily low", t=days + 3600 * temp.argmin(axis=1), y=temp.min(axis=1), color="#1f77b4", radius=4)

# Hours with more than 0.1 mm of precipitation, shown as intervals
rain = data["precipitation"] > 0.1
precipitation = vv.TimeseriesGraph(y_label="mm/h", **upper)
precipitation.add_interval_series(name="rain", t_start=t[rain], t_end=t[rain] + 3600, color="#9ecae1", alpha=0.6, border_color="none")
precipitation.add_line_series(name="precipitation", t=t, y=data["precipitation"], color="#08519c", width=1.5)

wind = vv.TimeseriesGraph(y_label="Wind (km/h)", legend_opts={"location": "northeast"}, **upper)
wind.add_line_series(name="gusts", t=t, y=data["wind_gusts_10m"], color="#7f7f7f", dash=[4, 3])
wind.add_line_series(name="wind speed", t=t, y=data["wind_speed_10m"], color="#2ca02c", width=1.5)

pressure = vv.TimeseriesGraph(y_label="Pressure (hPa)", **upper)
pressure.add_line_series(name="surface pressure", t=t, y=data["surface_pressure"], color="#9467bd", width=1.5)

humidity = vv.TimeseriesGraph(y_label="Humidity (%)", y_range=[0, 100])
humidity.add_line_series(name="relative humidity", t=t, y=data["relative_humidity_2m"], color="#17becf", width=1.5)

summary = vv.Markdown(f"""
### New York City, October 2026

Hourly weather at the grid point nearest Central Park. Time is counted from 1 October, 00:00 local time.

| October | |
|---|---|
| Warmest hour | {data["temperature_2m"].max():.1f} °C |
| Coldest hour | {data["temperature_2m"].min():.1f} °C |
| Mean temperature | {data["temperature_2m"].mean():.1f} °C |
| Total precipitation | {data["precipitation"].sum():.0f} mm |
| Hours with rain | {int(rain.sum())} |
| Strongest gust | {data["wind_gusts_10m"].max():.0f} km/h |
| Pressure range | {data["surface_pressure"].min():.0f} to {data["surface_pressure"].max():.0f} hPa |

Data: [Open-Meteo](https://open-meteo.com) historical weather API (ERA5 reanalysis, CC BY 4.0).
""")

panels = vv.Box(
    direction="vertical",
    items=[
        vv.LayoutItem(temperature, title="Temperature", stretch=1.6),
        vv.LayoutItem(precipitation, title="Precipitation", stretch=1),
        vv.LayoutItem(wind, title="Wind", stretch=1),
        vv.LayoutItem(pressure, title="Surface pressure", stretch=1),
        vv.LayoutItem(humidity, title="Relative humidity", stretch=1.3),
    ],
)
view = vv.Splitter(
    direction="horizontal",
    item1=vv.LayoutItem(summary),
    item2=vv.LayoutItem(panels),
    split_pos=0.28,
)

view.show(
    title="Weather in New York City, October 2026",
    script=open(__file__).read(),
    upload=True,
)
