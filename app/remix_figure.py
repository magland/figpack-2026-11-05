import json
import urllib.request
import numpy as np
import figpack.views as vv

# Figure for the REMIX slide: the weather in New York City in October 2026,
# hourly, from the Open-Meteo historical weather API (ERA5 reanalysis at the
# grid point nearest Central Park). Two panels, small enough to read in the
# preview of the remix page, with colors, labels and a legend in the default
# style for a remix to change.
url = (
    "https://archive-api.open-meteo.com/v1/archive?latitude=40.78&longitude=-73.97"
    "&start_date=2026-10-01&end_date=2026-10-31&timezone=America%2FNew_York"
    "&hourly=temperature_2m,precipitation"
)
with urllib.request.urlopen(url) as response:
    hourly = json.load(response)["hourly"]
data = {k: np.array(v, dtype=np.float32) for k, v in hourly.items() if k != "time"}

# Time in seconds since 1 October, 00:00 local time
t = np.arange(len(hourly["time"]), dtype=np.float32) * 3600
days = t.reshape(-1, 24)[:, 0]

temp = data["temperature_2m"].reshape(-1, 24)
temperature = vv.TimeseriesGraph(y_label="Temperature (°C)", legend_opts={"location": "northeast"}, hide_nav_toolbar=True, hide_time_axis_labels=True)
temperature.add_line_series(name="hourly", t=t, y=data["temperature_2m"], color="#d62728", width=1.5)
temperature.add_marker_series(name="daily high", t=days + 3600 * temp.argmax(axis=1), y=temp.max(axis=1), color="#8c1515", radius=4)
temperature.add_marker_series(name="daily low", t=days + 3600 * temp.argmin(axis=1), y=temp.min(axis=1), color="#1f77b4", radius=4)

# Hours with more than 0.1 mm of precipitation, shown as intervals
rain = data["precipitation"] > 0.1
precipitation = vv.TimeseriesGraph(y_label="mm/h", hide_nav_toolbar=True)
precipitation.add_interval_series(name="rain", t_start=t[rain], t_end=t[rain] + 3600, color="#9ecae1", alpha=0.6, border_color="none")
precipitation.add_line_series(name="precipitation", t=t, y=data["precipitation"], color="#08519c", width=1.5)

view = vv.Box(
    direction="vertical",
    items=[
        vv.LayoutItem(temperature, title="Temperature", stretch=1.6),
        vv.LayoutItem(precipitation, title="Precipitation", stretch=1),
    ],
)

view.show(
    title="Weather in New York City, October 2026",
    script=open(__file__).read(),
    upload=True,
)
