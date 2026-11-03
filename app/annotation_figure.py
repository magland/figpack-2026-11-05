import numpy as np
import figpack.views as vv

# Simulated 4-channel ephys: 30 seconds at 20 kHz
fs = 20_000
n = 30 * fs
rng = np.random.default_rng(0)
t = np.arange(n) / fs

# Noise plus a low-frequency modulation (uV)
data = 8 * rng.standard_normal((n, 4))
data += 25 * np.sin(2 * np.pi * 7 * t)[:, None]

# Spikes at random times on each channel
k = np.arange(40)
waveform = -np.sin(np.pi * k / 20) * np.exp(-k / 12)
for ch, amp in enumerate([140, 95, 180, 110]):
    for i in rng.integers(0, n - 40, size=300):
        data[i:i + 40, ch] += amp * waveform

graph = vv.TimeseriesGraph(y_label="Voltage (uV)", allow_annotation=True)
graph.add_uniform_series(
    name="ephys",
    start_time_sec=0,
    sampling_frequency_hz=fs,
    data=data.astype(np.float32),
    channel_names=["Ch 1", "Ch 2", "Ch 3", "Ch 4"],
    colors=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
    channel_spacing=300,
)
graph.show(title="Simulated 4-channel ephys (annotation)", upload=True)
