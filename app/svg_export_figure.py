import numpy as np
import figpack.views as vv

# Simulated 4-channel ephys, 30 seconds at 20 kHz, in a layout of views that
# all support Export as SVG: raw traces, a spectrogram, spike rates and the
# low-frequency signal.
fs = 20_000
duration = 30
n = duration * fs
rng = np.random.default_rng(1)
t = np.arange(n) / fs
colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

# Low-frequency oscillation whose frequency drifts between about 5 and 10 Hz
freq = 7.5 + 2.5 * np.sin(2 * np.pi * t / 20)
lfp = 25 * np.sin(2 * np.pi * np.cumsum(freq) / fs) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 7))

data = 8 * rng.standard_normal((n, 4)) + lfp[:, None]

# Spikes with slowly varying rates, a biphasic waveform on each channel
k = np.arange(40)
waveform = -np.sin(np.pi * k / 20) * np.exp(-k / 12)
spike_times = []
for ch, (amp, rate, phase) in enumerate([(140, 10, 0.0), (95, 14, 1.5), (180, 6, 3.0), (110, 12, 4.5)]):
    r = rate * (1 + 0.8 * np.sin(2 * np.pi * t / 12 + phase))
    idx = np.nonzero(rng.random(n) < r / fs)[0]
    idx = idx[idx < n - 40]
    for i in idx:
        data[i:i + 40, ch] += amp * waveform
    spike_times.append(idx / fs)

# Raw traces
raw = vv.TimeseriesGraph(y_label="Voltage (uV)", hide_nav_toolbar=True)
raw.add_uniform_series(
    name="ephys",
    start_time_sec=0,
    sampling_frequency_hz=fs,
    data=data.astype(np.float32),
    channel_names=["Ch 1", "Ch 2", "Ch 3", "Ch 4"],
    colors=colors,
    channel_spacing=300,
)

# Low-frequency signal: channel mean, averaged down to 1 kHz
lfp_1k = data.mean(axis=1).reshape(-1, 20).mean(axis=1)
low = vv.TimeseriesGraph(y_label="uV")
low.add_uniform_series(
    name="low-frequency signal",
    start_time_sec=0,
    sampling_frequency_hz=1000,
    data=lfp_1k.astype(np.float32),
    colors=["#1c2954"],
)

# Spectrogram of the low-frequency signal: 1 s windows every 50 ms, 0-30 Hz
win, hop = 1000, 50
frames = np.lib.stride_tricks.sliding_window_view(lfp_1k, win)[::hop] * np.hanning(win)
power = np.abs(np.fft.rfft(frames, axis=1)[:, :31]) ** 2
spectrogram = vv.Spectrogram(
    start_time_sec=win / 2000,
    sampling_frequency_hz=1000 / hop,
    frequency_min_hz=0,
    frequency_delta_hz=1,
    data=np.log10(power + 1).astype(np.float32),
)

# Spike rate per channel in 250 ms bins
edges = np.arange(0, duration + 0.25, 0.25)
rates = vv.TimeseriesGraph(y_label="Spikes/s", hide_nav_toolbar=True)
for ch, times in enumerate(spike_times):
    counts, _ = np.histogram(times, edges)
    rates.add_line_series(
        name=f"Ch {ch + 1}",
        t=(edges[:-1] + 0.125).astype(np.float32),
        y=(counts / 0.25).astype(np.float32),
        color=colors[ch],
        width=2,
    )

view = vv.Box(
    direction="vertical",
    items=[
        vv.LayoutItem(raw, title="Raw traces", stretch=1.4),
        vv.LayoutItem(
            vv.Box(
                direction="horizontal",
                items=[
                    vv.LayoutItem(spectrogram, title="Spectrogram (0-30 Hz)"),
                    vv.LayoutItem(rates, title="Spike rate"),
                ],
            ),
            stretch=1.6,
        ),
        vv.LayoutItem(low, title="Low-frequency signal", stretch=1),
    ],
)
view.show(title="Simulated ephys: layout for SVG export", upload=True)
