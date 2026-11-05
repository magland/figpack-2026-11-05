"""
Simulated hippocampal place cells on a linear track, with theta phase
precession and sharp-wave ripple replay, decoded with a Bayesian decoder.

A virtual rat runs laps on a 2 m track and pauses at each end. While it runs,
each of 120 place cells fires near its preferred location, and its spikes
shift to earlier theta phases as the rat crosses the field (phase
precession). While the rat pauses, brief sharp-wave ripple events replay
compressed trajectories along the track, in forward or reverse order.

Place fields are re-estimated from the running periods of the simulated
spikes (as one would with real data), and the position is decoded from the
spikes in 25 ms windows. During ripples the decoded position sweeps across
the track while the rat sits still. Everything here is synthetic.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import figpack.views as vv
import figpack_experimental.views as fpj
import figpack_spike_sorting.views as ssv

rng = np.random.default_rng(1)
L = 200.0  # track length (cm)
fs = 1000.0  # simulation resolution (Hz)
T = 600.0  # session length (s)
n = int(T * fs)
t = np.arange(n) / fs

# ---------------------------------------------------------------------------
# Trajectory: run end to end with a smooth speed profile, pause, repeat
pos = np.zeros(n)
rest_intervals, run_intervals = [], []
k, x0 = 0, 0.0
while k < n:
    rest = int(rng.uniform(4, 11) * fs)  # pause at the current end
    rest_intervals.append((k / fs, min(k + rest, n) / fs))
    jitter = ndimage.gaussian_filter1d(rng.standard_normal(rest), 300) * 40
    pos[k:k + rest] = (x0 + np.sign(L / 2 - x0) * (3 + jitter))[: n - k]
    k += rest
    if k >= n:
        break
    run = int(L / rng.uniform(35, 55) * fs)
    s = np.arange(run) / run
    x1 = L - x0
    pos[k:k + run] = (x0 + (x1 - x0) * (1 - np.cos(np.pi * s)) / 2)[: n - k]
    run_intervals.append((k / fs, min(k + run, n) / fs))
    k += run
    x0 = x1
pos = np.clip(pos, 0, L)
vel = np.gradient(ndimage.gaussian_filter1d(pos, 50)) * fs
speed = np.abs(vel)
running = speed > 5

# Theta rhythm: about 8 Hz, slightly faster at higher speed
theta_freq = 7.5 + 0.02 * speed
theta_phase = 2 * np.pi * np.cumsum(theta_freq) / fs

# ---------------------------------------------------------------------------
# Sharp-wave ripples during pauses, each replaying a compressed trajectory
events = []  # (t_start, t_end, x_from, x_to)
for a, b in rest_intervals:
    tt = a + rng.exponential(1.0)
    while tt < b - 0.5:
        dur = rng.uniform(0.08, 0.2)
        here = pos[int(tt * fs)]
        far = L - here if here < L / 2 else 0.0
        frac = rng.uniform(0.6, 1.0)  # some replays cover part of the track
        if rng.random() < 0.6:  # starts where the rat is and sweeps away
            x_from, x_to = here, here + frac * (far - here)
        else:  # sweeps toward the rat
            x_from, x_to = here + frac * (far - here), here
        events.append((tt, tt + dur, x_from, x_to))
        tt += dur + 0.3 + rng.exponential(1.2)
replay_pos = np.full(n, np.nan)
for a, b, xf, xt in events:
    i0, i1 = int(a * fs), int(b * fs)
    replay_pos[i0:i1] = np.linspace(xf, xt, i1 - i0)

# ---------------------------------------------------------------------------
# Place cells: Gaussian fields, theta phase precession, replay participation
n_cells = 120
centers = np.sort(rng.uniform(5, L - 5, n_cells))
widths = rng.uniform(6, 14, n_cells)  # Gaussian sigma (cm)
peaks = rng.uniform(10, 30, n_cells)  # peak in-field rate (Hz)
kappa = 1.5  # concentration of theta phase locking
# Each cell takes part in a random 60% of replay events, with variable gain
event_gain = (rng.random((len(events) + 1, n_cells)) < 0.6) * rng.uniform(0.5, 1.5, (len(events) + 1, n_cells))
event_gain[-1] = 0  # index -1: no event
event_index = np.full(n, -1)
for e, (a, b, _, _) in enumerate(events):
    event_index[int(a * fs):int(b * fs)] = e
spike_trains = []
for i in range(n_cells):
    g = np.exp(-0.5 * ((pos - centers[i]) / widths[i]) ** 2)
    # Phase goes from 360 to 0 degrees across the field in the running direction
    u = np.sign(vel) * (pos - centers[i]) / (2 * widths[i])
    pref = np.pi * (1 - np.clip(u, -1, 1))
    theta_mod = np.exp(kappa * (np.cos(theta_phase - pref) - 1)) / np.i0(kappa) * np.exp(kappa)
    rate = np.where(running, peaks[i] * g * theta_mod, 0.0) + 0.1
    gr = np.nan_to_num(np.exp(-0.5 * ((replay_pos - centers[i]) / widths[i]) ** 2))
    rate += 60 * gr * event_gain[event_index, i]
    spikes = t[rng.random(n) < rate / fs] + rng.uniform(0, 1 / fs)
    spike_trains.append(np.sort(spikes))

# ---------------------------------------------------------------------------
# LFP at 1250 Hz: speed-dependent theta, sharp waves with ~190 Hz ripples, 1/f noise
lfp_fs = 1250.0
idx = (np.arange(int(T * lfp_fs)) / lfp_fs * fs).astype(int)
lfp_t = idx / fs
theta_amp = 0.02 + 0.3 * np.clip(speed[idx] / 30, 0, 1.2)
noise = np.cumsum(rng.standard_normal(len(idx)))
noise = signal.lfilter([1, -1], [1, -0.995], noise)  # remove drift, keep a 1/f-like spectrum
lfp = theta_amp * np.cos(theta_phase[idx]) + 0.006 * noise + 0.02 * rng.standard_normal(len(idx))
for a, b, _, _ in events:
    c, d = (a + b) / 2, b - a
    env = np.exp(-0.5 * ((lfp_t - c) / (d / 4)) ** 2)
    lfp += -0.4 * np.exp(-0.5 * ((lfp_t - c) / 0.04) ** 2)
    lfp += 0.2 * env * np.sin(2 * np.pi * rng.uniform(170, 210) * (lfp_t - a))
lfp = lfp.astype(np.float32)

# ---------------------------------------------------------------------------
# Rate maps from running periods only (2 cm bins)
bins = np.linspace(0, L, 101)
grid = (bins[:-1] + bins[1:]) / 2
occupancy = np.histogram(pos[running], bins)[0] / fs
rate_maps = np.zeros((n_cells, len(grid)))
for i, st in enumerate(spike_trains):
    j = (st * fs).astype(int)
    j = j[running[j]]
    counts = np.histogram(pos[j], bins)[0]
    rate_maps[i] = ndimage.gaussian_filter1d(counts / np.maximum(occupancy, 1e-3), 1.5)
rate_maps += 0.05

# Bayesian decoding with a uniform prior: 25 ms windows every 10 ms
dec_fs, win = 100.0, 0.025
n_bins = int(T * dec_fs)
counts5 = np.zeros((int(T * 200), n_cells))  # 5 ms counts
for i, st in enumerate(spike_trains):
    counts5[:, i] = np.bincount((st * 200).astype(int), minlength=counts5.shape[0])[: counts5.shape[0]]
counts25 = ndimage.uniform_filter1d(counts5, 5, axis=0, mode="constant") * 5
counts_dec = counts25[1::2][:n_bins]  # window centered on each 10 ms step
log_post = counts_dec @ np.log(rate_maps) - win * rate_maps.sum(axis=0)
log_post -= log_post.max(axis=1, keepdims=True)
posterior = np.exp(log_post)
posterior /= posterior.sum(axis=1, keepdims=True)
dec_t = (np.arange(n_bins) + 0.5) / dec_fs
obs = np.interp(dec_t, t, pos)

# ---------------------------------------------------------------------------
# Views
decode = fpj.LinearDecode(
    start_time_sec=float(dec_t[0]),
    sampling_frequency_hz=dec_fs,
    data=posterior.astype(np.float32),
    observed_positions=obs.astype(np.float32),
    position_grid=grid.astype(np.float32),
)

raster = ssv.RasterPlot(
    start_time_sec=0.0,
    end_time_sec=T,
    plots=[ssv.RasterPlotItem(unit_id=i + 1, spike_times_sec=st.astype(np.float32))
           for i, st in enumerate(spike_trains)],
)

swr_start = np.array([e[0] for e in events], dtype=np.float32)
swr_end = np.array([e[1] for e in events], dtype=np.float32)

lfp_graph = vv.TimeseriesGraph(y_label="LFP (mV)", hide_nav_toolbar=True)
lfp_graph.add_interval_series(name="ripple events", t_start=swr_start, t_end=swr_end, color="#f4a261", alpha=0.4)
lfp_graph.add_uniform_series(name="LFP", start_time_sec=0.0, sampling_frequency_hz=lfp_fs, data=lfp, colors=["#264653"])

ds = 20  # position at 50 Hz
behavior = vv.TimeseriesGraph(y_label="Position (cm)", y_range=[-5, 205], hide_nav_toolbar=True)
behavior.add_interval_series(name="running", t_start=np.array([r[0] for r in run_intervals], dtype=np.float32),
                             t_end=np.array([r[1] for r in run_intervals], dtype=np.float32), color="#8ecae6", alpha=0.3)
behavior.add_uniform_series(name="position", start_time_sec=0.0, sampling_frequency_hz=fs / ds,
                            data=pos[::ds].astype(np.float32), colors=["#023047"], width=2)

# Spectrogram of the LFP: 0.4 s windows every 50 ms, up to 250 Hz
f_spec, t_spec, sxx = signal.spectrogram(lfp, fs=lfp_fs, nperseg=500, noverlap=500 - 62, window="hann")
keep = f_spec <= 250
spectrogram = vv.Spectrogram(
    start_time_sec=float(t_spec[0]),
    sampling_frequency_hz=float(1 / (t_spec[1] - t_spec[0])),
    frequency_min_hz=0.0,
    frequency_delta_hz=float(f_spec[1]),
    # power relative to the median at each frequency (log10), so theta and
    # ripples stand out against the 1/f background
    data=np.clip(np.log10(sxx[keep].T / np.median(sxx[keep], axis=1)), 0, None).astype(np.float32),
    hide_nav_toolbar=True,
)

# Static summary: rate maps sorted by field center, and theta phase precession for one cell
plt.rcParams["font.size"] = 12
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
norm_maps = rate_maps / rate_maps.max(axis=1, keepdims=True)
axes[0].imshow(norm_maps, aspect="auto", origin="lower", extent=[0, L, 0.5, n_cells + 0.5], cmap="magma")
axes[0].set_xlabel("Position (cm)")
axes[0].set_ylabel("Cell (sorted by field center)")
axes[0].set_title("Rate maps estimated from running periods")
cell = int(np.argmin(np.abs(centers - L / 2)))
st = spike_trains[cell]
j = (st * fs).astype(int)
sel = running[j] & (vel[j] > 0)
ph = np.degrees(theta_phase[j[sel]] % (2 * np.pi))
axes[1].scatter(np.r_[pos[j[sel]], pos[j[sel]]], np.r_[ph, ph + 360], s=4, color="#264653")
axes[1].set_xlim(centers[cell] - 3 * widths[cell], centers[cell] + 3 * widths[cell])
axes[1].set_ylim(0, 720)
axes[1].set_yticks([0, 180, 360, 540, 720])
axes[1].set_xlabel("Position (cm), rightward runs")
axes[1].set_ylabel("Theta phase (deg)")
axes[1].set_title(f"Phase precession, cell {cell + 1}")
fig.tight_layout()
summary = vv.MatplotlibFigure(fig)

table = ssv.UnitsTable(
    columns=[
        ssv.UnitsTableColumn(key="center", label="Field center (cm)", dtype="float"),
        ssv.UnitsTableColumn(key="width", label="Field sigma (cm)", dtype="float"),
        ssv.UnitsTableColumn(key="peak", label="Peak rate (Hz)", dtype="float"),
        ssv.UnitsTableColumn(key="n_spikes", label="Spikes", dtype="int"),
    ],
    rows=[ssv.UnitsTableRow(unit_id=i + 1, values={
        "center": round(float(centers[i]), 1), "width": round(float(widths[i]), 1),
        "peak": round(float(peaks[i]), 1), "n_spikes": int(len(spike_trains[i]))})
        for i in range(n_cells)],
)

view = vv.Splitter(
    direction="horizontal",
    item1=vv.LayoutItem(vv.Box(direction="vertical", items=[
        vv.LayoutItem(decode, title="Decoded position (posterior) with the actual position", stretch=2),
        vv.LayoutItem(raster, title="120 place cells, sorted by field center", stretch=2),
        vv.LayoutItem(lfp_graph, title="LFP with ripple events", stretch=1),
        vv.LayoutItem(spectrogram, title="LFP spectrogram (power relative to median)", stretch=1),
        vv.LayoutItem(behavior, title="Position, running periods shaded", stretch=0.8),
    ])),
    item2=vv.LayoutItem(vv.Box(direction="vertical", items=[
        vv.LayoutItem(summary, title="Place fields and phase precession", stretch=1),
        vv.LayoutItem(table, title="Cells", stretch=1),
    ])),
    split_pos=0.72,
)

description = """
Synthetic data. A virtual rat runs laps on a 2 m linear track (35 to 55 cm/s)
and pauses for 4 to 11 s at each end, for 10 minutes. 120 place cells have
Gaussian fields (sigma 6 to 14 cm, peak 10 to 30 Hz, non-directional) and
fire as inhomogeneous Poisson processes, with theta phase precession while
running. During pauses, sharp-wave ripple events (80 to 200 ms) replay
compressed trajectories in forward or reverse order (each cell joins a random 60% of events). The LFP is a sum of
speed-dependent theta, sharp waves with 170 to 210 Hz ripples, and 1/f noise.
Rate maps are estimated from the running periods, and position is decoded
with a Bayesian decoder (uniform prior, 25 ms windows every 10 ms).
"""

view.show(
    title="Place cells, phase precession and replay (simulated)",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
