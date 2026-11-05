"""
Sorted units from one Neuropixels probe in an Allen Brain Observatory
Visual Coding session (DANDI:000021), streamed directly from the archive.

Probe D passes through visual cortex (VISl), hippocampus (CA1, CA3, DG) and
visual thalamus (LGd). The figure covers the full ~2.6 hour session: units
table, raster plot and spike amplitudes sorted by depth, running speed with
the stimulus blocks, and per-unit average waveforms, autocorrelograms and
locations on the probe. All views are linked by time and by unit selection.
"""

import os
import h5py
import numpy as np
import remfile
import figpack.views as vv
import figpack_spike_sorting.views as ssv

url = "https://api.dandiarchive.org/api/assets/58703c97-c0a9-4736-b684-73c85c1a444a/download/"
probe = "probeD"

# Cache the remote byte ranges locally so that re-running is fast
os.makedirs("remfile_cache", exist_ok=True)
f = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache("remfile_cache")), "r")
units = f["units"]
electrodes = f["general/extracellular_ephys/electrodes"]

# Electrodes on this probe, indexed by local channel index (0-383)
probe_rows = np.nonzero(electrodes["group_name"][:].astype(str) == probe)[0]
local_index = electrodes["local_index"][:][probe_rows]
order = np.argsort(local_index)
probe_rows, local_index = probe_rows[order], local_index[order]
assert np.array_equal(local_index, np.arange(384))
ch_x = electrodes["probe_horizontal_position"][:][probe_rows].astype(float)
ch_y = electrodes["probe_vertical_position"][:][probe_rows].astype(float)
ch_region = electrodes["location"][:][probe_rows].astype(str)
channel_locations = {str(c): [ch_x[c], ch_y[c]] for c in range(384)}

# Units on this probe that passed Allen's quality control
electrode_ids = electrodes["id"][:]
row_of_electrode = {eid: k for k, eid in enumerate(electrode_ids)}
probe_row_set = set(probe_rows)
peak_rows = np.array([row_of_electrode[e] for e in units["peak_channel_id"][:]])
quality = units["quality"][:].astype(str)
unit_rows = np.nonzero(np.isin(peak_rows, probe_rows) & (quality == "good"))[0]
peak_channel = electrodes["local_index"][:][peak_rows[unit_rows]]

# Sort units by position along the probe, tip first. The raster plot draws the
# first unit at the bottom, so cortex ends up at the top.
depth_order = np.argsort(ch_y[peak_channel], kind="stable")
unit_rows, peak_channel = unit_rows[depth_order], peak_channel[depth_order]
unit_ids = units["id"][:][unit_rows]
print(f"{len(unit_ids)} units on {probe}")

# Read the ragged spike-time, amplitude and waveform arrays in one block each
def read_ragged(name, rows):
    index = units[f"{name}_index"][:]
    lo, hi = index[rows.min() - 1] if rows.min() > 0 else 0, index[rows.max()]
    block = units[name][lo:hi]
    return [block[(index[r - 1] if r > 0 else 0) - lo:index[r] - lo] for r in rows]

spike_times = read_ragged("spike_times", unit_rows)
spike_amplitudes = [a * 1e6 for a in read_ragged("spike_amplitudes", unit_rows)]  # V to uV
waveforms = read_ragged("waveform_mean", unit_rows)  # each is 384 channels x 82 samples, uV
# Start the raster at t = 0 so it covers the running speed and stimulus
# intervals too, which begin a little before the first spike
t_start = 0.0
t_end = float(np.ceil(max(st[-1] for st in spike_times)))
print(f"{sum(len(st) for st in spike_times)} spikes, up to {t_end:.0f} s")

# Units table
metric = {k: units[k][:][unit_rows] for k in [
    "firing_rate", "snr", "amplitude", "isi_violations", "presence_ratio", "waveform_duration"]}
columns = [
    ssv.UnitsTableColumn(key="region", label="Region", dtype="str"),
    ssv.UnitsTableColumn(key="depth", label="Probe position (um)", dtype="int"),
    ssv.UnitsTableColumn(key="firing_rate", label="Rate (Hz)", dtype="float"),
    ssv.UnitsTableColumn(key="amplitude", label="Amplitude (uV)", dtype="float"),
    ssv.UnitsTableColumn(key="snr", label="SNR", dtype="float"),
    ssv.UnitsTableColumn(key="isi_violations", label="ISI viol.", dtype="float"),
    ssv.UnitsTableColumn(key="waveform_duration", label="Spike width (ms)", dtype="float"),
]
rows = []
for i, uid in enumerate(unit_ids):
    rows.append(ssv.UnitsTableRow(unit_id=int(uid), values={
        "region": ch_region[peak_channel[i]],
        "depth": int(ch_y[peak_channel[i]]),
        **{k: round(float(v[i]), 3) for k, v in metric.items()},
    }))
units_table = ssv.UnitsTable(columns=columns, rows=rows)

# Raster plot and spike amplitudes over the whole session
raster = ssv.RasterPlot(
    start_time_sec=t_start,
    end_time_sec=t_end,
    plots=[ssv.RasterPlotItem(unit_id=int(uid), spike_times_sec=st.astype(np.float32))
           for uid, st in zip(unit_ids, spike_times)],
)
amplitudes = ssv.SpikeAmplitudes(
    start_time_sec=t_start,
    end_time_sec=t_end,
    plots=[ssv.SpikeAmplitudesItem(unit_id=int(uid), spike_times_sec=st.astype(np.float32),
                                   spike_amplitudes=a.astype(np.float32))
           for uid, st, a in zip(unit_ids, spike_times, spike_amplitudes)],
)

# Running speed, resampled to 30 Hz, with the stimulus blocks as intervals
speed_t = f["processing/running/running_speed/timestamps"][:]
speed = f["processing/running/running_speed/data"][:]
fs_speed = 30
t_uniform = np.arange(speed_t[0], speed_t[-1], 1 / fs_speed)
behavior = vv.TimeseriesGraph(y_label="Running speed (cm/s)")
stimulus_colors = {
    "spontaneous": "#bbbbbb",
    "gabors": "#e15759",
    "flashes": "#f28e2b",
    "drifting_gratings": "#4e79a7",
    "natural_movie_three": "#59a14f",
    "natural_movie_one": "#8cd17d",
    "static_gratings": "#b07aa1",
    "natural_scenes": "#edc948",
}
for stim, color in stimulus_colors.items():
    table = f[f"intervals/{stim}_presentations"]
    starts, stops = table["start_time"][:], table["stop_time"][:]
    # Merge consecutive presentations into blocks
    breaks = np.nonzero(starts[1:] - stops[:-1] > 5)[0]
    block_starts = np.concatenate([[starts[0]], starts[breaks + 1]])
    block_stops = np.concatenate([stops[breaks], [stops[-1]]])
    behavior.add_interval_series(
        name=stim.replace("_", " "), t_start=block_starts.astype(np.float32),
        t_end=block_stops.astype(np.float32), color=color, alpha=0.35,
    )
behavior.add_uniform_series(
    name="running speed", start_time_sec=float(t_uniform[0]), sampling_frequency_hz=fs_speed,
    data=np.interp(t_uniform, speed_t, speed).astype(np.float32), colors=["black"],
)

# Average waveforms on the 16 channels nearest each unit's peak channel
waveform_items = []
for uid, w, pc in zip(unit_ids, waveforms, peak_channel):
    chans = np.arange(max(0, pc - 8), min(384, pc + 8))
    waveform_items.append(ssv.AverageWaveformItem(
        unit_id=int(uid), channel_ids=[str(c) for c in chans],
        waveform=w[chans].T.astype(np.float32),
    ))
average_waveforms = ssv.AverageWaveforms(
    average_waveforms=waveform_items, channel_locations=channel_locations)

# Autocorrelograms, +/- 50 ms in 1 ms bins
edges = np.arange(-0.05, 0.0505, 0.001)
acg_items = []
for uid, st in zip(unit_ids, spike_times):
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for lag in range(1, len(st)):
        d = st[lag:] - st[:-lag]
        d = d[d < edges[-1]]
        if len(d) == 0:
            break
        d = d[d > 0]  # skip duplicate spike times
        counts += np.histogram(d, edges)[0] + np.histogram(-d, edges)[0]
    acg_items.append(ssv.AutocorrelogramItem(unit_id=int(uid), bin_edges_sec=edges, bin_counts=counts))
autocorrelograms = ssv.Autocorrelograms(autocorrelograms=acg_items)

# Unit locations: center of mass of waveform peak-to-peak amplitude over
# channels within 50 um of the peak channel
location_items = []
for uid, w, pc in zip(unit_ids, waveforms, peak_channel):
    near = np.hypot(ch_x - ch_x[pc], ch_y - ch_y[pc]) <= 50
    weight = np.ptp(w, axis=1)[near] ** 2
    location_items.append(ssv.UnitLocationsItem(
        unit_id=int(uid),
        x=float(np.sum(weight * ch_x[near]) / np.sum(weight)),
        y=float(np.sum(weight * ch_y[near]) / np.sum(weight)),
    ))
unit_locations = ssv.UnitLocations(
    units=location_items, channel_locations=channel_locations, disable_auto_rotate=True)

# Layout
regions = " / ".join(dict.fromkeys(ch_region[peak_channel][::-1]))
main_column = vv.Box(direction="vertical", items=[
    vv.LayoutItem(vv.TabLayout(items=[
        vv.TabLayoutItem(raster, label="Raster plot"),
        vv.TabLayoutItem(amplitudes, label="Spike amplitudes"),
    ]), stretch=2),
    vv.LayoutItem(behavior, title="Running speed and stimulus blocks", stretch=1),
    vv.LayoutItem(vv.Box(direction="horizontal", items=[
        vv.LayoutItem(average_waveforms, title="Average waveforms"),
        vv.LayoutItem(autocorrelograms, title="Autocorrelograms"),
    ]), stretch=1.5),
])
view = vv.Splitter(
    direction="horizontal",
    item1=vv.LayoutItem(units_table, title=f"{len(unit_ids)} units ({regions})"),
    item2=vv.LayoutItem(vv.Box(direction="horizontal", items=[
        vv.LayoutItem(main_column),
        vv.LayoutItem(unit_locations, title="Probe", max_size=160),
    ])),
    split_pos=0.28,
)

description = """
Sorted units from Neuropixels probe D in Allen Brain Observatory Visual Coding
session 715093703 ([DANDI:000021](https://dandiarchive.org/dandiset/000021)),
streamed from the archive. Data: Allen Institute for Brain Science.
Units are those labeled "good" by Allen's quality control, sorted by depth.
Unit locations are the amplitude-weighted center of mass over nearby channels.
"""

view.show(
    title="Allen Neuropixels: sorted units across cortex, hippocampus and thalamus",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
