"""
Two-photon calcium imaging of 475 neurons in mouse primary visual cortex
(VISp) from an Allen Brain Observatory Visual Coding session (DANDI:000728),
streamed directly from the archive.

The figure shows the cell outlines over the maximum-intensity projection, a
two-minute clip of the raw (motion-corrected) imaging movie and the natural
movie the mouse was watching at the time, and, over the whole ~64 minute
session, detected calcium events for every cell, dF/F traces of the most
active cells, and running speed with the stimulus epochs.
"""

import io
import os
import h5py
import numpy as np
import remfile
from PIL import Image
import figpack.views as vv
import figpack_experimental.views as fpj
import figpack_nwb.views as fpn
import figpack_spike_sorting.views as ssv

processed_url = "https://api.dandiarchive.org/api/assets/25f50ae3-00ae-40b4-839c-ec331403e5a2/download/"
movie_url = "https://api.dandiarchive.org/api/assets/8e4e0009-99a3-4834-abf0-45461239b726/download/"

# Cache the remote byte ranges locally so that re-running is fast
os.makedirs("remfile_cache", exist_ok=True)
cache = remfile.DiskCache("remfile_cache")
f = h5py.File(remfile.File(processed_url, disk_cache=cache), "r")
ophys = f["processing/ophys"]

# Calcium imaging time base (about 30 Hz)
t = ophys["DfOverF/DfOverF/timestamps"][:]
fs = (len(t) - 1) / (t[-1] - t[0])

# Maximum-intensity projection as a contrast-stretched grayscale image
mip = ophys["SummaryImages/maximum_intensity_projection"][:].astype(float)
lo, hi = np.percentile(mip, [0.5, 99.8])
mip8 = (255 * np.clip((mip - lo) / (hi - lo), 0, 1)).astype(np.uint8)
png = io.BytesIO()
Image.fromarray(mip8.T).save(png, format="PNG")
projection = vv.Image(png.getvalue())

# Cell masks, with per-cell selection (read by the view from the NWB file).
# The view draws pixel_mask x horizontally; in this file x is the image row,
# so the projection and movie are shown transposed to match it.
segmentation = fpn.PlaneSegmentation(
    nwb=processed_url, path="/processing/ophys/ImageSegmentation/PlaneSegmentation")

# Detected calcium events for every cell, shown as a raster
roi_ids = ophys["ImageSegmentation/PlaneSegmentation/id"][:]
events = ophys["DfOverF/DfOverFEvents/data"][:]
raster = ssv.RasterPlot(
    start_time_sec=0.0,
    end_time_sec=float(np.ceil(t[-1])),
    plots=[ssv.RasterPlotItem(unit_id=int(rid), spike_times_sec=t[events[:, i] > 0].astype(np.float32))
           for i, rid in enumerate(roi_ids)],
)
print(f"{len(roi_ids)} cells, {int((events > 0).sum())} events")

# dF/F traces of the 40 cells with the most events, most active at the top
dff = ophys["DfOverF/DfOverF/data"][:]
top = np.argsort(-(events > 0).sum(axis=0))[:40][::-1]
traces = vv.TimeseriesGraph(y_label="dF/F (offset per cell)")
traces.add_uniform_series(
    name="dF/F",
    start_time_sec=float(t[0]),
    sampling_frequency_hz=fs,
    data=dff[:, top].astype(np.float32),
    channel_names=[f"Cell {roi_ids[i]}" for i in top],
    channel_spacing=1.5,
)

# Running speed with the stimulus epochs as intervals
speed = f["processing/behavior/BehavioralTimeSeries/running_speed"]
behavior = vv.TimeseriesGraph(y_label="Running speed (cm/s)")
epochs = f["intervals/epochs"]
epoch_type = epochs["stimulus_type"][:].astype(str)
epoch_colors = {
    "drifting_gratings": "#4e79a7",
    "natural_movie_one": "#8cd17d",
    "natural_movie_three": "#59a14f",
    "spontaneous": "#bbbbbb",
}
for stim, color in epoch_colors.items():
    sel = epoch_type == stim
    behavior.add_interval_series(
        name=stim.replace("_", " "),
        t_start=epochs["start_time"][:][sel].astype(np.float32),
        t_end=epochs["stop_time"][:][sel].astype(np.float32),
        color=color, alpha=0.35,
    )
speed_t, speed_v = speed["timestamps"][:], speed["data"][:]
ok = np.isfinite(speed_v)
behavior.add_line_series(
    name="running speed", t=speed_t[ok].astype(np.float32),
    y=speed_v[ok].astype(np.float32), color="black",
)

# Two minutes of the raw motion-corrected movie during natural movie one,
# averaged in pairs of frames (15 Hz) and contrast-stretched
clip_start, clip_dur = 1293.0, 120.0
i0, i1 = np.searchsorted(t, [clip_start, clip_start + clip_dur])
i1 = i0 + 2 * ((i1 - i0) // 2)
movie = h5py.File(remfile.File(movie_url, disk_cache=cache), "r")
raw = movie["acquisition/MotionCorrectedTwoPhotonSeries/data"]
frames = np.zeros(((i1 - i0) // 2, 512, 512), dtype=np.float32)
step = 190  # a multiple of the 19-frame chunk size
for j in range(i0, i1, step):
    block = raw[j:min(j + step, i1)].astype(np.float32)
    block = block[: 2 * (len(block) // 2)]
    frames[(j - i0) // 2:(j - i0) // 2 + len(block) // 2] = block.reshape(-1, 2, 512, 512).mean(axis=1)
    print(f"movie frames {j - i0 + len(block)} / {i1 - i0}")
lo, hi = np.percentile(frames[::20], [1, 99.5])
gray = (255 * np.clip((frames - lo) / (hi - lo), 0, 1)).astype(np.uint8).transpose(0, 2, 1)
del frames
# Synchronized with the traces: frame k averages imaging frames i0 + 2k and
# i0 + 2k + 1, so it sits at the midpoint of the pair
calcium_movie = fpj.LossyVideo(
    data=np.repeat(gray[..., None], 3, axis=3), fps=15,
    start_time_sec=float(t[i0]) + 0.5 / fs, sampling_frequency_hz=fs / 2,
)

# The natural movie on screen at the time of each clip frame (the 30 s movie
# is shown 10 times in this epoch), on the same time base as the clip
templates = f["stimulus/templates/natural_movie_one/data"][:]
presented = f["stimulus/presentation/natural_movie_one_stimulus"]
shown_t, shown_index = presented["timestamps"][:], presented["data"][:]
clip_t = float(t[i0]) + 0.5 / fs + np.arange(len(gray)) / (fs / 2)
k = np.clip(np.searchsorted(shown_t, clip_t, side="right") - 1, 0, len(shown_t) - 1)
stimulus_movie = fpj.LossyVideo(
    data=np.repeat(templates[shown_index[k]][..., None], 3, axis=3), fps=15,
    start_time_sec=float(clip_t[0]), sampling_frequency_hz=fs / 2,
)

# Layout
images = vv.Box(direction="horizontal", items=[
    vv.LayoutItem(projection, title="Maximum-intensity projection"),
    vv.LayoutItem(segmentation, title=f"{len(roi_ids)} cells (click to select)"),
    vv.LayoutItem(vv.TabLayout(items=[
        vv.TabLayoutItem(calcium_movie, label=f"Imaging movie ({clip_start:.0f}-{clip_start + clip_dur:.0f} s)"),
        vv.TabLayoutItem(stimulus_movie, label="What the mouse saw"),
    ])),
])
timeline = vv.Box(direction="vertical", items=[
    vv.LayoutItem(raster, title="Calcium events, all cells", stretch=2),
    vv.LayoutItem(traces, title="dF/F, 40 most active cells", stretch=1.6),
    vv.LayoutItem(behavior, title="Running speed and stimulus epochs", stretch=1),
])
view = vv.Splitter(
    direction="vertical",
    item1=vv.LayoutItem(images),
    item2=vv.LayoutItem(timeline),
    split_pos=0.35,
)

description = """
Two-photon calcium imaging (GCaMP6) of 475 neurons in mouse primary visual
cortex, Allen Brain Observatory Visual Coding session 688678766
([DANDI:000728](https://dandiarchive.org/dandiset/000728)), streamed from
the archive. Data: Allen Institute for Brain Science. Calcium events are
Allen's detected dF/F events. The imaging movie clip is motion corrected,
averaged to 15 Hz and contrast stretched. Images are shown transposed
(rows as x) to match the orientation of the cell mask panel.
"""

view.show(
    title="Allen two-photon imaging: 475 neurons in visual cortex",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
