"""
Synchronized views: two minutes of an Allen Brain Observatory two-photon
session (DANDI:000728), streamed from the archive.

The natural movie the mouse was watching, the imaging movie, dF/F traces of
the most active cells and the running speed share one timeline. Playing either
video moves the other video and the time cursor in the traces; clicking or
zooming in a trace moves the videos.
"""

import os
import h5py
import numpy as np
import remfile
import figpack.views as vv
import figpack_experimental.views as fpj

processed_url = "https://api.dandiarchive.org/api/assets/25f50ae3-00ae-40b4-839c-ec331403e5a2/download/"
movie_url = "https://api.dandiarchive.org/api/assets/8e4e0009-99a3-4834-abf0-45461239b726/download/"

# Cache the remote byte ranges locally so that re-running is fast
os.makedirs("remfile_cache", exist_ok=True)
cache = remfile.DiskCache("remfile_cache")
f = h5py.File(remfile.File(processed_url, disk_cache=cache), "r")
ophys = f["processing/ophys"]

# Calcium imaging time base (about 30 Hz), and a two-minute clip during the
# natural movie one epoch
t = ophys["DfOverF/DfOverF/timestamps"][:]
fs = (len(t) - 1) / (t[-1] - t[0])
clip_start, clip_dur = 1293.0, 120.0
i0, i1 = np.searchsorted(t, [clip_start, clip_start + clip_dur])
i1 = i0 + 2 * ((i1 - i0) // 2)

# dF/F of the 12 cells with the most calcium events in the clip, most active at the top
roi_ids = ophys["ImageSegmentation/PlaneSegmentation/id"][:]
events = ophys["DfOverF/DfOverFEvents/data"][i0:i1]
top = np.argsort(-(events > 0).sum(axis=0))[:12][::-1]
dff = ophys["DfOverF/DfOverF/data"][i0:i1][:, top]
traces = vv.TimeseriesGraph(y_label="dF/F (offset per cell)", hide_nav_toolbar=True, hide_time_axis_labels=True)
traces.add_uniform_series(
    name="dF/F",
    start_time_sec=float(t[i0]),
    sampling_frequency_hz=fs,
    data=dff.astype(np.float32),
    channel_names=[f"Cell {roi_ids[i]}" for i in top],
    channel_spacing=1.5,
)

# Running speed over the clip
speed = f["processing/behavior/BehavioralTimeSeries/running_speed"]
speed_t, speed_v = speed["timestamps"][:], speed["data"][:]
sel = (speed_t >= t[i0]) & (speed_t <= t[i1 - 1]) & np.isfinite(speed_v)
behavior = vv.TimeseriesGraph(y_label="cm/s")
behavior.add_line_series(
    name="running speed", t=speed_t[sel].astype(np.float32),
    y=speed_v[sel].astype(np.float32), color="black", width=1.5,
)

# The raw motion-corrected movie, averaged in pairs of frames (15 Hz) and
# contrast-stretched
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
gray = (255 * np.clip((frames - lo) / (hi - lo), 0, 1)).astype(np.uint8)
del frames
# Frame k averages imaging frames i0 + 2k and i0 + 2k + 1, so it sits at the
# midpoint of the pair
clip_t = float(t[i0]) + 0.5 / fs + np.arange(len(gray)) / (fs / 2)
calcium_movie = fpj.LossyVideo(
    data=np.repeat(gray[..., None], 3, axis=3), fps=15,
    start_time_sec=float(clip_t[0]), sampling_frequency_hz=fs / 2,
)

# The natural movie on screen at the time of each clip frame (the 30 s movie
# is shown 10 times in this epoch), on the same time base as the clip
templates = f["stimulus/templates/natural_movie_one/data"][:]
presented = f["stimulus/presentation/natural_movie_one_stimulus"]
shown_t, shown_index = presented["timestamps"][:], presented["data"][:]
k = np.clip(np.searchsorted(shown_t, clip_t, side="right") - 1, 0, len(shown_t) - 1)
stimulus_movie = fpj.LossyVideo(
    data=np.repeat(templates[shown_index[k]][..., None], 3, axis=3), fps=15,
    start_time_sec=float(clip_t[0]), sampling_frequency_hz=fs / 2,
)

videos = vv.Box(direction="horizontal", items=[
    vv.LayoutItem(stimulus_movie, title="What the mouse saw"),
    vv.LayoutItem(calcium_movie, title="Two-photon imaging (primary visual cortex)"),
])
timeline = vv.Box(direction="vertical", items=[
    vv.LayoutItem(traces, title="dF/F, 12 most active cells", stretch=2),
    vv.LayoutItem(behavior, title="Running speed", stretch=1.2),
])
view = vv.Splitter(
    direction="vertical",
    item1=vv.LayoutItem(videos),
    item2=vv.LayoutItem(timeline),
    split_pos=0.5,
)

description = """
Two minutes of two-photon calcium imaging (GCaMP6) in mouse primary visual
cortex, Allen Brain Observatory Visual Coding session 688678766
([DANDI:000728](https://dandiarchive.org/dandiset/000728)), streamed from the
archive. Data: Allen Institute for Brain Science. The imaging movie is motion
corrected, averaged to 15 Hz and contrast stretched. All views share one
timeline.
"""

view.show(
    title="Synchronized views: stimulus, imaging and behavior",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
