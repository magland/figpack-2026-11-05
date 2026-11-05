"""
Spiral wave breakup in an excitable medium, with a simulated electrode array.

We simulate the Bar-Eiswirth variant of the Barkley reaction-diffusion model,
a standard caricature of excitable tissue such as cardiac muscle or cortex.
The excitability parameter eps is ramped slowly upward. At first a single
spiral rotates steadily; past a threshold it breaks up into many short-lived
spirals (spatiotemporal chaos, a model of fibrillation).

The figure shows the wave field as a movie, an 8 x 8 grid of virtual
electrodes "recording" the simulation (as a movie at the electrode positions
and as traces), a spectrogram of the center electrode, and the number of
spiral tips over time. For display, one model time unit is shown as 20 ms.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib
import figpack.views as vv
import figpack_experimental.views as fpj

# Model: du/dt = u (1 - u) (u - (v + b) / a) / eps + lap(u),  dv/dt = f(u) - v
N, L = 256, 128.0  # grid points per side, domain size
dx = L / N
a, b = 0.84, 0.07
dt = 0.01
T_ramp, T_total = 600.0, 800.0
eps_start, eps_end = 0.04, 0.08


def eps_at(t):
    return eps_start + (eps_end - eps_start) * min(t / T_ramp, 1.0)


def laplacian(z):  # no-flux boundaries
    zp = np.pad(z, 1, mode="edge")
    return (zp[2:, 1:-1] + zp[:-2, 1:-1] + zp[1:-1, 2:] + zp[1:-1, :-2] - 4 * z) / dx**2


def f(u):  # Bar-Eiswirth recovery term
    return np.where(u < 1 / 3, 0.0, np.where(u > 1, 1.0, 1 - 6.75 * u * (u - 1) ** 2))


def count_tips(u, v):
    # A spiral tip is where the u = 0.5 and v = a / 2 - b contours cross; count
    # the connected groups of 2 x 2 cells in which both fields change sign
    s1, s2 = u > 0.5, v > a / 2 - b
    def mixed(s):
        c = s[:-1, :-1].astype(int) + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:]
        return (c > 0) & (c < 4)
    return ndimage.label(mixed(s1) & mixed(s2))[1]


# Initial condition: an excited quadrant next to a refractory one, whose
# shared free end curls up into a spiral
u = np.zeros((N, N))
v = np.zeros((N, N))
u[: N // 2, : N // 2] = 1.0
v[N // 2:, : N // 2] = 0.6


def step_model(u, v, eps):  # forward Euler
    return (u + dt * (u * (1 - u) * (u - (v + b) / a) / eps + laplacian(u)),
            v + dt * (f(u) - v))


# Let the spiral form before recording starts
for _ in range(int(100 / dt)):
    u, v = step_model(u, v, eps_start)

# Virtual 8 x 8 electrode array; each electrode averages a 5 x 5 patch
n_side = 8
spacing = N // n_side
centers = np.arange(spacing // 2, N, spacing)
electrode_rc = [(r, c) for r in centers for c in centers]
electrode_xy = np.array([[c * dx, (N - r) * dx] for r, c in electrode_rc])

# For display, one model time unit is shown as 20 ms. This is an arbitrary
# choice; it puts the spiral rotation at roughly 12 Hz.
sec_per_unit = 0.02

frame_every = 50  # steps between movie frames (0.5 time units)
sample_every = 5  # steps between electrode samples (0.05 time units)
n_steps = int(round(T_total / dt))
cmap = matplotlib.colormaps["magma"]
font = ImageFont.load_default(size=20)
frames, samples, tip_t, tip_n = [], [], [], []
for step in range(n_steps + 1):
    t = step * dt
    if step % sample_every == 0:
        samples.append([u[r - 2:r + 3, c - 2:c + 3].mean() for r, c in electrode_rc])
    if step % frame_every == 0:
        rgb = (cmap(np.clip(u, 0, 1))[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb).resize((2 * N, 2 * N), Image.BILINEAR)
        ImageDraw.Draw(img).text((10, 6), f"t = {t * sec_per_unit:.2f} s   eps = {eps_at(t):.3f}",
                                 fill=(255, 255, 255), font=font)
        frames.append(np.array(img))
        tip_t.append(t * sec_per_unit)
        tip_n.append(count_tips(u, v))
    if step % 10000 == 0:
        print(f"t = {t:.0f}")
    u, v = step_model(u, v, eps_at(t))

fs = 1 / (sample_every * dt * sec_per_unit)  # electrode samples per second
rng = np.random.default_rng(0)
signals = np.array(samples, dtype=np.float32)
signals += 0.02 * rng.standard_normal(signals.shape).astype(np.float32)

# Activation times: upward crossings of 0.5 on each electrode
up = (signals[1:] >= 0.5) & (signals[:-1] < 0.5)
act_frame, act_channel = np.nonzero(up)
order = np.argsort(act_frame)
act_frame, act_channel = act_frame[order] + 1, act_channel[order]

# Wave field movie, synchronized with the other views (frame i is at model
# time i * frame_every * dt)
movie = fpj.LossyVideo(
    data=np.array(frames), fps=30, start_time_sec=0.0,
    sampling_frequency_hz=1 / (frame_every * dt * sec_per_unit),
)

# Electrode array movie with activation markers
mea = fpj.MEAMovie(
    raw_data=np.round(signals * 1000).astype(np.int16),
    electrode_coords=electrode_xy,
    start_time_sec=0.0,
    sampling_frequency_hz=fs,
    spike_channel_indices=act_channel.astype(np.uint16),
    spike_frame_indices=act_frame.astype(np.uint32),
)

# Traces for one row of electrodes across the middle of the array
row = [k for k, (r, c) in enumerate(electrode_rc) if r == centers[n_side // 2]]
traces = vv.TimeseriesGraph(y_label="u (offset per electrode)")
traces.add_uniform_series(
    name="electrodes", start_time_sec=0.0, sampling_frequency_hz=fs,
    data=signals[:, row], channel_names=[f"E{k}" for k in row], channel_spacing=1.2,
)

# Spectrogram of an electrode near the center: 2 s windows every 40 ms
center = row[n_side // 2]
win, hop = int(2 * fs), int(0.04 * fs)
x = signals[:, center] - signals[:, center].mean()
segments = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)
freqs = np.fft.rfftfreq(win, 1 / fs)
keep = freqs <= 30
power = np.abs(np.fft.rfft(segments, axis=1)[:, keep]) ** 2
spectrogram = vv.Spectrogram(
    start_time_sec=win / fs / 2,
    sampling_frequency_hz=1 / (hop / fs),
    frequency_min_hz=0,
    frequency_delta_hz=float(freqs[1]),
    data=np.log10(power + 1e-3).astype(np.float32),
)

# Order parameter: number of spiral tips, alongside the eps ramp
tips = vv.TimeseriesGraph(y_label="Spiral tips", y_range=[0, 1.1 * max(tip_n)])
tips.add_line_series(name="spiral tips", t=np.array(tip_t, dtype=np.float32),
                     y=np.array(tip_n, dtype=np.float32), color="#d62728", width=2)
eps_graph = vv.TimeseriesGraph(y_label="eps", y_range=[0.03, 0.09])
eps_graph.add_line_series(name="eps", t=np.array([0, T_ramp, T_total], dtype=np.float32) * sec_per_unit,
                          y=np.array([eps_start, eps_end, eps_end], dtype=np.float32),
                          color="#1f77b4", width=2)

simulation = vv.Box(direction="vertical", items=[
    vv.LayoutItem(vv.Box(direction="horizontal", items=[
        vv.LayoutItem(movie, title="Wave field u(x, y)"),
        vv.LayoutItem(mea, title="Virtual 8 x 8 electrode array"),
    ]), stretch=1.6),
    vv.LayoutItem(traces, title="Electrodes across the middle row", stretch=1.2),
    vv.LayoutItem(spectrogram, title="Spectrogram, center electrode", stretch=0.8),
    vv.LayoutItem(vv.Box(direction="horizontal", items=[
        vv.LayoutItem(tips, title="Number of spiral tips"),
        vv.LayoutItem(eps_graph, title="Control parameter eps"),
    ]), stretch=1),
])

# Numbers quoted in the explanation, measured from this run
tip_t, tip_n = np.array(tip_t), np.array(tip_n)
t_breakup = tip_t[np.nonzero(tip_n > 5)[0][0]]
eps_breakup = eps_at(t_breakup / sec_per_unit)
before = signals[int(1 * fs):int(7 * fs), center]
spectrum = np.abs(np.fft.rfft(before - before.mean())) ** 2
f_rotation = np.fft.rfftfreq(len(before), 1 / fs)[1 + np.argmax(spectrum[1:])]
tips_after = int(np.median(tip_n[tip_t > 12]))

explanation = vv.Markdown(f"""
# What we are seeing

## The model

An *excitable medium* is one in which each point sits at rest until a
neighbor excites it, then fires once and is refractory for a while before it
can fire again. Cardiac muscle is the classic example. Waves in such a medium
annihilate when they collide rather than passing through each other, and a
wave with a free end curls around that end into a rotating *spiral wave*.

We simulate the Barkley model with the Bar-Eiswirth recovery term, a standard
two-variable caricature of excitable tissue:

- du/dt = u (1 - u) (u - (v + b) / a) / eps + laplacian(u)
- dv/dt = f(u) - v

Here u is the fast excitation variable (shown in the movie) and v is the slow
recovery variable. The parameter eps sets how much faster excitation is than
recovery. For small eps a single spiral rotates steadily. As eps increases,
the back of the wave becomes unstable and the spiral breaks up into many
short-lived spirals, a regime called spatiotemporal chaos. In cardiac tissue,
spiral breakup of this kind is thought to underlie the transition from
tachycardia (one rotating wave) to fibrillation (many).

In this run eps is ramped slowly from {eps_start} to {eps_end} over the first
{T_ramp * sec_per_unit:.0f} s and then held for {(T_total - T_ramp) * sec_per_unit:.0f} s.

## The panels

- **Wave field.** The excitation u over the whole 256 x 256 domain. Bright
  bands are wavefronts. The current eps is also written on each frame.
- **Virtual electrode array.** An 8 x 8 grid of electrodes, each reporting the
  average of u over a small patch, with a little added noise. Markers show
  activation times (upward crossings of u = 0.5).
- **Electrodes across the middle row.** The same signals as traces for the 8
  electrodes in one row.
- **Spectrogram.** Power spectrum over time for an electrode near the center.
- **Number of spiral tips.** A tip is where the u = 0.5 and v = a/2 - b contours
  cross. A single spiral has one tip.
- **Control parameter eps.** The ramp described above.

All panels share one timeline. Playing the wave field movie or the electrode
array movie moves the time cursor in the other views, and clicking a time in
the traces or spectrogram moves both movies to that time.

## What to look for

- Before breakup there is a single spiral. Every electrode fires periodically,
  at about {f_rotation:.1f} Hz, and the spectrogram shows a sharp line at that
  frequency (with a weaker harmonic).
- Breakup begins at about t = {t_breakup:.1f} s (eps = {eps_breakup:.3f}),
  where the tip count first rises above 5. Within a few seconds it settles
  around {tips_after} tips.
- After breakup the electrode traces become irregular and the spectrogram
  line spreads into a broad band.

## Caveats

This is a caricature, not a model of any particular tissue. Time is in model
units; we display one unit as 20 ms, an arbitrary choice that places the
rotation in a physiologically plausible range. The electrode signals are local
averages of u, not simulated extracellular potentials. The moment of breakup
depends on the ramp rate, the domain size and the numerics (forward Euler,
dt = {dt}, grid spacing {dx}, no-flux boundaries).
""")

view = vv.TabLayout(items=[
    vv.TabLayoutItem(simulation, label="Simulation"),
    vv.TabLayoutItem(explanation, label="What we are seeing"),
])

description = """
Simulation of the Bar-Eiswirth variant of the Barkley model of an excitable
medium (256 x 256 grid, forward Euler, dt = 0.01). The parameter eps is ramped
from 0.04 to 0.08 over the first 600 time units and then held for 200 more. A
single rotating spiral breaks up into spatiotemporal chaos. Electrodes average
u over 5 x 5 grid patches, with small added noise. For display, one model time
unit is shown as 20 ms (an arbitrary choice).
"""

view.show(
    title="Spiral wave breakup in an excitable medium",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
