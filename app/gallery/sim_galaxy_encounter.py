"""
Tidal tails from a close encounter of two disk galaxies (restricted N-body).

Following Toomre and Toomre (1972), each galaxy is a softened point mass
surrounded by a disk of massless test stars on circular orbits. The two cores
attract each other and move on a bound, eccentric orbit; the stars feel both
cores but not each other. One disk lies in the orbital plane, the other is
tilted by 60 degrees. The close passages pull out long tidal tails and a
bridge of stars between the galaxies.

Everything is integrated with the kick-drift-kick leapfrog (symplectic), so
the energy of the two-core orbit is conserved up to a small bounded error.
Units: G = 1, core mass 1, initial disk radius 1.
"""

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import figpack.views as vv
import figpack_experimental.views as fpj

rng = np.random.default_rng(3)
M = 1.0  # mass of each core
soft = 0.1  # Plummer softening length
n_stars = 8000  # per galaxy
dt = 0.005
T = 80.0
frame_every = 20  # steps between movie frames (0.1 time units)
t_snapshot = 15.0  # 3D snapshot, after the first passage

# ---------------------------------------------------------------------------
# Core orbit: bound ellipse with eccentricity 0.8 and pericenter distance 1.2,
# starting at separation 6 on the way in
mu, e, q = 2 * M, 0.8, 1.2
p = q * (1 + e)
r0 = 6.0
f0 = -np.arccos((p / r0 - 1) / e)  # true anomaly (negative: approaching)
rel_x = r0 * np.array([np.cos(f0), np.sin(f0), 0.0])
rel_v = np.sqrt(mu / p) * np.array([-np.sin(f0), e + np.cos(f0), 0.0])
core_x = np.array([-0.5 * rel_x, 0.5 * rel_x])
core_v = np.array([-0.5 * rel_v, 0.5 * rel_v])


def make_disk(center, velocity, tilt_deg):
    # Radii uniform in [0.2, 1] (surface density ~ 1/r), circular velocities
    # for the softened potential, rotating in the same sense as the orbit
    r = rng.uniform(0.2, 1.0, n_stars)
    phi = rng.uniform(0, 2 * np.pi, n_stars)
    vc = np.sqrt(M * r**2 / (r**2 + soft**2) ** 1.5)
    x = np.stack([r * np.cos(phi), r * np.sin(phi), np.zeros(n_stars)], axis=1)
    v = np.stack([-vc * np.sin(phi), vc * np.cos(phi), np.zeros(n_stars)], axis=1)
    a = np.radians(tilt_deg)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return x @ rot.T + center, v @ rot.T + velocity


xa, va = make_disk(core_x[0], core_v[0], 0)
xb, vb = make_disk(core_x[1], core_v[1], 60)
xs = np.concatenate([xa, xb])
vs = np.concatenate([va, vb])
origin = np.repeat([0, 1], n_stars)


def accel_from_cores(x, cores):
    a = np.zeros_like(x)
    for c in cores:
        d = c - x
        a += M * d / ((d**2).sum(axis=1, keepdims=True) + soft**2) ** 1.5
    return a


def core_accel(cx):
    d = cx[1] - cx[0]
    f = M * d / (d @ d + soft**2) ** 1.5
    return np.array([f, -f])


def core_energy(cx, cv):
    d = cx[1] - cx[0]
    return 0.5 * M * (cv**2).sum() - M * M / np.sqrt(d @ d + soft**2)


def bound_state(x, v, cx, cv):
    # 0: bound to core A, 1: bound to core B, 2: unbound (specific energy
    # relative to each core, ignoring the other)
    eps = []
    for k in range(2):
        d = x - cx[k]
        eps.append(0.5 * ((v - cv[k]) ** 2).sum(axis=1) - M / np.sqrt((d**2).sum(axis=1) + soft**2))
    eps = np.array(eps)
    state = np.argmin(eps, axis=0)
    state[eps.min(axis=0) >= 0] = 2
    return state


# ---------------------------------------------------------------------------
# Rendering: face-on (x-y) and tilted views, stars colored by galaxy of origin
W = 544  # pixels per panel
extent = 9.0
col_a = np.array([0.35, 0.7, 1.0])
col_b = np.array([1.0, 0.65, 0.3])
font = ImageFont.load_default(size=20)
tilt = np.radians(-70)
view_rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])


def render_panel(p2, cores2):
    rgb = np.zeros((W, W, 3))
    for g, col in [(0, col_a), (1, col_b)]:
        h = np.histogram2d(p2[origin == g, 1], p2[origin == g, 0], bins=W,
                           range=[[-extent, extent], [-extent, extent]])[0]
        h = ndimage.gaussian_filter(h, 0.8)
        rgb += np.log1p(8 * h)[..., None] * col
    for c in cores2:
        yy, xx = np.mgrid[:W, :W]
        cx, cy = (c[0] + extent) / (2 * extent) * W, (c[1] + extent) / (2 * extent) * W
        rgb += 1.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 18)[..., None]
    rgb = 1 - np.exp(-0.9 * rgb)
    return (np.flipud(rgb) * 255).astype(np.uint8)


def render(x, cx, t):
    face = render_panel(x[:, :2], cx[:, :2])
    tilted = render_panel((x @ view_rot.T)[:, :2], (cx @ view_rot.T)[:, :2])
    img = Image.fromarray(np.concatenate([face, tilted], axis=1))
    d = ImageDraw.Draw(img)
    d.line([(W, 0), (W, W)], fill=(70, 70, 70), width=2)
    d.text((10, 8), f"t = {t:.1f}", fill=(255, 255, 255), font=font)
    d.text((10, W - 30), "face-on (orbital plane)", fill=(180, 180, 180), font=font)
    d.text((W + 12, W - 30), "tilted 70 degrees", fill=(180, 180, 180), font=font)
    return np.array(img)


# ---------------------------------------------------------------------------
# Leapfrog (kick-drift-kick)
n_steps = int(round(T / dt))
frames, rec_t, separation, energy, fractions = [], [], [], [], []
E0 = core_energy(core_x, core_v)
a_s = accel_from_cores(xs, core_x)
a_c = core_accel(core_x)
for step in range(n_steps + 1):
    t = step * dt
    if step % frame_every == 0:
        frames.append(render(xs, core_x, t))
        rec_t.append(t)
        separation.append(np.linalg.norm(core_x[1] - core_x[0]))
        energy.append((core_energy(core_x, core_v) - E0) / abs(E0))
        s = bound_state(xs, vs, core_x, core_v)
        fractions.append([np.mean(s == origin), np.mean((s != origin) & (s < 2)), np.mean(s == 2)])
        if abs(t - t_snapshot) < dt / 2:
            snap_x, snap_core = xs.copy(), core_x.copy()
        if step % 2000 == 0:
            print(f"t = {t:.0f}  separation = {separation[-1]:.2f}")
    if step == n_steps:
        break
    vs += 0.5 * dt * a_s
    core_v += 0.5 * dt * a_c
    xs += dt * vs
    core_x += dt * core_v
    a_s = accel_from_cores(xs, core_x)
    a_c = core_accel(core_x)
    vs += 0.5 * dt * a_s
    core_v += 0.5 * dt * a_c

rec_t = np.array(rec_t, dtype=np.float32)
fractions = np.array(fractions, dtype=np.float32)

# ---------------------------------------------------------------------------
# Views
# Synchronized with the graphs: 1 s on the shared time axis = 1 simulation time unit
movie = fpj.LossyVideo(data=np.array(frames), fps=30, start_time_sec=0.0, sampling_frequency_hz=1 / (frame_every * dt))

# State after the first passage in 3D (a random subset of stars, for responsiveness)
sub = rng.choice(len(xs), 6000, replace=False)
fig3d = go.Figure()
for g, name, col in [(0, "galaxy A (in orbital plane)", "#59b3ff"), (1, "galaxy B (tilted 60 degrees)", "#ffa64d")]:
    k = sub[origin[sub] == g]
    fig3d.add_trace(go.Scatter3d(x=snap_x[k, 0], y=snap_x[k, 1], z=snap_x[k, 2], mode="markers", name=name,
                                 marker=dict(size=1.5, color=col, opacity=0.7)))
fig3d.add_trace(go.Scatter3d(x=snap_core[:, 0], y=snap_core[:, 1], z=snap_core[:, 2], mode="markers", name="cores",
                             marker=dict(size=6, color="white", line=dict(color="black", width=1))))
lim = [-9, 9]
fig3d.update_layout(
    template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), legend=dict(x=0, y=1),
    scene=dict(xaxis=dict(range=lim), yaxis=dict(range=lim), zaxis=dict(range=lim), aspectmode="cube"),
)
snapshot = vv.PlotlyFigure(fig3d)

sep_graph = vv.TimeseriesGraph(y_label="Core separation")
sep_graph.add_line_series(name="separation", t=rec_t, y=np.array(separation, dtype=np.float32), color="#2a9d8f", width=2)

frac_graph = vv.TimeseriesGraph(y_label="Fraction of stars", y_range=[0, 1])
frac_graph.add_line_series(name="bound to own galaxy", t=rec_t, y=fractions[:, 0], color="#264653", width=2)
frac_graph.add_line_series(name="captured by the other", t=rec_t, y=fractions[:, 1], color="#e9c46a", width=2)
frac_graph.add_line_series(name="unbound", t=rec_t, y=fractions[:, 2], color="#e76f51", width=2)

energy_graph = vv.TimeseriesGraph(y_label="Relative energy error")
energy_graph.add_line_series(name="core orbit energy error", t=rec_t, y=np.array(energy, dtype=np.float32), color="#6a4c93", width=1.5)

view = vv.Box(direction="vertical", items=[
    vv.LayoutItem(vv.Box(direction="horizontal", items=[
        vv.LayoutItem(movie, title="Encounter movie (linked to the graphs below)", stretch=2),
        vv.LayoutItem(snapshot, title="t = 15, after the first passage, in 3D (drag to rotate)", stretch=1),
    ]), stretch=2.2),
    vv.LayoutItem(vv.Box(direction="horizontal", items=[
        vv.LayoutItem(sep_graph, title="Distance between the cores"),
        vv.LayoutItem(frac_graph, title="Where the stars end up"),
        vv.LayoutItem(energy_graph, title="Leapfrog energy conservation (core orbit)"),
    ]), stretch=1),
])

description = """
Synthetic simulation in the spirit of Toomre and Toomre (1972). Two softened
point masses (G = M = 1, softening 0.1) on a bound orbit with eccentricity 0.8
and pericenter 1.2, each carrying 8000 massless test stars on circular orbits
between radius 0.2 and 1. Galaxy A's disk lies in the orbital plane, galaxy
B's is tilted by 60 degrees; both rotate in the same sense as the orbit.
Kick-drift-kick leapfrog, dt = 0.005, t = 0 to 80. On the shared time axis, 1 s corresponds to one simulation time unit, and the movie is synchronized with the graphs. A star is counted as bound
to a core if its specific energy relative to that core alone is negative.
Because the stars have no mass and the cores feel no dynamical friction, the
galaxies do not merge; real encounters lose orbital energy to their dark
matter halos.
"""

view.show(
    title="Tidal tails in a galaxy encounter (simulated)",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
