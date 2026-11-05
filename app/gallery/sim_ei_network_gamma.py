"""
Gamma oscillations in a simulated network of 5000 spiking neurons.

A randomly connected network of 4000 excitatory (E) and 1000 inhibitory (I)
leaky integrate-and-fire neurons, with exponentially decaying synaptic
currents and 1 to 2 ms transmission delays. Each neuron also receives a noisy
external drive. At low drive the network fires sparsely and irregularly with
no dominant rhythm. When the drive increases, E and I cells lock into an
alternating rhythm in the gamma range (E cells excite I cells, which then
silence both populations for a few milliseconds; the PING mechanism), and
during a slow ramp of the drive the rhythm speeds up.

The figure shows a raster of 100 neurons, the population rates, an LFP proxy
(the summed synaptic currents onto E cells) and its spectrogram, membrane
potentials of four neurons, and the drive. Everything here is synthetic.
"""

import numpy as np
import numba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import figpack.views as vv
import figpack_spike_sorting.views as ssv

rng = np.random.default_rng(0)
NE, NI = 4000, 1000
N = NE + NI
CE, CI = 400, 100  # excitatory and inhibitory inputs per neuron
J_EE, J_IE, J_EI, J_II = 0.5, 2.0, -2.0, -2.0  # mV jump in synaptic current; J_IE is E onto I
tau_m = np.r_[np.full(NE, 20.0), np.full(NI, 10.0)]  # membrane time constants (ms)
tau_syn_e, tau_syn_i = 2.0, 5.0  # synaptic current decay (ms)
V_th, V_reset = 20.0, 10.0  # mV relative to rest
ref_steps = np.r_[np.full(NE, 20), np.full(NI, 10)].astype(np.int32)  # 2 ms (E), 1 ms (I)
sigma = 5.0  # external noise (mV)
dt = 0.1  # ms
T = 10000.0  # ms
n_steps = int(T / dt)

# Fixed in-degree random connectivity, stored as outgoing lists per neuron
pre, post, w = [], [], []
for i in range(N):
    pre += [rng.choice(NE, CE, replace=False), NE + rng.choice(NI, CI, replace=False)]
    post += [np.full(CE, i), np.full(CI, i)]
    w += [np.full(CE, J_EE if i < NE else J_IE), np.full(CI, J_EI if i < NE else J_II)]
pre, post, w = np.concatenate(pre), np.concatenate(post), np.concatenate(w)
order = np.argsort(pre, kind="stable")
pre, post, w = pre[order], post[order].astype(np.int32), w[order]
ptr = np.searchsorted(pre, np.arange(N + 1)).astype(np.int64)
delay = np.round(rng.uniform(1.0, 2.0, len(post)) / dt).astype(np.int32)

# External drive to E cells (mV); I cells get 90% of it.
# 0-2 s baseline, 2-3.5 s step, 3.5-5 s baseline, 5-9 s slow ramp, 9-10 s baseline
t_ms = np.arange(n_steps) * dt
drive = np.full(n_steps, 12.0)
drive[(t_ms >= 2000) & (t_ms < 3500)] = 30.0
ramp = (t_ms >= 5000) & (t_ms < 9000)
drive[ramp] = 12.0 + (40.0 - 12.0) * (t_ms[ramp] - 5000) / 4000
recorded = np.array([0, 1, NE, NE + 1])  # membrane potentials of 2 E and 2 I cells


@numba.njit
def simulate(drive, ptr, post, w, delay, tau_m, ref_steps, recorded, seed):
    np.random.seed(seed)
    D = 32  # ring buffer length for delayed input (steps)
    V = np.random.rand(N) * V_th
    I_e = np.zeros(N)
    I_i = np.zeros(N)
    ref = np.zeros(N, np.int32)
    buf_e = np.zeros((D, N))
    buf_i = np.zeros((D, N))
    spike_step, spike_cell = [], []
    lfp = np.zeros(n_steps)
    vm = np.zeros((n_steps, len(recorded)))
    for s in range(n_steps):
        k = s % D
        I_e += buf_e[k]
        I_i += buf_i[k]
        buf_e[k] = 0.0
        buf_i[k] = 0.0
        acc = 0.0
        for i in range(N):
            if i < NE:
                acc += abs(I_e[i]) + abs(I_i[i])
            I_e[i] -= dt * I_e[i] / tau_syn_e
            I_i[i] -= dt * I_i[i] / tau_syn_i
            if ref[i] > 0:
                ref[i] -= 1
                continue
            mu = drive[s] if i < NE else 0.9 * drive[s]
            V[i] += dt * (-V[i] + mu + I_e[i] + I_i[i]) / tau_m[i] + sigma * np.sqrt(dt / tau_m[i]) * np.random.randn()
            if V[i] >= V_th:
                V[i] = V_reset
                ref[i] = ref_steps[i]
                spike_step.append(s)
                spike_cell.append(i)
                for q in range(ptr[i], ptr[i + 1]):
                    kk = (s + delay[q]) % D
                    if w[q] > 0:
                        buf_e[kk, post[q]] += w[q]
                    else:
                        buf_i[kk, post[q]] += w[q]
        lfp[s] = acc / NE
        for r in range(len(recorded)):
            j = recorded[r]
            vm[s, r] = V[j]
        # draw spikes on the recorded traces
        for q in range(len(spike_step) - 1, -1, -1):
            if spike_step[q] != s:
                break
            for r in range(len(recorded)):
                if recorded[r] == spike_cell[q]:
                    vm[s, r] = 40.0
    return np.array(spike_step), np.array(spike_cell), lfp, vm


spike_step, spike_cell, lfp_full, vm = simulate(drive, ptr, post, w, delay, tau_m, ref_steps, recorded, 1)
spike_t = spike_step * dt / 1000  # s
T_s = T / 1000
print(f"{len(spike_t)} spikes")

# Population rates in 1 ms bins (Hz per neuron), lightly smoothed
nb = int(T)
rate_e = np.bincount((spike_t[spike_cell < NE] * 1000).astype(int), minlength=nb)[:nb] / NE * 1000
rate_i = np.bincount((spike_t[spike_cell >= NE] * 1000).astype(int), minlength=nb)[:nb] / NI * 1000
rate_e = ndimage.gaussian_filter1d(rate_e.astype(float), 1)
rate_i = ndimage.gaussian_filter1d(rate_i.astype(float), 1)

# LFP proxy at 1 kHz
lfp = lfp_full.reshape(-1, 10).mean(axis=1)
lfp_fs = 1000.0

# ---------------------------------------------------------------------------
# Views
shown = np.r_[np.arange(80), NE + np.arange(20)]  # 80 E and 20 I cells
trains = {c: spike_t[spike_cell == c] for c in shown}
raster = ssv.RasterPlot(
    start_time_sec=0.0, end_time_sec=T_s,
    plots=[ssv.RasterPlotItem(unit_id=k + 1, spike_times_sec=trains[c].astype(np.float32)) for k, c in enumerate(shown)],
)

rates = vv.TimeseriesGraph(y_label="Rate (Hz)", hide_nav_toolbar=True)
rates.add_uniform_series(name="I population", start_time_sec=0.0, sampling_frequency_hz=1000.0,
                         data=rate_i.astype(np.float32), colors=["#d62728"])
rates.add_uniform_series(name="E population", start_time_sec=0.0, sampling_frequency_hz=1000.0,
                         data=rate_e.astype(np.float32), colors=["#1f77b4"])

lfp_graph = vv.TimeseriesGraph(y_label="LFP proxy", hide_nav_toolbar=True)
lfp_graph.add_uniform_series(name="LFP proxy", start_time_sec=0.0, sampling_frequency_hz=lfp_fs,
                             data=lfp.astype(np.float32), colors=["#2a2a2a"])

# Spectrogram: 250 ms windows every 10 ms, power relative to the median at
# each frequency during the first baseline period (log10)
f_spec, t_spec, sxx = signal.spectrogram(lfp - lfp.mean(), fs=lfp_fs, nperseg=250, noverlap=240, window="hann")
keep = (f_spec > 0) & (f_spec <= 150)
base = np.median(sxx[keep][:, t_spec < 2.0], axis=1, keepdims=True)
spectrogram = vv.Spectrogram(
    start_time_sec=float(t_spec[0]),
    sampling_frequency_hz=float(1 / (t_spec[1] - t_spec[0])),
    frequency_min_hz=float(f_spec[keep][0]),
    frequency_delta_hz=float(f_spec[1]),
    data=np.clip(np.log10(sxx[keep].T / base.T), 0, None).astype(np.float32),
    hide_nav_toolbar=True,
)

vm_graph = vv.TimeseriesGraph(y_label="Vm (mV, offset)", hide_nav_toolbar=True)
vm_graph.add_uniform_series(name="Vm", start_time_sec=0.0, sampling_frequency_hz=1000 / dt,
                            data=vm.astype(np.float32), channel_names=["E cell 1", "E cell 2", "I cell 1", "I cell 2"],
                            colors=["#1f77b4", "#4a98d0", "#d62728", "#e86a6b"], channel_spacing=50)

drive_graph = vv.TimeseriesGraph(y_label="Drive (mV)", y_range=[0, 45], hide_nav_toolbar=True)
drive_graph.add_uniform_series(name="drive to E cells", start_time_sec=0.0, sampling_frequency_hz=100.0,
                               data=drive[::100].astype(np.float32), colors=["#2a9d8f"], width=2)

# Static summary: LFP spectra at baseline and during the step, and the
# gamma peak frequency against drive during the ramp
plt.rcParams["font.size"] = 11
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for (a, b), label, col in [((0.3, 2.0), "baseline (drive 12 mV)", "#888888"), ((2.3, 3.5), "step (drive 30 mV)", "#e76f51")]:
    x = lfp[int(a * lfp_fs):int(b * lfp_fs)]
    f, p = signal.welch(x - x.mean(), fs=lfp_fs, nperseg=256)
    axes[0].semilogy(f, p, color=col, label=label, lw=2)
axes[0].set_xlim(0, 200)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("LFP proxy power")
axes[0].legend(frameon=False)
axes[0].set_title("Spectra")
centers, peak_f, drives = np.arange(5.25, 8.8, 0.25), [], []
for c in centers:
    x = lfp[int((c - 0.25) * lfp_fs):int((c + 0.25) * lfp_fs)]
    f, p = signal.welch(x - x.mean(), fs=lfp_fs, nperseg=256, nfft=1024)
    sel = (f > 20) & (f < 150)
    peak_f.append(f[sel][np.argmax(p[sel])])
    drives.append(drive[int(c * 1000 / dt)])
axes[1].plot(drives, peak_f, "o-", color="#264653")
axes[1].set_xlabel("Drive (mV)")
axes[1].set_ylabel("Peak frequency (Hz)")
axes[1].set_title("Rhythm speeds up during the ramp")
fig.tight_layout()
summary = vv.MatplotlibFigure(fig)


def cv_isi(st):
    isi = np.diff(st)
    return float(np.std(isi) / np.mean(isi)) if len(isi) > 2 else float("nan")


table = ssv.UnitsTable(
    columns=[
        ssv.UnitsTableColumn(key="type", label="Type", dtype="str"),
        ssv.UnitsTableColumn(key="rate", label="Rate (Hz)", dtype="float"),
        ssv.UnitsTableColumn(key="cv", label="CV of ISI", dtype="float"),
    ],
    rows=[ssv.UnitsTableRow(unit_id=k + 1, values={
        "type": "E" if c < NE else "I",
        "rate": round(len(trains[c]) / T_s, 2),
        "cv": round(cv_isi(trains[c]), 2)}) for k, c in enumerate(shown)],
)

view = vv.Splitter(
    direction="horizontal",
    item1=vv.LayoutItem(vv.Box(direction="vertical", items=[
        vv.LayoutItem(raster, title="Spikes of 80 E cells (1-80) and 20 I cells (81-100)", stretch=3),
        vv.LayoutItem(rates, title="Population rates (E blue, I red)", stretch=1),
        vv.LayoutItem(spectrogram, title="LFP proxy spectrogram (power relative to baseline)", stretch=1),
        vv.LayoutItem(lfp_graph, title="LFP proxy (summed synaptic currents onto E cells)", stretch=0.8),
        vv.LayoutItem(vm_graph, title="Membrane potentials", stretch=1.2),
        vv.LayoutItem(drive_graph, title="External drive", stretch=0.6),
    ])),
    item2=vv.LayoutItem(vv.Box(direction="vertical", items=[
        vv.LayoutItem(summary, title="Summary", stretch=1),
        vv.LayoutItem(table, title="Neurons shown in the raster", stretch=1.2),
    ])),
    split_pos=0.72,
)

description = """
Synthetic simulation. 4000 excitatory and 1000 inhibitory leaky
integrate-and-fire neurons (membrane time constants 20 and 10 ms, threshold
20 mV, reset 10 mV, refractory 2 and 1 ms), each receiving 400 E and 100 I
inputs chosen at random, with exponential synaptic currents (2 ms E, 5 ms I)
and delays of 1 to 2 ms. External drive: mean plus white noise (sigma 5 mV),
with a step from 12 to 30 mV at 2 to 3.5 s and a ramp from 12 to 40 mV over
5 to 9 s. Euler-Maruyama integration, dt = 0.1 ms. The LFP proxy is the mean
absolute synaptic current onto E cells.
"""

view.show(
    title="Gamma oscillations in a spiking E/I network (simulated)",
    description=description,
    script=open(__file__).read(),
    upload=True,
)
