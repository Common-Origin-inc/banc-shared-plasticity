"""
Shared constants and ring-bump analysis for BANC_BEST_BRAIN.

Neuron and synapse constants are the static v2 model's (inherited from Shiu et al. 2024 /
Kakaria & de Bivort 2017); the Kim et al. 2017 protocol timing is the one used by every test.
The analysis helpers decode the heading bump from the 45 ring EPG neurons (population vector,
localisation R, width, persistence). `CX` is an alias of this module so callers written against
the original two-module layout (`BS.CX.decode`, ...) keep working.
"""
import sys

import numpy as np
from brian2 import mV, ms, uF, cm

# ---- neuron (universal) ----
v_0, v_rst, v_th = -52 * mV, -52 * mV, -45 * mV        # rest, reset, threshold
t_mbr, tau_syn, t_rfc = 20 * ms, 5 * ms, 2.2 * ms       # membrane tau, synaptic tau, refractory
E_exc, E_inh = 0 * mV, -70 * mV                         # reversal potentials
C_M = 1.0 * uF / cm ** 2                                # specific capacitance; C_i = C_M * area_i
G_L_SPEC = C_M / t_mbr                                  # leak per area, so tau_m = 20 ms for all

# ---- Shiu et al. 2024 baseline (model.py default_params; plasticity.build_network model="shiu") ----
W_SYN_SHIU, T_DLY_SHIU = 0.275 * mV, 1.8 * ms           # voltage jump per synapse, synaptic delay
F_POI_SHIU = 250                                        # their Poisson-input scaling: an input event jumps v by 250 w_syn

# ---- drive ----
STIM_HZ = 150.0            # Poisson rate of a stimulus drive (x synapses-per-spike sets the mean)
SENS_KICK_F = 250          # synapses' worth per spontaneous event: one event = one sensory spike

# ---- Kim et al. 2017 protocol (Fig. 2C): one 45-deg EB sector for 1.047 s, then darkness ----
STIM_ON, STIM_OFF, T_END = 0.5, 0.5 + 1.047, 4.0

# ---- analysis ----
DT_BIN_MS, SMOOTH_MS, ACTIVE_HZ = 5.0, 50.0, 2.0


def bin_rates(st, si, n, duration_s, idx=None):
    """[len(idx), n_bins] Hz from spike times/indices; idx = which neurons (default all)."""
    n_bins = int(round(duration_s / (DT_BIN_MS / 1000)))
    if idx is None:
        idx = np.arange(n)
    pos = -np.ones(n, dtype=np.int64); pos[idx] = np.arange(len(idx))
    k = (pos[si] >= 0) & (st < duration_s)
    out = np.zeros((len(idx), n_bins))
    np.add.at(out, (pos[si[k]], np.floor(st[k] / (DT_BIN_MS / 1000)).astype(int)), 1)
    return out * (1000.0 / DT_BIN_MS)


def bump_fwhm_deg(profile):
    """Kim et al.'s bump width: contiguous EPGs above half maximum around the peak, x 360/45."""
    n = len(profile)
    if profile.max() <= 0:
        return 0.0
    k = int(np.argmax(profile)); half = 0.5 * profile[k]
    w = 1
    for step in (1, -1):
        j = k + step
        while w < n and profile[j % n] >= half:
            w += 1; j += step
    return w * 360.0 / n


def smooth(rates, window_ms=SMOOTH_MS):
    w = max(1, int(round(window_ms / DT_BIN_MS)))
    return np.apply_along_axis(lambda r: np.convolve(r, np.ones(w) / w, mode="same"), 1, rates)


def decode(rates_epg, angles_epg):
    """population vector: decoded angle (rad) and localisation R in [0, 1] per time bin"""
    z = (np.exp(1j * angles_epg)[:, None] * rates_epg).sum(axis=0)
    tot = rates_epg.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(tot > 0, np.abs(z) / tot, 0.0)
    return np.angle(z), r


def participation_ratio(rates_epg):
    s1, s2 = rates_epg.sum(axis=0), (rates_epg ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s2 > 0, s1 ** 2 / s2, 0.0)


def jaccard_vs_reference(rates_epg, t_axis, t_ref, thresh=ACTIVE_HZ):
    active = rates_epg > thresh
    ref = active[:, int(np.argmin(np.abs(t_axis - t_ref)))]
    inter = (active & ref[:, None]).sum(axis=0)
    union = (active | ref[:, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def analyse(rates, epg, order, angles, t_axis):
    """bump metrics after light-off (module STIM_ON / STIM_OFF / T_END give the protocol times)"""
    sm = smooth(rates)
    r_epg = sm[epg]
    theta, R = decode(sm[order], angles[order])
    pr = participation_ratio(r_epg)
    jac = jaccard_vs_reference(r_epg, t_axis, STIM_OFF)
    after = t_axis > STIM_OFF + 0.05
    live = after & (r_epg.sum(axis=0) > ACTIVE_HZ)        # shape metrics only where the ring is active
    at = lambda arr, t: float(arr[int(np.argmin(np.abs(t_axis - t)))])
    below = np.flatnonzero(after & (jac < 0.5))
    persist = float(t_axis[below[0]] - STIM_OFF) if len(below) else float(T_END - STIM_OFF)
    per_neuron = rates[epg][:, after].mean(1)
    return {
        "smooth": sm, "theta": theta, "R": R, "pr": pr, "jaccard": jac,
        "epg_hz_after": float(per_neuron.mean()),
        "active_hz_after": float(np.mean(per_neuron[per_neuron > ACTIVE_HZ]) if (per_neuron > ACTIVE_HZ).any() else 0.0),
        "n_active_after": int((per_neuron > ACTIVE_HZ).sum()),
        "epg_hz_before": float(rates[epg][:, t_axis < STIM_ON].mean()),
        "R_after": float(np.nanmean(R[live])) if live.any() else 0.0,
        "pr_after": float(np.nanmean(pr[live])) if live.any() else 0.0,
        "live_frac": float(live.sum() / after.sum()),
        "jaccard_+1.0": at(jac, STIM_OFF + 1.0),
        "jaccard_+2.0": at(jac, STIM_OFF + 2.0),
        "persist_s": persist,
    }


CX = sys.modules[__name__]
