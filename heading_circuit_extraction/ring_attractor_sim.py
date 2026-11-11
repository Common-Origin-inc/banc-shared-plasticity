"""BANC ring attractor — Experiment 1 of Kim et al. 2017 (localised E-PG
stimulation, then release) on the raw connectome.

THE MODEL
─────────
Conductance-based LIF, one compartment per neuron. Per-neuron quantities come
from BANC; everything else is a universal constant. No cell-type parameters.

  membrane   C_i dV/dt = g_L,i (V_rest - V) + g_e (E_exc - V) + g_i (E_inh - V)
             C_i   = c_m * A_i          c_m = 1 uF/cm^2, A_i = BANC membrane area
             g_L,i = C_i / tau_m        tau_m = 20 ms (Shiu / Kakaria)
             reset V = V_rest           standard LIF; Shiu's `g = 0` reset is NOT
                                        used -- it deletes queued input on every
                                        spike and kills any recurrent loop
  synapse    j spikes -> g_e[i] += n_ji * G_syn   (ACh)
                      or g_i[i] += n_ji * G_syn   (GABA / glutamate)
             decay tau_syn = 5 ms, E_exc = 0 mV, E_inh = -70 mV
             n_ji = BANC synapse count, transmitter = BANC NT prediction
  G_syn      20 pS per synapse, the one free constant (connectome gives counts,
             not siemens). Bump identical at 20 and 40 pS.

What each ingredient does (ablation.py): the reversal potential is what makes
the bump exist -- inhibition saturates as the neuron depolarises, excitation
does not; a voltage-jump synapse fails with or without area. Area sets width
and robustness (4 flickering neurons without it, 10 stable with it).

WHAT IT DOES
────────────
Stimulate 8 contiguous EPG for 0.5 s, remove the stimulus: a bump of ~10 EPG,
contiguous on the recovered ring, holds its position with zero external
input for the rest of the run, every seed. The network never sees the ring
angles. That is the attractor.

WHAT IT DOES NOT DO
───────────────────
The bump neurons fire at ~200 Hz, the refractory ceiling; real EPGs fire at
tens of Hz. Tests 6-8 showed that uniform adaptation, per-transmitter
receptor timescales (GABA 15 / Glu 30 ms), a slow ACh component (up to
100 ms), and their combinations all move along the same rate/stability
tradeoff without breaking it: below ~110 Hz the bump loses position or dies.
The cause is that a deterministic, noise-free, homogeneous loop can only run
as an all-or-nothing synchronous oscillator; a graded low-rate state needs an
asynchronous (fluctuation-driven) regime, i.e. tonic background. That is left
for the whole-brain stage. Adaptation and receptor channels remain in build()
for ablation, off by default.

GEOMETRY IS ANALYSIS-ONLY. The arc seed uses recovered angles because the
experiment being replicated is "illuminate a contiguous region"; the network
itself never sees them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import *

OUT = Path(__file__).resolve().parent
prefs.codegen.target = "cython"

# ---- membrane: Shiu / Kakaria values where they are specific quantities ----
v_0, v_rst, v_th = -52 * mV, -52 * mV, -45 * mV
t_mbr, tau_syn, t_rfc = 20 * ms, 5 * ms, 2.2 * ms     # tau_syn: Shiu's, uniform
E_exc, E_inh = 0 * mV, -70 * mV
C_M = 1.0 * uF / cm ** 2               # universal specific capacitance
G_L_SPEC = C_M / t_mbr                 # 0.05 mS/cm^2, so tau_m = 20 ms for all

# ---- the ONE free constant: conductance per synapse ----
# 5-10 pS silent; 12-16 pS 55-150 Hz but drifting/dying; 18-40 pS stable at
# the ceiling with identical shape; 80 pS too wide. Shiu's 0.275 mV on a
# median EPG (g_L 2.8 nS, 52 mV driving force) is ~15 pS.
G_SYN_PS = 20.0

# ---- receptor channels (build() supports one conductance per transmitter;
# the model uses the same 5 ms for all, which is Shiu's value and what works) ----
CHANNELS = ["ach", "gaba", "glu"]
E_REV = {"ach": E_exc, "gaba": E_inh, "glu": E_inh}
TAU_UNIFORM = {"ach": 5.0, "gaba": 5.0, "glu": 5.0}          # ms  <- the model
TAU_PER_NT = {"ach": 5.0, "gaba": 15.0, "glu": 30.0}          # ms, literature;
#   tested (Test 7): 147 Hz and LESS stable. Kept for ablation only.
NT_TO_CHANNEL = {"acetylcholine": "ach", "gaba": "gaba", "glutamate": "glu"}
#   neuromodulators (tyramine, octopamine, serotonin; 76 upstream) -> ach

# ---- spike-frequency adaptation (off; kept for ablation) ----
# Tests 6/8: walks the rate down along the rate/stability tradeoff, does not
# break it. E_K, tau_a literature; b_adapt is a conductance so it scales
# with 1/A_i like everything else.
E_K = -80 * mV
tau_a = 200 * ms
B_ADAPT_NS = 0.0

F_POI = 250                            # stimulus: one drive spike = 250 synapses
R_POI = 150.0                          # Hz, Shiu's stimulation rate

N_SEEDS = 3
ARC_SIZE = 8
STIM_ON, STIM_OFF, T_END = 0.5, 1.0, 3.5
DT_BIN_MS, SMOOTH_MS, ACTIVE_HZ = 5.0, 50.0, 2.0
MAX_ACTIVE_HZ = 60.0                   # bump neurons above this = ceiling


# ==========================================================================
# network
# ==========================================================================
def channel_of(nt):
    """Receptor channel for each neuron's predicted transmitter."""
    return np.array([NT_TO_CHANNEL.get(str(x), "ach") for x in nt])


def build(w_matrix, area_nm2, g_syn_ps, stim_schedule, duration_s,
          drive_dt=0.005, uniform_leak=False, voltage_synapse=False,
          b_adapt_ns=0.0, nt=None, tau_table=None, charge_fixed=True):
    """Conductance-based LIF on the BANC matrix.

    With the defaults this builds THE MODEL (see module docstring):
        tau_m dV/dt = (V_rest - V) + [g_e (E_exc - V) + g_i (E_inh - V)] / g_L,i
        dg_e/dt = -g_e / tau_syn ;  dg_i/dt = -g_i / tau_syn
        j spikes -> g_e[i] += n_ji G_syn (ACh) or g_i[i] += n_ji G_syn (GABA/Glu)
    g_syn_ps    : G_syn, conductance per synapse.
    nt          : per-neuron transmitter (npz 'nt'); decides E vs I per edge.
                  None = use the sign of the count (same thing).
    Everything else is for ablation.py and is off by default:
    tau_table   : {channel: ms}; unequal values switch to one conductance per
                  transmitter (ach / gaba / glu), each with its own decay.
    charge_fixed: in that case, scale each channel's G by 5 ms / tau.
    b_adapt_ns  : > 0 adds a spike-triggered K+ conductance (adaptation).
      uniform_leak    : every neuron gets the median area, so g_L is the
                        same for all -- removes the morphology dependence
      voltage_synapse : Shiu's synapse (g in volts, added straight to the
                        drive, no reversal potential); g_syn_ps is then read
                        as W_SYN in mV, scaled by g_L_median / g_L_i so
                        area can still act on the jump when uniform_leak is
                        False
    """
    n = w_matrix.shape[0]
    area = np.full(n, np.median(area_nm2)) if uniform_leak else area_nm2
    g_L = (area * 1e-18) * meter ** 2 * G_L_SPEC              # per neuron
    taus = dict(TAU_UNIFORM if tau_table is None else tau_table)
    ns = dict(NAMESPACE, **{f"tau_{c}": taus[c] * ms for c in CHANNELS},
              **{f"E_{c}": E_REV[c] for c in CHANNELS})

    pre, post = np.nonzero(w_matrix)
    counts = w_matrix[pre, post]
    # route each edge to the receptor channel of its presynaptic neuron; with
    # no transmitter array fall back to the sign of the count
    if nt is None:
        edge_ch = np.where(counts > 0, "ach", "gaba")
    else:
        edge_ch = channel_of(nt)[pre]
    syn_objs = []

    # THE MODEL: two conductances, E and I, same decay; no adaptation term.
    # The three-channel / adaptation variants below exist only for ablation
    # (tau_table with unequal values, or b_adapt_ns > 0).
    split_channels = len(set(taus.values())) > 1
    adapt = b_adapt_ns > 0

    if not voltage_synapse and not split_channels and not adapt:
        eqs = """
        dv/dt = (v_0 - v + (g_e*(E_exc - v) + g_i*(E_inh - v)) / g_L) / t_mbr : volt (unless refractory)
        dg_e/dt = -g_e / tau_syn : siemens
        dg_i/dt = -g_i / tau_syn : siemens
        g_L : siemens (constant)
        """
        G = NeuronGroup(n, eqs, threshold="v > v_th", reset="v = v_rst",
                        refractory=t_rfc, method="exponential_euler", name="cx",
                        namespace=dict(ns, tau_syn=taus["ach"] * ms))
        G.v, G.g_e, G.g_i = v_0, 0 * siemens, 0 * siemens
        G.g_L = g_L
        w_unit = g_syn_ps * psiemens
        exc = edge_ch == "ach"
        s_e = Synapses(G, G, "w : siemens", on_pre="g_e_post += w", name="rec_e")
        s_e.connect(i=pre[exc], j=post[exc]); s_e.w = np.abs(counts[exc]) * w_unit
        s_i = Synapses(G, G, "w : siemens", on_pre="g_i_post += w", name="rec_i")
        s_i.connect(i=pre[~exc], j=post[~exc]); s_i.w = np.abs(counts[~exc]) * w_unit
        syn_objs = [s_e, s_i]
        stim_target = "g_e_post += w_ext"
    elif voltage_synapse:
        eqs = """
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau_syn          : volt (unless refractory)
        """
        G = NeuronGroup(n, eqs, threshold="v > v_th", reset="v = v_rst",
                        refractory=t_rfc, method="exact", name="cx",
                        namespace=ns)
        G.v, G.g = v_0, 0 * mV
        # per-neuron scale of the voltage jump: median-leak neuron gets
        # exactly W_SYN per synapse, bigger neurons less, smaller more
        scale = float(np.median(g_L / siemens)) / (g_L / siemens)
        w_unit = g_syn_ps * mV
        exc, inh = counts > 0, counts < 0
        s_e = Synapses(G, G, "w : volt", on_pre="g_post += w", name="rec_e")
        s_e.connect(i=pre[exc], j=post[exc])
        s_e.w = counts[exc] * scale[post[exc]] * w_unit
        s_i = Synapses(G, G, "w : volt", on_pre="g_post += w", name="rec_i")
        s_i.connect(i=pre[inh], j=post[inh])
        s_i.w = counts[inh] * scale[post[inh]] * w_unit        # negative
        syn_objs = [s_e, s_i]
        stim_target = "g_post += w_ext"
    else:
        # ABLATION ONLY: one conductance per receptor channel, each with its
        # own decay and reversal potential, plus an adaptation conductance
        current = " + ".join(f"g_{c}*(E_{c} - v)" for c in CHANNELS)
        eqs = (f"dv/dt = (v_0 - v + ({current} + g_a*(E_K - v)) / g_L) / t_mbr"
               " : volt (unless refractory)\n")
        eqs += "".join(f"dg_{c}/dt = -g_{c} / tau_{c} : siemens\n" for c in CHANNELS)
        eqs += """
        dg_a/dt = -g_a / tau_a : siemens
        g_L : siemens (constant)
        b_adapt : siemens (constant)
        """
        # g_a accumulates across spikes and is never reset: that is the
        # whole mechanism. It decays during the refractory period too.
        G = NeuronGroup(n, eqs, threshold="v > v_th",
                        reset="v = v_rst; g_a += b_adapt",
                        refractory=t_rfc, method="exponential_euler", name="cx",
                        namespace=ns)
        G.v, G.g_a = v_0, 0 * siemens
        for c in CHANNELS:
            setattr(G, f"g_{c}", 0 * siemens)
        G.g_L = g_L
        G.b_adapt = b_adapt_ns * nsiemens
        w_unit = g_syn_ps * psiemens
        for c in CHANNELS:
            m = edge_ch == c
            if not m.any():
                continue
            # charge_fixed: G_c * tau_c = G_ach * 5 ms for every channel
            g_c = w_unit * (TAU_UNIFORM["ach"] / taus[c] if charge_fixed else 1.0)
            s = Synapses(G, G, "w : siemens", on_pre=f"g_{c}_post += w",
                         name=f"rec_{c}")
            s.connect(i=pre[m], j=post[m])
            s.w = np.abs(counts[m]) * g_c
            syn_objs.append(s)
        stim_target = "g_ach_post += w_ext"

    # stimulus: named targets only, F_POI synapses' worth per drive spike
    steps = int(round(duration_s / drive_dt))
    rate = np.zeros((steps, n))
    for t0, t1, idxs, hz in stim_schedule:
        rate[int(t0 / drive_dt):int(t1 / drive_dt), idxs] = hz
    ta = TimedArray(rate * Hz, dt=drive_dt * second)
    P = PoissonGroup(n, rates="ta(t, i)", name="drive", namespace={"ta": ta})
    ext = Synapses(P, G, on_pre=stim_target, name="drive_syn",
                   namespace={"w_ext": F_POI * w_unit})
    ext.connect(j="i")

    spikes = SpikeMonitor(G)
    return Network(G, *syn_objs, P, ext, spikes), spikes


NAMESPACE = dict(v_0=v_0, v_rst=v_rst, v_th=v_th, t_mbr=t_mbr, tau_syn=tau_syn,
                 E_exc=E_exc, E_inh=E_inh, E_K=E_K, tau_a=tau_a)


def run_sim(w_matrix, area_nm2, g_syn_ps, schedule, duration_s, seed_val=1,
            **flags):
    start_scope()
    seed(seed_val)
    net, spikes = build(w_matrix, area_nm2, g_syn_ps, schedule, duration_s,
                        **flags)
    net.run(duration_s * second, namespace=NAMESPACE)
    n = w_matrix.shape[0]
    n_bins = int(round(duration_s / (DT_BIN_MS / 1000.0)))
    out = np.zeros((n, n_bins))
    t = np.asarray(spikes.t / second)
    i = np.asarray(spikes.i)
    k = (t >= 0) & (t < duration_s)
    np.add.at(out, (i[k], np.floor(t[k] / (DT_BIN_MS / 1000.0)).astype(int)), 1)
    return out * (1000.0 / DT_BIN_MS)


# ==========================================================================
# analysis
# ==========================================================================
def smooth(rates, window_ms=SMOOTH_MS):
    w = max(1, int(round(window_ms / DT_BIN_MS)))
    return np.apply_along_axis(
        lambda r: np.convolve(r, np.ones(w) / w, mode="same"), 1, rates)


def decode(rates_epg, angles_epg):
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
    sm = smooth(rates)
    r_epg = sm[epg]
    theta, R = decode(sm[order], angles[order])
    pr = participation_ratio(r_epg)
    jac = jaccard_vs_reference(r_epg, t_axis, STIM_OFF)
    after = t_axis > STIM_OFF + 0.05
    # shape metrics only over bins where EPG is actually active; averaging
    # R=0 / PR=0 from silent bins makes a dying bump look localised
    live = after & (r_epg.sum(axis=0) > ACTIVE_HZ)
    at = lambda arr, t: float(arr[int(np.argmin(np.abs(t_axis - t)))])
    below = np.flatnonzero(after & (jac < 0.5))
    persist = float(t_axis[below[0]] - STIM_OFF) if len(below) else float(T_END - STIM_OFF)
    per_neuron = rates[epg][:, after].mean(1)
    return {
        "smooth": sm, "theta": theta, "R": R, "pr": pr, "jaccard": jac,
        "epg_hz_after": float(per_neuron.mean()),
        # rate of the neurons that ARE the bump; the population mean hides a
        # 5-neuron bump firing at the refractory ceiling
        "active_hz_after": float(np.mean(per_neuron[per_neuron > ACTIVE_HZ])
                                 if (per_neuron > ACTIVE_HZ).any() else 0.0),
        "n_active_after": int((per_neuron > ACTIVE_HZ).sum()),
        "epg_hz_before": float(rates[epg][:, t_axis < STIM_ON].mean()),
        "R_after": float(np.nanmean(R[live])) if live.any() else 0.0,
        "pr_after": float(np.nanmean(pr[live])) if live.any() else 0.0,
        "live_frac": float(live.sum() / after.sum()),
        "jaccard_+1.0": at(jac, STIM_OFF + 1.0),
        "jaccard_+2.0": at(jac, STIM_OFF + 2.0),
        "persist_s": persist,
    }


def passes(r_):
    hz, pr = r_["active_hz_after"][0], r_["pr_after"][0]
    R, j1 = r_["R_after"][0], r_["jaccard_+1.0"][0]
    return (pr < 20) and (j1 > 0.5) and (5.0 <= hz <= MAX_ACTIVE_HZ) and (R > 0.35)


def verdict(r_):
    hz, pr, R, j1 = (r_["active_hz_after"][0], r_["pr_after"][0],
                     r_["R_after"][0], r_["jaccard_+1.0"][0])
    if passes(r_):        return "PASS — persistent localised bump"
    if r_["epg_hz_after"][0] < 0.5: return "silent after light-off"
    if pr > 30:           return "DIFFUSE — everything fires"
    if hz > MAX_ACTIVE_HZ and R > 0.35 and j1 > 0.5:
        return "PERSISTENT LOCALISED but saturated"
    if hz > MAX_ACTIVE_HZ: return "localised but too fast"
    if R > 0.35 and j1 > 0.5: return "bump but rate out of range"
    if R > 0.35:          return "localised but drifting"
    return "weak"


# ==========================================================================
def main() -> int:
    d = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
    with open(OUT / "banc_cx_circuit.json") as f:
        meta = json.load(f)
    W = d["W"].astype(np.float64)
    lab, angles = d["labels"].astype(str), d["angles"]
    area = d["area_nm2"].astype(np.float64)
    frag = d["is_fragment"].astype(bool)
    nt = d["nt"].astype(str)
    n = W.shape[0]
    ch = channel_of(nt)
    n_mod = int(np.count_nonzero([str(x) not in NT_TO_CHANNEL for x in nt]))
    print("receptor channel by presynaptic transmitter:  " +
          "  ".join(f"{c}: {int((ch == c).sum())} neurons" for c in CHANNELS) +
          f"   ({n_mod} neuromodulatory/unknown routed to ach)")

    epg = np.flatnonzero(lab == "EPG")
    d7 = np.flatnonzero(lab == "Delta7")
    pen = np.flatnonzero((lab == "PEN_a(PEN1)") | (lab == "PEN_b(PEN2)"))
    up = np.flatnonzero(lab == "upstream")
    order = epg[np.argsort(angles[epg])]
    t_axis = np.arange(int(round(T_END / (DT_BIN_MS / 1000)))) * DT_BIN_MS / 1000

    print(f"BANC {meta['datastack']} mat {meta['materialization']}   "
          f"{n} neurons, {int((W != 0).sum()):,} edges, "
          f"{int(frag.sum())} fragments at median area")
    a_um2 = area / 1e6
    g_l_ns = area * 1e-18 * float(G_L_SPEC / (siemens / meter ** 2)) * 1e9
    print("membrane area (um^2) / leak (nS), median by type:")
    for t in ["EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7", "upstream"]:
        m = (lab == t) & ~frag
        if m.any():
            print(f"  {t:<12} area {np.median(a_um2[m]):7.0f}   g_L {np.median(g_l_ns[m]):5.2f}"
                  f"   0.275 mV-equivalent G_syn {0.275/52*np.median(g_l_ns[m])*1e3:5.1f} pS")
    print()

    # arc seed on recovered ring, centred on the connectivity hub
    hub = epg[np.argmax(np.abs(W[np.ix_(epg, epg)]).sum(1))]
    hp = int(np.flatnonzero(order == hub)[0])
    seed_set = order[[(hp - ARC_SIZE // 2 + k) % len(order)
                      for k in range(ARC_SIZE)]].tolist()


    # ---------------- silence check ----------------
    r0 = run_sim(W, area, G_SYN_PS, [], T_END, seed_val=0, nt=nt,
                 tau_table=TAU_UNIFORM)
    tot = int(round(r0.sum() * DT_BIN_MS / 1000))
    print(f"silence check (no stimulus): {tot} spikes   "
          f"{'OK' if tot == 0 else 'FAIL'}\n")

    # ---------------- the model, N_SEEDS runs ----------------
    print(f"model: G_syn {G_SYN_PS:g} pS, tau_syn {TAU_UNIFORM['ach']:g} ms, "
          f"E_exc {E_exc/mV:g} / E_inh {E_inh/mV:g} mV, g_L from area, "
          f"adaptation {B_ADAPT_NS:g} nS, no background\n")
    runs = []
    aft = t_axis > STIM_OFF + .05
    for s in range(N_SEEDS):
        rr = run_sim(W, area, G_SYN_PS, [(STIM_ON, STIM_OFF, seed_set, R_POI)],
                     T_END, seed_val=100 + s, b_adapt_ns=B_ADAPT_NS,
                     nt=nt, tau_table=TAU_UNIFORM)
        a = analyse(rr, epg, order, angles, t_axis)
        a["rates"] = rr
        a["d7_after"] = float(rr[d7][:, aft].mean())
        a["pen_after"] = float(rr[pen][:, aft].mean())
        a["up_after"] = float(rr[up][:, aft].mean())
        per = rr[epg][:, aft].mean(1)
        a["active_rows"] = sorted(int(np.flatnonzero(order == e)[0])
                                  for e in epg[per > ACTIVE_HZ])
        runs.append(a)
        print(f"seed {s}: active EPG rows {a['active_rows']}  @ "
              f"{a['active_hz_after']:.0f} Hz   R {a['R_after']:.3f}  "
              f"PR {a['pr_after']:.1f}  J+1 {a['jaccard_+1.0']:.2f}  "
              f"alive {a['live_frac']:.2f}  persist {a['persist_s']:.2f} s")

    def agg(k_):
        v = np.array([r_[k_] for r_ in runs])
        return float(v.mean()), float(v.std(ddof=1))

    res = {"g_syn_ps": G_SYN_PS, "tau_ms": TAU_UNIFORM, "b_adapt_ns": B_ADAPT_NS,
           "runs": runs,
           **{k_: agg(k_) for k_ in (
               "epg_hz_before", "epg_hz_after", "active_hz_after",
               "n_active_after", "R_after", "pr_after",
               "jaccard_+1.0", "jaccard_+2.0", "persist_s", "live_frac",
               "d7_after", "pen_after", "up_after")}}
    seed_rows = sorted(int(np.flatnonzero(order == s_)[0]) for s_ in seed_set)
    print(f"\nstimulated rows {seed_rows}")
    print(f"mean over seeds: {res['n_active_after'][0]:.1f} active EPG @ "
          f"{res['active_hz_after'][0]:.0f} Hz, R {res['R_after'][0]:.3f}, "
          f"PR {res['pr_after'][0]:.1f}, J+1 {res['jaccard_+1.0'][0]:.2f}, "
          f"alive {res['live_frac'][0]:.2f}   | {verdict(res)}")

    # ---------------- figure: one column per seed ----------------
    INK, MUTED = "#1b2320", "#5d6b64"
    fig, axes = plt.subplots(4, N_SEEDS, figsize=(6.5 * N_SEEDS, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1]},
                             squeeze=False)
    for col, run0 in enumerate(runs):
        rr = run0["rates"]
        ax = axes[0, col]
        im = ax.imshow(rr[order], aspect="auto", origin="lower",
                       extent=[t_axis[0], t_axis[-1], 0, len(order)],
                       cmap="magma", vmax=max(1, np.percentile(rr[order], 99.5)))
        ax.axvspan(STIM_ON, STIM_OFF, color="cyan", alpha=.15)
        th = run0["theta"]; Rv = run0["R"]
        ring_pos = np.interp(th, angles[order], np.arange(len(order)) + .5)
        ring_pos[Rv < 0.35] = np.nan
        ax.plot(t_axis, ring_pos, ".", color="cyan", ms=1.5, alpha=.7)
        for sr in seed_rows:
            ax.plot(-0.04, sr + .5, ">", color="#2f6f8f", ms=6, clip_on=False,
                    transform=ax.get_yaxis_transform())
        ax.set_title(f"seed {col}   ·   {run0['n_active_after']} EPG @ "
                     f"{run0['active_hz_after']:.0f} Hz after light-off",
                     fontsize=11, fontweight="bold", color=INK, pad=10)
        ax.set_ylabel("EPG, sorted by\nrecovered angle")
        plt.colorbar(im, ax=ax, label="Hz", pad=.01)
        for row, (key, ylab, hline, ylim) in enumerate([
                ("R", "bump strength R", .35, (0, 1)),
                ("pr", "participation ratio", ARC_SIZE, (0, len(epg) + 3)),
                ("jaccard", "Jaccard vs light-off", .5, (0, 1.02))], start=1):
            ax = axes[row, col]
            ax.plot(t_axis, run0[key], lw=1.4, color="#2f6f8f")
            ax.axhline(hline, ls="--", lw=1, color=MUTED)
            ax.axvspan(STIM_ON, STIM_OFF, color="cyan", alpha=.10)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylab)
        axes[3, col].set_xlabel("time (s)")
    for ax in axes.ravel():
        ax.grid(True, alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"BANC ring attractor — Kim et al. 2017 Exp. 1   ·   G_syn "
                 f"{G_SYN_PS:g} pS, τ_syn {TAU_UNIFORM['ach']:g} ms, "
                 f"g_L from area, E/I from reversal, no adaptation, no background",
                 fontsize=12, fontweight="bold", color=INK, y=.995)
    fig.tight_layout()
    fig.savefig(OUT / "experiment1.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    with open(OUT / "experiment1_results.json", "w") as f:
        json.dump({
            "model": {
                "neuron": "conductance-based LIF, one compartment",
                "v_rest_mV": -52, "v_reset_mV": -52, "v_th_mV": -45,
                "t_rfc_ms": 2.2, "tau_m_ms": 20, "c_m_uF_per_cm2": 1.0,
                "g_L": "A_i * c_m / tau_m, A_i = BANC l2cache area_nm2",
                "reset": "v = v_rst (standard; no synaptic reset)",
                "synapse": "g += n_ji * G_syn on presynaptic spike, exponential decay",
                "G_syn_pS": G_SYN_PS, "tau_syn_ms": TAU_UNIFORM["ach"],
                "E_rev_mV": {c: float(E_REV[c] / mV) for c in CHANNELS},
                "nt_to_channel": NT_TO_CHANNEL,
                "n_neuromodulatory_routed_to_ach": n_mod,
                "adaptation_nS": B_ADAPT_NS, "background": "none",
                "cell_type_parameters": "none",
            },
            "protocol": {"stim_rows": seed_rows, "stim_hz": R_POI,
                         "stim_synapses_per_spike": F_POI,
                         "stim_on_s": STIM_ON, "stim_off_s": STIM_OFF, "t_end_s": T_END},
            "n_fragments_at_median_area": int(frag.sum()),
            "pass_criterion": f"PR<20, J+1>0.5, 5<=active-neuron Hz<="
                              f"{MAX_ACTIVE_HZ:g}, R>0.35",
            "silence_check_spikes": tot, "verdict": verdict(res), "passes": passes(res),
            "per_seed": [{k: v for k, v in r_.items()
                          if k not in ("rates", "smooth", "theta", "R", "pr", "jaccard")}
                         for r_ in runs],
            "mean": {k: v for k, v in res.items() if k != "runs"},
            "known_limitation": "bump neurons fire at the refractory ceiling "
                                "(~200 Hz); real EPGs fire at tens of Hz. See "
                                "module docstring.",
        }, f, indent=2)
    print(f"\nwrote {OUT / 'experiment1.png'}, experiment1_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
