"""
Unified Brain+VNC network (BANC 888: 76,683 neurons, 27.7M synapses) with local, unsupervised
plasticity. `build_network` builds the whole model; every option below is off unless requested,
so the plain call reproduces the static v2 model.

Plant
  neuron    conductance LIF, C_i = c_m·A_i (measured area), g_L = C_i/tau_m; V_rest = V_reset −52,
            V_th −45 mV, t_rfc 2.2 ms, E_exc 0, E_inh −70 mV; tau_syn 5 ms (inh), tau_e (exc,
            charge-preserving scaling)
  weights   w = n·G_E (exc), w = n·rho·G_E·E_i/I_i (inh), rho = (E_exc − V_th)/(V_th − E_inh) = 1.8:
            each neuron's inhibitory capacity balances its excitatory capacity at threshold
  escape    optional soft threshold: hazard lam0·exp((V − V_th)/Δ)            (`escape=(Δ_mV, lam0_Hz)`)
  also      adaptive threshold (`adapt`), two compartments (`two_comp`), reciprocity weightings
            (`recip`, `recip_ee`, `recip_een`) — diagnostics, see README
  traces    xt (tau 20 ms), r (tau 100 ms), m dopamine (tau_da)

Plasticity (all local; nothing task-related; targets = non-sensory neurons, `pl` flag)
  I -> E  Vogels et al. 2011, multiplicative, floored at the v2 prior:
      pre spike :  w += eta_i·w0·(xt_post − alpha)      alpha = 2·rho0·tau_x
      post spike:  w += eta_i·w0·xt_pre                 w in [w0, wmax_i·w0]
  E -> E  (a) rate-cap decay every 100 ms (`gamma_e`; the "brake" of p1-p8), and/or
          (b) pair STDP (`eta_e`): LTP on post spike ∝ xt_pre, LTD on pre spike ∝ a·xt_post;
              variants: LTP ∝ r_post/rho0 (triplet-like, default), saturating + band-gated
              (`ltp_sat`), or DOPAMINE-GATED (`da=` edge table): both terms × g(m_post),
              g(m) = clip((m − m_th)/(1 − m_th), 0, 1) — the three-factor rule of p25-p41.
  dopamine  `da` = the connectome's dopaminergic edges (build_da_edges.py): a DAN spike adds
            n_syn/da_norm to its targets' m; release laws `da_release` = None | supra | deplete.
  E -> I and I -> I fixed at v2.

The best brain (p37a) = escape (1 mV, 50 Hz) + tau_e 15 ms + I->E rule (rho0 25, bound 20x) +
dopamine-gated STDP (eta 8e-3, a 0.5, bound 3.5x, m_th 0.35, tau_da 0.3 s), warm-started from p8;
see reproduce_p37a.sh and WRITEUP_v2_to_p37a.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from brian2 import *                                   # noqa: F401,F403

OUT = Path(__file__).resolve().parent
import banc_common as BS                                  # noqa: E402  (constants, analysis)

prefs.codegen.target = "cython"
defaultclock.dt = 0.1 * ms

# constants (universal, as in v2)
G_SYN_PS = 30.0
STIM_HZ = BS.STIM_HZ
SENS_KICK_F = BS.SENS_KICK_F
# plasticity constants (declared; not from fly data)
TAU_X, TAU_R = 20 * ms, 100 * ms
RHO0_HZ = 5.0
ETA_I, WMAX_I = 3e-3, 10.0
GAMMA_E, FLOOR_E = 0.002, 0.1     # e-fold per (s x Hz over rho0): 50 Hz over -> 10 %/s; floor x w0
E_DECAY_START_S = 10.0            # inhibitory plasticity alone first
ETA_E, WMAX_E = 0.0, 5.0
LTD_RATIO = 1.1                   # STDP LTD/LTP amplitude ratio (slightly depression-dominated)
STDP_START_S = 30.0               # E->E STDP off until inhibitory plasticity has balanced the brain
# two-sided (dead-band) homeostasis: excitation scales UP for a plastic target that
# receives excitatory spikes but fires below RHO_LOW (Turrigiano 1998); nothing in
# [RHO_LOW, RHO0]; brake above RHO0 as before. Off unless --two-sided.
RHO_LOW_HZ, GAMMA_UP = 0.5, 0.01           # Hz; e-fold per second at r = 0
# intrinsic rate control (plant physics, universal, off by default):
#   adapt  = (alpha_mV, tau_a_ms): adaptive threshold, dvth/dt = (v_th - vth)/tau_a, vth += alpha at
#            a spike (MAT model, Kobayashi 2009; sAHP timescale) -- a rate-proportional negative
#            feedback that gives the recurrent loop a stable low-rate crossing
#   escape = (delta_mV, lam0_Hz): soft threshold, spike hazard lam0*exp((v - vth)/delta)
#            (Gerstner escape noise / GLM sharpness 1-3 mV: Jolivet 2006, Pozzorini 2015);
#            lam0 = 1/tau_m keeps v_th's meaning (hazard 1/tau_m at threshold)
ADAPT_TAU_MS = 200.0
ESCAPE_LAM0_HZ = float(1.0 / (BS.t_mbr / second))

NAMESPACE = dict(v_0=BS.v_0, v_rst=BS.v_rst, v_th=BS.v_th, t_mbr=BS.t_mbr, tau_syn=BS.tau_syn,
                 E_exc=BS.E_exc, E_inh=BS.E_inh)


def load_data(path=OUT / "banc_full.npz", vis_sens=False):
    """vis_sens: treat visual projection / centrifugal neurons as sensory-equivalents
    (they are the optic lobe's outputs and get no input with the optic lobe excluded):
    they join the spontaneous drive set and are excluded from plasticity targets."""
    d = np.load(path, allow_pickle=True)
    n = len(d["root_id"])
    sc = d["super_class"].astype(str)
    sens_mask = np.char.startswith(sc.astype(str), "sensory")
    if vis_sens:
        sens_mask |= np.isin(sc, ["visual_projection", "visual_centrifugal"])
    data = dict(n=n, root=d["root_id"].astype(np.int64), cell_type=d["cell_type"].astype(str),
                label=d["label"].astype(str), super_class=sc, region=d["region"].astype(str),
                side=d["side"].astype(str), sign=d["sign"].astype(np.int8),
                area=d["area_nm2"].astype(float), pre_idx=d["pre_idx"].astype(np.int64),
                post_idx=d["post_idx"].astype(np.int64), n_syn=d["n_syn"].astype(float),
                is_cx_core=d["is_cx_core"].astype(bool),
                sens_idx=np.flatnonzero(sens_mask))
    return data


def kim_stim(data, sector_deg=45, offset_deg=0.0):
    """the 6 EPGs of one 45-deg EB sector (Kim 2017), by root id from the CX circuit;
    offset_deg rotates the sector around the ring (for training kicks at every heading)"""
    cx = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
    cx_root = cx["root_id"].astype(np.int64); cx_lab = cx["labels"].astype(str); cx_ang = cx["angles"]
    cx_epg = np.flatnonzero(cx_lab == "EPG"); order = cx_epg[np.argsort(cx_ang[cx_epg])]
    Wcx = cx["W"].astype(float)
    hub = cx_epg[np.argmax(np.abs(Wcx[np.ix_(cx_epg, cx_epg)]).sum(1))]
    hp = int(np.flatnonzero(order == hub)[0]); k = int(round(sector_deg / 360.0 * len(order)))
    hp = (hp + int(round(offset_deg / 360.0 * len(order)))) % len(order)
    seed_ids = order[[(hp - k // 2 + j) % len(order) for j in range(k)]]
    pos = {int(r): i for i, r in enumerate(data["root"])}
    return np.array([pos[int(cx_root[e])] for e in seed_ids])


def v2_weights(data, g_syn_ps=G_SYN_PS):
    """per-edge conductance under the v2 rule; returns w (S) and the edge-class masks"""
    n, sign = data["n"], data["sign"]
    pre, post, ns = data["pre_idx"], data["post_idx"], data["n_syn"]
    exc_pre = sign[pre] > 0
    E_in = np.bincount(post[exc_pre], weights=ns[exc_pre], minlength=n)
    I_in = np.bincount(post[~exc_pre], weights=ns[~exc_pre], minlength=n)
    rho = float((BS.E_exc - BS.v_th) / (BS.v_th - BS.E_inh))
    gi_fac = np.ones(n)
    ok = (E_in > 0) & (I_in > 0)
    gi_fac[ok] = rho * E_in[ok] / I_in[ok]
    g = g_syn_ps * 1e-12
    w = ns * g * np.where(exc_pre, 1.0, gi_fac[post])
    exc_post = sign[post] > 0
    masks = dict(ee=exc_pre & exc_post, ei=exc_pre & ~exc_post, ie=~exc_pre & exc_post, ii=~exc_pre & ~exc_post)
    return w, masks, gi_fac


def reciprocity_factor(data, kappa):
    """candidate universal rule: w_ji = n_ji G (1 + kappa n_ij) -- a synapse is stronger the more
    its target projects back onto its source (reciprocally connected pairs are stronger, Song et
    al. 2005). Returns the per-edge factor (edge order of data)."""
    n = data["n"]
    pre, post, ns = data["pre_idx"].astype(np.int64), data["post_idx"].astype(np.int64), data["n_syn"]
    key = pre * n + post; order = np.argsort(key); ks = key[order]
    rev = post * n + pre
    j = np.minimum(np.searchsorted(ks, rev), len(ks) - 1)
    has = ks[j] == rev
    return 1.0 + kappa * np.where(has, ns[order][j], 0.0)


def edge_reciprocity(data):
    """rho_ij = n_ji / (n_ij + n_ji): 0 for a pure feed-forward edge, 1/2 for a symmetric pair.
    Used to make short-term depression a structural property (std=(U0, tau, "recip") -> U_ij = U0 rho_ij):
    recurrent loops depress, feed-forward pathways (ORN->PN, PN->KC, LC->GF) do not."""
    n = data["n"]
    pre, post, ns = data["pre_idx"].astype(np.int64), data["post_idx"].astype(np.int64), data["n_syn"]
    key = pre * n + post; order = np.argsort(key); ks = key[order]
    rev = post * n + pre
    j = np.minimum(np.searchsorted(ks, rev), len(ks) - 1)
    n_rev = np.where(ks[j] == rev, ns[order][j], 0.0)
    return n_rev / (ns + n_rev)


def reciprocity_factor_norm(data, kappa, ee_mask):
    """reciprocity as REDISTRIBUTION among a neuron's excitatory inputs: the E->E factor divided by
    its count-weighted mean over the postsynaptic neuron's E inputs, so every neuron's total E
    drive is conserved (brain-wide mean exactly 1). Returns the per-edge factor (1 off E->E)."""
    n = data["n"]
    post, ns = data["post_idx"], data["n_syn"]
    fac = reciprocity_factor(data, kappa)
    num = np.bincount(post[ee_mask], weights=(ns * fac)[ee_mask], minlength=n)
    den = np.bincount(post[ee_mask], weights=ns[ee_mask], minlength=n)
    mean_i = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0)
    return np.where(ee_mask, fac / mean_i[post], 1.0)


def build_network(data, g_syn_ps=G_SYN_PS, plastic=False, gains=None, sens_hz=0.0,
                  stim=None, stim2=None, duration_s=2.0, record_spikes=True,
                  eta_i=ETA_I, wmax_i=WMAX_I, eta_e=ETA_E, wmax_e=WMAX_E, floor_e=FLOOR_E,
                  tau_e=None, in_model=False, gamma_e=GAMMA_E, e_start_s=E_DECAY_START_S,
                  two_sided=False, rho_low=RHO_LOW_HZ, gamma_up=GAMMA_UP,
                  adapt=None, escape=None, two_comp=None, recip=None, ltp_sat=False, gate_fac=2.0,
                  recip_ee=None, recip_een=None, da=None, da_norm=100.0, tau_da_s=1.0, da_th=0.0,
                  da_release=None, std=None, brake_recip=False, model="cond", g_reset=True, stim_v=False):
    """stim / stim2 = (idx, on_s, off_s, f). gains = dict(ee=array, ie=array) multiplying
    the v2 weights. std = (U, tau_rec_s): Tsodyks-Markram short-term depression on every
    excitatory synapse (resource x, release U*x per spike, recovery tau_rec; event-driven).
    the v2 weights of those edge classes (frozen result of a plasticity run).
    tau_e: excitatory synaptic time constant (default = tau_syn, 5 ms); the per-spike
    conductance is scaled by tau_syn/tau_e so the charge per spike is unchanged.
    in_model: put the E->E homeostasis inside the model (synaptic run_regularly every
    100 ms, gate = slow trace of arriving excitation) so a standalone device (GPU) can
    run the whole settle in one run(); the runtime path applies it between chunks.
    adapt = (alpha_mV, tau_a_ms) adaptive threshold; escape = (delta_mV, lam0_Hz) soft
    threshold (see the constants block); both None = the v2 hard-threshold LIF.
    model = "shiu": the Shiu et al. 2024 formulation on the same graph and signs (build_shiu_network);
    g_reset = False keeps that formulation without the reset of g at a spike; stim_v = True delivers
    stimulus and background events as that paper does (a jump on v). Every option of the
    conductance model is refused in that mode."""
    if model != "cond":
        if model != "shiu":
            raise ValueError(f"unknown model {model!r} (cond | shiu)")
        given = dict(plastic=plastic, gains=gains, tau_e=tau_e, in_model=in_model, two_sided=two_sided, adapt=adapt,
                     escape=escape, two_comp=two_comp, recip=recip, ltp_sat=ltp_sat, recip_ee=recip_ee,
                     recip_een=recip_een, da=da, da_release=da_release, std=std, brake_recip=brake_recip)
        bad = [k for k, v in given.items() if v is not None and v is not False]
        if g_syn_ps != G_SYN_PS:
            bad.append("g_syn_ps")
        if bad:
            raise ValueError(f"model='shiu' has none of: {', '.join(bad)} (fixed voltage-jump weights, no plasticity)")
        return build_shiu_network(data, sens_hz=sens_hz, stim=stim, stim2=stim2, duration_s=duration_s,
                                  record_spikes=record_spikes, g_reset=g_reset, stim_v=stim_v)
    if not g_reset or stim_v:
        raise ValueError("g_reset=False and stim_v=True are defined for model='shiu' only")
    n = data["n"]
    tau_e_q = BS.tau_syn if tau_e is None else tau_e
    q_scale = float(BS.tau_syn / tau_e_q)                  # same charge per spike at any tau_e
    if two_comp is None:
        v_eq = "dv/dt = (v_0 - v + (g_e*(E_exc - v) + g_i*(E_inh - v)) / g_L) / t_mbr : volt (unless refractory)"
    else:
        # two compartments: the connectome's synapses on a dendrite (fraction 1 - f_s of the
        # membrane) coupled to the spike-initiating soma (fraction f_s) by g_c = rho * g_L.
        # Somatic drive is bounded by g_c (vd - v) whatever the fan-in; the somatic time
        # constant C_s/(g_Ls + g_c) is not shunted by the dendrite's synaptic conductance.
        # C = g_L * t_mbr (c_m * area), so both equations are written per unit g_L.
        v_eq = ("dv/dt = (f_s*(v_0 - v) + rho_c*(vd - v)) / (f_s*t_mbr) : volt (unless refractory)\n"
                "    dvd/dt = ((1 - f_s)*(v_0 - vd) + (g_e*(E_exc - vd) + g_i*(E_inh - vd)) / g_L + rho_c*(v - vd))"
                " / ((1 - f_s)*t_mbr) : volt")
    eqs = f"""
    {v_eq}
    dg_e/dt = -g_e / tau_e : siemens
    dg_i/dt = -g_i / tau_syn : siemens
    dxt/dt = -xt / tau_x : 1
    dr/dt = -r / tau_r : Hz
    dge_in/dt = -ge_in / tau_r : 1
    dm/dt = -m / tau_da : 1
    g_L : siemens (constant)
    pl : 1 (constant)
    """
    # m: dopamine trace (three-factor rule), incremented by the spikes of the dopaminergic neurons
    # that synapse onto the cell (da_edges.npz), in units of da_norm synapses; tau 1 s
    ns_n = dict(NAMESPACE, tau_x=TAU_X, tau_r=TAU_R, tau_e=tau_e_q, tau_da=float(tau_da_s) * second)
    if two_comp is not None:
        ns_n.update(rho_c=float(two_comp[0]), f_s=float(two_comp[1]))
    reset = "v = v_rst; xt += 1; r += 1/tau_r"
    if adapt is not None:
        eqs += "\n    dvth/dt = (v_th - vth) / tau_a : volt"
        reset += "; vth += alpha_a"
        ns_n.update(tau_a=adapt[1] * ms, alpha_a=adapt[0] * mV)
    thv = "vth" if adapt is not None else "v_th"
    threshold = f"v > {thv}"
    if escape is not None:
        threshold = f"(v > {thv}) or (rand() < lam0*exp((v - {thv})/delta_t)*dt)"
        ns_n.update(lam0=escape[1] * Hz, delta_t=escape[0] * mV)
    G = NeuronGroup(n, eqs, threshold=threshold, reset=reset,
                    refractory=BS.t_rfc, method="exponential_euler", name="net", namespace=ns_n)
    G.v = BS.v_0; G.g_L = (data["area"] * 1e-18) * meter ** 2 * BS.G_L_SPEC
    if two_comp is not None:
        G.vd = BS.v_0
    if adapt is not None:
        G.vth = BS.v_th
    # plastic targets: non-sensory neurons only (sensory rates are imposed by the drive)
    is_sens = np.zeros(n, bool); is_sens[data["sens_idx"]] = True
    G.pl = (~is_sens).astype(float)
    w, masks, gi_fac = v2_weights(data, g_syn_ps)
    if recip is not None:
        w = w * reciprocity_factor(data, recip)
    if recip_ee is not None:
        # E by structure: reciprocity weighting on excitatory->excitatory synapses only
        # (inhibition is left to the activity rule)
        fac_ee = reciprocity_factor(data, recip_ee)
        w = np.where(masks["ee"], w * fac_ee, w)
    if recip_een is not None:
        # E by structure, drive-conserving: reciprocity redistributes each neuron's E input
        w = w * reciprocity_factor_norm(data, recip_een, masks["ee"])
    if gains is not None:
        for k in ("ee", "ie", "ei", "ii"):               # ei/ii only from explicit scaling probes
            if k in gains:
                w[masks[k]] = w[masks[k]] * np.asarray(gains[k], float)
    if q_scale != 1.0:
        w[masks["ee"] | masks["ei"]] *= q_scale
    pre, post = data["pre_idx"], data["post_idx"]
    alpha = 2 * RHO0_HZ * float(TAU_X / second)
    ns_syn = dict(eta_i=eta_i, alpha=alpha, wmax_i=wmax_i, eta_e=eta_e, wmax_e=wmax_e, floor_e=floor_e,
                  a_ratio=LTD_RATIO, tau_r=TAU_R, t_stdp=STDP_START_S * second,
                  rho0=RHO0_HZ * Hz, rho_low=rho_low * Hz, t_e=e_start_s * second, dt_h=0.1 * second,
                  gamma_e=gamma_e / (second * Hz), gamma_up=gamma_up / second, two=float(two_sided),
                  m_th=float(da_th))                      # dopamine gate threshold (three-factor rule)
    syn = {}
    # depression classes: every excitatory synapse, or E->E only ("recip_ee": recurrent excitation
    # depresses, the E->I feedback brakes -- KC->APL, EPG->Delta7 -- are left intact)
    std_cls = ("ee",) if (std is not None and len(std) > 2 and std[2] == "recip_ee") else ("ee", "ei")
    for k, on_var in (("ee", "g_e"), ("ei", "g_e"), ("ie", "g_i"), ("ii", "g_i")):
        m = masks[k]
        model = "w : siemens\n w0 : siemens"
        pre_code, post_code = f"{on_var}_post += w", None
        if std is not None and k in std_cls:
            # short-term depression (Tsodyks & Markram 1997): the released fraction x recovers
            # between spikes; per-spike conductance w*x, then x -= U*x
            model += "\n dx/dt = (1 - x)/tau_rec : 1 (event-driven)"
            if len(std) > 2 and std[2].startswith("recip"):
                model += "\n U_s : 1"                          # per-edge release fraction U0 * rho_ij
                pre_code = f"{on_var}_post += w*x\n x -= U_s*x"
            else:
                pre_code = f"{on_var}_post += w*x\n x -= U_std*x"
                ns_syn["U_std"] = float(std[0])
            ns_syn["tau_rec"] = float(std[1]) * second
        if plastic and in_model and k in ("ee", "ei"):
            pre_code += "\n ge_in_post += 1"                        # excitation arriving (gate for the up rule)
        if brake_recip and k == "ee":
            model += "\n rho_s : 1"                                  # reciprocity gate on the brake: min(1, 2 rho_ij)
        if plastic and k == "ie":
            pre_code += "\n w = clip(w + pl_post*eta_i*w0*(xt_post - alpha), w0, wmax_i*w0)"
            post_code = "w = clip(w + pl_post*eta_i*w0*xt_pre, w0, wmax_i*w0)"
        if plastic and k == "ee" and eta_e > 0:
            # pair-based STDP (Litwin-Kumar & Doiron 2014 assembly formation):
            # post after pre -> LTP (pre trace); pre after post -> LTD (post trace),
            # LTD/LTP ratio a_ratio > 1 so uncorrelated activity nets to slight depression.
            # triplet form (Pfister & Gerstner 2006): LTP scales with the postsynaptic rate
            # trace r_post*tau_r (expected spikes per 100 ms) so potentiation dominates at
            # high rates (assemblies) and depression at low, uncorrelated rates.
            # gated on t > t_stdp so inhibitory plasticity balances the brain first
            if da is not None:
                # three-factor rule: pair STDP (LTP on post spike x pre trace, LTD on pre spike x
                # post trace) gated by the postsynaptic dopamine trace m (saturating at 1) --
                # excitatory synapses change only where and when dopamine arrives
                pre_code += ("\n w = clip(w - int(t > t_stdp)*pl_post*eta_e*a_ratio*w0*xt_post*clip((m_post - m_th)/(1 - m_th), 0, 1),"
                             " floor_e*w0, wmax_e*w0)")
            else:
                pre_code += "\n w = clip(w - int(t > t_stdp)*pl_post*eta_e*a_ratio*w0*xt_post, floor_e*w0, wmax_e*w0)"
            # LTP/LTD cross at r_post = a_ratio*rho0 (the brake's cap): only populations the
            # brake considers "high" can assemble
            if da is not None:
                post_code = ("w = clip(w + int(t > t_stdp)*pl_post*eta_e*w0*xt_pre*clip((m_post - m_th)/(1 - m_th), 0, 1),"
                             " floor_e*w0, wmax_e*w0)")
            elif ltp_sat:
                # coupled rule: LTP saturates at rho0 and is OFF above r_gate = gate_fac*rho0, so
                # excitation can only grow while the postsynaptic rate is inside the band (after
                # inhibition has caught up); a ceiling bump or a KC burst earns no potentiation
                post_code = ("w = clip(w + int(t > t_stdp)*pl_post*eta_e*w0*xt_pre*(clip(r_post, 0*Hz, rho0)/rho0)"
                             "*int(r_post < r_gate), floor_e*w0, wmax_e*w0)")
                ns_syn["r_gate"] = gate_fac * RHO0_HZ * Hz
            else:
                post_code = "w = clip(w + int(t > t_stdp)*pl_post*eta_e*w0*xt_pre*(r_post/rho0), floor_e*w0, wmax_e*w0)"
        S = Synapses(G, G, model, on_pre=pre_code, on_post=post_code, name=f"syn_{k}", namespace=ns_syn)
        S.connect(i=pre[m], j=post[m])
        S.w = w[m] * siemens; S.w0 = w[m] * siemens
        if brake_recip and k == "ee":
            S.rho_s = np.minimum(1.0, 2.0 * edge_reciprocity(data)[m])
        if std is not None and k in std_cls:
            S.x = 1.0
            if len(std) > 2 and std[2].startswith("recip"):
                S.U_s = float(std[0]) * edge_reciprocity(data)[m]
        if plastic and in_model and k == "ee":
            # E->E homeostasis inside the model, every 100 ms after t_e:
            # brake  : r_post > rho0            -> w *= exp(-dt gamma_e (r - rho0))
            # up     : r_post < rho_low & ge_in -> w *= exp(+dt gamma_up (rho_low - r)/rho_low)   (two-sided only)
            # brake_recip: the brake acts on an edge in proportion to its reciprocity (rho_s = min(1, 2 rho_ij)),
            # so recurrent loops are scaled and feed-forward chains (rho ~ 0) keep their v2 weights
            gate = "*rho_s" if brake_recip else ""
            S.run_regularly(
                f"w = clip(w * exp(-dt_h*gamma_e{gate}*pl_post*int(t > t_e)*clip(r_post - rho0, 0*Hz, 1e9*Hz))"
                "         * exp( dt_h*gamma_up*two*pl_post*int(t > t_e)*int(r_post < rho_low)*int(ge_in_post > 0.01)"
                "               *(rho_low - clip(r_post, 0*Hz, rho_low))/rho_low),"
                "         floor_e*w0, wmax_e*w0)", dt=100 * ms, when="end", name="ee_homeostasis")
        syn[k] = S
    objs = [G] + list(syn.values())
    if da is not None:
        # modulatory (dopaminergic) edges: no conductance, only the dopamine trace of the target
        # release nonlinearity (universal constants):
        #   None                 -> one quantum per DAN spike (linear in DAN rate)
        #   ("supra", r_base)    -> quantum ∝ DAN rate / r_base (cooperative Ca2+: cumulative release ~ rate^2)
        #   ("deplete", U, tau)  -> vesicle pool x, release ∝ x, x -= U x per spike, recovery tau
        #                           (Tsodyks-Markram; a ceiling burst releases ~one pool then stops)
        if da_release is None:
            S_da = Synapses(G, G, "n_da : 1", on_pre="m_post += n_da", name="syn_da")
        elif da_release[0] == "supra":
            S_da = Synapses(G, G, "n_da : 1", on_pre="m_post += n_da*(r_pre/r_base)", name="syn_da",
                            namespace=dict(r_base=float(da_release[1]) * Hz))
        elif da_release[0] == "deplete":
            S_da = Synapses(G, G, "n_da : 1\n dx/dt = (1 - x)/tau_rec : 1 (clock-driven)",
                            on_pre="m_post += n_da*x\n x -= U*x", name="syn_da", method="exact",
                            namespace=dict(U=float(da_release[1]), tau_rec=float(da_release[2]) * second))
        else:
            raise ValueError(f"unknown da_release {da_release}")
        S_da.connect(i=np.asarray(da["pre_idx"], np.int64), j=np.asarray(da["post_idx"], np.int64))
        S_da.n_da = np.asarray(da["n_syn"], float) / float(da_norm)
        if da_release is not None and da_release[0] == "deplete":
            S_da.x = 1.0
        objs.append(S_da)
    g_syn = g_syn_ps * psiemens
    # stim may be a list of (idx, ons, offs, f) tuples (e.g. training kicks at rotating sectors)
    stims = ([(f"drive{k}", s) for k, s in enumerate(stim)] if isinstance(stim, list) else [("drive", stim)]) \
        + [("drive2", stim2)]
    for tag, sp in stims:
        if sp is None or not len(sp[0]):
            continue
        idx, on, off, f = sp                       # on/off may be lists of windows
        steps = int(round(duration_s / 0.005))
        rr = np.zeros(steps)
        ons = np.atleast_1d(on); offs = np.atleast_1d(off)
        for a, b in zip(ons, offs):
            rr[int(round(a / 0.005)):int(round(b / 0.005))] = STIM_HZ
        gate = TimedArray(rr * Hz, dt=5 * ms)
        P = PoissonGroup(len(idx), rates="gate(t)", name=tag, namespace={"gate": gate})
        ext = Synapses(P, G, "w_ext : siemens", on_pre="g_e_post += w_ext", name=f"{tag}_syn")
        ext.connect(i=np.arange(len(idx)), j=np.asarray(idx)); ext.w_ext = f * g_syn * q_scale
        objs += [P, ext]
    if sens_hz > 0 and len(data["sens_idx"]):
        Sg = PoissonGroup(len(data["sens_idx"]), rates=sens_hz * Hz, name="sens")
        ss = Synapses(Sg, G, "w_s : siemens", on_pre="g_e_post += w_s", name="sens_syn")
        ss.connect(i=np.arange(len(data["sens_idx"])), j=data["sens_idx"]); ss.w_s = SENS_KICK_F * g_syn * q_scale
        objs += [Sg, ss]
    spikes = SpikeMonitor(G, record=record_spikes)
    net = Network(*objs, spikes)
    return net, G, syn, spikes, masks


def build_shiu_network(data, sens_hz=0.0, stim=None, stim2=None, duration_s=2.0, record_spikes=True, g_reset=True,
                       stim_v=False):
    """Shiu et al. 2024 (model.py) on this graph: dv/dt = (v_0 - v + g)/t_mbr, dg/dt = -g/tau_syn, g in
    volts, both frozen while refractory; a presynaptic spike adds w = n_syn * sign * W_SYN_SHIU to g
    after T_DLY_SHIU; at threshold v = v_rst and g = 0 (g_reset=False: v only). The same constants for
    every neuron (no area), signs from the v2 edge-class masks. stim / stim2 / sens_hz as in
    build_network; an event of f synapses adds f * W_SYN_SHIU to g (no delay). stim_v=True is the
    published stimulation instead (model.py PoissonInput on v, weight w_syn * f_poi): every event
    jumps v by F_POI_SHIU * W_SYN_SHIU whatever f is, so a stimulated neuron fires once per event
    outside its refractory period (the published code also sets that period to zero for stimulated
    neurons; here it is kept). Returns what build_network returns, with one Synapses object, syn["all"]."""
    n = data["n"]
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau_syn : volt (unless refractory)
    """
    reset = "v = v_rst; g = 0*mV" if g_reset else "v = v_rst"
    G = NeuronGroup(n, eqs, threshold="v > v_th", reset=reset, refractory=BS.t_rfc, method="exact",
                    name="net", namespace=dict(NAMESPACE))
    G.v = BS.v_0
    _, masks, _ = v2_weights(data)                        # edge classes only; the v2 conductances are not used
    exc = masks["ee"] | masks["ei"]
    S = Synapses(G, G, "w : volt", on_pre="g_post += w", delay=BS.T_DLY_SHIU, name="syn_all")
    S.connect(i=data["pre_idx"], j=data["post_idx"])
    S.w = data["n_syn"] * np.where(exc, 1.0, -1.0) * BS.W_SYN_SHIU
    objs = [G, S]
    # stim may be a list of (idx, ons, offs, f) tuples, as in build_network
    stims = ([(f"drive{k}", s) for k, s in enumerate(stim)] if isinstance(stim, list) else [("drive", stim)]) \
        + [("drive2", stim2)]
    for tag, sp in stims:
        if sp is None or not len(sp[0]):
            continue
        idx, on, off, f = sp                       # on/off may be lists of windows
        steps = int(round(duration_s / 0.005))
        rr = np.zeros(steps)
        ons = np.atleast_1d(on); offs = np.atleast_1d(off)
        for a, b in zip(ons, offs):
            rr[int(round(a / 0.005)):int(round(b / 0.005))] = STIM_HZ
        gate = TimedArray(rr * Hz, dt=5 * ms)
        P = PoissonGroup(len(idx), rates="gate(t)", name=tag, namespace={"gate": gate})
        ext = Synapses(P, G, "w_ext : volt", on_pre=f"{'v' if stim_v else 'g'}_post += w_ext", name=f"{tag}_syn")
        ext.connect(i=np.arange(len(idx)), j=np.asarray(idx)); ext.w_ext = (BS.F_POI_SHIU if stim_v else f) * BS.W_SYN_SHIU
        objs += [P, ext]
    if sens_hz > 0 and len(data["sens_idx"]):
        Sg = PoissonGroup(len(data["sens_idx"]), rates=sens_hz * Hz, name="sens")
        ss = Synapses(Sg, G, "w_s : volt", on_pre=f"{'v' if stim_v else 'g'}_post += w_s", name="sens_syn")
        ss.connect(i=np.arange(len(data["sens_idx"])), j=data["sens_idx"]); ss.w_s = SENS_KICK_F * BS.W_SYN_SHIU
        objs += [Sg, ss]
    spikes = SpikeMonitor(G, record=record_spikes)
    net = Network(*objs, spikes)
    return net, G, {"all": S}, spikes, masks


def apply_e_decay(G, S_ee, dt_s, gamma_e=GAMMA_E, floor_e=FLOOR_E, rho0=RHO0_HZ, post_idx=None,
                  two_sided=False, exc_in=None, rate_hz=None, rho_low=RHO_LOW_HZ, gamma_up=GAMMA_UP,
                  wmax_e=WMAX_E):
    """vectorised E->E homeostasis between chunks. Brake: r > rho0 -> multiplicative decay
    (exact exponential for constant r), floored. Two-sided: r < rho_low AND excitatory
    spikes arrived in the chunk (exc_in > 0) -> multiplicative growth, capped at wmax_e."""
    pl = np.asarray(G.pl)
    r = np.asarray(G.r / Hz) if rate_hz is None else rate_hz
    over = np.maximum(0.0, r - rho0) * pl                          # sensory targets excluded
    j = post_idx if post_idx is not None else np.asarray(S_ee.j)
    w = np.asarray(S_ee.w / siemens); w0 = np.asarray(S_ee.w0 / siemens)
    fac = np.exp(-dt_s * gamma_e * over)
    n_up = 0
    if two_sided and exc_in is not None:
        under = pl * (r < rho_low) * (exc_in > 0)
        n_up = int(under.sum())
        fac = fac * np.exp(dt_s * gamma_up * under * (rho_low - np.minimum(r, rho_low)) / rho_low)
    w_new = np.clip(w * fac[j], floor_e * w0, wmax_e * w0)
    S_ee.w = w_new * siemens
    return float(np.mean(w_new / w0)), n_up


def run(net, seconds):
    t0 = time.time()
    net.run(seconds * second, namespace=NAMESPACE)
    return time.time() - t0
