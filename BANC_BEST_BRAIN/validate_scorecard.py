"""
Held-out scorecard on the unified Brain+VNC graph, with frozen plasticity
gains (--gains w_final_TAG.npz) or the plain v2 initialisation (--gains none,
the control). Protocols: Kim 2017 bump, PEN_a rotation,
feeding (sugar / bitter / water / sugar+bitter), looming -> giant fiber (+
LC10a control), olfaction -> KC sparseness, resting state under 10 Hz sensory.

usage: validate_scorecard.py --gains none|FILE [--only kim,pen,feeding,looming,olfaction,rest] [--gsyn 30] [--model shiu[-vstim][-noreset]]
writes scorecard_<tag>.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
BULK = Path(os.environ.get("BANC_BULK", OUT.parent / "bulk"))   # BANC tables (banc_888_meta.feather, for feeding)


def arg(flag, default, cast=str):
    a = sys.argv
    return cast(a[a.index(flag) + 1]) if flag in a else default


def vals(flag):
    """numeric values following FLAG (stops at the first non-numeric token)"""
    a = sys.argv
    if flag not in a:
        return []
    out = []
    for tok in a[a.index(flag) + 1:]:
        try:
            out.append(float(tok))
        except ValueError:
            break
    return out


DEVICE = arg("--device", "runtime")                     # runtime | cpp_standalone | cuda_standalone
if DEVICE != "runtime":
    import brian2
    if DEVICE == "cuda_standalone":
        import brian2cuda                                # noqa: F401  (registers the device)
    brian2.set_device(DEVICE, build_on_run=False)
import plasticity as PL                                  # noqa: E402
from provenance import provenance                        # noqa: E402
from brian2 import second, siemens, start_scope, seed, device, prefs   # noqa: E402
BS, CX = PL.BS, PL.BS.CX
N_RUN = [0]
SEEDS = []                                               # seed of every sim() call, in call order
TAU_E = None                                             # --tau-e MS: excitatory synaptic time constant
LIVE = None                                              # --live: rule params (from the gains file) or None
T_PRE = 3.0                                              # --live: settle prefix before each protocol (s)
CORE_EE = None                                           # --live: CX core->core mask on the E->E edges
ADAPT = None                                             # --adapt ALPHA_MV TAU_MS: adaptive threshold
ESCAPE = None                                            # --escape DELTA_MV [LAM0_HZ]: soft threshold
TWO_COMP = None                                          # --two-comp RHO [F_S]: dendrite + soma neuron
RECIP = None                                             # --recip KAPPA: reciprocity rule (plasticity.reciprocity_factor)
SAVE_RATES = ""                                          # --save-rates FILE: kimrest saves per-neuron rates after light-off
RECIP_EE = None                                          # --recip-ee KAPPA: reciprocity rule on E->E synapses only
RECIP_EEN = None                                         # --recip-ee-norm KAPPA: same, per-neuron E drive conserved
STD = None                                               # --std U,TAU_REC_S: short-term depression on excitatory synapses
MODEL = "cond"                                           # --model shiu|shiu-noreset: Shiu et al. 2024 formulation (without the g reset)


def sim(data, gains, T, stim=None, stim2=None, sens=0.0, seed_val=1, gsyn=30.0):
    if DEVICE != "runtime":
        # standalone: one compiled program per test
        device.reinit(); device.activate(build_on_run=False, directory=f"/tmp/brian_sc_{N_RUN[0]}")
        if DEVICE == "cpp_standalone" and "--openmp" in sys.argv:
            prefs.devices.cpp_standalone.openmp_threads = int(arg("--openmp", 8))
        N_RUN[0] += 1
    start_scope(); seed(seed_val); SEEDS.append(seed_val)
    if LIVE is None:
        net, G, syn, spikes, masks = PL.build_network(data, g_syn_ps=gsyn, plastic=False, gains=gains,
                                                      sens_hz=sens, stim=stim, stim2=stim2, duration_s=T,
                                                      tau_e=TAU_E, adapt=ADAPT, escape=ESCAPE, two_comp=TWO_COMP,
                                                      recip=RECIP, recip_ee=RECIP_EE, recip_een=RECIP_EEN, std=STD,
                                                      model=MODEL.split("-")[0], g_reset=not MODEL.endswith("noreset"), stim_v="vstim" in MODEL)
    else:
        # the rule keeps running during the experiment (warm start = the saved weights, which
        # become the rule's floor w0); every test runs in the resting brain the rule was tuned in,
        # after a T_PRE settle; the protocol windows are shifted by T_PRE and spike times shifted back
        p = LIVE
        shift = lambda sp: None if sp is None else (sp[0], np.atleast_1d(sp[1]) + T_PRE, np.atleast_1d(sp[2]) + T_PRE, sp[3])
        net, G, syn, spikes, masks = PL.build_network(
            data, g_syn_ps=gsyn, plastic=True, gains=gains, sens_hz=p["sens"], stim=shift(stim), stim2=shift(stim2),
            duration_s=T + T_PRE, eta_i=p["eta_i"], wmax_i=p["wmax_i"], eta_e=p["eta_e"], floor_e=p["floor_e"],
            tau_e=TAU_E, in_model=True, gamma_e=p["gamma_e"], e_start_s=0.0,
            two_sided=p["two_sided"], rho_low=p["rho_low"], gamma_up=p["gamma_up"],
            adapt=ADAPT, escape=ESCAPE, two_comp=TWO_COMP, recip=RECIP, recip_ee=RECIP_EE, recip_een=RECIP_EEN, std=STD,
            model=MODEL.split("-")[0], g_reset=not MODEL.endswith("noreset"), stim_v="vstim" in MODEL)
        T = T + T_PRE
    tr = PL.run(net, T)
    if DEVICE != "runtime":
        import time as _t; t0 = _t.time()
        device.build(run=True)
        tr = _t.time() - t0
    st, si = np.asarray(spikes.t / second), np.asarray(spikes.i)
    if LIVE is not None:
        # how much the experiment rewrote the brain (gain relative to the warm start)
        w_ee = np.asarray(syn["ee"].w / siemens) / np.asarray(syn["ee"].w0 / siemens)
        w_ie = np.asarray(syn["ie"].w / siemens) / np.asarray(syn["ie"].w0 / siemens)
        print(f"   drift | <wE> {w_ee.mean():.3f} (CX core {w_ee[CORE_EE].mean():.3f})  <wI> {w_ie.mean():.3f} "
              f"(p90 {np.percentile(w_ie, 90):.2f}) over {T:g} s incl. {T_PRE:g} s settle", flush=True)
        k = st >= T_PRE; st, si = st[k] - T_PRE, si[k]
    return st, si, tr


def rates(st, si, n, a, b):
    k = (st > a) & (st < b)
    return np.bincount(si[k], minlength=n) / (b - a)


def main() -> int:
    global TAU_E
    gpath = arg("--gains", "none"); gsyn = arg("--gsyn", 30.0, float)
    if "--tau-e" in sys.argv:
        from brian2 import ms
        TAU_E = arg("--tau-e", 5.0, float) * ms
        print(f"excitatory tau_syn = {TAU_E}")
    if "--vrest" in sys.argv:                            # V_rest = V_reset override (mV)
        from brian2 import mV
        BS.v_0 = BS.v_rst = arg("--vrest", -52.0, float) * mV
        PL.NAMESPACE.update(v_0=BS.v_0, v_rst=BS.v_rst)
        print(f"V_rest = V_reset = {BS.v_0}")
    if "--einh" in sys.argv:                             # inhibitory reversal override (mV); balance rule follows
        from brian2 import mV
        BS.E_inh = arg("--einh", -70.0, float) * mV
        PL.NAMESPACE.update(E_inh=BS.E_inh)
        print(f"E_inh = {BS.E_inh} (balance factor {(BS.E_exc - BS.v_th) / (BS.v_th - BS.E_inh):.2f})")
    if "--trfc" in sys.argv:                             # absolute refractory period override (ms)
        from brian2 import ms
        BS.t_rfc = arg("--trfc", 2.2, float) * ms
        print(f"t_rfc = {BS.t_rfc}")
    only = arg("--only", "kim,kimrest,pen,feeding,looming,olfaction,rest").split(",")
    tag = "v2init" if gpath == "none" else Path(gpath).stem.replace("w_final_", "")
    dpath = arg("--data", str(OUT / "banc_full.npz"))
    global LIVE, CORE_EE, ADAPT, ESCAPE, TWO_COMP, STD, MODEL
    if "--model" in sys.argv:                            # the published formulation as a baseline (plasticity.build_shiu_network)
        MODEL = arg("--model", "shiu")
        assert MODEL in ("shiu", "shiu-noreset", "shiu-vstim", "shiu-vstim-noreset"), "--model takes shiu[-vstim][-noreset]"
        tag += f"_{MODEL}"
        print(f"model: Shiu et al. 2024 voltage-jump synapses, w_syn {BS.W_SYN_SHIU}, delay {BS.T_DLY_SHIU}, uniform leak, "
              f"g reset {'off' if MODEL.endswith('noreset') else 'on'}, stimulation on {'v (published)' if 'vstim' in MODEL else 'g (f synapses)'}; G_E, gains and the conductance-model options do not apply")
    if "--std" in sys.argv:                              # short-term depression (plant physics, universal)
        parts = arg("--std", "0.2,0.2").split(",")            # U,TAU[,recip]: recip -> U_ij = U * rho_ij (edge reciprocity)
        STD = (float(parts[0]), float(parts[1])) + ((parts[2],) if len(parts) > 2 else ())   # recip | recip_ee
        tag += f"_std{STD[2] if len(STD) > 2 else ''}U{STD[0]:g}t{STD[1]:g}"
        print(f"short-term depression on excitatory synapses: U {STD[0]:g}{' x edge reciprocity (' + STD[2] + ')' if len(STD) > 2 else ''}, tau_rec {STD[1]:g} s")
    if "--two-comp" in sys.argv:                         # dendrite + soma neuron (plant physics)
        v = vals("--two-comp"); TWO_COMP = (v[0], v[1] if len(v) > 1 else 0.5)
        print(f"two-compartment neuron: rho = g_c/g_L {TWO_COMP[0]:g}, somatic fraction f_s {TWO_COMP[1]:g}")
    if "--adapt" in sys.argv:                            # intrinsic rate control (plant physics)
        v = vals("--adapt"); ADAPT = (v[0], v[1] if len(v) > 1 else PL.ADAPT_TAU_MS)
        print(f"adaptive threshold: alpha {ADAPT[0]:g} mV, tau_a {ADAPT[1]:g} ms")
    if "--escape" in sys.argv:
        v = vals("--escape"); ESCAPE = (v[0], v[1] if len(v) > 1 else PL.ESCAPE_LAM0_HZ)
        print(f"escape noise: delta {ESCAPE[0]:g} mV, lam0 {ESCAPE[1]:g} Hz")
    if "--live" in sys.argv:
        # the plasticity rule stays ON during every test, with the constants the settle used
        assert gpath != "none", "--live needs a gains file (its params hold the rule constants)"
        p = json.loads(str(np.load(gpath)["params"]))
        LIVE = dict(sens=float(p.get("sens", 10.0)), eta_i=float(p["eta_i"]), wmax_i=float(p["wmax_i"]),
                    eta_e=float(p.get("eta_e", 0.0)), floor_e=float(p["floor_e"]), gamma_e=float(p["gamma_e"]),
                    two_sided=bool(p.get("two_sided", False)), rho_low=float(p.get("rho_low", PL.RHO_LOW_HZ)),
                    gamma_up=float(p.get("gamma_up", PL.GAMMA_UP)), rho0=float(p["rho0"]),
                    vis_sens=bool(p.get("vis_sens", False)), tau_e_ms=float(p.get("tau_e_ms", 5.0)))
        PL.RHO0_HZ = LIVE["rho0"]; PL.STDP_START_S = 0.0     # module constants the rule reads
        if ADAPT is None and p.get("adapt"):
            ADAPT = tuple(p["adapt"])
        if ESCAPE is None and p.get("escape"):
            ESCAPE = tuple(p["escape"])
        if TAU_E is None and LIVE["tau_e_ms"] != 5.0:
            from brian2 import ms
            TAU_E = LIVE["tau_e_ms"] * ms
        tag += "_live"
        print(f"LIVE: rule on during every test; eta_i {LIVE['eta_i']:g} wmax_i {LIVE['wmax_i']:g} rho0 {LIVE['rho0']:g} Hz "
              f"gamma_e {LIVE['gamma_e']:g} eta_e {LIVE['eta_e']:g}"
              + (f" two-sided {LIVE['rho_low']:g}-{LIVE['rho0']:g} Hz gamma_up {LIVE['gamma_up']:g}" if LIVE["two_sided"] else "")
              + f"; tau_e {TAU_E if TAU_E is not None else '5 ms'}; resting brain (sensory {LIVE['sens']:g} Hz"
              + (" + visual projection neurons" if LIVE["vis_sens"] else "") + f"), {T_PRE:g} s settle before each protocol")
    data = PL.load_data(dpath, vis_sens=LIVE["vis_sens"] if LIVE else False); n = data["n"]
    print(f"graph: {Path(dpath).name}")
    ct, sc, side = data["cell_type"], data["super_class"], data["side"]
    cts = pd.Series(ct)
    gains = None
    if gpath != "none":
        g = np.load(gpath); gains = dict(ee=g["ee"], ie=g["ie"])
        print(f"gains from {gpath}: <ee> {g['ee'].mean():.3f}  <ie> {g['ie'].mean():.3f} (p90 {np.percentile(g['ie'],90):.2f})")
        if "--gains-class" in sys.argv:                  # ablation: keep one learned class, the other at v2 (ones)
            cls = arg("--gains-class", "ie"); other = "ee" if cls == "ie" else "ie"
            gains[other] = np.ones_like(gains[other]); tag += f"_g{cls}"
            print(f"gains class {cls} only ({other} = 1)")
    exc = data["sign"][data["pre_idx"]] > 0
    ee = exc & (data["sign"][data["post_idx"]] > 0); ie = ~exc & (data["sign"][data["post_idx"]] > 0)
    core_edge = data["is_cx_core"][data["pre_idx"]] & data["is_cx_core"][data["post_idx"]]
    if LIVE:
        CORE_EE = core_edge[ee]
    if "--scale" in sys.argv:
        # fitting probe (NOT the model): --scale PRE>POST:K, repeatable. PRE/POST = core | exc | inh |
        # any | regex on cell_type. Scales the v2 weight of every edge whose pre and post match.
        _, masks_all, _ = PL.v2_weights(data, gsyn)
        if gains is None:
            gains = {}
        gains = {k: (np.asarray(gains[k], float).copy() if k in gains else np.ones(int(masks_all[k].sum())))
                 for k in ("ee", "ie", "ei", "ii")}

        def match(pat):
            if pat == "core":
                return data["is_cx_core"]
            if pat == "exc":
                return data["sign"] > 0
            if pat == "inh":
                return data["sign"] < 0
            if pat == "any":
                return np.ones(n, bool)
            import re
            return np.array([bool(re.fullmatch(pat, c)) for c in ct])
        specs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--scale"]
        for spec in specs:
            parts = spec.split(":")                       # PRE,POST:K[:CLASS]  (CLASS in ee/ei/ie/ii)
            pre_pat, post_pat = parts[0].split(","); kk = float(parts[1])
            classes = (parts[2],) if len(parts) > 2 else ("ee", "ie", "ei", "ii")
            em = match(pre_pat)[data["pre_idx"]] & match(post_pat)[data["post_idx"]]
            n_e = 0
            for k in classes:
                sel = em[masks_all[k]]
                gains[k][sel] *= kk; n_e += int(sel.sum())
            print(f"scale {pre_pat} > {post_pat} x {kk:g} [{','.join(classes)}]: {n_e:,} edges")
        tag += "_" + "_".join(s.replace(">", "-").replace(":", "x").replace("*", "").replace(".", "")
                              .replace("(", "").replace(")", "") for s in specs)
    if "--recip-ee" in sys.argv:
        # E by structure: reciprocity on E->E synapses only (inhibition left to iSTDP)
        global RECIP_EE
        RECIP_EE = arg("--recip-ee", 0.3, float)
        fac = PL.reciprocity_factor(data, RECIP_EE); pre_i, post_i = data["pre_idx"], data["post_idx"]
        ee_m = (data["sign"][pre_i] > 0) & (data["sign"][post_i] > 0)
        core_e = data["is_cx_core"][pre_i] & data["is_cx_core"][post_i] & ee_m
        tag += f"_recipee{RECIP_EE:g}"
        print(f"reciprocity on E->E only, kappa {RECIP_EE:g}: CX core->core {fac[core_e].mean():.2f}, "
              f"KC->KC {fac[np.char.startswith(ct[pre_i], 'KC') & np.char.startswith(ct[post_i], 'KC')].mean():.2f}, "
              f"PN->KC {fac[np.char.endswith(ct[pre_i], 'PN') & np.char.startswith(ct[post_i], 'KC')].mean():.2f}, "
              f"all E->E {fac[ee_m].mean():.2f}")
    if "--recip-ee-norm" in sys.argv:
        # E by structure, drive-conserving: reciprocity redistributes each neuron's E input
        global RECIP_EEN
        RECIP_EEN = arg("--recip-ee-norm", 1.0, float)
        pre_i, post_i = data["pre_idx"], data["post_idx"]
        ee_m = (data["sign"][pre_i] > 0) & (data["sign"][post_i] > 0)
        fn = PL.reciprocity_factor_norm(data, RECIP_EEN, ee_m); ns_ = data["n_syn"]
        def wm(sel):
            return float(np.average(fn[sel], weights=ns_[sel])) if sel.any() else float("nan")
        core_e = data["is_cx_core"][pre_i] & data["is_cx_core"][post_i] & ee_m
        tag += f"_recipeen{RECIP_EEN:g}"
        print(f"drive-conserving reciprocity on E->E, kappa {RECIP_EEN:g}: CX core->core {wm(core_e):.2f}, "
              f"PEN->EPG {wm(np.char.startswith(ct[pre_i], 'PEN') & (ct[post_i] == 'EPG') & ee_m):.2f}, "
              f"PEG->EPG {wm((ct[pre_i] == 'PEG') & (ct[post_i] == 'EPG') & ee_m):.2f}, "
              f"KC->KC {wm(np.char.startswith(ct[pre_i], 'KC') & np.char.startswith(ct[post_i], 'KC') & ee_m):.2f}, "
              f"PN->KC {wm(np.char.endswith(ct[pre_i], 'PN') & np.char.startswith(ct[post_i], 'KC') & ee_m):.2f}, "
              f"all E->E {wm(ee_m):.3f}")
    if "--recip" in sys.argv:
        # candidate universal rule (plasticity.reciprocity_factor): w_ji = n_ji G (1 + kappa n_ij).
        # Applied inside build_network as part of the plant (so a settle and the scorecard agree).
        global RECIP
        kappa = arg("--recip", 0.2, float); RECIP = kappa
        fac = PL.reciprocity_factor(data, kappa); has = fac > 1.0
        pre_i, post_i = data["pre_idx"], data["post_idx"]
        tag += f"_recip{kappa:g}"
        core_edge_ = data["is_cx_core"][pre_i] & data["is_cx_core"][post_i]
        print(f"reciprocity rule kappa {kappa:g}: reciprocal edges {int(has.sum()):,} of {len(has):,}; "
              f"mean factor all {fac.mean():.2f}, CX core->core {fac[core_edge_].mean():.2f}, "
              f"ExR->EPG {fac[np.char.startswith(ct[pre_i], 'ExR') & (ct[post_i] == 'EPG')].mean():.2f}, "
              f"Delta7->EPG {fac[(ct[pre_i] == 'Delta7') & (ct[post_i] == 'EPG')].mean():.2f}, "
              f"ER->EPG {fac[np.char.startswith(ct[pre_i], 'ER') & (ct[post_i] == 'EPG')].mean():.2f}, "
              f"KC->KC {fac[np.char.startswith(ct[pre_i], 'KC') & np.char.startswith(ct[post_i], 'KC')].mean():.2f}, "
              f"APL->KC {fac[(ct[pre_i] == 'APL') & np.char.startswith(ct[post_i], 'KC')].mean():.2f}")
    if "--fbfrac" in sys.argv:
        # candidate universal rule 2: a neuron's output synapses are scaled by its FEEDBACK FRACTION
        # F_j = (synapses j receives from neurons it projects to) / (all synapses j receives):
        # w_ji = n_ji G (1 + kappa F_j). Inhibitors driven by their own targets (ExR6, Δ7) are
        # strengthened; feed-forward ones (ER from TuBu) are not. One constant.
        kappa = arg("--fbfrac", 10.0, float)
        _, masks_all, _ = PL.v2_weights(data, gsyn)
        if gains is None:
            gains = {}
        gains = {k: (np.asarray(gains[k], float).copy() if k in gains else np.ones(int(masks_all[k].sum())))
                 for k in ("ee", "ie", "ei", "ii")}
        pre_i, post_i, ns_e = data["pre_idx"], data["post_idx"], data["n_syn"]
        key = pre_i.astype(np.int64) * n + post_i; order = np.argsort(key); ks = key[order]
        rev = post_i.astype(np.int64) * n + pre_i
        j = np.minimum(np.searchsorted(ks, rev), len(ks) - 1); has = ks[j] == rev
        # for edge j->i with a reverse edge i->j: the reverse synapses are input to j from its target i
        fb_in = np.bincount(pre_i, weights=np.where(has, ns_e[order][j], 0.0), minlength=n)
        tot_in = np.bincount(post_i, weights=ns_e, minlength=n)
        F = np.where(tot_in > 0, fb_in / np.maximum(tot_in, 1), 0.0)
        fac = 1.0 + kappa * F[pre_i]
        for k in ("ee", "ie", "ei", "ii"):
            gains[k] *= fac[masks_all[k]]
        tag += f"_fb{kappa:g}"
        def _F(t):
            m = ct == t
            return float(F[m].mean()) if m.any() else float("nan")
        print(f"feedback-fraction rule kappa {kappa:g}: F(ExR6) {_F('ExR6'):.2f} F(Delta7) {_F('Delta7'):.2f} "
              f"F(ER4d) {_F('ER4d'):.2f} F(EPG) {_F('EPG'):.2f} F(PEN_a) {_F('PEN_a(PEN1)'):.2f} "
              f"F(APL) {_F('APL'):.2f} F(KC mean) {float(F[np.char.startswith(ct, 'KC')].mean()):.2f} "
              f"F(all) {F.mean():.2f}; mean factor {fac.mean():.2f}")
    if "--save-rates" in sys.argv or "--actbal" in sys.argv:
        global SAVE_RATES
        SAVE_RATES = arg("--save-rates", "", str)
        if "--actbal" in sys.argv:
            # candidate universal rule 3: activity-weighted balance. The v2 rule balances E and I
            # synapse COUNTS at threshold (assumes equal presynaptic rates); here the balance is
            # evaluated on the measured rates of a reference state (r from --save-rates), with each
            # inhibitory synapse weighted by its presynaptic rate:
            #   w_ji (inh) = n_ji rho G_E (sum_exc n r) / (sum_inh n r^2) * r_j,   r -> r + r0 (floor 1 Hz)
            # = the static fixed point of Vogels' rule for that state. Applied to ie and ii.
            rf = arg("--actbal", "", str); r0 = arg("--r0", 1.0, float)
            r = np.load(rf)["hz"].astype(float) + r0
            _, masks_all, _ = PL.v2_weights(data, gsyn)
            if gains is None:
                gains = {}
            gains = {k: (np.asarray(gains[k], float).copy() if k in gains else np.ones(int(masks_all[k].sum())))
                     for k in ("ee", "ie", "ei", "ii")}
            pre_i, post_i, ns_e = data["pre_idx"], data["post_idx"], data["n_syn"]
            exc_pre = data["sign"][pre_i] > 0
            E_act = np.bincount(post_i[exc_pre], weights=(ns_e * r[pre_i])[exc_pre], minlength=n)
            I_act = np.bincount(post_i[~exc_pre], weights=(ns_e * r[pre_i] ** 2)[~exc_pre], minlength=n)
            E_cnt = np.bincount(post_i[exc_pre], weights=ns_e[exc_pre], minlength=n)
            I_cnt = np.bincount(post_i[~exc_pre], weights=ns_e[~exc_pre], minlength=n)
            ok = (I_act > 0) & (I_cnt > 0) & (E_cnt > 0)
            # factor relative to the v2 count-balance weight n rho G_E E_cnt/I_cnt
            per_post = np.ones(n)
            per_post[ok] = (E_act[ok] / I_act[ok]) / (E_cnt[ok] / I_cnt[ok])
            fac = np.where(exc_pre, 1.0, per_post[post_i] * r[pre_i])
            for k in ("ie", "ii"):
                gains[k] *= fac[masks_all[k]]
            tag += "_actbal"
            def _f(tp, tt):
                m = (ct[pre_i] == tp) & (ct[post_i] == tt)
                return float(fac[m].mean()) if m.any() else float("nan")
            print(f"activity-weighted balance from {Path(rf).name} (r0 {r0:g} Hz): factor ExR6->EPG {_f('ExR6', 'EPG'):.2f} "
                  f"Delta7->EPG {_f('Delta7', 'EPG'):.2f} ER4d->EPG {_f('ER4d', 'EPG'):.2f} ExR6->PEN_a {_f('ExR6', 'PEN_a(PEN1)'):.2f} "
                  f"APL->KC {float(fac[(ct[pre_i] == 'APL') & np.char.startswith(ct[post_i], 'KC')].mean()):.2f} "
                  f"all I {float(fac[~exc_pre].mean()):.2f} (median {float(np.median(fac[~exc_pre])):.2f})")
    if "--kcx" in sys.argv:
        # ONE circuit-specific number (stated as such): scale the CX-core loop's E and I synapses
        # together (balance kept) -- the fluctuation-driven state needs the loop stronger, not weaker
        kcx = arg("--kcx", 1.0, float)
        if gains is None:
            gains = dict(ee=np.ones(int(ee.sum())), ie=np.ones(int(ie.sum())))
        gains = dict(ee=gains["ee"].copy(), ie=gains["ie"].copy())
        kcx_i = arg("--kcx-i", 1.0, float)               # extra factor on the loop's I->E only (2nd number)
        gains["ee"][core_edge[ee]] *= kcx; gains["ie"][core_edge[ie]] *= kcx * kcx_i
        tag += f"_kcx{kcx:g}" + (f"i{kcx_i:g}" if kcx_i != 1.0 else "")
        print(f"k_CX = {kcx:g} on {int(core_edge[ee].sum()):,} E->E and {int(core_edge[ie].sum()):,} I->E CX-core edges"
              + (f"; I->E x {kcx_i:g} extra" if kcx_i != 1.0 else ""))
    print(f"scorecard [{tag}] on {n:,} neurons, G_E {gsyn:g} pS\n")
    res = {}
    prov = provenance()

    # ---- CX mapping (as brain_sim.py)
    cx = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
    cx_root = cx["root_id"].astype(np.int64); cx_lab = cx["labels"].astype(str); cx_ang = cx["angles"]
    cx_epg = np.flatnonzero(cx_lab == "EPG"); cx_order = cx_epg[np.argsort(cx_ang[cx_epg])]
    Wcx = cx["W"].astype(float)
    hub = cx_epg[np.argmax(np.abs(Wcx[np.ix_(cx_epg, cx_epg)]).sum(1))]
    hp = int(np.flatnonzero(cx_order == hub)[0]); n_stim = int(round(45 / 360.0 * len(cx_order)))
    cx_seed = cx_order[[(hp - n_stim // 2 + k) % len(cx_order) for k in range(n_stim)]]
    pos = {int(r): i for i, r in enumerate(data["root"])}
    epg = np.array([pos[int(cx_root[e])] for e in cx_order]); angles = cx_ang[cx_order]
    stim_epg = np.array([pos[int(cx_root[e])] for e in cx_seed])
    KIM_ON, KIM_OFF, T_KIM = 0.5, 1.547, 4.0
    KIM_F = arg("--stim-f", 250.0, float)                # Kim drive, synapses per stimulus spike (titration)
    if KIM_F != 250.0:
        print(f"Kim drive: {KIM_F:g} synapses per stimulus spike (default 250)")

    if "kim" in only:
        st, si, tr = sim(data, gains, T_KIM, stim=(stim_epg, KIM_ON, KIM_OFF, KIM_F), seed_val=100, gsyn=gsyn)
        t_axis = np.arange(int(round(T_KIM / (BS.DT_BIN_MS / 1000)))) * BS.DT_BIN_MS / 1000
        r_epg = BS.bin_rates(st, si, n, T_KIM, epg)
        a = CX.analyse(r_epg, np.arange(len(epg)), np.arange(len(epg)), angles, t_axis)
        aft = t_axis > KIM_OFF + 0.05
        fwhm = BS.bump_fwhm_deg(r_epg[:, aft].mean(1))
        hz_a = rates(st, si, n, KIM_OFF + 0.05, T_KIM)
        res["kim"] = dict(n_epg=int(a["n_active_after"]), hz=float(a["active_hz_after"]), fwhm=fwhm,
                          R=float(a["R_after"]), alive=float(a["live_frac"]), active_after=int((hz_a > 2).sum()))
        print(f"KIM      | EPG {res['kim']['n_epg']:2d} active @ {res['kim']['hz']:4.0f} Hz  FWHM {fwhm:3.0f} deg  "
              f"R {res['kim']['R']:.2f} alive {res['kim']['alive']:.2f} | brain active after {res['kim']['active_after']:,} "
              f"| target: persistent, ~90 deg, <160 Hz  [{tr:.0f}s]", flush=True)

    if "kimrest" in only:
        # the same protocol inside the resting brain the weights were tuned for
        # (spontaneous drive on, incl. visual projection neurons if the run used them)
        vis = (bool(json.loads(str(np.load(gpath)["params"])).get("vis_sens", False)) if gpath != "none" else False) \
            or ("--vis" in sys.argv)
        print(f"resting brain: sensory 10 Hz{' + visual projection neurons' if vis else ''}")
        data_r = PL.load_data(dpath, vis_sens=vis)
        st, si, tr = sim(data_r, gains, T_KIM, stim=(stim_epg, KIM_ON, KIM_OFF, KIM_F), sens=10.0,
                         seed_val=arg("--seed", 100, int), gsyn=gsyn)
        t_axis = np.arange(int(round(T_KIM / (BS.DT_BIN_MS / 1000)))) * BS.DT_BIN_MS / 1000
        r_epg = BS.bin_rates(st, si, n, T_KIM, epg)
        a = CX.analyse(r_epg, np.arange(len(epg)), np.arange(len(epg)), angles, t_axis)
        pre = rates(st, si, n, 0.3, KIM_ON); aft_r = rates(st, si, n, KIM_OFF + 0.05, T_KIM)   # skip onset transient
        fwhm = BS.bump_fwhm_deg(r_epg[:, t_axis > KIM_OFF + 0.05].mean(1))
        # does the bump stay where it was put? decoded angle just after light-off vs the end
        th, _ = CX.decode(CX.smooth(r_epg), angles)
        ang = lambda t: float(np.degrees(th[int(np.argmin(np.abs(t_axis - t)))]))
        drift = (ang(T_KIM - 0.05) - ang(KIM_OFF + 0.1) + 180) % 360 - 180
        res["kimrest"] = dict(n_epg=int(a["n_active_after"]), hz=float(a["active_hz_after"]), fwhm=fwhm,
                              R=float(a["R_after"]), alive=float(a["live_frac"]), drift_deg=float(drift),
                              epg_before=float(pre[epg].mean()), epg_after=float(aft_r[epg].mean()),
                              active_before=int((pre > 2).sum()), active_after=int((aft_r > 2).sum()))
        k = res["kimrest"]
        print(f"KIM@rest | EPG before light {k['epg_before']:4.1f} Hz | after: {k['n_epg']:2d} active @ {k['hz']:4.0f} Hz  "
              f"FWHM {fwhm:3.0f} deg  R {k['R']:.2f} alive {k['alive']:.2f} drift {drift:+4.0f} deg | brain active before "
              f"{k['active_before']:,} after {k['active_after']:,} | target: persistent bump 10-40 Hz  [{tr:.0f}s]", flush=True)
        # ring profile (Hz per EPG, in ring order) before the light and after light-off: is it a bump
        # on a silent ring, or a modulation on a tonically active one?
        prof_b, prof_a = pre[epg], aft_r[epg]
        _, R_b = CX.decode(prof_b[:, None], angles)
        res["kimrest"].update(profile_before=prof_b.tolist(), profile_after=prof_a.tolist(), R_before=float(np.ravel(R_b)[0]))
        print(f"    ring | before (R {res['kimrest']['R_before']:.2f}): " + " ".join(f"{v:.0f}" for v in prof_b))
        print(f"    ring | after : " + " ".join(f"{v:.0f}" for v in prof_a), flush=True)
        if SAVE_RATES:
            np.savez_compressed(SAVE_RATES, hz=aft_r, window=(KIM_OFF + 0.05, T_KIM))
            print(f"saved per-neuron rates after light-off to {SAVE_RATES}")

    if "pen" in only:
        out = {}
        for sd in ("left", "right"):
            pen = np.flatnonzero((ct == "PEN_a(PEN1)") & (side == sd))
            st, si, tr = sim(data, gains, T_KIM, stim=(stim_epg, KIM_ON, KIM_OFF, KIM_F),
                             stim2=(pen, 2.0, 3.0, 50), seed_val=arg("--pen-seed", 100, int), gsyn=gsyn)
            t_axis = np.arange(int(round(T_KIM / (BS.DT_BIN_MS / 1000)))) * BS.DT_BIN_MS / 1000
            r_epg = CX.smooth(BS.bin_rates(st, si, n, T_KIM, epg))
            th, R = CX.decode(r_epg, angles)
            ang = lambda t: float(np.degrees(th[int(np.argmin(np.abs(t_axis - t)))]))
            d = (ang(3.0) - ang(1.9) + 180) % 360 - 180
            Rw = float(R[(t_axis > 1.9) & (t_axis < 3.0)].mean())      # a bump must exist for a shift to mean anything
            # trajectory during the drive: decoded angle every 100 ms and the unwrapped rotation
            # (sum of wrapped 100 ms steps), so a +178 endpoint shift can be told from a -182 one
            ts = np.arange(1.9, 3.05, 0.1); traj = np.array([ang(t) for t in ts])
            steps = (np.diff(traj) + 180) % 360 - 180; unwrapped = float(steps.sum())
            out[sd] = d; out[sd + "_R"] = Rw; out[sd + "_unwrapped"] = unwrapped; out[sd + "_traj"] = traj.tolist()
            print(f"PEN {sd:<5}| shift during PEN drive {d:+6.1f} deg  (bump R {Rw:.2f})  [{tr:.0f}s]", flush=True)
            print(f"         | angle 1.9->3.0 s every 100 ms: " + " ".join(f"{a:+.0f}" for a in traj)
                  + f" | unwrapped {unwrapped:+.0f} deg, max 100 ms step {np.abs(steps).max():.0f} deg", flush=True)
        res["pen"] = dict(out, opposite=bool(out["left"] * out["right"] < 0 and min(abs(out["left"]), abs(out["right"])) > 5
                                             and min(out["left_R"], out["right_R"]) > 0.5))
        print(f"         | target: opposite directions -> {'PASS' if res['pen']['opposite'] else 'FAIL'}")

    if "feeding" in only and not (BULK / "banc_888_meta.feather").exists():
        # the feeding test identifies taste neurons from the BANC metadata table, which is not redistributed
        print(f"FEED     | skipped: {BULK / 'banc_888_meta.feather'} not found (BANC data release; see README)", flush=True)
        res["feeding"] = None
    elif "feeding" in only:
        meta = pd.read_feather(BULK / "banc_888_meta.feather")
        meta = meta.set_index(meta.root_888.astype(np.int64)).reindex(data["root"])
        func = meta.cell_function_detailed.astype(str).str.lower()
        lab = meta.cell_sub_class.astype(str).str.contains("labellum").to_numpy()
        grn = {k: np.flatnonzero(lab & func.str.contains(k).to_numpy()) for k in ("sugar", "bitter", "water")}
        mn9 = {s: np.flatnonzero((ct == "MN9") & (side == s)) for s in ("left", "right")}
        is_kc = cts.str.startswith("KC").to_numpy()
        leg = (sc == "motor") & (data["region"] == "ventral_nerve_cord")
        out = {}
        for name, idx in (("sugar", grn["sugar"]), ("bitter", grn["bitter"]), ("water", grn["water"]),
                          ("sugar+bitter", np.concatenate([grn["sugar"], grn["bitter"]]))):
            st, si, tr = sim(data, gains, 2.0, stim=(idx, 0.2, 1.2, 25), gsyn=gsyn)
            hz = rates(st, si, n, 0.25, 1.2); after = rates(st, si, n, 1.25, 2.0)
            out[name] = dict(mn9=float((hz[mn9["left"]].mean() + hz[mn9["right"]].mean()) / 2),
                             active=int((hz > 2).sum()), after=int((after > 2).sum()),
                             kc=float(hz[is_kc].mean()), legmn=float(hz[leg].mean()))
            print(f"FEED {name:<12}| MN9 {out[name]['mn9']:6.1f} Hz | active {out[name]['active']:>6,} after {out[name]['after']:>6,} "
                  f"| KC {out[name]['kc']:.2f} Hz  legMN {out[name]['legmn']:.1f} Hz  [{tr:.0f}s]", flush=True)
        s, b, w, sb = out["sugar"]["mn9"], out["bitter"]["mn9"], out["water"]["mn9"], out["sugar+bitter"]["mn9"]
        res["feeding"] = dict(out, verdict=dict(sugar=s > 5, bitter_silent=b < 2, water=w > 5,
                                                suppress=(s > 5) and (sb < 0.7 * s),
                                                decays=out["sugar"]["after"] < 0.3 * max(out["sugar"]["active"], 1)))
        print("         | " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in res["feeding"]["verdict"].items()))

    if "looming" in only or "loom1" in only:
        gf = np.flatnonzero(ct == "DNp01"); mdn = np.flatnonzero(ct == "MDN")
        dnp09 = np.flatnonzero(ct == "DNp09"); dna02 = np.flatnonzero(ct == "DNa02")
        ttm = np.flatnonzero(cts.str.contains(r"^TTM|^DLM|^DVM").to_numpy())
        out = {}
        cases = [("LC4+LPLC2", r"^(LC4|LPLC2)$", 100), ("LC10a", r"^LC10a$", 25)] if "looming" in only else []
        if "loom1" in only:                              # experiment 8: single LC types alone, x 25 syn
            cases += [("LC4", r"^LC4$", 25), ("LPLC2", r"^LPLC2$", 25), ("LC6", r"^LC6$", 25), ("LC16", r"^LC16$", 25)]
        for name, pat, f in cases:
            idx = np.flatnonzero(cts.str.contains(pat).to_numpy())
            st, si, tr = sim(data, gains, 2.0, stim=(idx, 0.2, 1.2, f), gsyn=gsyn)
            hz = rates(st, si, n, 0.25, 1.2); after = rates(st, si, n, 1.25, 2.0)
            out[name] = dict(gf=float(hz[gf].mean()), mdn=float(hz[mdn].mean()), dnp09=float(hz[dnp09].mean()),
                             dna02=float(hz[dna02].mean()), active=int((hz > 2).sum()),
                             after=int((after > 2).sum()), ttm=float(hz[ttm].mean()) if len(ttm) else float("nan"))
            print(f"LOOM {name:<10}| GF {out[name]['gf']:5.1f} Hz  MDN {out[name]['mdn']:4.1f}  DNp09 {out[name]['dnp09']:4.1f}  "
                  f"DNa02 {out[name]['dna02']:4.1f}  VNC(TTM/DLM/DVM) {out[name]['ttm']:5.1f} Hz "
                  f"| active {out[name]['active']:>6,} after {out[name]['after']:>6,}  [{tr:.0f}s]", flush=True)
    if "looming" in only:
        res["looming"] = dict(out, verdict=dict(gf_fires=out["LC4+LPLC2"]["gf"] > 5,
                                                specific=out["LC10a"]["gf"] < 2 and out["LC4+LPLC2"]["mdn"] < 2))
        print("         | " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in res["looming"]["verdict"].items()))

    if "olfaction" in only:
        kc = np.flatnonzero(cts.str.startswith("KC").to_numpy()); apl = np.flatnonzero(ct == "APL")
        orn = np.flatnonzero(ct == "ORN_DA1"); pn = np.flatnonzero(cts.str.contains(r"^DA1_(ad|l|lv)PN").to_numpy())
        out = {}
        for f in (3, 8):
            st, si, tr = sim(data, gains, 2.0, stim=(orn, 0.2, 1.2, f), gsyn=gsyn)
            hz = rates(st, si, n, 0.25, 1.2)
            out[f"DA1_f{f}"] = dict(orn=float(hz[orn].mean()), pn=float(hz[pn].mean()),
                                    kc_frac=float((hz[kc] > 2).mean()), kc_hz=float(hz[kc].mean()), apl=float(hz[apl].mean()))
            o = out[f"DA1_f{f}"]
            print(f"OLF DA1 x{f:<3} | ORN {o['orn']:4.0f} Hz  PN {o['pn']:4.0f} Hz  KC {100*o['kc_frac']:4.1f}% active ({o['kc_hz']:.1f} Hz)  "
                  f"APL {o['apl']:4.0f} Hz | target 5-10 %  [{tr:.0f}s]", flush=True)
        res["olfaction"] = dict(out, verdict=dict(sparse=all(0.01 < v["kc_frac"] < 0.15 for v in out.values()),
                                                  pn_fires=all(v["pn"] > 5 for v in out.values())))
        print("         | " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in res["olfaction"]["verdict"].items()))

    if "olf4" in only:                                   # experiment 9: four glomeruli, one at a time
        kc = np.flatnonzero(cts.str.startswith("KC").to_numpy()); apl = np.flatnonzero(ct == "APL")
        out = {}
        for glom in ("DA1", "DM1", "VA1v", "DL3"):
            orn = np.flatnonzero(ct == f"ORN_{glom}")
            pn = np.flatnonzero(cts.str.contains(rf"^{glom}_.*PN").to_numpy())
            if not len(orn):
                print(f"OLF {glom}: no ORN_{glom} in the graph"); continue
            st, si, tr = sim(data, gains, 2.0, stim=(orn, 0.2, 1.2, 3), gsyn=gsyn)
            hz = rates(st, si, n, 0.25, 1.2)
            out[glom] = dict(n_orn=int(len(orn)), orn=float(hz[orn].mean()), pn=float(hz[pn].mean()) if len(pn) else float("nan"),
                             kc_frac=float((hz[kc] > 2).mean()), kc_hz=float(hz[kc].mean()), apl=float(hz[apl].mean()),
                             active=int((hz > 2).sum()))
            o = out[glom]
            print(f"OLF {glom:<5}x3 | ORN {o['orn']:4.0f} Hz ({o['n_orn']})  PN {o['pn']:4.0f} Hz  KC {100*o['kc_frac']:4.1f}% active "
                  f"({o['kc_hz']:.1f} Hz)  APL {o['apl']:4.0f} Hz  active {o['active']:,} | target 5-10 %  [{tr:.0f}s]", flush=True)
        res["olf4"] = out

    if "rest" in only:
        st, si, tr = sim(data, gains, 2.0, sens=10.0, seed_val=3, gsyn=gsyn)
        non_sens = ~np.char.startswith(sc.astype(str), "sensory")
        hz = rates(st, si, n, 0.5, 2.0); h = hz[non_sens]
        act = np.flatnonzero((hz > 2) & non_sens)
        cvs = []
        for i_ in act[:1500]:
            t_i = np.sort(st[(si == i_) & (st > 0.5)])
            if len(t_i) >= 5:
                isi = np.diff(t_i); cvs.append(isi.std() / isi.mean())
        res["rest"] = dict(active=int(len(act)), mean=float(h.mean()), ceiling=float((h > 200).mean()),
                           cv=float(np.median(cvs)) if cvs else float("nan"), silent=float((h < 0.5).mean()))
        r = res["rest"]
        print(f"REST     | non-sens active {r['active']:,} mean {r['mean']:.2f} Hz  >200 Hz {100*r['ceiling']:.1f}%  "
              f"silent {100*r['silent']:.0f}%  ISI CV {r['cv']:.2f} | target: sparse, irregular, ~1-5 Hz  [{tr:.0f}s]", flush=True)

    res["provenance"] = dict(prov, seeds=SEEDS)
    with open(OUT / f"scorecard_{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nwrote scorecard_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
