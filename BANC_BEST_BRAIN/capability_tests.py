"""
Capability tests for a frozen brain, designed to show what the ring attractor can do and to
compare p37a against the original static v2 on the same protocols. All in the resting brain
(sensory + visual projection neurons at 10 Hz) unless --silent.

  persist    Kim kick 0.5-1.547 s, then darkness to --seconds (20 s): rate / width / R / decoded
             angle per 2 s window; alive fraction, total drift, ISI CV and Fano factor of the bump
             neurons. Kim 2017: the bump persists in darkness for tens of seconds with slow drift.
  jump       bump established by kick A; second stimulus at A+--offset deg (90 or 180) with
             --f synapses/spike at 3-4 s: does the bump move to the new sector? (Kim 2017 Fig. 4:
             the bump jumps to a second cue; a rigid ring does not.)
  compete    two sectors (A and A+180) stimulated simultaneously: number of bumps after light-off
             (a ring attractor is winner-take-all: one bump).
  invariance single kick at --f in {50, 100, 250, 500}: width and rate after light-off vs drive
             (an attractor normalises its amplitude: width/rate independent of drive).
  rest       resting brain 4 s: rate distribution (median, mean, log-rate skew), ISI CV, pairwise
             correlation (sample), fraction at the ceiling.

  odortrials one odour (DA1) presented --trials times (6), --isi s between onsets (25), 1 s pulses, as in
             Honegger et al. 2011; per-KC spike counts per presentation saved to results_cap/kctrials_<tag>.npz
usage: capability_tests.py --brain v2|p37a --test NAME [--seed S] [--f F] [--offset DEG] [--seconds T]
                           [--cue-offset DEG] [--silent] [--device cuda_standalone] [--model shiu[-vstim][-noreset]]
--cue-offset rotates sector A (the training sector by default) around the ring (protocol 14).
brain v2   = --gains none, no escape noise, tau_e 5 ms (the published static model)
brain p37a = --gains w_final_p37a.npz, escape 1 mV, tau_e 15 ms
writes results_cap/cap_<brain>_<test>[_<cond>]_s<seed>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
RES = OUT / "results_cap"; RES.mkdir(exist_ok=True)


def arg(flag, default, cast=str):
    a = sys.argv
    return cast(a[a.index(flag) + 1]) if flag in a else default


DEVICE = arg("--device", "runtime")
if DEVICE != "runtime":
    import brian2
    if DEVICE == "cuda_standalone":
        import brian2cuda  # noqa: F401
    brian2.set_device(DEVICE, build_on_run=False)
import plasticity as PL  # noqa: E402
from provenance import provenance  # noqa: E402
from brian2 import ms, second, start_scope, seed, device  # noqa: E402
BS = PL.BS; CX = BS.CX
KIM_ON, KIM_OFF = 0.5, 1.547


def ring(data, offset_deg=0.0):
    cx = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
    cx_root = cx["root_id"].astype(np.int64); cx_lab = cx["labels"].astype(str); cx_ang = cx["angles"]
    cx_epg = np.flatnonzero(cx_lab == "EPG"); order = cx_epg[np.argsort(cx_ang[cx_epg])]
    W = cx["W"].astype(float)
    hub = cx_epg[np.argmax(np.abs(W[np.ix_(cx_epg, cx_epg)]).sum(1))]
    hp = int(np.flatnonzero(order == hub)[0]); k = int(round(45 / 360.0 * len(order)))
    hp = (hp + int(round(offset_deg / 360.0 * len(order)))) % len(order)
    seed_e = order[[(hp - k // 2 + j) % len(order) for j in range(k)]]
    pos = {int(r): i for i, r in enumerate(data["root"])}
    epg = np.array([pos[int(cx_root[e])] for e in order]); ang = cx_ang[order]
    return epg, ang, np.array([pos[int(cx_root[e])] for e in seed_e])


def n_bumps(profile, frac=0.2):
    """contiguous groups of EPGs above frac*max, with wrap-around"""
    if profile.max() <= 0:
        return 0
    on = profile > frac * profile.max()
    if on.all():
        return 1
    k = int(np.flatnonzero(~on)[0]); on = np.roll(on, -k)     # start at an off position
    return int(np.sum((on[1:] & ~on[:-1])) + (1 if on[0] else 0))


def circ_mean_deg(a_deg):
    z = np.exp(1j * np.radians(a_deg)); return float(np.degrees(np.angle(z.mean()))), float(np.abs(z.mean()))


def main() -> int:
    brain = arg("--brain", "p37a"); test = arg("--test", "persist"); sd = arg("--seed", 1, int)
    f = arg("--f", 250.0, float); offset = arg("--offset", 90.0, float); T = arg("--seconds", 20.0 if arg("--test", "persist") == "persist" else 6.0, float)
    silent = "--silent" in sys.argv
    if brain == "v2":
        gains, escape, tau_e = None, None, 5.0 * ms
    else:
        g = np.load(OUT / "w_final_p37a.npz"); gains = dict(ee=g["ee"], ie=g["ie"]); escape = (1.0, PL.ESCAPE_LAM0_HZ); tau_e = 15.0 * ms
    # ablation overrides on top of the brain preset (tag suffix records them)
    ov = ""
    if "--gains" in sys.argv:
        gp = arg("--gains", "none")
        if gp == "none":
            gains = None; ov += "_gnone"
        else:
            g = np.load(OUT / gp); gains = dict(ee=g["ee"], ie=g["ie"]); ov += "_g" + Path(gp).stem.replace("w_final_", "")
    if "--gains-class" in sys.argv and gains is not None:      # keep one learned class, the other at v2 (ones)
        cls = arg("--gains-class", "ie"); other = "ee" if cls == "ie" else "ie"
        gains[other] = np.ones_like(gains[other]); ov += f"_g{cls}"
    if "--no-escape" in sys.argv:
        escape = None; ov += "_noesc"
    if "--tau-e" in sys.argv:
        tau_e = arg("--tau-e", 15.0, float) * ms; ov += f"_taue{arg('--tau-e', 15.0, float):g}"
    model = arg("--model", "cond")                          # shiu | shiu-noreset: Shiu et al. 2024 formulation (plasticity.build_shiu_network)
    if model != "cond":
        assert model in ("shiu", "shiu-noreset", "shiu-vstim", "shiu-vstim-noreset"), "--model takes shiu[-vstim][-noreset]"
        assert brain == "v2" and "--tau-e" not in sys.argv, "--model shiu needs --brain v2 and no --tau-e (no gains, escape noise or tau_e in that model)"
        tau_e = None; ov += f"_{model}"                     # the v2 preset's 5 ms is that model's own tau
    no_vis = "--no-vis" in sys.argv
    if no_vis:
        ov += "_novis"
    data = PL.load_data(str(OUT / "banc_full.npz"), vis_sens=not no_vis); n = data["n"]
    if "--no-orn" in sys.argv:                              # resting brain without the ORNs in the spontaneous set
        is_orn = np.char.startswith(data["cell_type"].astype(str), "ORN_")
        data["sens_idx"] = data["sens_idx"][~is_orn[data["sens_idx"]]]; ov += "_noorn"
    cue = arg("--cue-offset", 0.0, float)                  # protocol 14: rotate the cue sector A around the ring
    epg, ang, stimA = ring(data, cue); _, _, stimB = ring(data, cue + offset)
    cue_ang = circ_mean_deg(np.degrees(ang[np.isin(epg, stimA)]))[0]   # ring angles are in radians
    sens = 0.0 if silent else 10.0
    cond = f"_c{cue:g}" if cue else ""
    stim, stim2 = (stimA, KIM_ON, KIM_OFF, f), None
    if test == "persist":
        stim = (stimA, KIM_ON, KIM_OFF, 250.0)
    elif test == "jump":
        stim = (stimA, KIM_ON, KIM_OFF, 250.0); stim2 = (stimB, 3.0, 4.0, f); cond += f"_o{offset:g}_f{f:g}"
    elif test == "compete":
        _, _, stimB = ring(data, cue + 180.0); stim = (np.concatenate([stimA, stimB]), KIM_ON, KIM_OFF, 250.0)
    elif test == "invariance":
        stim = (stimA, KIM_ON, KIM_OFF, f); cond += f"_f{f:g}"
    elif test == "rest":
        stim = None; T = 4.0
    elif test == "spont":                                   # spontaneous bump: no cue at all (Seelig & Jayaraman 2015)
        stim = None; T = 8.0
    elif test == "odortrials":                              # one odour presented repeatedly (Honegger et al. 2011: 1 s pulse,
        n_tr = arg("--trials", 6, int); isi_s = arg("--isi", 25.0, float)   # 25 s between onsets, first onset at 8 s)
        tr_on = 8.0 + isi_s * np.arange(n_tr)
        stim, stim2, T = (np.flatnonzero(data["cell_type"] == "ORN_DA1"), tr_on, tr_on + 1.0, 3.0), None, float(n_tr * isi_s)
        cond = f"_n{n_tr}"
    elif test in ("odors", "conc", "evoked"):
        ct = data["cell_type"]
        orn = lambda g: np.flatnonzero(ct == f"ORN_{g}")
        if test == "odors":                                 # two odours: sparse AND distinct KC codes
            stim, stim2, T = (orn("DA1"), 0.5, 1.5, 3.0), (orn("DL3"), 3.0, 4.0, 3.0), 5.0
        elif test == "conc":                                # concentration series: graded, sub-saturating KC recruitment
            stim, stim2, T = (orn("DA1"), 0.5, 1.5, f), None, 2.0; cond = f"_f{f:g}"
        else:                                               # evoked activity is bounded and decays: odour then looming
            lc = np.flatnonzero(np.isin(ct, ["LC4", "LPLC2"]))
            stim, stim2, T = (orn("DA1"), 0.5, 1.5, 3.0), (lc, 3.0, 4.0, 100.0), 5.5
    live = "--live" in sys.argv and brain != "v2"          # iSTDP (I->E) keeps running during the test
    eta_i = arg("--eta-i", None, float)                     # --live only: override the settle's learning rate
    pre = arg("--pre", 0.0, float)                          # frozen runs: settle this long before the protocol
    std = None                                              # --std U,TAU_REC_S[,recip]  (recip: U_ij = U * rho_ij)
    if "--std" in sys.argv:
        parts = arg("--std", "").split(","); std = (float(parts[0]), float(parts[1])) + ((parts[2],) if len(parts) > 2 else ())   # recip | recip_ee
    tag = (f"{brain}_{test}{cond}_s{sd}{'_silent' if silent else ''}{'_live' if live else ''}"
           f"{f'_eta{eta_i:g}' if (live and eta_i is not None) else ''}{f'_pre{pre:g}' if (pre and not live) else ''}"
           + ((f"_std{std[2] if len(std) > 2 else ''}U{std[0]:g}t{std[1]:g}") if std else "") + ov)
    if "--precheck" in sys.argv:
        # no network: what release fraction the rule assigns, by pathway (E edges only)
        rho = PL.edge_reciprocity(data); ct = data["cell_type"]; ex = data["sign"] > 0
        pre_i, post_i, ns = data["pre_idx"], data["post_idx"], data["n_syn"]
        core = data["is_cx_core"]
        def cls(name, pm, qm):
            e = ex[pre_i] & pm[pre_i] & qm[post_i]
            u = float(np.average(rho[e], weights=ns[e])) * (std[0] if std else 1.0) if e.any() else float("nan")
            print(f"   precheck | {name:<14} edges {int(e.sum()):>7,}  syn {int(ns[e].sum()):>9,}  count-weighted U {u:.3f}"
                  f"  (rho: mean {rho[e].mean():.3f}, frac>0 {(rho[e] > 0).mean():.2f})" if e.any() else f"   precheck | {name}: no edges", flush=True)
        s = lambda p: np.char.startswith(ct.astype(str), p)
        cls("core->core", core, core); cls("EPG->PEN/PEG", s("EPG"), s("PEN") | s("PEG")); cls("PEN/PEG->EPG", s("PEN") | s("PEG"), s("EPG"))
        cls("EPG->ExR/D7", s("EPG"), s("ExR") | s("Delta7")); cls("ORN->PN", s("ORN_"), np.array([("PN" in c) for c in ct]))
        cls("PN->KC", np.array([("PN" in c) for c in ct]), s("KC")); cls("KC->KC", s("KC"), s("KC")); cls("KC->APL", s("KC"), ct == "APL")
        cls("LC4/LPLC2->GF", s("LC4") | s("LPLC2"), ct == "DNp01"); cls("all E edges", np.ones(n, bool), np.ones(n, bool))
        return 0
    print(f"[{tag}] T {T:g} s, sens {sens:g} Hz, escape {escape}, tau_e {tau_e}", flush=True)
    if DEVICE != "runtime":
        device.reinit(); device.activate(build_on_run=False, directory=f"/tmp/brian_cap_{tag}")
    start_scope(); seed(sd)
    T_PRE = 0.0
    if not live:
        T_PRE = pre
        shift = lambda sp: None if sp is None else (sp[0], sp[1] + T_PRE, sp[2] + T_PRE, sp[3])
        net, G, syn, spikes, masks = PL.build_network(data, g_syn_ps=30.0, plastic=False, gains=gains, sens_hz=sens,
                                                      stim=shift(stim), stim2=shift(stim2), duration_s=T + T_PRE,
                                                      tau_e=tau_e, escape=escape, std=std,
                                                      model=model.split("-")[0], g_reset=not model.endswith("noreset"), stim_v="vstim" in model)
    else:
        # the inhibitory rule (Vogels iSTDP, the settle's own constants) stays on; no E rule, no brake.
        # warm start = the saved weights (the rule's floor w0); 3 s settle before the protocol.
        p = json.loads(str(np.load(OUT / "w_final_p37a.npz")["params"])); T_PRE = 3.0
        PL.RHO0_HZ = float(p["rho0"])
        if eta_i is not None:
            p["eta_i"] = eta_i
        shift = lambda sp: None if sp is None else (sp[0], sp[1] + T_PRE, sp[2] + T_PRE, sp[3])
        print(f"   live: iSTDP eta_i {p['eta_i']:g} wmax_i {p['wmax_i']:g} rho0 {p['rho0']:g} Hz; eta_e 0, gamma_e 0; "
              f"{T_PRE:g} s settle before the protocol", flush=True)
        net, G, syn, spikes, masks = PL.build_network(
            data, g_syn_ps=30.0, plastic=True, gains=gains, sens_hz=sens, stim=shift(stim), stim2=shift(stim2),
            duration_s=T + T_PRE, eta_i=float(p["eta_i"]), wmax_i=float(p["wmax_i"]), eta_e=0.0,
            floor_e=float(p["floor_e"]), tau_e=tau_e, in_model=True, gamma_e=0.0, e_start_s=0.0, escape=escape, std=std)
    PL.run(net, T + T_PRE)
    if DEVICE != "runtime":
        device.build(run=True)
    st, si = np.asarray(spikes.t / second), np.asarray(spikes.i)
    res = dict(brain=brain, test=test, seed=sd, f=f, offset=offset, T=T, silent=silent, live=live,
               cue_offset=cue, cue_angle=cue_ang)
    res["provenance"] = provenance()
    if live:
        from brian2 import siemens
        w_ie = np.asarray(syn["ie"].w / siemens) / np.asarray(syn["ie"].w0 / siemens)
        onto_epg = np.isin(data["post_idx"][masks["ie"]], epg)
        res.update(wI_drift=float(w_ie.mean()), wI_drift_epg=float(w_ie[onto_epg].mean()), wI_p90=float(np.percentile(w_ie, 90)))
        print(f"   drift | <wI> {res['wI_drift']:.3f} (onto ring EPGs {res['wI_drift_epg']:.3f}, p90 {res['wI_p90']:.2f}) "
              f"over {T + T_PRE:g} s", flush=True)
    if T_PRE > 0:
        k = st >= T_PRE; st, si = st[k] - T_PRE, si[k]

    if test in ("odors", "conc", "evoked"):
        ct = data["cell_type"]; kc = np.flatnonzero(np.char.startswith(ct.astype(str), "KC"))
        pn = np.flatnonzero(np.array([c.startswith("DA1_") and c.endswith("PN") for c in ct]))
        win = lambda a, b: np.bincount(si[(st >= a) & (st < b)], minlength=n) / (b - a)
        if test == "odors":
            hA, hB = win(0.55, 1.5), win(3.05, 4.0); A, B = hA[kc] > 2, hB[kc] > 2
            h0, h1 = win(0.05, 0.5), win(2.0, 2.9)                    # KC baseline before and between odours
            jac = float((A & B).sum() / max((A | B).sum(), 1))
            res.update(kc_base=float((h0[kc] > 2).mean()), kc_between=float((h1[kc] > 2).mean()),
                       kc_A=float(A.mean()), kc_B=float(B.mean()), jaccard=jac, kc_hz_A=float(hA[kc].mean()), kc_hz_B=float(hB[kc].mean()),
                       n_A=int(A.sum()), n_B=int(B.sum()), n_both=int((A & B).sum()),
                       sparse=bool(0.01 <= A.mean() <= 0.2 and 0.01 <= B.mean() <= 0.2), distinct=bool(jac < 0.5))
            print(f"CAP {tag:<22}| KC at rest {100*res['kc_base']:4.1f}% (between odours {100*res['kc_between']:4.1f}%) | "
                  f"DA1: KC {100*A.mean():4.1f}% ({A.sum()}) | DL3: KC {100*B.mean():4.1f}% ({B.sum()}) | "
                  f"shared {int((A & B).sum())}  Jaccard {jac:.2f} | sparse {'PASS' if res['sparse'] else 'FAIL'}  "
                  f"distinct {'PASS' if res['distinct'] else 'FAIL'}", flush=True)
        elif test == "conc":
            h = win(0.55, 1.5); o = data["cell_type"] == "ORN_DA1"
            res.update(orn=float(h[o].mean()), pn=float(h[pn].mean()) if len(pn) else float("nan"), kc_frac=float((h[kc] > 2).mean()),
                       kc_hz=float(h[kc].mean()), active=int((h > 2).sum()))
            print(f"CAP {tag:<22}| ORN {res['orn']:5.0f} Hz  PN {res['pn']:5.0f} Hz  KC {100*res['kc_frac']:5.1f}% ({res['kc_hz']:.1f} Hz)  "
                  f"active {res['active']:,}", flush=True)
        else:
            gf = np.flatnonzero(ct == "DNp01"); nA, nL = len(stim[0]), len(stim2[0])
            base, dA, aA, dL, aL = win(0.0, 0.5), win(0.55, 1.5), win(1.6, 2.6), win(3.05, 4.0), win(4.1, 5.1)
            c = lambda h: int((h > 2).sum())
            res.update(base=c(base), odor_during=c(dA), odor_after=c(aA), loom_during=c(dL), loom_after=c(aL),
                       odor_gain=float((c(dA) - c(base)) / nA), loom_gain=float((c(dL) - c(base)) / nL), gf=float(dL[gf].mean()),
                       odor_decays=bool(c(aA) - c(base) < 0.3 * max(c(dA) - c(base), 1)),
                       loom_decays=bool(c(aL) - c(base) < 0.3 * max(c(dL) - c(base), 1)))
            print(f"CAP {tag:<22}| baseline {c(base):,} | odour: during {c(dA):,} (+{res['odor_gain']:.1f}/stim neuron) after 1 s {c(aA):,} "
                  f"{'decays' if res['odor_decays'] else 'PERSISTS'} | looming: GF {res['gf']:.0f} Hz, during {c(dL):,} "
                  f"(+{res['loom_gain']:.1f}/stim neuron) after 1 s {c(aL):,} {'decays' if res['loom_decays'] else 'PERSISTS'}", flush=True)
    elif test == "odortrials":
        # per Kenyon cell and presentation: spike counts 0.5-4.5 s after odour onset (Honegger's response window)
        # and in the 4 s before onset (baseline). Responder at level k: at least k spikes more than baseline,
        # on at least half of the presentations (their reliability criterion). The counts are saved for offline analysis.
        kc = np.flatnonzero(np.char.startswith(data["cell_type"].astype(str), "KC"))
        cnt = lambda a, b_: np.bincount(si[(st >= a) & (st < b_)], minlength=n)[kc]
        resp = np.stack([cnt(t0 + 0.5, t0 + 4.5) for t0 in tr_on], 1); base = np.stack([cnt(t0 - 4.5, t0 - 0.5) for t0 in tr_on], 1)
        puls = np.stack([cnt(t0 + 0.05, t0 + 1.0) for t0 in tr_on], 1)
        RES.mkdir(exist_ok=True)
        np.savez_compressed(RES / f"kctrials_{tag}.npz", kc=kc, onsets=tr_on, resp=resp, base=base, pulse=puls)
        for k_ in (1, 2, 5, 10):
            res[f"frac_k{k_}"] = float((((resp - base) >= k_).mean(1) >= 0.5).mean())
        res.update(n_kc=int(len(kc)), trials=int(n_tr), isi=float(isi_s),
                   frac_2hz_per_trial=[float((puls[:, j] / 0.95 > 2).mean()) for j in range(n_tr)],
                   kc_base_hz=float(base.mean() / 4.0))
        print(f"CAP {tag:<22}| KC responders on >= half of {n_tr} presentations: >=1 spike {100*res['frac_k1']:.1f}%  >=2 {100*res['frac_k2']:.1f}%  "
              f">=5 {100*res['frac_k5']:.1f}%  >=10 {100*res['frac_k10']:.1f}% | single-trial >2 Hz: "
              + " ".join(f"{100*x:.1f}" for x in res["frac_2hz_per_trial"]) + f" % | KC baseline {res['kc_base_hz']:.2f} Hz", flush=True)
    elif test == "rest":
        non_sens = ~np.char.startswith(data["super_class"].astype(str), "sensory"); non_sens[data["sens_idx"]] = False
        hz = np.bincount(si[st > 0.5], minlength=n) / (T - 0.5); h = hz[non_sens]
        act = np.flatnonzero((hz > 2) & non_sens); rng = np.random.default_rng(0)
        cvs = []
        for i_ in rng.choice(act, min(2000, len(act)), replace=False) if len(act) else []:
            t_i = np.sort(st[(si == i_) & (st > 0.5)])
            if len(t_i) >= 5:
                isi = np.diff(t_i); cvs.append(isi.std() / isi.mean())
        # pairwise correlation of 50 ms counts among 300 random active neurons
        samp = rng.choice(act, min(300, len(act)), replace=False) if len(act) else np.array([], int)
        edges = np.arange(0.5, T + 1e-9, 0.05)
        M = np.array([np.histogram(st[si == i_], bins=edges)[0] for i_ in samp], float)
        if len(M) > 1:
            Cc = np.corrcoef(M); corr = float(np.nanmean(Cc[np.triu_indices(len(M), 1)]))
        else:
            corr = float("nan")
        lr = np.log10(h[h > 0.1])
        res.update(active=int(len(act)), mean_hz=float(h.mean()), median_active_hz=float(np.median(hz[act])) if len(act) else 0.0,
                   ceiling=float((h > 200).mean()), silent=float((h < 0.5).mean()), cv=float(np.median(cvs)) if cvs else float("nan"),
                   corr=corr, lograte_skew=float(((lr - lr.mean()) ** 3).mean() / max(lr.std() ** 3, 1e-9)) if len(lr) > 10 else float("nan"))
        print(f"CAP {tag:<22}| active {res['active']:,} mean {res['mean_hz']:.2f} Hz median(active) {res['median_active_hz']:.1f} "
              f">200Hz {100*res['ceiling']:.1f}% silent {100*res['silent']:.0f}% CV {res['cv']:.2f} corr {res['corr']:.3f} "
              f"lograte skew {res['lograte_skew']:+.2f}", flush=True)
    else:
        r_epg = CX.smooth(BS.bin_rates(st, si, n, T, epg)); t_axis = np.arange(r_epg.shape[1]) * BS.DT_BIN_MS / 1000
        th, R = CX.decode(r_epg, ang); th = np.degrees(th)
        wins = [(a, min(a + 2.0, T)) for a in np.arange(KIM_OFF + 0.05, T - 0.5, 2.0)] if test == "persist" else \
               [(KIM_OFF + 0.05, 3.0), (4.1, T)] if test == "jump" else [(1.0, T)] if test == "spont" else [(KIM_OFF + 0.05, T)]
        rows = []
        for a, b in wins:
            kb = (t_axis >= a) & (t_axis < b)
            hz = np.bincount(si[(st >= a) & (st < b)], minlength=n) / (b - a)
            prof = r_epg[:, kb].mean(1); act = hz[epg] > 2
            m_ang, m_R = circ_mean_deg(th[kb])
            rows.append(dict(t=(float(a), float(b)), n_epg=int(act.sum()), hz=float(hz[epg][act].mean()) if act.any() else 0.0,
                             fwhm=float(BS.bump_fwhm_deg(prof)), R=float(R[kb].mean()), angle=m_ang, alive=float((R[kb] > 0.5).mean()),
                             n_bumps=n_bumps(prof), active_brain=int((hz > 2).sum())))
        res["windows"] = rows
        if test == "persist":
            th_w = np.array([r["angle"] for r in rows])
            res.update(alive=float(np.mean([r["alive"] for r in rows])), drift_total=float((th_w[-1] - th_w[0] + 180) % 360 - 180),
                       drift_step_med=float(np.median(np.abs((np.diff(th_w) + 180) % 360 - 180))) if len(th_w) > 1 else 0.0,
                       hz_mean=float(np.mean([r["hz"] for r in rows])), fwhm_mean=float(np.nanmean([r["fwhm"] for r in rows])))
            # spike statistics of the bump neurons over the whole dark period
            kb = (t_axis >= KIM_OFF + 0.05)
            hz_all = np.bincount(si[st >= KIM_OFF + 0.05], minlength=n) / (T - KIM_OFF - 0.05)
            bump = epg[hz_all[epg] > 0.5 * hz_all[epg].max()] if hz_all[epg].max() > 0 else np.array([], int)
            cvs, fanos = [], []
            for i_ in bump:
                t_i = np.sort(st[(si == i_) & (st >= KIM_OFF + 0.05)])
                if len(t_i) >= 10:
                    isi = np.diff(t_i); cvs.append(isi.std() / isi.mean())
                    c = np.histogram(t_i, bins=np.arange(KIM_OFF + 0.05, T, 0.1))[0]; fanos.append(c.var() / max(c.mean(), 1e-9))
            res.update(isi_cv=float(np.median(cvs)) if cvs else float("nan"), fano=float(np.median(fanos)) if fanos else float("nan"),
                       n_bump=int(len(bump)), hz_bump=float(hz_all[bump].mean()) if len(bump) else 0.0,
                       hz_ring_max=float(hz_all[epg].max()))
            print(f"CAP {tag:<22}| alive {res['alive']:.2f} over {T:g} s | rate {res['hz_mean']:5.1f} Hz (bump EPGs {res['hz_bump']:.0f}, "
                  f"peak {res['hz_ring_max']:.0f})  FWHM {res['fwhm_mean']:3.0f} deg | "
                  f"drift total {res['drift_total']:+5.0f} deg, median step/2s {res['drift_step_med']:4.1f} deg | "
                  f"ISI CV {res['isi_cv']:.2f}  Fano {res['fano']:.2f} ({res['n_bump']} bump EPG) | windows: "
                  + " ".join(f"{r['hz']:.0f}Hz/{r['fwhm']:.0f}d/{r['n_bumps']}b" for r in rows), flush=True)
        elif test == "jump":
            before, after = rows
            shift = (after["angle"] - before["angle"] + 180) % 360 - 180
            target = offset
            res.update(shift=float(shift), target=float(target), jumped=bool(abs((shift - target + 180) % 360 - 180) < 30 and after["R"] > 0.5),
                       stayed=bool(abs(shift) < 20 and after["R"] > 0.5))
            print(f"CAP {tag:<22}| before {before['n_epg']} EPG @ {before['hz']:.0f} Hz angle {before['angle']:+.0f} | "
                  f"after {after['n_epg']} EPG @ {after['hz']:.0f} Hz angle {after['angle']:+.0f} R {after['R']:.2f} bumps {after['n_bumps']} | "
                  f"shift {shift:+.0f} deg (target {target:+.0f}) -> {'JUMPED' if res['jumped'] else ('stayed' if res['stayed'] else 'other')}", flush=True)
        else:
            r = rows[0]
            res.update(**{k: r[k] for k in ("n_epg", "hz", "fwhm", "R", "alive", "n_bumps", "active_brain")})
            print(f"CAP {tag:<22}| {r['n_epg']} EPG @ {r['hz']:.0f} Hz  FWHM {r['fwhm']:.0f} deg  R {r['R']:.2f}  alive {r['alive']:.2f}  "
                  f"bumps {r['n_bumps']}  brain active {r['active_brain']:,}", flush=True)
    with open(RES / f"cap_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
