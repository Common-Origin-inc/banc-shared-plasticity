"""
Staged plasticity run on the unified Brain+VNC graph under spontaneous
sensory drive only (no stimulus, no task). Logs every chunk; checkpoints the
per-edge gains; writes w_final.npz (gains ee, ie relative to v2) at the end.

usage: run_native_plasticity.py [--seconds 100] [--chunk 1] [--checkpoint 10]
        [--sens 10] [--gsyn 30] [--eta-i 1e-3] [--gamma-e 0.05] [--floor-e 0.1]
        [--rho0 5] [--wmax-i 10] [--enable-hebbian-e ETA_E] [--no-e-decay]
        [--e-decay-start S] [--tag NAME]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))


def arg(flag, default, cast=float):
    a = sys.argv
    return cast(a[a.index(flag) + 1]) if flag in a else default


DEVICE = arg("--device", "runtime", str)                # runtime | cpp_standalone | cuda_standalone
if DEVICE != "runtime":
    import brian2
    if DEVICE == "cuda_standalone":
        import brian2cuda                                # noqa: F401
    brian2.set_device(DEVICE, build_on_run=False, directory="/tmp/brian_settle")
import plasticity as PL                                  # noqa: E402
from provenance import provenance                        # noqa: E402
from brian2 import Hz, second, siemens, StateMonitor, device, prefs   # noqa: E402


def main() -> int:
    seconds = arg("--seconds", 100.0); chunk = arg("--chunk", 1.0); ckpt = arg("--checkpoint", 10.0)
    sens = arg("--sens", 10.0); gsyn = arg("--gsyn", 30.0)
    eta_i = arg("--eta-i", PL.ETA_I); gamma_e = arg("--gamma-e", PL.GAMMA_E); floor_e = arg("--floor-e", PL.FLOOR_E)
    PL.RHO0_HZ = arg("--rho0", PL.RHO0_HZ); wmax_i = arg("--wmax-i", PL.WMAX_I)
    eta_e = arg("--enable-hebbian-e", 0.0); e_decay = "--no-e-decay" not in sys.argv
    brake_recip = "--brake-recip" in sys.argv          # E->E brake gated by edge reciprocity (feed-forward edges spared)
    e_start = arg("--e-decay-start", PL.E_DECAY_START_S); tag = arg("--tag", "run", str)
    seed_val = int(arg("--seed", 7, int))
    prov = provenance()                                    # git commit + package versions, saved with the weights

    dpath = arg("--data", str(OUT / "banc_full.npz"), str)
    vis_sens = "--vis-sens" in sys.argv                    # visual projection neurons join the drive
    bump_every = arg("--bump-every", 0.0)                  # kick the Kim bump every S s (0 = off)
    bump = "--bump" in sys.argv or bump_every > 0          # (--bump alone = once at t = 2 s)
    two_sided = "--two-sided" in sys.argv                  # dead-band homeostasis (see plasticity.py)
    data = PL.load_data(dpath, vis_sens=vis_sens)
    n = data["n"]; sc = data["super_class"]; ct = data["cell_type"]
    is_sens = np.zeros(n, bool); is_sens[data["sens_idx"]] = True
    non_sens = ~is_sens
    print(f"graph: {Path(dpath).name}; drive set {int(is_sens.sum()):,} neurons"
          + (" (incl. visual projection/centrifugal)" if vis_sens else "")
          + (f"; Kim bump every {bump_every:g} s from 2 s" if bump_every > 0 else ("; Kim bump at 2 s" if bump else ""))
          + ("; TWO-SIDED (dead band %.1f-%.0f Hz, gamma_up %g)" % (PL.RHO_LOW_HZ, PL.RHO0_HZ, PL.GAMMA_UP) if two_sided else ""))
    pops = {"EPG": ct == "EPG", "PEN_a": ct == "PEN_a(PEN1)", "D7": ct == "Delta7",
            "ER": np.char.startswith(ct.astype(str), "ER"), "ExR6": ct == "ExR6",
            "KC": np.char.startswith(ct.astype(str), "KC"), "APL": ct == "APL",
            "MN9": ct == "MN9", "PN": np.char.endswith(ct.astype(str), "PN"),
            "legMN": (sc == "motor") & (data["region"] == "ventral_nerve_cord"),
            "DN": sc == "descending", "VNCint": sc == "ventral_nerve_cord_intrinsic"}
    print(f"[{tag}] {n:,} neurons, {len(data['n_syn']):,} edges; sens {sens:g} Hz, G_E {gsyn:g} pS; "
          f"eta_i {eta_i:g}, wmax_i {wmax_i:g}, gamma_e {gamma_e:g} (start {e_start:g} s, {'on' if e_decay else 'off'}), "
          f"floor_e {floor_e:g}, eta_e {eta_e:g}, rho0 {PL.RHO0_HZ:g} Hz; {seconds:g} s in {chunk:g} s chunks", flush=True)
    from brian2 import seed, start_scope
    start_scope(); seed(seed_val)
    stim = None
    if bump:
        ons = np.arange(2.0, seconds, bump_every) if bump_every > 0 else np.array([2.0])
        n_rot = int(arg("--bump-rotate", 0, int))          # kicks cycle through N sectors around the ring
        if n_rot > 1:
            stim = [(PL.kim_stim(data, offset_deg=360.0 * k / n_rot), ons[k::n_rot], ons[k::n_rot] + 1.047, 250)
                    for k in range(n_rot)]
            print(f"  training kicks rotate through {n_rot} sectors ({len(ons)} kicks)")
        else:
            stim = (PL.kim_stim(data), ons, ons + 1.047, 250)
    # excitatory input arriving per neuron (for the two-sided rule): edge list of E synapses
    exc_e = data["sign"][data["pre_idx"]] > 0
    pre_e, post_e, ns_e = data["pre_idx"][exc_e], data["post_idx"][exc_e], data["n_syn"][exc_e]
    tau_e = arg("--tau-e", 0.0)
    tau_e = tau_e * PL.ms if tau_e > 0 else None
    PL.LTD_RATIO = arg("--ltd-ratio", PL.LTD_RATIO)
    PL.STDP_START_S = arg("--stdp-start", PL.STDP_START_S)
    # intrinsic rate control (plant physics): --adapt ALPHA_MV [TAU_MS], --escape DELTA_MV [LAM0_HZ]
    def _vals(flag):
        out = []
        if flag in sys.argv:
            for tok in sys.argv[sys.argv.index(flag) + 1:]:
                try:
                    out.append(float(tok))
                except ValueError:
                    break
        return out
    adapt = escape = None
    if _vals("--adapt"):
        v = _vals("--adapt"); adapt = (v[0], v[1] if len(v) > 1 else PL.ADAPT_TAU_MS)
        print(f"  adaptive threshold: alpha {adapt[0]:g} mV, tau_a {adapt[1]:g} ms")
    if _vals("--escape"):
        v = _vals("--escape"); escape = (v[0], v[1] if len(v) > 1 else PL.ESCAPE_LAM0_HZ)
        print(f"  escape noise: delta {escape[0]:g} mV, lam0 {escape[1]:g} Hz")
    two_comp = None
    if _vals("--two-comp"):
        v = _vals("--two-comp"); two_comp = (v[0], v[1] if len(v) > 1 else 0.5)
        print(f"  two-compartment neuron: rho {two_comp[0]:g}, f_s {two_comp[1]:g}")
    recip = arg("--recip", 0.0)
    recip = recip if recip > 0 else None                   # reciprocity rule, plasticity.reciprocity_factor
    if recip:
        print(f"  reciprocity rule: kappa {recip:g}")
    ltp_sat = "--ltp-sat" in sys.argv; gate_fac = arg("--gate-fac", 2.0)   # coupled rule (band-gated LTP)
    if ltp_sat:
        print(f"  coupled rule: LTP saturates at rho0 and is off above {gate_fac:g} x rho0")
    recip_ee = arg("--recip-ee", 0.0)
    recip_ee = recip_ee if recip_ee > 0 else None          # E by structure: reciprocity on E->E only
    if recip_ee:
        print(f"  reciprocity on E->E only: kappa {recip_ee:g}")
    recip_een = arg("--recip-ee-norm", 0.0)
    recip_een = recip_een if recip_een > 0 else None       # same, per-neuron E drive conserved
    if recip_een:
        print(f"  drive-conserving reciprocity on E->E: kappa {recip_een:g}")
    da_file = arg("--da", "", str); da = None                # three-factor rule: dopaminergic edges
    tau_da = arg("--tau-da", 1.0)                            # dopamine trace time constant (s)
    da_th = arg("--da-th", 0.0)                              # dopamine gate threshold (0 = none)
    da_release = None                                        # --da-release supra:R_BASE | deplete:U:TAU_REC
    rel = arg("--da-release", "", str)
    if rel:
        p_ = rel.split(":")
        da_release = (p_[0],) + tuple(float(v) for v in p_[1:])
        print(f"  dopamine release: {da_release}")
    if da_file:
        dz = np.load(da_file); da = dict(pre_idx=dz["pre_idx"], post_idx=dz["post_idx"], n_syn=dz["n_syn"])
        print(f"  three-factor rule: E->E STDP gated by the dopamine trace (tau {tau_da:g} s); "
              f"{len(da['pre_idx']):,} DA edges from {da_file}")
    # --warm FILE: start from a previous run's gains (they become the rules' floor w0); --wmax-e W
    warm = arg("--warm", "", str); gains = None
    if warm:
        gw = np.load(warm); gains = dict(ee=gw["ee"], ie=gw["ie"])
        print(f"  warm start from {warm}: <ee> {gw['ee'].mean():.3f}  <ie> {gw['ie'].mean():.3f}")
    wmax_e = arg("--wmax-e", PL.WMAX_E)
    standalone = DEVICE != "runtime"
    net, G, syn, spikes, masks = PL.build_network(
        data, g_syn_ps=gsyn, plastic=True, gains=gains, sens_hz=sens, stim=stim, duration_s=seconds,
        record_spikes=False, eta_i=eta_i, wmax_i=wmax_i, eta_e=eta_e, wmax_e=wmax_e, floor_e=floor_e, tau_e=tau_e,
        in_model=standalone, gamma_e=gamma_e if e_decay else 0.0, e_start_s=e_start,
        two_sided=two_sided, rho_low=PL.RHO_LOW_HZ, gamma_up=PL.GAMMA_UP, adapt=adapt, escape=escape,
        two_comp=two_comp, recip=recip, ltp_sat=ltp_sat, gate_fac=gate_fac, recip_ee=recip_ee, recip_een=recip_een,
        da=da, tau_da_s=tau_da, da_th=da_th, da_release=da_release, brake_recip=brake_recip)
    if standalone:
        # one compiled run; per-second log from a rate-trace monitor on a neuron sample
        if DEVICE == "cpp_standalone" and "--openmp" in sys.argv:
            prefs.devices.cpp_standalone.openmp_threads = int(arg("--openmp", 16, int))
        rng = np.random.default_rng(0)
        samp = np.flatnonzero(non_sens); samp = np.sort(rng.choice(samp, min(4000, len(samp)), replace=False))
        pop_idx = np.unique(np.concatenate([np.flatnonzero(m)[:400] for m in pops.values()] + [samp]))
        mon = StateMonitor(G, "r", record=pop_idx, dt=1 * second, name="ratemon")
        net.add(mon)
        t_wall = time.time()
        print(f"  standalone {DEVICE}: building and running {seconds:g} s ...", flush=True)
        PL.run(net, seconds)
        device.build(run=True)
        t_run = time.time() - t_wall
        r_t = np.asarray(mon.r / Hz); t_ax = np.asarray(mon.t / second)
        pos = {int(i_): k for k, i_ in enumerate(pop_idx)}
        w_ie = np.asarray(syn["ie"].w / siemens) / np.asarray(syn["ie"].w0 / siemens)
        w_ee = np.asarray(syn["ee"].w / siemens) / np.asarray(syn["ee"].w0 / siemens)
        if gains is not None:                              # compose with the warm start: saved gains are vs v2
            w_ie = w_ie * np.asarray(gains["ie"], float); w_ee = w_ee * np.asarray(gains["ee"], float)
        log = []
        s_rows = [pos[int(i_)] for i_ in samp]
        for k, t_sim in enumerate(t_ax):
            h = r_t[s_rows, k]
            rec = dict(t=float(t_sim), mean_hz=float(h.mean()), active=int((h > 2).mean() * non_sens.sum()),
                       ceiling=float((h > 200).mean()), silent=float((h < 0.5).mean()),
                       pops={p: float(r_t[[pos[int(i_)] for i_ in np.flatnonzero(m)[:400]], k].mean())
                             for p, m in pops.items()})
            log.append(rec)
            if k % 5 == 0 or k == len(t_ax) - 1:
                print(f"  t {t_sim:6.1f} s | non-sens mean {rec['mean_hz']:6.2f} Hz  active ~{rec['active']:>6,}  "
                      f">200 Hz {100*rec['ceiling']:4.1f}%  silent {100*rec['silent']:4.1f}% | "
                      + " ".join(f"{p} {v:5.1f}" for p, v in rec["pops"].items()), flush=True)
        total_hz = np.asarray(spikes.count, float) / seconds
        print(f"  final: <wI/w0> {w_ie.mean():.3f} (p90 {np.percentile(w_ie, 90):.2f})  <wE/w0> {w_ee.mean():.3f}  "
              f"(p90 {np.percentile(w_ee, 90):.2f}, p99 {np.percentile(w_ee, 99):.2f})  "
              f"overall non-sens mean {total_hz[non_sens].mean():.2f} Hz | wall {t_run/60:.1f} min "
              f"({t_run/seconds:.1f} s per simulated s incl. compile)", flush=True)
        # did the EB loop assemble? E->E gains among CX core types vs the rest
        pre_ee = np.asarray(syn["ee"].i); post_ee_ = np.asarray(syn["ee"].j)
        core = data["is_cx_core"]
        loop = core[pre_ee] & core[post_ee_]
        is_kc_n = np.char.startswith(ct.astype(str), "KC")
        kc_kc = is_kc_n[pre_ee] & is_kc_n[post_ee_]
        pn_kc = np.char.endswith(ct.astype(str), "PN")[pre_ee] & is_kc_n[post_ee_]
        print(f"  E->E gain: CX core->core {w_ee[loop].mean():.3f} (n {int(loop.sum()):,}) | "
              f"EPG->PEN/PEG {w_ee[(ct[pre_ee]=='EPG') & np.isin(ct[post_ee_], ['PEN_a(PEN1)','PEN_b(PEN2)','PEG'])].mean():.3f} | "
              f"PEN/PEG->EPG {w_ee[np.isin(ct[pre_ee], ['PEN_a(PEN1)','PEN_b(PEN2)','PEG']) & (ct[post_ee_]=='EPG')].mean():.3f} | "
              f"KC->KC {w_ee[kc_kc].mean():.3f} | PN->KC {w_ee[pn_kc].mean():.3f} | all {w_ee.mean():.3f}", flush=True)
        with open(OUT / f"log_{tag}.json", "w") as f:
            json.dump(log, f)
        np.savez_compressed(OUT / f"w_final_{tag}.npz", ee=w_ee, ie=w_ie, t=seconds,
                            params=json.dumps(dict(sens=sens, gsyn=gsyn, eta_i=eta_i, wmax_i=wmax_i, gamma_e=gamma_e,
                                                   floor_e=floor_e, eta_e=eta_e, rho0=PL.RHO0_HZ, seconds=seconds,
                                                   vis_sens=vis_sens, bump=bump, bump_every=bump_every,
                                                   bump_rotate=int(arg("--bump-rotate", 0, int)), brake_recip=brake_recip,
                                                   two_sided=two_sided, rho_low=PL.RHO_LOW_HZ, gamma_up=PL.GAMMA_UP,
                                                   tau_e_ms=float(tau_e / PL.ms) if tau_e is not None else 5.0,
                                                   adapt=adapt, escape=escape, two_comp=two_comp, recip=recip, wmax_e=wmax_e,
                                                   ltp_sat=ltp_sat, gate_fac=gate_fac, recip_ee=recip_ee, recip_een=recip_een,
                                                   da=Path(da_file).name if da_file else None, tau_da=tau_da, da_th=da_th,
                                                   da_release=rel or None,
                                                   ltd_ratio=PL.LTD_RATIO, stdp_start=PL.STDP_START_S,
                                                   warm=Path(warm).name if warm else None,
                                                   device=DEVICE, data=Path(dpath).name,
                                                   seed=seed_val, provenance=prov)))
        print(f"wrote w_final_{tag}.npz", flush=True)
        return 0
    post_ee = np.asarray(syn["ee"].j)
    w0_ie = np.asarray(syn["ie"].w0 / siemens); w0_ee = np.asarray(syn["ee"].w0 / siemens)
    cnt_prev = np.zeros(n); log = []
    t_wall = time.time()
    n_chunks = int(round(seconds / chunk))
    for k in range(n_chunks):
        t_sim = (k + 1) * chunk
        tr = PL.run(net, chunk)
        cnt = np.asarray(spikes.count, float); hz = (cnt - cnt_prev) / chunk; cnt_prev = cnt
        mean_e, n_up = 1.0, 0
        if e_decay and t_sim > e_start:
            exc_in = np.bincount(post_e, weights=ns_e * hz[pre_e], minlength=n) if two_sided else None
            mean_e, n_up = PL.apply_e_decay(G, syn["ee"], chunk, gamma_e=gamma_e, floor_e=floor_e,
                                            rho0=PL.RHO0_HZ, post_idx=post_ee, two_sided=two_sided,
                                            exc_in=exc_in, rate_hz=hz)
        else:
            mean_e = float(np.mean(np.asarray(syn["ee"].w / siemens) / w0_ee))
        w_ie = np.asarray(syn["ie"].w / siemens) / w0_ie
        h = hz[non_sens]
        rec = dict(t=t_sim, mean_hz=float(h.mean()), active=int((h > 2).sum()),
                   ceiling=float((h > 200).mean()), silent=float((h < 0.5).mean()),
                   median_active=float(np.median(h[h > 2])) if (h > 2).any() else 0.0,
                   wI=float(w_ie.mean()), wI_p90=float(np.percentile(w_ie, 90)), wE=mean_e, n_up=n_up,
                   pops={p: float(hz[m].mean()) for p, m in pops.items()}, wall=tr)
        log.append(rec)
        print(f"  t {t_sim:6.1f} s | non-sens mean {rec['mean_hz']:6.2f} Hz  active {rec['active']:>6,}  "
              f">200 Hz {100*rec['ceiling']:4.1f}%  silent {100*rec['silent']:4.1f}% | wI {rec['wI']:.3f} (p90 {rec['wI_p90']:.2f})  "
              f"wE {rec['wE']:.3f}" + (f" up {n_up:,}" if two_sided else "") + " | "
              + " ".join(f"{p} {v:5.1f}" for p, v in rec["pops"].items()) + f"  [{tr:.0f}s]", flush=True)
        if (t_sim % ckpt) < 1e-9 or k == n_chunks - 1:
            np.savez_compressed(OUT / f"w_{tag}_t{int(t_sim)}.npz", ee=np.asarray(syn["ee"].w / siemens) / w0_ee,
                                ie=w_ie, t=t_sim, seed=seed_val, provenance=json.dumps(prov))
            with open(OUT / f"log_{tag}.json", "w") as f:
                json.dump(log, f)
    np.savez_compressed(OUT / f"w_final_{tag}.npz", ee=np.asarray(syn["ee"].w / siemens) / w0_ee, ie=w_ie,
                        t=seconds, params=json.dumps(dict(sens=sens, gsyn=gsyn, eta_i=eta_i, wmax_i=wmax_i,
                                                          gamma_e=gamma_e, floor_e=floor_e, eta_e=eta_e,
                                                          rho0=PL.RHO0_HZ, seconds=seconds, vis_sens=vis_sens,
                                                          bump=bump, bump_every=bump_every, two_sided=two_sided,
                                                          rho_low=PL.RHO_LOW_HZ, gamma_up=PL.GAMMA_UP,
                                                          adapt=adapt, escape=escape, wmax_e=wmax_e,
                                                          ltd_ratio=PL.LTD_RATIO, stdp_start=PL.STDP_START_S,
                                                          warm=Path(warm).name if warm else None,
                                                          data=Path(dpath).name,
                                                          seed=seed_val, provenance=prov)))
    print(f"done in {(time.time()-t_wall)/60:.1f} min; wrote w_final_{tag}.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
