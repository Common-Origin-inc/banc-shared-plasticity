"""
Lattice-field-theory fingerprint of the heading bump (Franchini & Bardella, arXiv 2604.05251;
Bardella et al., Entropy 26, 495, 2024), applied to the SIMULATED ring-EPG raster.

The paper's model: binary raster Ω_iα (N neurons x T bins of one refractory period), action
    A(Ω) = Σ I_i Ω_iα + Σ A_ij Ω_iα Ω_jα + Σ B_αβ Ω_iα Ω_iβ
with {A, B, I} inferred from the raster's rates, spatial covariance ⟨δΦ⟩ and temporal covariance
⟨δΠ⟩ (bi-stationary case). Here the inversion is the standard maximum-entropy (naive mean-field)
one, A = −(⟨δΦ⟩ + λ1)^{-1} off the diagonal, I_i = logit(m_i) − Σ_j A_ij m_j, and B is read from
the lag-autocovariance averaged over the active neurons.

What is compared: the fingerprint of v2 (ceiling bump), p37a (rule-settled, 70 Hz) and the hand
fit (25 Hz), as a rate-independent test of ring-attractor structure — A_ij as a function of the
angular distance between EPGs (a ring attractor gives a cosine-like profile: positive coupling
between neighbours, negative across the ring) — plus the memory kernel's refractory dip and decay
time. There is NO experimental spike raster of the fly ring to fit (Kim 2017 is wedge-level
calcium imaging), so the comparison to Kim is only at the level of the observables Kim reports.

usage: lft_bump.py --tag NAME --gains none|w_final_X.npz [--scale PRE,POST:K[:CLASS] ...]
                   [--escape 1] [--tau-e 15] [--seconds 6] [--seed 1] [--device cuda_standalone]
writes results_lft/lft_<tag>.npz / .json / .png
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
RES = OUT / "results_lft"; RES.mkdir(exist_ok=True)


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
from brian2 import ms, second, start_scope, seed, device  # noqa: E402
BS = PL.BS; CX = BS.CX


def ring_epgs(data):
    cx = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
    cx_root = cx["root_id"].astype(np.int64); cx_lab = cx["labels"].astype(str); cx_ang = cx["angles"]
    cx_epg = np.flatnonzero(cx_lab == "EPG"); order = cx_epg[np.argsort(cx_ang[cx_epg])]
    W = cx["W"].astype(float)
    hub = cx_epg[np.argmax(np.abs(W[np.ix_(cx_epg, cx_epg)]).sum(1))]
    hp = int(np.flatnonzero(order == hub)[0]); k = int(round(45 / 360.0 * len(order)))
    seed_e = order[[(hp - k // 2 + j) % len(order) for j in range(k)]]
    pos = {int(r): i for i, r in enumerate(data["root"])}
    epg = np.array([pos[int(cx_root[e])] for e in order]); ang = cx_ang[order]
    stim = np.array([pos[int(cx_root[e])] for e in seed_e])
    return epg, ang, stim


def apply_scales(data, gains, gsyn, specs):
    """the --scale fitting probe of validate_scorecard (PRE,POST:K[:CLASS])"""
    _, masks, _ = PL.v2_weights(data, gsyn)
    n = data["n"]; ct = data["cell_type"]
    g = {k: (np.asarray(gains[k], float).copy() if gains and k in gains else np.ones(int(masks[k].sum())))
         for k in ("ee", "ie", "ei", "ii")}

    def match(pat):
        if pat == "core": return data["is_cx_core"]
        if pat == "exc": return data["sign"] > 0
        if pat == "inh": return data["sign"] < 0
        if pat == "any": return np.ones(n, bool)
        return np.array([bool(re.fullmatch(pat, c)) for c in ct])
    for spec in specs:
        parts = spec.split(":"); pre_pat, post_pat = parts[0].split(","); kk = float(parts[1])
        classes = (parts[2],) if len(parts) > 2 else ("ee", "ie", "ei", "ii")
        em = match(pre_pat)[data["pre_idx"]] & match(post_pat)[data["post_idx"]]
        for k in classes:
            g[k][em[masks[k]]] *= kk
    return g


def fingerprint(omega, ang, dt_bin, lam=0.05, max_lag_s=0.5):
    """omega: (N, T) binary. Returns rates, spatial covariance, MaxEnt couplings A and fields I,
    the coupling profile vs angular distance, and the temporal kernel from lag autocovariance."""
    N, T = omega.shape
    m = omega.mean(1)                                          # rates per bin
    C = np.cov(omega)                                          # spatial covariance ⟨δΦ⟩ (N x N)
    active = m > 1e-3
    # naive mean-field MaxEnt inversion on the active subset, shrinkage-regularised
    Ca = C[np.ix_(active, active)]
    Ca_reg = Ca + lam * np.trace(Ca) / max(len(Ca), 1) * np.eye(len(Ca))
    A = np.full((N, N), np.nan)
    Ia = np.full(N, np.nan)
    if active.sum() >= 3:
        Jinv = np.linalg.inv(Ca_reg)
        Aa = -Jinv; np.fill_diagonal(Aa, 0.0)
        A[np.ix_(active, active)] = Aa
        ma = np.clip(m[active], 1e-4, 1 - 1e-4)
        Ia[active] = np.log(ma / (1 - ma)) - Aa @ ma
    # coupling profile vs angular distance (only pairs of active neurons)
    ai = np.flatnonzero(active)
    d, a = [], []
    for u in range(len(ai)):
        for v in range(u + 1, len(ai)):
            dd = abs((np.degrees(ang[ai[u]] - ang[ai[v]]) + 180) % 360 - 180)
            d.append(dd); a.append(A[ai[u], ai[v]])
    d = np.array(d); a = np.array(a)
    edges = np.arange(0, 181, 22.5)
    prof = [float(np.mean(a[(d >= lo) & (d < hi)])) if ((d >= lo) & (d < hi)).any() else float("nan")
            for lo, hi in zip(edges[:-1], edges[1:])]
    # cosine fit A(Δθ) ≈ a0 + a1 cos(Δθ) (a1 > 0 = ring-attractor sign structure)
    if len(d) > 3:
        X = np.column_stack([np.ones_like(d), np.cos(np.radians(d))])
        coef, *_ = np.linalg.lstsq(X, a, rcond=None); a0, a1 = float(coef[0]), float(coef[1])
        r2 = 1 - np.sum((a - X @ coef) ** 2) / max(np.sum((a - a.mean()) ** 2), 1e-12)
    else:
        a0 = a1 = r2 = float("nan")
    # temporal kernel: lag autocovariance averaged over active neurons, normalised at lag 0
    L = int(max_lag_s / dt_bin)
    ac = np.zeros(L + 1); cnt = 0
    for i in ai:
        x = omega[i] - m[i]
        v = np.dot(x, x) / T
        if v <= 0: continue
        for lag in range(L + 1):
            ac[lag] += np.dot(x[:T - lag], x[lag:]) / (T - lag) / v
        cnt += 1
    ac = ac / max(cnt, 1)
    lags = np.arange(L + 1) * dt_bin
    # refractory dip = min over lags 1..5 bins; decay time = first lag where ac < 1/e after the dip
    dip = float(ac[1:6].min()) if L >= 5 else float("nan")
    tail = ac[6:]
    below = np.flatnonzero(tail < np.exp(-1))
    tau_dec = float(lags[6 + below[0]]) if len(below) else float("nan")
    return dict(rates_hz=(m / dt_bin), m=m, active=active, C=C, A=A, I=Ia, prof_deg=edges[:-1].tolist(),
                prof=prof, cos_a0=a0, cos_a1=a1, cos_r2=float(r2), ac=ac, lags=lags, dip=dip, tau_dec_s=tau_dec)


def main() -> int:
    tag = arg("--tag", "run"); gpath = arg("--gains", "none"); T_END = arg("--seconds", 6.0, float)
    tau_e = arg("--tau-e", 15.0, float) * ms; gsyn = arg("--gsyn", 30.0, float)
    escape = (arg("--escape", 1.0, float), PL.ESCAPE_LAM0_HZ) if "--escape" in sys.argv else None
    specs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--scale"]
    seed_val = arg("--seed", 1, int)
    std = None                                              # --std U,TAU[,recip|recip_ee] (short-term depression)
    if "--std" in sys.argv:
        parts = arg("--std", "").split(","); std = (float(parts[0]), float(parts[1])) + ((parts[2],) if len(parts) > 2 else ())
    data = PL.load_data(str(OUT / "banc_full.npz"), vis_sens=True); n = data["n"]
    gains = None
    if gpath != "none":
        g = np.load(gpath); gains = dict(ee=g["ee"], ie=g["ie"])
    if specs:
        gains = apply_scales(data, gains, gsyn, specs)
    epg, ang, stim = ring_epgs(data)
    KIM_ON, KIM_OFF = 0.5, 1.547
    print(f"[{tag}] gains {gpath}{' + ' + ' '.join(specs) if specs else ''}; escape {escape}; {T_END:g} s; seed {seed_val}", flush=True)
    if DEVICE != "runtime":
        device.reinit(); device.activate(build_on_run=False, directory=f"/tmp/brian_lft_{tag}")
    start_scope(); seed(seed_val)
    net, G, syn, spikes, masks = PL.build_network(data, g_syn_ps=gsyn, plastic=False, gains=gains, sens_hz=10.0,
                                                  stim=(stim, KIM_ON, KIM_OFF, 250), duration_s=T_END,
                                                  tau_e=tau_e, escape=escape, std=std)
    PL.run(net, T_END)
    if DEVICE != "runtime":
        device.build(run=True)
    st, si = np.asarray(spikes.t / second), np.asarray(spikes.i)
    # raster of the 45 ring EPGs, bins of one refractory period, persistence window after light-off
    dt_bin = float(BS.t_rfc / second)
    t0, t1 = KIM_OFF + 0.05, T_END
    nb = int((t1 - t0) / dt_bin)
    omega = np.zeros((len(epg), nb), np.uint8)
    for k, i in enumerate(epg):
        tt = st[(si == i) & (st >= t0) & (st < t0 + nb * dt_bin)]
        omega[k, ((tt - t0) / dt_bin).astype(int)] = 1
    fp = fingerprint(omega, ang, dt_bin)
    # Kim-level observables on the same window
    r_epg = CX.smooth(BS.bin_rates(st, si, n, T_END, epg)); t_axis = np.arange(r_epg.shape[1]) * BS.DT_BIN_MS / 1000
    kb = t_axis > t0
    th, R = CX.decode(r_epg, ang)
    fwhm = BS.bump_fwhm_deg(r_epg[:, kb].mean(1))
    a = CX.analyse(r_epg, np.arange(len(epg)), np.arange(len(epg)), ang, t_axis)
    ang_t = np.degrees(th[kb]); drift = float((ang_t[-1] - ang_t[0] + 180) % 360 - 180)
    act = fp["active"]
    print(f"LFT {tag:<8}| active EPG {int(act.sum()):2d}  rate {fp['rates_hz'][act].mean():6.1f} Hz | "
          f"A(Δθ) profile {' '.join(f'{p:+.2f}' if np.isfinite(p) else 'nan' for p in fp['prof'])} | "
          f"cos fit a1 {fp['cos_a1']:+.3f} (a0 {fp['cos_a0']:+.3f}, R2 {fp['cos_r2']:.2f}) | "
          f"B: dip {fp['dip']:+.2f} tau {1e3*fp['tau_dec_s']:.0f} ms | FWHM {fwhm:.0f} deg  R {float(R[kb].mean()):.2f}  "
          f"alive {a['live_frac']:.2f}  drift {drift:+.0f} deg", flush=True)
    np.savez_compressed(RES / f"lft_{tag}.npz", omega=omega, ang=ang, dt_bin=dt_bin, C=fp["C"], A=fp["A"], I=fp["I"],
                        ac=fp["ac"], lags=fp["lags"], rates_hz=fp["rates_hz"], r_epg=r_epg, t_axis=t_axis)
    with open(RES / f"lft_{tag}.json", "w") as f:
        json.dump(dict(tag=tag, gains=gpath, scales=specs, n_active=int(act.sum()),
                       rate_active_hz=float(fp["rates_hz"][act].mean()) if act.any() else 0.0,
                       prof_deg=fp["prof_deg"], prof=fp["prof"], cos_a0=fp["cos_a0"], cos_a1=fp["cos_a1"],
                       cos_r2=fp["cos_r2"], B_dip=fp["dip"], B_tau_s=fp["tau_dec_s"], fwhm_deg=fwhm,
                       R=float(R[kb].mean()), alive=float(a["live_frac"]), drift_deg=drift,
                       kim_2017=dict(fwhm_deg=90, persists_after_light_off=True, raster="none (calcium imaging, 16 wedges)")),
                  f, indent=1, default=float)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
        ax[0].imshow(omega, aspect="auto", cmap="Greys", interpolation="nearest",
                     extent=[t0, t0 + nb * dt_bin, len(epg), 0]); ax[0].set_title("EPG raster (ring order)"); ax[0].set_xlabel("s")
        Ad = np.where(np.isfinite(fp["A"]), fp["A"], 0.0)
        v = np.nanmax(np.abs(Ad)) or 1
        ax[1].imshow(Ad, cmap="RdBu_r", vmin=-v, vmax=v); ax[1].set_title("inferred A_ij (ring order)")
        ax[2].plot(fp["prof_deg"], fp["prof"], "o-"); ax[2].axhline(0, color="k", lw=0.5)
        ax[2].set_xlabel("angular distance (deg)"); ax[2].set_title(f"A vs Δθ  (cos a1 {fp['cos_a1']:+.2f})")
        ax[3].plot(1e3 * fp["lags"], fp["ac"]); ax[3].axhline(0, color="k", lw=0.5)
        ax[3].set_xlabel("lag (ms)"); ax[3].set_title("temporal kernel (autocorr.)")
        fig.suptitle(f"{tag}: {int(act.sum())} active EPG @ {fp['rates_hz'][act].mean():.0f} Hz, FWHM {fwhm:.0f} deg")
        fig.tight_layout(); fig.savefig(RES / f"lft_{tag}.png", dpi=110)
    except Exception as e:
        print(f"(no figure: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
