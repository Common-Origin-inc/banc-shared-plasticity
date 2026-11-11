"""
Protocol 14 diagnostic (added after the first heading-map results were seen, 2026-10-16).
The decoded bump angle after the cue did not track the cue in network P. This records, for one
trial, the rate of every ring EPG neuron during and after the cue and the decoded angle every
100 ms from stimulus onset, so the location of the bump can be compared with the stimulated
neurons directly (rows = position in the ring order used by every test).

usage: diag_trace.py --net P|K|v2 [--cue DEG] [--seed S] [--seconds T] [--device cuda_standalone]
writes protocols/14-heading-map/diag/trace_<net>_c<cue>_s<seed>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BB = HERE.parents[1] / "BANC_BEST_BRAIN"
sys.path.insert(0, str(BB))


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
import capability_tests as CT  # noqa: E402  (ring(), the same cue and decoder as the persistence test)
from brian2 import ms, second, start_scope, seed, device  # noqa: E402

BS = PL.BS
net = arg("--net", "P"); cue = arg("--cue", 0.0, float); sd = arg("--seed", 1, int); T = arg("--seconds", 6.0, float)
if net == "v2":
    gains, escape, tau_e, std = None, None, 5.0 * ms, None
else:
    gp = BB / ("w_final_p37a.npz" if net == "P" else f"w_final_rep{net}_p37a.npz")
    g = np.load(gp); gains = dict(ee=g["ee"], ie=g["ie"]); escape = (1.0, PL.ESCAPE_LAM0_HZ); tau_e = 15.0 * ms
    std = (0.2, 0.2, "recip_ee")
data = PL.load_data(str(BB / "banc_full.npz"), vis_sens=True); n = data["n"]
epg, ang, stimA = CT.ring(data, cue)
tag = f"{net}_c{cue:g}_s{sd}"
if DEVICE != "runtime":
    device.reinit(); device.activate(build_on_run=False, directory=f"/tmp/brian_diag_{tag}")
start_scope(); seed(sd)
netw, G, syn, spikes, masks = PL.build_network(data, g_syn_ps=30.0, plastic=False, gains=gains, sens_hz=10.0,
                                               stim=(stimA, CT.KIM_ON, CT.KIM_OFF, 250.0), stim2=None, duration_s=T,
                                               tau_e=tau_e, escape=escape, std=std)
PL.run(netw, T)
if DEVICE != "runtime":
    device.build(run=True)
st, si = np.asarray(spikes.t / second), np.asarray(spikes.i)
r = BS.smooth(BS.bin_rates(st, si, n, T, epg)); t_axis = np.arange(r.shape[1]) * BS.DT_BIN_MS / 1000
th, R = BS.decode(r, ang)
step = int(round(100 / BS.DT_BIN_MS))
win = lambda a, b: (np.bincount(si[(st >= a) & (st < b)], minlength=n)[epg] / (b - a)).round(1).tolist()
stim_rows = sorted(int(np.flatnonzero(epg == s)[0]) for s in stimA)
zc = np.exp(1j * ang[stim_rows]).mean()
out = dict(net=net, cue_offset=cue, seed=sd, T=T, ring_angle_deg=np.degrees(ang).round(1).tolist(), stim_rows=stim_rows,
           cue_angle_deg=float(np.degrees(np.angle(zc))),
           rows_hz=dict(before=win(0.0, 0.5), stim=win(0.6, 1.5), after_0=win(1.6, 2.6), after_1=win(2.6, 4.6), after_2=win(4.6, T)),
           trace_t=t_axis[::step].round(2).tolist(), trace_angle_deg=np.degrees(th[::step]).round(1).tolist(),
           trace_R=R[::step].round(3).tolist())
(HERE / "diag").mkdir(exist_ok=True)
(HERE / "diag" / f"trace_{tag}.json").write_text(json.dumps(out))
peak = lambda v: int(np.argmax(v))
print(f"[{tag}] cue rows {stim_rows} cue angle {out['cue_angle_deg']:+.0f} | peak row during stim {peak(out['rows_hz']['stim'])}, "
      f"after {peak(out['rows_hz']['after_0'])}/{peak(out['rows_hz']['after_1'])}/{peak(out['rows_hz']['after_2'])}")
