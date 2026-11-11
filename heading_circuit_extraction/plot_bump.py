"""One run, one figure: what the bump looks like.

Uses ring_attractor_sim.run_sim at a single config (no sweep) and draws
  1. spike raster of all 45 EPG, sorted by recovered ring angle
  2. bump profile (rate vs ring position) at several times after light-off
  3. the same profiles on a polar plot, which is what the ring actually is
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ring_attractor_sim as S

OUT = Path(__file__).resolve().parent
G_SYN_PS = S.G_SYN_PS                        # the model, as in ring_attractor_sim
B_ADAPT_NS = S.B_ADAPT_NS
TAU_TABLE = S.TAU_UNIFORM
CHARGE_FIXED = True
SEED = 100
SNAPS = [0.75, 1.2, 1.7, 2.5, 3.3]          # s; first one is during the light

d = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
W = d["W"].astype(float); lab = d["labels"].astype(str); ang = d["angles"]
area = d["area_nm2"].astype(float); nt = d["nt"].astype(str)
epg = np.flatnonzero(lab == "EPG")
order = epg[np.argsort(ang[epg])]
hub = epg[np.argmax(np.abs(W[np.ix_(epg, epg)]).sum(1))]
hp = int(np.flatnonzero(order == hub)[0])
seed_set = order[[(hp - S.ARC_SIZE // 2 + k) % len(order)
                  for k in range(S.ARC_SIZE)]].tolist()
seed_rows = sorted(int(np.flatnonzero(order == s)[0]) for s in seed_set)

# ---- run, keeping the raw spike times this time
S.start_scope(); S.seed(SEED)
net, spikes = S.build(W, area, G_SYN_PS,
                      [(S.STIM_ON, S.STIM_OFF, seed_set, S.R_POI)], S.T_END,
                      b_adapt_ns=B_ADAPT_NS, nt=nt, tau_table=TAU_TABLE,
                      charge_fixed=CHARGE_FIXED)
net.run(S.T_END * S.second, namespace=S.NAMESPACE)
st = np.asarray(spikes.t / S.second); si = np.asarray(spikes.i)
row_of = {int(nrn): k for k, nrn in enumerate(order)}
mask = np.isin(si, order)
st_e, rows_e = st[mask], np.array([row_of[int(x)] for x in si[mask]])

# smoothed rates for the profiles
n_bins = int(round(S.T_END / (S.DT_BIN_MS / 1000)))
rates = np.zeros((len(order), n_bins))
np.add.at(rates, (rows_e, np.floor(st_e / (S.DT_BIN_MS / 1000)).astype(int)), 1)
rates *= 1000.0 / S.DT_BIN_MS
sm = S.smooth(rates, 100.0)
t_axis = np.arange(n_bins) * S.DT_BIN_MS / 1000
theta, R = S.decode(sm, ang[order])

INK, MUTED = "#1b2320", "#5d6b64"
cols = plt.cm.viridis(np.linspace(0.15, 0.95, len(SNAPS)))

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.6, 1])
ax_r = fig.add_subplot(gs[0, :])
ax_p = fig.add_subplot(gs[1, 0])
ax_o = fig.add_subplot(gs[1, 1], projection="polar")

# 1. raster
ax_r.axvspan(S.STIM_ON, S.STIM_OFF, color="cyan", alpha=.15, lw=0,
             label="light on 8 EPG")
ax_r.plot(st_e, rows_e, "|", color=INK, ms=4, mew=.6)
for r in seed_rows:
    ax_r.plot(-0.03, r, ">", color="#2f6f8f", ms=6, clip_on=False,
              transform=ax_r.get_yaxis_transform())
pos = np.interp(theta, ang[order], np.arange(len(order)))
pos[R < 0.35] = np.nan
ax_r.plot(t_axis, pos, "-", color="#d1495b", lw=2, label="decoded bump position")
for t0, c in zip(SNAPS, cols):
    ax_r.axvline(t0, color=c, lw=1.4, ls="--")
ax_r.set_xlim(0, S.T_END); ax_r.set_ylim(-1, len(order))
ax_r.set_ylabel("EPG neuron, sorted by recovered ring angle")
ax_r.set_xlabel("time (s)")
ax_r.set_title(f"BANC ring attractor — EPG spikes   ·   G_syn {G_SYN_PS:g} pS, "
               f"τ ach/gaba/glu {TAU_TABLE['ach']:g}/{TAU_TABLE['gaba']:g}/"
               f"{TAU_TABLE['glu']:g} ms, b_adapt {B_ADAPT_NS:g} nS",
               fontsize=13, fontweight="bold", color=INK, pad=12)
ax_r.legend(loc="upper right", frameon=False, fontsize=9)

# 2. profiles
x = np.arange(len(order))
for t0, c in zip(SNAPS, cols):
    k = int(np.argmin(np.abs(t_axis - t0)))
    tag = "during light" if t0 < S.STIM_OFF else f"+{t0 - S.STIM_OFF:.1f} s after"
    ax_p.plot(x, sm[:, k], "-o", color=c, ms=3.5, lw=1.6, label=f"t = {t0:.2f} s  ({tag})")
    ax_o.plot(np.r_[ang[order], ang[order][0]], np.r_[sm[:, k], sm[0, k]],
              "-", color=c, lw=1.6)
    ax_o.fill(ang[order], sm[:, k], color=c, alpha=.12)
for r in seed_rows:
    ax_p.axvline(r, color="#2f6f8f", lw=6, alpha=.12)
ax_p.set_xlabel("ring position (EPG index by recovered angle)")
ax_p.set_ylabel("rate (Hz, 100 ms window)")
ax_p.set_title("bump profile — blue band = the 8 stimulated EPG",
               fontsize=11, fontweight="bold", color=INK)
ax_p.legend(frameon=False, fontsize=8.5)
ax_p.set_xlim(-0.5, len(order) - 0.5)

ax_o.set_theta_zero_location("N"); ax_o.set_theta_direction(-1)
ax_o.set_yticklabels([]); ax_o.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
ax_o.set_title("same profiles on the ring", fontsize=11, fontweight="bold", color=INK)
for ax in (ax_r, ax_p):
    ax.grid(True, alpha=.22); ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "bump.png", dpi=170, bbox_inches="tight", facecolor="white")
after = t_axis > S.STIM_OFF + .05
act = rates[:, after].mean(1) > S.ACTIVE_HZ
print(f"active EPG after light-off: rows {np.flatnonzero(act).tolist()} "
      f"@ {rates[act][:, after].mean():.0f} Hz   (stimulated rows {seed_rows})")
print(f"wrote {OUT / 'bump.png'}")
