# Method: how the best brain is built

Principle: **no number is set per cell type.** Inputs are the BANC connectome (who connects to
whom, with how many synapses; transmitter annotations), each neuron's measured surface area, and
universal constants that apply to every neuron and every synapse alike.

## 0. Static base (v2)

- Conductance-based leaky integrate-and-fire neuron per BANC neuron: C_i = c_m · A_i (c_m 1 µF/cm²,
  A_i the L2-mesh surface area), g_L = C_i / τ_m with τ_m 20 ms, rest = reset −52 mV, threshold
  −45 mV, refractory 2.2 ms, E_exc 0 mV, E_inh −70 mV, τ_syn 5 ms.
- Every excitatory synapse: w_ij = n_ij · G_E, one unit conductance G_E = 30 pS per synapse count.
- Every inhibitory synapse: w_ij = n_ij · G_E · ρ · E_i / I_i, where E_i and I_i are neuron i's total
  excitatory and inhibitory synapse counts and ρ = (E_exc − V_th)/(V_th − E_inh) = 1.8: each
  neuron's total inhibition balances its total excitation *at threshold*. Derived, not fitted.
- Transmitter sign from the BANC annotations (verified, else predicted): GABA, glutamate, histamine
  inhibitory; acetylcholine and unknown excitatory; synapses of verified modulator neurons (DA,
  OA, 5-HT, TA, NO) are not fast conductances and are dropped (12 % of synapses).
- Result: feed-forward reflexes work (sugar → MN9, looming → giant fibre) but the brain seizes at
  rest, any odour ignites the mushroom body, and the heading bump fires at the refractory ceiling
  (~280 Hz). Every recurrent circuit is bistable: silent or maximal.

## 1. Slow excitatory synapses

τ_exc = 15 ms with the per-spike conductance divided by 3 (charge per spike unchanged). Smoother
excitation lets the ring loop survive the end of the stimulus (with 5 ms it dies). Ablation C.

## 2. Plasticity, stage 1 (p8): 100 s of homeostasis

Brain under spontaneous drive (all sensory and visual-projection neurons Poisson at 10 Hz), a Kim
kick every 10 s. Two universal rules on the v2 weights:

- **Vogels inhibitory STDP** on every inhibitory→excitatory synapse (Vogels et al. 2011):
  pre spike: w += η_i w0 (x_post − α); post spike: w += η_i w0 x_pre; traces τ_x 20 ms,
  α = 2 ρ0 τ_x, target ρ0 = 30 Hz, η_i 3·10⁻³, bounds [w0, 10 w0]. Inhibition onto a neuron
  grows while it fires above ρ0.
- **Excitatory rate cap**: every 100 ms, E→E synapses onto neurons firing above ρ0 decay,
  w ← w·exp(−0.1 s · 0.002 · (r − ρ0)/Hz), floor 0.1 w0, from t = 10 s.

Outcome: a resting brain (sparse, irregular, a few Hz), sparse odour responses, specific looming.
Cost: the sugar → MN9 reflex is lost here and never recovered by any later step.

## 3. Plasticity, stage 2 (p37a): 500 s with dopamine

Warm start from the p8 weights; kick every 3 s; seed 11.

- Vogels iSTDP continues (ρ0 25 Hz, bounds [w0, 20 w0]).
- E→E plasticity becomes **three-factor, dopamine-gated pair STDP**: LTP on a post spike ∝ pre
  trace, LTD on a pre spike ∝ post trace × 0.5, η_e 8·10⁻³, bounds [0.1 w0, 3.5 w0] — multiplied
  by a gate g(m) = clip((m − 0.35)/(1 − 0.35), 0, 1) on the postsynaptic neuron's dopamine trace m.
  m is incremented by n_syn/100 whenever one of the neuron's dopaminergic presynaptic partners
  fires (`da_edges.npz`: 380 verified dopaminergic neurons, 33,054 edges, read from the
  connectome) and decays with τ 0.3 s. The dopamine is produced by the network itself.
- Selectivity is anatomical: an EPG receives ~100 dopaminergic synapses (ER4m, LPsP — driven by
  the ring loop), a Kenyon cell ~2 (PAM/PPL1). The same rule everywhere potentiates the compass
  loop (E→E gains ×2–2.5) and leaves the mushroom body alone (KC→KC 0.89).

After 600 s the weights are saved (`w_final_p37a.npz`) and never change again.

## 4. Escape noise

Soft threshold: a spike is emitted with hazard λ₀·exp((V − V_th)/Δ), Δ = 1 mV, λ₀ = 1/τ_m = 50 Hz
(Gerstner escape noise; GLM sharpness 1–3 mV). The developed loop's mean drive sits just under
threshold; without the noise it fires only in fits. Ablation B.

## 5. Visual projection neurons in the spontaneous drive set

The optic lobe is excluded from the graph, so its outputs (visual projection and centrifugal
neurons) receive the same 10 Hz spontaneous drive as sensory neurons. This gives the ring
neurons (ER/ExR) their tonic activity and the bump its width. Ablation K.

## 6. Short-term depression scaled by reciprocity (the final piece)

Tsodyks–Markram depression on **excitatory→excitatory** synapses only: resource x per synapse,
dx/dt = (1 − x)/τ_rec, each spike releases U_ij · x and depletes x by the same amount; τ_rec 0.2 s.
The release fraction is set by the edge's reciprocity in the connectome:

    U_ij = U₀ · ρ_ij,   ρ_ij = n_ji / (n_ij + n_ji),   U₀ = 0.2

A feed-forward edge (ORN→PN 0.004, PN→KC 0.004, LC4/LPLC2→GF 0.017) barely depresses; a loop
edge (EPG→PEN 0.13, PEN→EPG 0.17) depresses at ~15 % per spike. This removes the ring's ceiling
fixed point (the frozen p37a bump was 70 Hz and climbed to ~200 Hz on 60 % of 20 s draws)
without touching sensory pathways or the E→I feedback brakes (KC→APL, EPG→Δ7 — which the
all-synapse version depressed, igniting the resting brain). Depression is not learning: x resets
every run, the frozen weights stay frozen.

## Parameters

| parameter | value | provenance |
|---|---|---|
| c_m, τ_m, V_rest = V_reset, V_th, t_rfc, E_exc, E_inh, τ_syn | 1 µF/cm², 20 ms, −52, −45 mV, 2.2 ms, 0, −70 mV, 5 ms | literature (Shiu 2024 / Kakaria 2017) |
| balance ratio ρ | 1.8 | derived from reversal potentials |
| escape λ₀ | 1/τ_m | derived |
| transmitter sign policy; E→E-only depression; reciprocity form | — | structural choices |
| **G_E** | 30 pS | swept (10–60) |
| **τ_exc** | 15 ms | swept (5–50) |
| **Δ** | 1 mV | swept (1–2) |
| **ρ0** | 30 Hz (stage 1), 25 Hz (stage 2) | swept (5–30) |
| **η_e** | 8·10⁻³ | swept (2·10⁻³, 8·10⁻³) |
| **wmax_e** | 3.5 | swept (2.5–5) |
| **τ_da** | 0.3 s | swept (0.3, 1.0) |
| **m_th** | 0.35 | swept (0–0.5) |
| **U₀** | 0.2 | swept (0.1–0.5) |
| **τ_rec** | 0.2 s | swept (0.1–0.5) |
| η_i, iSTDP bounds, τ_x, τ_r, γ_e, floor, LTD/LTP 0.5, DA increment, kick intervals, durations | see above | declared, not swept |

Eleven universal constants were swept (bold). None is per cell type.

## Known limitations

- The bump does not rotate smoothly under PEN drive (it jumps), is narrower than the fly's 90°,
  and sits at the bottom of the 10–40 Hz band; capture by a second cue and competition between
  cues are partial.
- Sugar → MN9 is lost at stage 1; water → MN9 never worked in any brain.
- Under the uniform 10 Hz resting background the Kenyon cells are ~68 % active (in every brain,
  v2 included); the sparse odour codes hold in the silent brain.
- The two-stage 600 s pipeline is historical; whether one stage with the stage-2 rules would reach
  the same brain is untested. Stage 2 ignites the mushroom body on some random draws.
- One connectome, one animal; the benchmarks are qualitative reproductions of published
  experiments, not fits to recordings.
