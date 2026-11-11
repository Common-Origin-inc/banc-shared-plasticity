# Results

All numbers are from the JSONs in `results/` (scorecards, capability battery, LFT) and the run
log `results/capability/cap.log`. Best brain = `w_final_p37a.npz` with `--escape 1 --tau-e 15
--vis --std 0.2,0.2,recip_ee`; v2 = `--gains none`, no noise, τ_e 5 ms (its published form).

## Summary scorecard (criteria in TESTS.md)

| # | test | v2 | best |
|---|---|---|---|
| 1 | Kim bump, silent brain | ✗ (280 Hz) | ✓ (10 Hz, alive 0.79) |
| 2 | Kim bump, resting brain | ✗ (seized) | ✓ (10–11 Hz, R 0.95–0.97, alive 0.90–0.98) |
| 3 | 20 s persistence | ✗ | ✓ (22–33 Hz, alive 0.87–0.90, 5/5 seeds) |
| 4 | spontaneous bump | ✗ | ✓ (12–15 Hz, R 0.84–0.89) |
| 5 | PEN rotation | ✗ (rigid) | ✗ (jumps) |
| 6 | second cue captures bump | ✗ | ✗ (+36 to +46°) |
| 7 | competition → one bump | ✗ | ✗ (1 of 2 seeds) |
| 8 | drive invariance | ✗ | ✓ (14 / 15 / 14 / 13 Hz) |
| 9 | bump width | ✓ (96°) | ✗ (56–80°) |
| 10 | LFT coupling structure | ✗ (flat) | ✓ |
| 11 | sugar → MN9 | ✓ (150–182 Hz) | ✗ (0) |
| 12 | bitter silent | ✓ | ✓ |
| 13 | bitter suppresses sugar | ✗ | ✗ |
| 14 | water → MN9 | ✗ | ✗ |
| 15 | post-sugar activity decays | ✗ | ✓ |
| 16 | looming → GF, specific | ✓ (44–52 Hz) | ✓ (16 Hz) |
| 17 | single LC types silent | ✓ | ✓ |
| 18 | looming evoked activity decays | ✗ | ✓ |
| 19 | olfaction DA1 sparse | ✗ (45–80 %) | ✓ (3.4 / 9.1 %) |
| 20 | two odours sparse + distinct | ✗ (Jaccard 0.78) | ✓ (Jaccard 0.01) |
| 21 | concentration series graded | ✗ (saturated) | ✓ (1.1 / 3.3 / 8.0 / 10.4 %) |
| 22 | odour evoked activity decays | ✗ | ✓ |
| 23 | resting state | ✗ (31k @ 43 Hz) | ✓ (13k @ 4.8 Hz, CV 1.0) |
| 24 | rate distribution | ✗ | ✓ |
| | **total** | **5 / 24** | **17 / 24** |

## Full suite, best brain (`results/scorecards/scorecard_p37a_stdrecip_eeU0.2t0.2_full_seed1.json`, `..._kimrest_seed2/3.json`, `results/capability/`)

| experiment | best brain | frozen p37a (no depression) |
|---|---|---|
| Kim bump, silent | 26 EPG @ 10 Hz, 32°, R 0.97, alive 0.79 | 116 Hz, 32° |
| Kim bump, resting, seeds 1/2/3 | 30–36 EPG @ 10–11 Hz, R 0.95–0.97, alive 0.90/0.98/0.95; a spontaneous bump exists before the kick and the kick relocates it | 68/75/75 Hz over 4 s; metastable over 20 s |
| 20 s persistence, resting, seeds 1–5 | bump EPGs 22–33 Hz (peak 30–48), alive 0.87–0.90, 56–73°, step 6–14°/2 s | ~40 % of draws hold, the rest climb to ~200 Hz within 2–8 s |
| 20 s persistence, silent, 3 seeds | 11–15 Hz, alive 0.73–0.82, 29–35°, fragments (2–4 bumps) | 100–121 Hz, stable |
| PEN rotation (seeds 100, 101) | left −90 / right +178; left +96 / right +94; trajectories jump 120–180° per 100 ms | +0.3 / −1.2° (rigid) |
| second cue 90/180°, f 50/250 | +36 / +45, +4 / +46 | +33 / +90 |
| competition (2 seeds) | 1 bump / 2 bumps | 1 bump at ceiling |
| drive invariance f 50/100/250/500 | 14 / 15 / 14 / 13 Hz, 80/80/80/56° | 189 / 192 / 90 / 100 Hz |
| sugar / bitter / water / sugar+bitter → MN9 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| looming LC4+LPLC2 → GF | 15.8 Hz; MDN, DNp09, DNa02 0 | 41.6 Hz |
| single LC types | GF 0–0.5 Hz | silent |
| olfaction DA1 ×3 / ×8 | KC 3.4 / 9.1 % | 5.8 / 11.3 % |
| olfaction DM1 / VA1v / DL3 | 22.7 / 0 / 3.7 % | 76 / 69 / 3.7 % |
| resting state | 13.3k @ 4.8 Hz, CV 1.01, 0.2 % ceiling; capability: 16.8k @ 6.7 Hz, corr 0.001 | 17.8k @ 5.75 Hz |
| spontaneous bump (2 seeds) | 25–27 EPG @ 12–15 Hz, R 0.84–0.89 | — |
| two odours, silent | 3.0 % / 3.5 %, 4 shared, Jaccard 0.01 | — |
| concentration ×1/3/8/16 | KC 1.1 / 3.3 / 8.0 / 10.4 %; PN 12 / 43 / 73 / 94 Hz | — |
| evoked decay, silent | odour +3.6 neurons per stimulated neuron, decays; looming +9.6, GF 16 Hz, decays | — |
| LFT (1.6–6 s) | 29 EPG @ 14 Hz; A(Δθ) +12.4 +2.1 +0.1 −0.2 +0.2 −0.1 −0.1 −0.1; FWHM 80°, R 0.91 | +4.2 → −0.1, FWHM 88° |

v2 on the same tests (`results/capability/cap_v2_*.json`, LFT `results/lft/lft_v2.*`): silent
bump 277–286 Hz, 96°, ISI CV 0.03; resting brain 31.3k non-sensory @ 43 Hz, 6.6 % at ceiling;
odours 56 % / 48 % of KCs, Jaccard 0.78; concentration 52 / 56 / 56 / 77 %; evoked +18.7 and
+20.9 neurons per stimulated neuron with ~2,900 still on after 1 s; LFT profile flat.

## Depression variants (how the final rule was reached)

| variant | bump (20 s, 3 seeds) | brain | verdict |
|---|---|---|---|
| none (frozen p37a) | 70 Hz, ~40 % of draws hold, the rest climb to ~200 Hz | rest 5.8 Hz, KC 5.8 %, GF 42 Hz | metastable |
| uniform STD on all E synapses, U 0.2, τ 0.2 (`scorecard_p37a_stdU0.2t0.2.json`) | 57 Hz, 92°, stable, drive-invariant | **sugar 0, GF 0.5 Hz, KC 1 %, PN 11 Hz** — propagation dies | rejected |
| reciprocity-scaled, all E synapses, U₀ 0.4 (`scorecard_p37a_stdrecipU0.4t0.2.json`) | 67–71 Hz, ring fragments into 2–6 bumps | **rest 28k @ 35 Hz**, KC 17–28 % — the E→I feedback brakes (KC→APL U 0.23, EPG→Δ7 0.085) are depressed too | rejected |
| reciprocity-scaled, E→E only, U₀ 0.4 (`..._stdrecip_eeU0.4t0.2.json`) | 10–14 Hz, alive 0.56–0.58 (dim) | rest 4.5 Hz, KC 3.1 / 8.1 %, GF 9 Hz | too strong |
| E→E only, U₀ 0.3 | 16–18 Hz, alive 0.73–0.78 | — | too strong |
| **E→E only, U₀ 0.2** | **29–33 Hz, alive 0.88–0.89** | rest 4.8 Hz, KC 3.4 / 9.1 %, GF 16 Hz | adopted |
| E→E only, U₀ 0.1 / τ 0.1 | alternates 20 ↔ 85 Hz windows | — | too weak |

Precheck of the rule (count-weighted U at U₀ 0.4): core→core 0.127, EPG→PEN/PEG 0.132,
PEN/PEG→EPG 0.165, KC→KC 0.087, ORN→PN 0.004, PN→KC 0.004, LC4/LPLC2→GF 0.017; KC→APL 0.228,
EPG→ExR/Δ7 0.085 (why the E→I restriction is needed).

## Ablation

One component removed, everything else at the best setting (`results/scorecards/scorecard_ablate_*.json`, `results/capability/cap_p37a_persist_s*_stdrecip_eeU0.2t0.2_<flag>.json`).

| removed | bump (3 seeds, 20 s) | rest | KC | GF | verdict |
|---|---|---|---|---|---|
| — (reference) | 29–33 Hz, alive 0.88–0.89 | 4.8 Hz | 3.4 % | 16 Hz | — |
| E. all learning (v2 weights + depression) | 0 Hz | 14.7 Hz, 0.5 % ceiling | 39.9 % | 26 Hz | FAIL — no bump, MB ignites |
| B. escape noise | 17–27 Hz, alive 0.36–0.65 | 4.0 Hz | 1.3 % | 8 Hz | FAIL — bump dies for whole windows |
| C. τ_exc 15 ms (→ 5 ms) | 10–11 Hz, alive 0.51–0.55 | 3.6 Hz | 6.5 % | 13 Hz | FAIL — dim, half-alive |
| D. stage-2 settle (p8 weights) | 0 Hz | 4.7 Hz | 4.0 % | 17 Hz | FAIL — no loop E, no bump |
| F. learned excitation (inhibition only) | 0 Hz | 11.3 Hz | 35.2 % | 25 Hz | FAIL |
| G. learned inhibition (excitation only) | 141–152 Hz, alternating 52 ↔ 164 | 7.1 Hz | 7.6 % | 16 Hz | FAIL — ceiling bump even with depression |
| K. visual-projection drive | 11–15 Hz, alive 0.66–0.71, fragments | 4.8 Hz | 3.4 % | 16 Hz | FAIL |

Every component is necessary; none is harmful. The learned halves are complementary: E alone
gives a ceiling bump, I alone gives none.

## Negative results kept for the record

- **Inhibitory rule left running during the tests** (`--live`): rate held in band (31–33 Hz, 20 s,
  3/3 seeds) but the bump hops 90–100° every 2–6 s; slowing the rule 10× raises the rate without
  lengthening the dwell (`results/capability/*_live*.json`).
- **Reciprocity-gated homeostatic brake in the settle** (`--brake-recip`; `scorecard_p8r.json`,
  `scorecard_p37ar*.json`): does not recover sugar (0 in p8r), kills the ring in p8r, and the
  p37ar draw leaks the mushroom body (KC→KC 1.93; KC 62–88 % on every glomerulus).
- **Resting brain without ORNs in the spontaneous set** (`*_noorn*.json`): KCs still 75 % active at
  rest — the mushroom body is driven by the other spontaneously active inputs; the uniform 10 Hz
  background is the limitation, not the ORNs.
