# Protocol 02 results

Measured values, one row per learned network. No threshold is applied.

| network | rest_mean_hz | rest_ceiling_pct | epg_hz | alive | fwhm_deg | hz_bump | kc_da1_pct | gf_hz | mn9_sugar_hz | sc_rest_mean_hz |
|---|---|---|---|---|---|---|---|---|---|---|
| published (stage 1 seed 7, stage 2 seed 11) | 6.66 | 0.21 | 14.96 / 21.61 / 23.73 | 0.87 / 0.89 / 0.90 | 56.00 / 70.22 / 72.89 | 22.19 / 29.42 / 32.68 | 3.38 | 15.79 | 0.00 | 4.77 |
| committed table, replicate 1 (seeds 101 / 201) | 8.06 | 0.22 | 11.72 / 12.35 / 12.65 | 0.66 / 0.76 / 0.81 | 60.44 / 72.89 / 81.78 | 10.57 / 13.13 / 17.05 | 23.62 | 16.32 | 0.00 | 6.28 |
| committed table, replicate 2 (seeds 102 / 202) | 8.10 | 0.22 | 15.09 / 18.75 / 20.71 | 0.79 / 0.83 / 0.86 | 69.33 / 76.44 / 84.44 | 19.16 / 27.38 / 29.33 | 22.52 | 17.37 | 0.00 | 6.32 |
| committed table, replicate 3 (seeds 103 / 203) | 9.43 | 0.40 | 17.77 / 19.98 / 23.41 | 0.78 / 0.86 / 0.88 | 55.11 / 68.44 / 84.44 | 25.30 / 29.09 / 31.82 | 50.83 | 15.26 | 0.00 | 7.35 |
| committed table, replicate 4 (seeds 104 / 204) | 9.40 | 0.34 | 11.61 / 13.53 / 14.72 | 0.80 / 0.85 / 0.86 | 57.78 / 74.67 / 80.00 | 13.89 / 15.89 / 17.73 | 49.80 | 15.79 | 0.00 | 7.49 |
| committed table, replicate 5 (seeds 105 / 205) | 6.53 | 0.21 | 22.37 / 26.25 / 34.80 | 0.91 / 0.94 / 0.97 | 48.00 / 52.44 / 56.00 | 33.45 / 39.26 / 44.23 | 2.88 | 17.37 | 0.00 | 4.63 |
| first-token table, replicate 1 (seeds 101 / 201) | 7.90 | 0.23 | 4.87 / 7.05 / 7.47 | 0.23 / 0.23 / 0.29 | 33.78 / 39.11 / 53.33 | 3.91 / 5.43 / 7.44 | 23.86 | 16.84 | 0.00 | 5.99 |
| first-token table, replicate 2 (seeds 102 / 202) | 6.38 | 0.21 | 4.35 / 4.62 / 5.64 | 0.12 / 0.15 / 0.23 | 27.56 / 32.89 / 36.44 | 2.55 / 4.05 / 5.43 | 3.19 | 16.32 | 0.00 | 4.55 |
| first-token table, replicate 3 (seeds 103 / 203) | 9.03 | 0.41 | 3.56 / 5.27 / 6.24 | 0.09 / 0.19 / 0.23 | 30.22 / 38.22 / 47.11 | 3.24 / 4.74 / 6.87 | 51.52 | 14.21 | 0.00 | 7.09 |
| first-token table, replicate 4 (seeds 104 / 204) | 6.36 | 0.21 | 4.17 / 6.94 / 9.38 | 0.10 / 0.19 / 0.25 | 24.89 / 40.00 / 45.33 | 2.92 / 4.45 / 7.03 | 3.25 | 16.32 | 0.00 | 4.57 |
| first-token table, replicate 5 (seeds 105 / 205) | 9.09 | 0.36 | 6.88 / 8.30 / 13.13 | 0.21 / 0.32 / 0.42 | 49.78 / 55.11 / 62.22 | 3.70 / 7.97 / 8.96 | 50.75 | 15.79 | 0.00 | 7.18 |

## Columns

- `rest_mean_hz`: non-driven mean rate (Hz), capability rest seed 1
- `rest_ceiling_pct`: non-driven neurons above 200 Hz (%)
- `epg_hz`: window-active EPG rate (Hz), persistence seeds 1 to 5: min / median / max
- `alive`: fraction of bins with R > 0.5, same seeds: min / median / max
- `fwhm_deg`: bump width (deg), same seeds: min / median / max
- `hz_bump`: bump-neuron rate (Hz), same seeds: min / median / max
- `kc_da1_pct`: Kenyon cells active, DA1 at 3 synapse equivalents (%)
- `gf_hz`: giant fibre rate, LC4+LPLC2 (Hz)
- `mn9_sugar_hz`: MN9 rate, sugar (Hz)
- `sc_rest_mean_hz`: scorecard rest mean (Hz): sensory-only background, 2 s, seed 3

## Checks

- Stage 1 with seed 7 reproduces the published `w_final_p8.npz` exactly (`ee` and `ie` arrays identical).
- Stage 1 checkpoints are identical across the two tables for every replicate.
- Replicates 2 to 5 of both tables were also run on separate GPUs of the same model. For the same seeds,
  stage-2 gains differed between machines by at most 4.4e-16 in any gain (last-bit rounding) in seven of
  eight comparisons and were identical in one; every primary outcome was identical.
- `ablate_check.py` exits nonzero when it prints FAIL under its historical thresholds; those verdicts are
  descriptive, not errors.
