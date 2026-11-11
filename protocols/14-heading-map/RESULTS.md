# Protocol 14 results

Persistence (20 s) after a cue at eight positions of the ring, three evaluation seeds each.
Cells: median localization over seeds / median signed angle of the bump in the last 2-s window
relative to the cue angle (deg). A dash: no trial with localization above 0.5.

| network | cue 0 | cue 45 | cue 90 | cue 135 | cue 180 | cue 225 | cue 270 | cue 315 | trials alive | alive and within 30 deg of cue | last-window angle, circular mean (resultant) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 0.89 / +163 | 0.86 / +111 | 0.89 / +80 | 0.89 / +32 | 0.88 / -14 | 0.86 / -54 | 0.88 / -112 | 0.90 / -155 | 24 of 24 | 3 of 24 | +15 (0.99) |
| 1 | 0.76 / +155 | 0.76 / +86 | 0.73 / +64 | 0.81 / +21 | 0.79 / -74 | 0.77 / -94 | 0.76 / -149 | 0.78 / -159 | 24 of 24 | 4 of 24 | -13 (0.90) |
| 2 | 0.84 / +156 | 0.84 / +113 | 0.84 / +86 | 0.79 / +12 | 0.78 / -9 | 0.85 / -42 | 0.84 / -118 | 0.83 / -146 | 24 of 24 | 6 of 24 | +19 (0.96) |
| 3 | 0.86 / +152 | 0.85 / +92 | 0.84 / +70 | 0.87 / +13 | 0.86 / -20 | 0.80 / -60 | 0.81 / -112 | 0.86 / -172 | 24 of 24 | 5 of 24 | +5 (0.98) |
| 4 | 0.86 / +157 | 0.87 / +112 | 0.88 / +5 | 0.88 / -5 | 0.88 / -73 | 0.86 / -62 | 0.87 / -138 | 0.87 / +141 | 24 of 24 | 5 of 24 | -18 (0.87) |
| 5 | 0.93 / +88 | 0.98 / +17 | 0.97 / -13 | 0.95 / -66 | 0.94 / -106 | 0.96 / -148 | 0.95 / +155 | 0.96 / +116 | 24 of 24 | 5 of 24 | -73 (0.98) |

Spontaneous bump (no cue, protocol 02 / archived 8-s trials, seeds 1 and 2) against the preferred position (circular mean of the last-window angles above), and distance of the preferred position from the training sector (cue offset 0, -154 deg):

| network | spontaneous angle, seeds 1, 2 (deg) | preferred position (deg) | spontaneous minus preferred (deg) | preferred minus training sector (deg) |
|---|---|---|---|---|
| P | -7, -12 | +15 | -22, -27 | +169 |
| 1 | -25, -16 | -13 | -12, -4 | +141 |
| 2 | +3, +7 | +19 | -16, -11 | +172 |
| 3 | -5, -6 | +5 | -10, -11 | +159 |
| 4 | -32, -39 | -18 | -14, -21 | +136 |
| 5 | -72, -74 | -73 | +0, -1 | +81 |

Resultant near 1 with many cues alive: bumps end at one position whatever the cue (a preferred position).
Cue angles are the circular mean of the six stimulated EPG neurons' ring angles, recomputed here: offset 0: -154, offset 45: -93, offset 90: -66, offset 135: -8, offset 180: +30, offset 225: +71, offset 270: +125, offset 315: +163 deg. The `cue_angle` field in the result files of these runs is wrong (radians read as degrees) and is not used.
