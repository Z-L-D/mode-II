# Post hoc follow-ups on existing data: results (2026-11-11)

Plan `PLAN.md` (sha 27d056e4, frozen 2026-11-10 11:51Z). Every check was defined in writing before it was run but after the data
were seen, so every row is post hoc. Scripts in `scripts/` (`ph_common.py`, `ph_A_atlas.py`, `ph_A_atlas_dec.py`, `ph_BC.py`),
tables in `out/`. Models are compared at each model's default operating point. Nothing in the papers has been changed yet.

## Recommended additions (ranked)

**A. Round 17 models as a replication set for the atlas (Mode-I §3-4, one new table "Atlas rules replayed on four models captured
for another purpose").** These four models were never analysed beyond the tier test.
| Check | Rule replayed | Result |
|---|---|---|
| RC34 | R11-5 backbone (PC1 ≥ 0.70, beats 999 depth-shift nulls) | 6 of 6 channels pass (PC1 0.72 to 0.91; uncond 6 of 6). |
| RC35 | B3 two components ≥ 0.98; B2 anisotropy g/g_MP < 0.10 in ≥ 95% of cells; B4 ΔR² ≥ 0.30 | B3 5 of 6 (Cosmos MLP 0.967). B4 6 of 6 (0.51 to 0.67). **B2 fails in all six**: medians 0.057 to 0.074 attention, 0.114 PixArt MLP, 0.058 Cosmos MLP; 10 to 55% of cells are ≥ 0.10. |
| RC36 | B8 plateau onset in [0.73, 0.92] | 6 of 10 in window. PixArt attention 0.66, Cosmos 0.94 / 0.97, Z1 attention 0.68. Observed range widens to 0.66 to 0.97. |
| RC37 | S4 clock envelope signs | PixArt and Cosmos, attention and MLP: `+,+,−` in 100% of bootstraps. SD3 M and SD3.5 M (R17) attention: `+,+,+` in 100%. No `+,−,+` outside Z-Image Turbo MLP. |
| RC38 | Layer-axis crossings | Not stable: the count changes with row set in 5 of 6 channels (for example PixArt MLP 4 / 5 / 8). The three-crossing pattern appears for Cosmos attention and SD3.5 M (all rows) only. |
| RC39 | Prompt decoding from the ordered triple | Conditional branch: 6 of 6 beat every baseline (0.37 to 0.48; scramble max 0.22 to 0.25, p = 0.01). Unconditional branch at chance (0.083), as expected for one seed. |
| RC40 | Cross-model prompt geometry (S2 Mantel, partial on text and length) | Unrelated attention pairs 0.04 to 0.49 (median 0.29), 96% significant after partialling out text. PixArt ↔ Cosmos 0.36 / 0.33 / 0.44. The triple's text RSA is ≤ 0.11 in every model. |
| RC41 | R15-1 handoff | 4 of 6 pass. PixArt fails the 10%-of-steps criterion (0.24 / 0.31 at step 2) and crosses 0.5 at step 4 of 20 (t = 0.96), ending at 0.995. |

What this changes in wording: (1) "strong anisotropy" is an attention and caption-row MLP result; image-token MLPs sit at 0.06 to
0.11 of the isotropic value, 30 to 80 times closer to isotropy than Z-Image's MLP (0.0014 to 0.0019). The sentence that the U-Net
MLP is about a hundred times closer to isotropy than "the transformer MLPs" should name Z-Image. (2) The attention envelope
`+,+,−` is not universal: the SD3 family conditional branch is `+,+,+`. (3) The plateau window is a Z-Image/SD3.5 range, not a
general constant. (4) The backbone, the two-component rule, autonomy, decoding and prompt geometry replicate on models that
played no part in forming them.

**B. Centred and pruned statistics outside Z-Image (Mode-I §5, Mode-II §7; answers the deferred "row-split rechecks" item).**
| Check | Result |
|---|---|
| RC42 tangent by variant (native grid) | Successor SD3 M ↔ SD3.5 M 0.926 all / 0.848 centred / 0.911 pruned. SD3.5 M two sessions 0.980 / 0.953 / 0.979. PixArt ↔ Cosmos attention 0.196 / 0.130 / 0.220, all below the permutation p95. **PixArt ↔ Cosmos MLP 0.572 all, 0.267 centred (p95 0.199, surrogate 0.234), 0.757 pruned.** |
| RC43 mean-direction energy | Last-layer MLP energy share of the mean direction: 0.9993 PixArt, 0.94 Cosmos (attention medians 0.35 to 0.48). |
| RC44 identity gate by row set | Z-Image Base vs Turbo attention: all rows 0.655, image rows 0.945, caption rows 0.537, centred 0.959. MLP: 0.585 / 0.930 / 0.904 / 0.884. Split-half same weights ≥ 0.997. SD3 M vs SD3.5 M 0.940 / 0.841 centred. |
| RC45 decoding by row set | Z2 and ZITSV: every row set beats every baseline (0.29 to 0.47); centred is lowest for Z2 (0.33 / 0.32). |

Reading: the unrelated MLP value of 0.57 that broke the MLP bottom tier is mostly a shared mean-direction effect. With the mean
removed it falls to 0.27, a margin of 0.07 over the null, while the related pairs keep 0.85 to 0.95. The tier split is robust to
centring; the MLP bottom tier is reduced, not restored. Separately, the gate's Base / Turbo separation comes from caption rows;
on image rows it lands in the undecided band (0.93 to 0.95), which agrees with the R18 result that the gate reads SD3.5 L vs L
Turbo as 0.986.

**C. Lineage sections.**
| Check | Result |
|---|---|
| RC46 decoding on the real Base | Z1 0.54 attention / 0.46 MLP on 479 generations (scramble max 0.15 / 0.14); Z3 0.54 / 0.55. Matches §6.2; the Table caption that says it was not recomputed is stale. |
| RC47 successor by depth half | SD3 M ↔ SD3.5 M 0.940 on blocks 0 to 12 (the MMDiT-X blocks) and 0.912 on 12 to 23; same weights 0.990 / 0.971. The shortfall is not concentrated in the changed blocks. |
| RC48 matrix regression on prompts | Lineage ZIT ↔ ZIB β 0.56 to 0.62; unrelated 0.13 to 0.43, all 24 cells positive at p ≤ 0.01. |
| RC49 tangent per denoising state | Every related attention pair exceeds the largest unrelated per-state value (0.252) and its own p95 at 100% of states (Z1 ↔ ZIT 0.385 to 0.693, rising along the clock; successor 0.889 to 0.952). Unrelated attention exceeds p95 at 0 to 5% of states. MLP: PixArt ↔ Cosmos 0.30 to 0.67 above p95 at all states. |
| RC50 to RC52 | Not run (each needs a port of older code to the new cubes; none would change a claim). |

RC49 is the strongest single addition for the tier section: the attention separation holds state by state, not only on average.

## Lower value, report in the supplement or leave out
RC38 (unstable), RC41 (mixed), RC48 (repeats S2), RC45.

## Stale statements found in the sweep (fix in the same pass)
- Mode-I §3.5 says "the six permutation p-values"; Table 1 uses seven.
- "Code and data" lists Round 11 to 14 preregistrations and "Round 11 to 15" numbers; should run to 17 (18 after this pass).
- Limitations: "MLP hooks exist only for Z-Image" (PixArt and Cosmos have MLP hooks).
- Decode table caption says decoding was not recomputed on real Base; §6.2 gives 0.54 / 0.46.
- 24-point values without a grid label: Fig. 3 caption (0.71), Fig. 11 caption (0.71 / 0.72), §depthcal (0.30 ... 0.69), supplement (0.71 / 0.72).
- README: "149 of 149"; Round 12 row "0.71 / 0.72"; RC22 to RC33 and review rounds 4 to 7 not described.
- DATA.md: archive and R3 rows say "Z-Image Base" where the weights are Turbo; R16 / R17 cubes are in no derived bundle.
- Two "Float too large" warnings (Mode-I Table 1, 35 pt; Mode-II line 573, 11 pt).
- Release check: archives verify 7 of 7, both papers compile (82 / 54 pp), `check_numbers.py` 175 of 175.
