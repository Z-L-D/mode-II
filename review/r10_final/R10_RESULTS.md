# Round 10: final correlation sweep over Mode-I and Mode-II (results)

**Preregistration:** `R10_PREREGISTRATION.md`, SHA-256 `2338e502a682df61d7b409d38c560a05bec8b0f274b1c2237595bcde806ef1a8` (see `R10_PREREG_SHA256.txt` for the freeze time). **Computed:** 2026-10-09 22:33Z. Scripts: `scripts/r10_a_sweep.py` (R10-1 to R10-3), `scripts/r10_b_consistency.py` (R10-4). Outputs in `out/`. No PSTF data.

Two decision rules: **0 held, 2 tested, not supported.** R10-1 and R10-4 are descriptive.

| Test | Result | Verdict |
|---|---|---|
| R10-1 [desc] redundancy of the triple in the archive | Spearman ρ(log g, H) over all cells 0.93–0.97 in 8 cubes; ρ(log g, log gap) -0.79 to -0.20; ρ(H, log gap) -0.75 to -0.06. Mean depth-profile corr(z̃_g, z̃_H) 0.92–0.98 | descriptive: g and H are nearly monotone in each other in the archive; gap carries the distinct information |
| R10-2 plateau onset vs seed-predictability onset (7 ch.) | ρ 0.56, LOO 0.46–0.60 | **tested, not supported** (below the 0.7 rule) |
| R10-3 VR4/walk vs seed-predictability onset | without ZIT-attn (6 ch.) ρ -0.03, LOO -0.50 to 0.20; all 7: ρ -0.29 | **tested, not supported** |
| R10-4 [tab] cross-paper numeric consistency | 36 of 41 strings in both papers. The 5 others (Qwen −0.12 / −0.06, n-stability 0.946 / 0.968, "token 7") are quoted in Mode-I only; Mode-II states them in words ("near zero", "within 0.01") or not at all. No conflicting number | consistent |

## Reading
- The two onset statistics (plateau of the layer-mean trajectory; seed-predictability crossing) do not order the channels the same way, and persistence does not order them with either. The papers make no such claim, so nothing changes.
- R10-1 is a useful qualification of Mode-II's dimension-measures remark: although g, gap and H are not monotone transformations of each other mathematically, in the archive log g and H are nearly so (ρ ≥ 0.93 over all cells; depth profiles correlate at 0.92–0.98). The triple is effectively two-dimensional, with gap the second coordinate. Added as one sentence to the Mode-II remark (descriptive, preregistered as such).

## Paper changes
- Mode-II remark (§2): the R10-1 sentence.
- Both prereg tables: a Round 10 row (SHA 2338e502, 2 rules, 0 / -- / 2).
- Also this round (not R10 tests): Clark 2026 Lyapunov-spectrum paper and the linear-equivalence result cited in Mode-II §2 remark, §7 and §open (positioning only).
