# Round 12 preregistration: real Base, lineage tiers, and a second distillation pair

Frozen 2026-10-18T06:21:27Z, before any Round 12 capture exists. Arms and gates: `RERUN_SPEC.md`.

**What I have already seen, disclosed so that a reader can discount for it.** I saw the May 2026 Base workflow-test run: 120 prompts × 1, 100 steps, CFG 4 pooled (`../out/may_base_hierarchy.csv`). On it:
- Base↔Turbo-8 tangent was 0.736 (attn) and 0.735 (MLP).
- Base↔Turbo-25 (pooled) tangent was 0.642 (attn) and 0.630 (MLP).
- Turbo-8↔Turbo-25 (same weights) was 0.942 (attn) and 0.967 (MLP).
- The largest unrelated tangent in the archive is 0.240.
- The layer-permutation null p95 is 0.17–0.31.

The thresholds below were set with these numbers in view. They are wide on purpose: the May run has a different step count, capture and sample size. No Round 12 data, and no ACE Base data of any kind, has been seen.

**Methods (unchanged from the paper):**
- Tangent and shape: Round 6 S1 (`review/round6/scripts/r6c_common.py`, `tangent()`), on attn and MLP separately.
- Null: B's layer order permuted (the same permutation at every σ), 1000 draws, p95.
- Turbo-8 is the Z-Image Turbo archive (480 generations, CFG 1, cond-only). For Base, only the **cond** branch is used.
- The ACE Turbo side is the existing 120-generation ACE-Step 1.5 Turbo run (`review/r3_ace/data`).
- Layer-axis crossings use the B6 method of `review/r3_ace/R3B_SCORING_PLAN.md` (analyzer `plot_beta_layer`, 29 bins, ≥ 10 points, 100 prompt bootstraps). The cubes are rebuilt from the raw CSVs, keeping only complete generations.

| ID | Test | Supports the paper if | Refuted if |
|---|---|---|---|
| R12-0 | Identity gate (`fingerprint_check.py`) on every Base arm | Z-Image Base: step-0 depth-profile r with Turbo < 0.95 in both submodules. ACE Base: r < 0.85. | Any Base arm reads TURBO. The arm is then discarded, not analysed. |
| R12-1 | Z1 cond ↔ Turbo-8 tangent, attn and MLP | above the null p95 in both | below the null p95 in either |
| R12-2 | Lineage sits **below** same weights: Z1 cond ↔ Turbo-8 tangent ≤ 0.85 (attn) and ≤ 0.87 (MLP), i.e. at least 0.10 under 0.942 / 0.967 | both hold | either is > the bound |
| R12-3 | Lineage sits **above** unrelated: Z1 cond ↔ Turbo-8 tangent ≥ 0.30 + the largest unrelated Z-Image attn tangent (Z1 or Turbo-8 against Mochi, SD3.5 M, SD3.5 L, recomputed). For MLP, where no unrelated model has an MLP channel: ≥ null p95 + 0.30. | both hold | either fails |
| R12-4 | Z3 (Turbo 25 steps CFG 4, cond) ↔ Z1 cond: the same pair at identical settings | R12-2 and R12-3 bounds hold with Z3 in place of Turbo-8. For R12-2 the same-weights reference becomes Z3 ↔ Turbo-8, recomputed. | either bound fails. Not scored if Z3 is not run. |
| R12-5 | Z2 (Base CFG 1) ↔ Turbo-8 | R12-1 to R12-3 hold at CFG 1 | any fails |
| R12-6 | ACE Base ↔ ACE Turbo tangent (cond branches) | ≥ null p95 + 0.30 in both submodules, and ≤ 0.95 | any fails |
| R12-7 | ACE Base layer-axis crossings, attn | **No direction is predicted.** Decision rule fixed now: if ACE Base attn shows 3 crossings (+−+−) in ≥ 50% of bootstraps, the single crossing of ACE Turbo is attributed to distillation and the audio caveat is withdrawn. If it shows 1 crossing in ≥ 50%, the absence is attributed to modality. Anything else: unresolved. | n/a |
| R12-8 | Z1 Base attn and MLP layer-axis crossings | descriptive (the paper's +−+− claims will be restated from Z1, not from Turbo-25) | n/a |

**Consequences for the papers, fixed now:**
- If R12-1 to R12-3 pass, the lineage claim is restated as a middle tier: same weights > distillation lineage > unrelated, with the Z1 numbers.
- If R12-2 fails (the lineage pair matches same weights), the papers say lineage is indistinguishable from fixed weights at this resolution.
- If R12-1 or R12-3 fails, the lineage claim is withdrawn.
- R12-6 adds or withholds the second pair. A single-pair statement then says so.
- Every former "ZIB" channel result is restated either from Z1 or as Turbo-25 (see `../IMPACT_AUDIT.md`).
