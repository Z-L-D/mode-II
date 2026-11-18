# Audit of the second reviewer's Round 3 analysis (ledger v0.13 and the v0.14 B7 correction)

Audited 2026-10-06 by Claude against `review/round3/R3_PREREGISTRATION.md` (SHA-256 d29fdfa6…23f4a0) and the raw captures in `data/r3_captures/`. The reviewer's v0.13 scripts were not in the upload, so every number below was recomputed independently with the Round 2/3 estimators (`review/round3/audit/scripts/a1_reproduce.py`, output `review/round3/audit/out/A1_reproduce.json`). Ledger v0.14 HTML SHA-256 matches the stated 7e6769e3…885ff9.

## Verdict

The reviewer's audit is sound. Every reported number except one reproduces to the printed precision. The formal scorecard is right. The one number that does not reproduce looks like an arm label swap, and fixing it makes the reviewer's own conclusion stronger. Two interpretations need narrowing, and the v0.14 B7 correction was the right call (Round 4 P4 now settles most of it from existing data).

## Reproduction table

| Reviewer claim | Recomputed | Status |
|---|---|---|
| hook_mode=passive, branch=all, no prompt_influence in every arm | Same in all 8 arms; root cause traced to the split flag arriving False (see `capture_code_r2/README_R2.md` §0) | Reproduced |
| 60 prompts run vs 30 frozen; 29/30 present; sea-otter prompt missing; 31 extra | Same | Reproduced |
| One duplicate archive and one nonfinite 512 generation excluded | `-2.zip` is byte-identical to the S1001 CFG 4.5 1024 zip (md5); 512 animals_wildlife gen 3 nonfinite | Reproduced |
| Prompt hashes are md5-derived | `preflight_check.py` finds 0 mismatches | Reproduced |
| B8a tangent 0.999187 attn / 0.999535 mlp; shape r 0.999731 / 0.999611 | 0.999187 / 0.999535; 0.999731 / 0.999611 | Exact |
| B2a onset ZIB CFG 1 attn 0.749700; mlp 0.778579; SD35M CFG 1 0.84058 | 0.749700; 0.778579; 0.840580 | Exact |
| B3 seed share CFG 1 0.239→0.295; CFG 4.5 0.631→0.390; B3b range 0.257392 | 0.238770→0.294712; 0.630822→0.389669; 0.257392 | Exact |
| VR(2/4/8) clean CFG 1 = 1.415 / 1.844 / 2.144 | 1.4148 / 1.8438 / 2.1444 | Exact |
| σ50 ≈ 0.991 clean CFG 1 | 0.991063 | Exact |
| Seed identity 100% decodable at step 0 (held-out prompts), both arms | CFG 4.5: 100%; CFG 1: 95.8% with an L2 logistic classifier (C=1) | Reproduced (classifier-dependent at CFG 1) |
| Topology +−+−, 3 crossings, TRI 0.0296 / 0.0937 / 0.0146 / 0.0189 (A1–A4) | +−+−, 3 crossings; TRI 0.029639 / 0.093745 / 0.014552 / 0.018854 | Exact |
| B6 spread ratio by quarter 1.109 / 0.773 / 0.558 / 0.434; last-three-quarter median 0.589 | 1.1095 / 0.7725 / 0.5584 / 0.4337; 0.5889 | Exact |
| "Clean ZIB-CFG1 accounts for only 0.278 (attn) / 0.250 (mlp) of the old offset" | ZIB **CFG 1**: 0.046–0.081 attn, 0.049–0.059 mlp. ZIB **CFG 4** (branch-mean): 0.264–0.276 attn, 0.224–0.232 mlp | **Not reproduced as labelled.** The reported values match the CFG 4 arm. Likely an arm label swap |

## Scorecard check

29 NOT SCORABLE + B8a CONFIRMED + B3b REFUTED + B2a REFUTED (boundary). Agreed, row by row:
- A1–A3, C0–C4, D0–D6: no data.
- B1a–c, B5, B5b, B6, B7a–b, B8b, B8c, B9: these need cond/uncond rows. B2b needs ≥ 3 cond arms. B3a and B4 need "every arm" or "≥ 3 of 4 arms" and only one clean arm has two seeds.
- B2a: a single failing arm refutes "every arm". The E3 estimator returns midpoints of the ZIB schedule (… 0.8053, 0.7786, 0.7497 …), so the miss is one grid cell. Formal REFUTED is correct. On the CFG > 1 arms not scored in Round 3, every onset falls in range (Round 4 P5: 0.777–0.810).
- B3b: refuted by the one clean arm. The rule says "every arm", so one failure suffices.

No aggregate hit rate. Agreed.

## Interpretations that need narrowing

1. **B8c offset (label swap).** With the right arm, distillation accounts for about 5% of the old ZIB↔ZIT absolute-g offset, not about 28%. Round 4 P2 decomposes the rest. About 72% is pooled capture; about 23% is guidance plus the uncond branch in the branch-mean.
2. **"Capture mixing almost certainly contributed to the old jitter appearance."** Directionally supported, but capture is confounded with guidance in the R3 arms. Round 4 P3 found:
   - the shared seed component does not create the walk-like VR (seed-centred VR(4) is still 1.79);
   - the 2-seeds-per-prompt design does not create it either (archive SD35M subsampled to 2 runs per prompt: VR(4) 0.51);
   - the pooled CFG 4.5 arm with the same seeds is jitter-like (VR(4) 0.74).
   
   So the persistence difference tracks the arm (CFG 1 cond-only vs CFG 4.5 mixed), not the seed design. Guidance and capture can only be separated by the per-branch Round 4 capture (prereg R1e).
3. **Seed fate.** The reviewer correctly withdrew σ50≈0.991. Round 4 P3 adds the replacement estimate. Seed-centred inside each CV fold, CFG 1 σ50 is **0.742** (bootstrap 95% CI 0.69–0.97), and step-0 R² falls from 0.445 to 0.127. The archive's 0.89–0.94 came from mixed-capture CFG 4.5 runs. Subsampling the archive to 2 runs per prompt only moves it to 0.90 (0.83–0.93). The clean onset may therefore be later than the archive value. This is one arm with a wide CI, so the R4 per-branch run (R1d) decides it.

## The v0.14 B7 correction

Right to withdraw "not robust to passive capture", since 100 vs 40 steps and the prompt were confounded. Round 4 P4 then used existing data to split most of it:
- **Level and trend.** The res-suite functional sits 10.7× below passive (median log ratio −2.37 at matched σ and layer) and falls over steps (ρ −0.99), where passive rises (ρ +1.00).
- **Robustness.** −+−+− survives smooth σ×layer multiplicative perturbations up to 20% RMS in ≥ 82% of draws. Round 2 P4 measured matched-σ step-count effects of ≤ 2%.
- **Prompt.** No single passive prompt gives −+−+−: all 60 start '+', 42/60 are +−+−. All 30 single legacy generations give −+−+−.
- **CFG study.** The legacy-schema CFG study uses the same pyramids prompt and the same 100 steps at 256 px. It gives **+−+−**, and its g *rises* over steps, like passive.

Every −+−+− dataset is the res suite, whose functional is 10× lower and falling. The prompt and the 100-step traversal together do not produce it. Traversal is therefore not excluded by experiment, but it is now an unlikely explanation. The 100-step passive run stays recommended because it is cheap and decisive; the expected outcome is +−+−.

New caveat from this audit: the two "legacy-schema" files are not one instrument. The res suite falls and sits 10× low; the CFG study rises at an intermediate level. Treat their provenance separately until the A/B exists.
