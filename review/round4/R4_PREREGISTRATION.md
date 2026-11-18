# Mode Series Round 4: preregistered predictions

Frozen 2026-10-06 before any of the calculations below were run. The SHA-256 of this file and the UTC time are in `R4_PREREG_SHA256.txt`.

**What was already computed when this was frozen (audit only, reviewer-reported quantities):** the intake table, prompt membership, B8a (ZIB CFG1↔ZIT tangent), the B8c post-hoc ratio against the old pooled ZIB archive, B2a onsets on the CFG 1 arms, E6 seed share on SD35M CFG 1/4.5 at 1024, VR(2/4/8) on SD35M CFG 1 (uncentered), uncentered σ50 and step-0 seed-identity accuracy on SD35M CFG 1/4.5, the 24-bin topology and TRI of the four SD35M arms, the B6 spread ratios, and the capture regime per step in every campaign (seq_len metadata). Nothing in sections P or R has been computed.

**Design facts used below (metadata, read before freezing):**
- SD35M archive (CFG 4.5, 40 steps): step 0 pooled (seq 8192), step 1 mixed, steps ≥ 2 branch-mean (seq 4096: separate cond and uncond forwards, g averaged).
- R3 SD35M CFG 4.5 at 1024 and 512: pooled at every step. R3 768: pooled except 4 steps of one generation.
- ZIB archive (CFG 4, 25 steps): pooled at every step (seq 8320).
- R3 ZIB CFG 4: branch-mean at every step (separate forwards: cond 4160, uncond 4128 rows).
- R3 CFG 1 arms (SD35M, ZIB) and the ZIT archive: cond only.
- The 60 R3 prompts are all in the SD35M and ZIT archives; 57 are also in the ZIB archive.
- R3 reuses seeds 1001 and 1002 for every prompt.

**Rules (same as Round 3):** score every row, including NOT SCORABLE ones with the reason; no edits after results; new analyses are labelled post hoc; report every filter; σ is the clock across models; use the estimators named in each row exactly as in the Round 2/3 scripts unless the row defines a new one.

---

## P. Calculations on data already in hand

### P1. What pooling does to the passive observables (same model, prompts and step count; capture regime differs)

Per (prompt, step, layer): mean over that prompt's generations of log g (and log gap). Matched prompts only.

| ID | Prediction | Decision rule |
|---|---|---|
| P1a | SD35M: pooled g exceeds branch-mean g. | R3 CFG 4.5 1024 (pooled) vs archive SD35M (branch-mean), steps ≥ 2, 60 prompts: median log(g_pooled / g_BM) > +0.03. |
| P1b | ZIB: the same direction. | Archive ZIB (pooled) vs R3 ZIB CFG 4 (branch-mean), all steps, 57 prompts, attn and mlp: median log(g_pooled / g_BM) > +0.03 in both. |
| P1c | Pooling lowers the spectral gap. | median log(gap_pooled / gap_BM) < 0 in P1a and in both submodules of P1b. |
| P1d | Negative control: when both captures are pooled, they agree. | SD35M step 0 (pooled in both campaigns): \|median log(g_R3 / g_archive)\| ≤ 0.03. |

### P2. Was the old ZIB↔ZIT lineage gap (0.943) pooling or guidance?

E10 estimator (`e10_tangent.py`); offsets as in the B8c row of Round 3 (median over σ-matched (layer, ZIT step) of \|Δ mean log g\|, 57 shared prompts).

| ID | Prediction | Decision rule |
|---|---|---|
| P2a | Pooling, not the guided trajectory, produced most of the old tangent gap. | Tangent ZIB CFG 4 branch-mean (R3) ↔ ZIT ≥ 0.98 in attn. ≥ 0.98 = pooling; ≤ 0.96 = guidance; in between = both. Report mlp the same way (old pooled value 0.967). |
| P2b | Guidance leaves ZIB's own depth geometry nearly unchanged. | ZIB CFG 1 ↔ ZIB CFG 4 branch-mean (both R3, same 60 prompts and seed): tangent ≥ 0.98 and shape r ≥ 0.98, attn and mlp. |
| P2c | Most of the old absolute-g offset was pooling. | offset(ZIB4-BM ↔ ZIT) / offset(ZIB4-pooled archive ↔ ZIT) ≤ 0.5, attn and mlp. |

### P3. Seed fate and persistence without the global-seed shortcut (R3 SD35M 1024)

Seed-centering: at each step, subtract from each generation's prompt-centred residual the mean residual of its seed (1001 or 1002) over prompts. For σ50 this is done inside each cross-validation fold from the training prompts only, so held-out prompts are never used to estimate the seed component. Otherwise the E5 estimator (`e5_seed_fate.py`: ridge α=100, 5-fold prompt-grouped CV) and the E5b variance ratio on steps ≥ 1.

| ID | Prediction | Decision rule |
|---|---|---|
| P3a | The σ50 ≈ 0.991 of the clean CFG 1 arm is the shared seed component. Without it, fate onset falls in the archive range. | Seed-centred σ50 (CFG 1) lies in [0.85, 0.96]. |
| P3b | The step-0 predictability goes away with the seed component. | Seed-centred R²(step 0 → final) < 0.20 (uncentred: 0.445). |
| P3c | The walk-like VR of the clean arm is the shared seed drift, not single-branch capture. | Seed-centred VR(4), CFG 1 < 1.0 (uncentred: 1.84). |
| P3d | Pooled capture with the same shared seeds is also walk-like. This separates the two explanations for the archive's jitter (VR(4) 0.41–0.51). | Uncentred VR(4), CFG 4.5 pooled > 1.0. If P3c and P3d both hold, the seed design, not capture mixing, explains the R3-vs-archive VR difference. If CFG 4.5 VR(4) ≤ 0.6, capture mixing is favoured, as the reviewer proposed. |

### P4. The four-crossing confound (B7), as far as existing data can split it

| ID | Prediction | Decision rule |
|---|---|---|
| P4a | Instrument level and trend: the legacy functional is a different quantity. | SD35M 1024 CFG 1, legacy res-suite vs R3 passive CFG 1, nearest σ, same layer: median log(g_legacy / g_passive) ≤ −1.0, and the Spearman ρ(step, layer-mean g) of the population mean is negative for legacy and positive for passive. |
| P4b | Step-count-sized changes cannot remove the fourth crossing. Round 2 P4 bounded step-count effects at matched σ at ≤ 2% for g (ZIT 8 vs 100, both instruments). | Multiply the legacy 1024 g field by (1 + ε(σ, layer)), ε a smooth random field (Legendre degree ≤ 2 in σ and in layer, Gaussian coefficients, scaled to a target RMS), 200 draws per RMS in {1, 2, 3, 5, 7.5, 10, 15, 20}%. The smallest RMS at which ≥ 50% of draws lose the −+−+− sequence is ≥ 5%. |
| P4c | The legacy and passive curves differ at the low-g end for every prompt and seed, not as a population average. | First sign of the 24-bin β curve: '+' for ≥ 90% of single prompts in R3 SD35M CFG 1 (2 generations each), '−' for ≥ 90% of single legacy generations at 1024 (thinned to the 40 nearest σ of the passive schedule, so point counts are comparable). Report the full-sequence frequencies. |
| P4d | Within the legacy instrument the topology is resolution- and size-robust. | The legacy CFG study (256 px, pyramids prompt, CFG 1.0, 100 steps): −+−+− with 4 crossings. CFG 5.5: report only. |

Interpretation fixed in advance: if P4a, P4b and P4c all hold, the capture functional is the leading explanation and a 100-step traversal effect would need to be at least 2.5× larger in SD35M than the measured step-count effect in ZIT. That still does not replace the R4 100-step passive capture (R2 below); it sets its expected outcome.

### P5. Plateau onset on the arms not scored in Round 3 (post hoc on the B2a estimator)

| ID | Prediction | Decision rule |
|---|---|---|
| P5 | Onset σ lies in [0.75, 0.90] on every remaining arm: SD35M CFG 4.5 at 512, 768, 1024 (pooled) and ZIB CFG 4 attn and mlp (branch-mean). | E3 `onset_sigma` on each arm's population mean. Report the ZIB values against the schedule's discrete σ grid. |

---

## R. Round 4 captures (predictions for data not yet taken)

Instrument: `capture_code_r2/nodes.py` after the 2026-10-06 fix (per-branch rows always on). Every CSV must pass `preflight_check.py` before analysis.

### R1. Per-branch core: SD35M 1024, 40 steps, CFG 1.0 and 4.5, the frozen 30 prompts, one unique seed pair per prompt shared by both CFG arms

| ID | Prediction | Decision rule |
|---|---|---|
| R1a | Preflight passes. | hook_mode passive-branch; branches {cond, uncond} at CFG 4.5 and {cond} at CFG 1; finite prompt_influence at CFG 4.5; 24 layers every step. |
| R1b | The cond branch at CFG 4.5 equals the CFG 1 run at step 0 (same input, same conditioning). | Median \|log(g_cond,4.5 / g_1.0)\| at step 0 ≤ 1e-3. |
| R1c | Seed identity is no longer decodable. | Step-0 seed-identity accuracy from held-out prompts ≤ 0.65 (unique seeds per prompt). |
| R1d | Seed fate with unique seeds sits in the archive range. | E5 σ50, cond branch, CFG 1 and CFG 4.5: both in [0.85, 0.96]. |
| R1e | With unique seeds, single-branch seed deviations are jitter-plus-walk like the archive. | VR(4), cond branch, CFG 1: ≤ 1.0. If > 1.2, single-branch capture itself is walk-like and P3c's explanation fails. |
| R1f | Guidance contracts across-seed spread (Round 3 B6, now on the cond branch). | Median over prompts and the last three quarters of steps of SD(cond, 4.5)/SD(1.0) < 0.7. |
| R1g | Cond and uncond are distinguishable. | prompt_influence < 0.999 somewhere at σ ≤ 0.9 (Round 3 B9). |
| R1h | Pooled-vs-branch effect from P1 is reproduced inside one run. | Pooled g recomputed from the two branches is not possible from scalars, so: median log(g_BM / g_cond) at steps ≥ 2 is reported, no prediction. |

### R2. Decisive B7 capture: SD35M 1024, CFG 1.0, 100 steps, current passive instrument, pyramids prompt, ≥ 30 seeds

| ID | Prediction | Decision rule |
|---|---|---|
| R2a | The passive 100-step run shows the passive topology, not the legacy one. | 24-bin β: first sign '+', sequence ≠ −+−+−, ≤ 3 crossings. If −+−+− with 4 crossings appears, traversal (or prompt) explains B7 and the four-crossing card is reinstated for passive data. |
| R2b | Same level as the 40-step passive arm at matched σ. | Median \|log(g_100 / g_40)\| ≤ 0.05 against R3 CFG 1 at nearest σ (prompt differs; report the prompt-level spread of R3 as the yardstick). |

### R3. Instrument A/B (Round 3 A1–A3, unchanged)

Run only if the four-crossing card stays in Mode-I/II after R2. The Round 3 A1–A3 rules apply unchanged.

### R4. G4 LLM and PSTF

Round 3 C0–C4 and D0–D6 apply unchanged to the first G4 and PSTF data.
