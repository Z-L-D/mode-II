# Mode Series Round 3: preregistered predictions for the new captures

Frozen 2026-10-05, before any of the new data exists. The SHA-256 of this file is in `R3_PREREG_SHA256.txt`.

**Scope.** These predictions cover the captures Zac is running now with the instrument in `capture_code_r2/`:
- the instrument A/B;
- the matched diffusion study, Tier 1 (`R2_MATCHED_STUDY.md`);
- the G4 LLM run with `cos_pafs_llm_measure_v2.py`;
- the PSTF arms (Mode-III, kept out of the Mode-I/II papers).

A second reviewer analyzes the data. Claude audits that analysis against this file.

**Rules for the analyst:**
- Score every prediction below, including ones that cannot be computed. Mark those NOT SCORABLE and give the reason.
- A prediction may not be edited after data arrives. New analyses are welcome, but label them post hoc.
- Report every filter: dropped generations, incomplete runs, and any branch or regime rows excluded.
- Use σ (`sigma` column) as the clock for every diffusion comparison, never step index across models.
- Use the `cond` branch for anything compared with a CFG 1.0 run, unless the prediction says otherwise.

---

## A. Instrument A/B (old Diagnostic EMAG guider at emag_scale 0 vs the patched Passive guider)

Settings: SD3.5-M, pyramids prompt, 512 px, CFG 1.0, 100 steps, 3 seeds, the two guiders on identical seeds.

| ID | Prediction | Decision rule |
|---|---|---|
| A1 | The images are identical between guiders for each seed. At emag_scale 0 the EMA-swapped pass only feeds the guidance term, which is weighted by 0. | Max pixel difference ≤ 2/255 on every seed. |
| A2 | The legacy guider's g differs from the passive g at matched (step, layer). | Median \|log(g_legacy / g_passive)\| ≥ 0.5 over (step, layer). If < 0.1, the instrument is cleared and A3 is moot. |
| A3 | The legacy guider's g falls over steps and the passive g rises, as in the archive comparison (Round 2 G0b note). | Sign of the Spearman ρ(step, layer-mean g) is negative for legacy and positive for passive on ≥ 2 of 3 seeds. |

Interpretation fixed in advance:
- **A2 true:** every SD3.5 rebound card (four crossings, TRI, the complex eigenpair, the CFG card) is a property of the EMA-perturbed pass. Those cards move to a "legacy instrument" appendix.
- **A1 false:** the old build changed the generated images too. Report it, because it would also affect the resolution-suite images.

## B. Matched diffusion study, Tier 1 (234 gens; arms 1–6 in `R2_MATCHED_STUDY.md`)

### B1. What the archive's CFG numbers actually were (arm 1 vs the archive SD35M campaign, shared prompts)

| ID | Prediction | Decision rule |
|---|---|---|
| B1a | From step 2 onward, the archive was branch-mean, so its g equals the average of the new cond and uncond g. | Per (prompt, step, layer), compare the archive's mean over its 4 runs with the new mean over 2 seeds of (g_cond + g_uncond)/2. Median \|Δlog g\| ≤ 0.03. |
| B1b | At step 0, the archive was pooled, and the pooled g exceeds the new per-branch cond g. | Archive step-0 g / new cond step-0 g > 1.10, median over layers and prompts. |
| B1c | Depth shape is preserved between cond and uncond. | Shape r(cond, uncond) ≥ 0.98, median over steps, using `common.depthnorm` on [log g, log gap, H]. |

### B2. σ-plateau universality (all six arms, cond branch)

| ID | Prediction | Decision rule |
|---|---|---|
| B2a | The transient-to-plateau onset lies at σ 0.75–0.90 in every arm. The arms span CFG 1–4.5, 512–1024 px and two models. | Onset estimated with the E3 estimator (`e3_sigma_registration.py`, `onset_sigma`), applied to each arm's population-mean field. |
| B2b | The onset is tighter in σ than in step fraction. | SD of onset σ / SD of onset step fraction < 0.5 across arms. |

### B3. Seed variance at σ≈1 was a pooling artifact (arms 1, 2, 5, 6, cond branch)

| ID | Prediction | Decision rule |
|---|---|---|
| B3a | On per-branch cond rows, the step 0→1 drop in seed variance share is small. | Seed share (E6 estimator, `e6_varcomp.py`, prompt and seed components) drops by ≤ 0.10 from step 0 to step 1 in every arm. The archive pooled campaigns dropped 0.19–0.34. |
| B3b | After step 1, seed share is flat. | max − min of seed share over steps ≥ 2 is ≤ 0.10 in every arm. |

### B4. The step-to-step jitter in seed deviations was a capture artifact (arms 1, 2, 5, 6, cond branch)

| ID | Prediction | Decision rule |
|---|---|---|
| B4 | On single-branch rows, seed deviations are walk-like, as ZIT-attn was (VR 0.97/0.92). This is the artifact hypothesis. | Variance ratio VR(4) ≥ 0.7 (E5b estimator, `e5b_persistence.py`) in ≥ 3 of the 4 arms. If VR(4) ≤ 0.55 in ≥ 3 arms (archive ZIB 0.41, SD35M 0.51), the jitter is real dynamics and the Round-2 caveat becomes a finding. |

### B5. Passive seed fate on the new instrument (arms 1 and 2, cond branch, 2 seeds per prompt)

| ID | Prediction | Decision rule |
|---|---|---|
| B5 | SD35M cond-branch seed-fate R² first reaches 0.5 at σ 0.88–0.96 in arm 1 (archive: 0.936). | E5 estimator (`e5_seed_fate.py`, ridge, prompt-grouped CV). Report the bootstrap CI. |
| B5b | Guidance moves fate onset earlier in σ, so the CFG 4.5 σ50 exceeds the CFG 1.0 σ50. | σ50(arm 1) − σ50(arm 2) > 0, and the bootstrap CI of the difference excludes 0. This is a weak prior; either sign is informative. |

### B6. Guidance contracts across-seed spread on the passive instrument (arms 1 vs 2, paired noise)

| ID | Prediction | Decision rule |
|---|---|---|
| B6 | The legacy-instrument result (spread ratio 0.21–0.31 after the first quarter of steps) replicates in direction. | Ratio of the across-seed SD of layer-mean log g (cond branch), CFG 4.5 / CFG 1.0, computed per prompt and step. Its median over prompts and over steps in the last three quarters is < 0.7. |

### B7. Does the SD3.5 rebound topology exist on passive data at any resolution? (arms 1, 3, 4, cond branch)

| ID | Prediction | Decision rule |
|---|---|---|
| B7a | No arm shows the legacy −+−+− four-crossing topology. Passive SD35M gave +−+− with TRI ≈ 0.01–0.02. | Frozen 24-bin estimator (`beta24.py`, as in `e12_beta_topology.py`): the sign sequence ≠ −+−+− and the crossing count ≤ 3 at 512, 768 and 1024. |
| B7b | Terminal rebound is weak. | TRI < 0.10 at every resolution, over all steps. |

If B7a fails at any resolution, the rebound is a model property on that arm. That would reinstate the SD3.5 rebound card for that resolution regardless of A.

### B8. Lineage tangent without the guidance confound (arms 5, 6 vs the archive ZIT)

| ID | Prediction | Decision rule |
|---|---|---|
| B8a | ZIB(CFG 1.0, cond)↔ZIT tangent cosine at matched σ is ≥ 0.94 (archive ZIB(4.0, pooled)↔ZIT: 0.943 attn). | E10 estimator (`e10_tangent.py`), attn. Also report mlp if the hook includes it. |
| B8b | ZIB CFG 1.0↔CFG 4.0 on the cond branch is a same-model pair. | Tangent ≥ 0.97 and shape r ≥ 0.98. |
| B8c | Distillation, not guidance, explains most of the ZIB↔ZIT absolute-g offset. | \|Δlog g\| ZIB(1.0)↔ZIT ≥ 0.5 × \|Δlog g\| ZIB(4.0 cond)↔ZIT, median over σ-matched (layer, state). This is a weak prior. |

### B9. Conditioning influence (new observable): exploratory, no directional prediction

- Report prompt_influence = cos(mean_cond, mean_uncond) per layer × σ for arms 1, 3, 4 and 5.
- One registered expectation: **B9.** prompt_influence < 0.999 somewhere at σ ≤ 0.9 in every CFG > 1 arm. That means cond and uncond are distinguishable in the passive telemetry.

## C. G4 LLM run (Qwen2.5 1.5B / 3B / 7B Instruct, 120 prompts, greedy, v2 script)

| ID | Prediction | Decision rule |
|---|---|---|
| C0 | The instrument reproduces v1. | `--verify_v1` max relative error ≤ 1e-2 (bf16) on the smoke test. |
| C1 | The windowed statistic is no longer a context clock. | Median within-generation \|Spearman ρ(token_idx, layer-mean g_w64)\| over token_idx ≥ 64 is ≤ 0.5 for every size. The v1 value was 1.000. |
| C2 | The Qwen β ladder does not survive on g_w64. | The 24-bin estimator on g_w64 (token_idx ≥ 64) finds ≤ 1 crossing per size. |
| C3 | The size-pair ordering of depth-derivative alignment survives. | Mean depth-derivative correlation of g_w64 profiles at matched token_idx: 1.5B↔3B ≥ 0.4; 1.5B↔7B and 3B↔7B ≤ 0.3 (v1 matched-n values were +0.63 / −0.12 / −0.06). |
| C4 | Diagnostic, with no prediction: was the archive prompted with the Llama-3.1 template? | Tokenize 10 archive prompts with the Qwen native template and with `LLAMA31_TEMPLATE`. Report which token count matches the archive's `prompt_len`. |

## D. PSTF system prompt (Mode-III appendix, not for the Mode-I/II papers)

Arms: `ptsf`, `ptsf_shuffled`, `default` (plus an optional neutral long system prompt of equal token length via `--system_prompt_file`). Llama-3.1-8B-Instruct, `pstf_24_prompts.txt`, greedy, 256 tokens.

Novelty is `nov_out_frac` per layer, averaged over token_idx ≥ 1. Ratios are paired by prompt, with 95% CIs from a prompt bootstrap. Layer bands as in `review/pstf_llama/novelty_bands_paired.csv`: 0–2, 3–7, 8–14, 15–22, 23–30, 31.

| ID | Prediction | Decision rule |
|---|---|---|
| D1 | The early-layer drop is vocabulary or length, not meaning. | Both ptsf/default and shuffled/default in L0–2 have CIs below 1, and the ptsf/shuffled CI includes 1. |
| D2 | The middle-layer rise is the content candidate. No direction is predicted; the decision rule is fixed. | **Content effect** if ptsf/shuffled in L15–22 has a CI excluding 1. **Length or format effect** if ptsf/shuffled includes 1 and shuffled/default excludes 1. |
| D3 | Windowed geometry. | Report g_w64, gap_w64 and H_w64 at token_idx ≥ 64 for all arms, with paired ratios. No prediction. |
| D4 | Behavior. | The analyst scores the saved texts for the behavior Zac observed, blind to arm. No prediction. |

---

## Amendment 1 (2026-10-05, before any new data; after reading the PSTF documents core → A → C.1 → D → E → F → G)

**Background.** The run's `PTSF_SYSTEM_PROMPT` (9,068 characters, in `cos_pafs_llm_measure.py`) is not the teaching prompt in the PSTF core document. It is a self-directed distillation of five sources:
- core v2 (the three phases and the transitions);
- Addendum A (the three modes);
- C.1 (honest scaffolding);
- E (creative oscillation);
- D (extended introspection);
- plus a closing list of explicit permissions, such as "permitted to explore before answering" and "not required to answer immediately".

Addenda F (dated May 20) and G postdate the run and are not in it.

The prompt instructs specific behavior. A change in the model's output is therefore the expected effect of instruction-following. The Mode-III question is whether the geometry change is specific to this content, or comes from any long behavioral instruction.

**Added arm and predictions:**

| ID | Prediction | Decision rule |
|---|---|---|
| D0 | Manipulation check: PTSF outputs carry the instructed behavior. | Rates per 256 tokens of four text measures: phase vocabulary (explor*, converg*, consolidat*, framing, mode), hedges (might, may, perhaps, possibly, could, uncertain), questions ('?'), and announced strategy shifts ("I notice", "I will now", "let me"). The ptsf rate exceeds both default and ptsf_shuffled on phase vocabulary and on hedges (paired by prompt, CI excludes 0). Lexicons are frozen as listed. |
| D5 | Content specificity. Optional arm: `instruct_matched`, a non-PSTF behavioral system prompt of equal token length (±5%) asking for a different style. Example: a detailed conventional tutoring or assistant style guide with no phase, mode or permission language. It is passed via `--system_prompt_file`. | **PSTF-specific geometry** if ptsf/instruct_matched in L15–22 novelty has a CI excluding 1 *and* D2 shows a content effect. **Generic long-instruction effect** if ptsf/instruct_matched includes 1 while ptsf/shuffled excludes 1. |
| D6 | Response length: PTSF (with "not required to answer immediately") reaches the 256-token cap at least as often as default. | Cap rate (ptsf) ≥ cap rate (default). The v1 run hit the cap in 96% (ptsf) and ~100% (default), so this mainly guards against a ceiling hiding a length effect. If both are ≥ 95%, rerun D0–D2 with `--max_tokens 512` on 8 prompts. |
