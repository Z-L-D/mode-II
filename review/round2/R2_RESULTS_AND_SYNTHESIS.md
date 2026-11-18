# Mode Series Round 2: results and synthesis

2026-10-05.

**Inputs:**
- Preregistration: `R2_PREREGISTRATION.md`. Frozen SHA 2343fd02…df67; amendments B1–B6, SHA 74252005…ed0.
- Gates: `R2_GATES.md`.
- Literature: `R2_LITERATURE.md`.
- Scripts: `scripts/`. Outputs: `out/`. Figures: `fig/`.

Each script runs from the scratchpad, which holds the `pq/` parquet files and the `F_*.pkl` field caches built by `review/scripts/build_fields.py`, plus `common.py` and `beta24.py` (copied into `scripts/`).

---

## 1. What needs Zac

1. **Which guider and emag_scale produced the legacy-schema files?** This covers the SD3.5 resolution suite, the CFG 1.0/5.5 study and the old ZIT 8/16/32 CSVs. Their files lack submodule and φ columns and are tagged hook_mode 'attn', which is the EMAG guider's vocabulary. In the current nodes.py that guider computes g, gap and H only inside the EMA-perturbed [uncond, cond] forward pass (`_make_emag_hook`, called from `predict_noise` at around line 2293).
   - At the same model and resolution, the passive SD35M campaign's g is about 10× higher (median 0.064 vs 0.005) and rises over steps, where the legacy g falls.
   - If the legacy runs were EMAG-perturbed, every SD3.5 rebound card is a property of the perturbed pass: the four-crossing β(g), TRI, the complex eigenpair and the CFG card. **Gate G0b.**
2. **Capture batch 1 (G0 + G0b)** is now the single most valuable action. Specs are in `R2_GATES.md`.

---

## 2. Scorecard (preregistered)

| P | Hypothesis | Result | Score |
|---|---|---|---|
| P1 | H1a: R_max not above the monotone-warp null in ≥5/7 channels | Not above in 4/7 at warp α=1 (ZIT×2 p≤0.012, ZIB-attn p=0.049 exceed); not above in 7/7 at α=10 | PARTIAL |
| | H1b: K(t) and C(t) have no interior max in ≥5/7 | No interior max in 7/7 for both | CONFIRMED |
| | H1c: integrated K above its shift null in 7/7 | 7/7 (0.27–0.52 vs 0.14–0.39) | CONFIRMED |
| | Falsifier of the Round-1 critique | Not met | critique stands |
| P2 | H2a: SD35L regime shift \|Δlog g\| ≥ 0.05 | 0.013 (gap −0.073 and H +0.044 move in opposite directions, as predicted). SD35M step-1 mixed cells: +0.25 | REFUTED for SD35L |
| | H2b: decoding ≥3× chance within regime-homogeneous subsets | Pooled 4.0–4.7×; branch-mean 2.3–3.3× (SD35L, 4 domains), 3.0–3.4× (SD35M) | PARTIAL |
| | H2c: shape r between regimes ≥ 0.98 | Median 0.997 (SD35L) and 0.983 (SD35M); noise floor 0.998–0.999 | CONFIRMED |
| P3 | H3a: equal-σ loses ≤0.01 r vs the optimized monotone warp in ≥90% of prompts | 99% (attn) and 100% (mlp); median loss 0.003/0.002 | CONFIRMED |
| | H3b: plateau onset tighter in log-SNR than in step fraction | SD ratio 0.25. Onset σ is 0.78–0.87 in all 8 channels | CONFIRMED |
| P4 | H4a: ZIT 8 vs 100 steps at matched σ: g ≤5%, gap and H ≤3% | g 1.9%/1.4%, gap 1.4%/0.35%, H 0.17%/0.48% (attn/mlp). Legacy 8/16/32: 1.7%/1.5%/0.25% | CONFIRMED |
| | H4b: Mochi 30 vs 100 per-σ speed within 25% below σ 0.9 | Ratios 0.81–0.92 (path) and 0.85–1.04 (chord) | CONFIRMED |
| P5 | H5a: seed-fate beats the diffusive increment-shuffle surrogate at τ 0.5 in ≥5/7 | 0/7. Observed is **below** the surrogate in all 7 | REFUTED |
| | H5b: no single-step jump >0.25 after the first transition | Max 0.22 | CONFIRMED |
| | H5c: CI width of τ(R²=0.5) ≤0.25 in ≥5/7 | 5/7 (ZIB ×2 fail at 0.41/0.60) | CONFIRMED |
| P6 | H6a: prompt share ≥0.5 at every step (σ=1 exempt for ZIB and SD3.5) | All pass except SD35M step 1 (σ 0.992, the mixed-regime step) at 0.495 | PARTIAL |
| | H6b: seed share non-monotone in ≥3 channels | 0/7 at the 0.05 amplitude criterion. Shallow late rises of 0.01–0.05 | REFUTED |
| P7 | H7a: \|ρ(n, PR/n)\| ≤ 0.9 | ρ = −1.000 in all 7 LLMs | REFUTED |
| | H7b: g/g₀ still decreasing | ρ = −1.000 | CONFIRMED (trivially) |
| | H7c: Qwen depth-derivative \|r\| ≤ 0.4 at matched n | 1.5B↔7B −0.12, 3B↔7B −0.06 (1.5B↔3B +0.63) | CONFIRMED |
| | H7d: β-ladder loses a crossing or shifts >20% on the normalized observable | 7B: 3 → 2 crossings. 1.5B and 3B have only 1 crossing even on raw g | CONFIRMED |
| P8 | H8a: weight-only median \|Δlog g\| ≤ 0.005 | 0.0013 (prefill, identical text, 120 prompts) | CONFIRMED |
| | H8b: argmax layer in depth fraction 0.3–0.9 | Argmax is the final layer (31), with 4× the effect of any other layer. Layers 0–5 are exactly 0 | REFUTED |
| P9 | H9a: cross-type φ beats φ_skip for held-out β_g (Llama-Base) | Neither adds anything (ΔR² −0.006 to +0.0002). The same estimator on Z-Image gives ≤0.0014 | REFUTED |
| P10 | H10a/b: ZIT↔ZIB tangent ≥0.8; unrelated ≤0.3; margin ≥0.5 vs shape ≤0.2 | 0.943/0.968; unrelated max 0.241 (all inside the layer-permutation null); margin 0.70/0.73 vs 0.15 | CONFIRMED |
| P11 | H11a: ZIT loop in ≥50% of layers, others <50% | ZIT 7%; ZIB 77%, Mochi 29%, Mochi-100 81%, SD35M 79%, SD35L 58%. At least one layer qualifies in 99–100% of generations everywhere | REFUTED |
| P12 | H12a: 120-prompt SD35M has ≥3 crossings, shifted >20% | 3 crossings (bootstrap mode 3), shifted. Sign sequence is +−+− rather than −+−+−, and TRI ≈ 0.02 | CONFIRMED (letter); see §3.6 |
| P13 | H13a: TRI 0.866→0.874 ±0.05 | 0.863 → 0.883 | CONFIRMED |
| | H13b: eigenpair modulus ±0.05 of 0.878/0.848 | 0.878 [0.862, 0.891] → 0.849 [0.836, 0.861]; angle 0.112 → 0.179 | CONFIRMED |
| | H13c: spread ratio < 0.5 | Median 0.26. First quarter of steps 0.73 (at the 1/√2 averaging bound), later 0.21–0.31 | CONFIRMED |

**Tally (28 hypotheses):** 18 CONFIRMED, 3 PARTIAL, 7 REFUTED.
- The refutations cluster in four places: LLM context (H7a), passive-commitment persistence (H5a), the ZIT-specific loop (H11a) and φ sector information (H9a).
- Each of those refutations removes a claim. None adds one.

---

## 3. Synthesis

### 3.1 There is no mid-run synchronization event (Mode-I §sync, Mode-II C1)

- The z-scored R peak is a by-product of z-scoring each layer over time. Against a null that keeps each layer's own monotone trajectory and randomizes its timing, R_max is at the null in 4/7 channels at α=1 and in 7/7 at the gentler α=10.
- The three channels that edge over the α=1 null are ZIT×2 (8 steps) and ZIB-attn (p=0.049). They show that layers share a σ-clock. They do not show a coordination event.
- Two centering-free measures of cross-layer co-movement, K(t) and C(t), are real: integrated K is above its shift null in 7/7. Both are **highest at the endpoints and lowest mid-run, with no interior maximum in any channel**. K has a U shape: high in the σ≈1 transient, low on the plateau, and rising again at the end.
- **Mode-I wording:** "Layers co-move strongly in the high-noise transient and again at the end of sampling. There is no interior synchronization peak once temporal centering is removed."
- D16–D18 are withdrawn. C1 should not be used as a landmark, including for G3.

### 3.2 σ is the clock, and the plateau onset is universal in σ (Mode-I backbone, Mode-II)

- For every one of the 105 shared prompts, the ZIT↔ZIB registration loses at most 0.011 in r when the per-prompt optimized monotone warp is replaced by the equal-σ map. The learned "25→8 clock" is σ.
- The speed plateau begins at σ = 0.78–0.87 in all 8 channels, which span four architectures, CFG 1–4.5, all three capture regimes and 8–100 steps. In step fraction the same onsets spread over 0.32–0.64.
- **Candidate Mode-II invariant:** a transient-to-plateau boundary near σ≈0.8 (log-SNR ≈ −2.5 to −3.9) for rectified-flow DiTs with these schedules. Two caveats:
  - The four model families share the flow-matching σ convention.
  - Schedule shift differs between them, and the onset survives it.
- Step-count invariance at matched σ is clean on the passive instrument (H4a, ≤2% for g). There is new structure at low σ: in the 100-step runs, both ZIT g and Mochi path speed change again below σ≈0.2 (Mochi chord speed 7.5 vs 4.6 on the plateau). The ZIT 8-step and Mochi 30-step schedules stop at σ 0.30 and 0.17, so they never record it.
- **Mode-I wording:** "a high-noise transient, a σ-invariant plateau from σ≈0.8, and a late re-acceleration below σ≈0.2 that short schedules do not sample". This replaces "late slow flow".

### 3.3 Passive commitment: early in σ and not persistent (Mode-I §fate, G3)

- In the CFG>1 base models, seed-fate R² crosses 0.5 at σ ≈ 0.89–0.94, i.e. log-SNR −4.1 to −5.4. Tight CIs: SD35M 0.936, SD35L 0.910, Mochi 0.886, ZIB-attn 0.915.
- That is at or before the plateau onset, and it is numerically near the speciation log-SNR (−5 to −6) of Biroli et al. 2024. The parameterizations differ, so this is descriptive only (B5).
- ZIT commits late in σ (0.66 / 0.50). Its seed share is tiny (5–8%), so the target is mostly noise. This does not contradict the finding of Gandikota & Bau that distilled models commit at step 1. It means that in ZIT the seed carries almost no telemetry-visible variance.
- **H5a refuted, in the opposite direction.** Seed deviations are *less* predictable than a random walk with the same per-step increments. The variance-ratio test explains why:
  - VR(k) falls with lag (ZIB 1 → 0.57 → 0.41 → 0.34 at k=1,2,4,8; SD35M → 0.42), between pure jitter (1/k) and a walk (1). Step-to-step increments carry a large non-persistent component on top of a persistent but bounded deviation.
  - ZIT-attn, the only cond-only channel, is nearly walk-like (VR 0.97/0.92). The jitter is largest in the pooled and branch-mean channels, which points to the CFG capture functional as at least part of it.
- **Fair null for the next round:** a walk with jitter (VR-matched), not a pure increment shuffle.
- **G3 consequence:** clone points at σ ∈ {0.97, 0.93, 0.85, 0.79}. Score fate on images.

### 3.4 Variance components and the σ=1 seed peak are partly a capture artifact (Mode-I)

- Prompt share is ≥0.5 nearly everywhere (ZIT ≥0.83, the others ≥0.5 after step 1). Seed share drops sharply from step 0 to step 1 only in the campaigns whose step 0 is pooled: ZIB 0.55→0.21, SD35M 0.45→0.26, SD35L 0.45→0.21. It does not drop in Mochi (branch-mean throughout, 0.28→0.30) or ZIT (cond-only, 0.08→0.07).
- A pooled matrix includes the prompt-independent uncond branch, which dilutes prompt information. The "σ=1 seed dominance" in ZIB and SD3.5 should be treated as a regime artifact until G0.
- After step 1, seed share is flat to slightly rising (U-shaped with amplitude <0.05). Neither "seed collapse" nor "non-monotone seed share" is supported. Report "flat after the transient".

### 3.5 Capture regime: a big effect at σ≈1, small later (Mode-I methods)

- At SD35M step 1 (σ 0.99), pooled g is +28% (Δlog g +0.25) over branch-mean.
- In SD35L the same comparison, made within domain over steps 1–27, gives Δlog g +0.013, gap −7%, H +0.044.
- Depth-profile shape is preserved (r ≥ 0.98), and domain decoding survives within regime-homogeneous subsets at ≥3× chance in most cells.
- **Practical reading:** absolute g at high σ is regime-contaminated; shapes and the late trajectory mostly are not. This narrows G0's urgency for shape-based claims and keeps it urgent for every σ≈1 and CFG-comparison claim.

### 3.6 SD3.5 rebound: reproducible on the legacy files, not established as a model property (Mode-I §rebound, Mode-II)

- The frozen 24-bin estimator reproduces the ledger's resolution-suite crossings to within about 10% and its CFG-study numbers exactly (TRI 0.863/0.883, modulus 0.878/0.849, angle 0.112/0.179). The ledger's arithmetic is right.
- **Three limits:**
  1. The CFG-study complex mode exists only in steps 0–9. Over all 100 steps, CFG 1.0 has no complex pair and CFG 5.5 has one in 35% of bootstraps. In steps 0–9 the sign sequence is a single +− crossing, not the four-crossing topology.
  2. On the passive 120-prompt SD35M campaign, the topology is +−+− with TRI ≈ 0.01–0.02 over all steps (0.17 in steps 2–10). The low-g negative branch never appears, because passive g never gets that low.
  3. Points 1–2 are confounded by instrument (G0b). The legacy files are very likely EMAG-perturbed-pass measurements.
- **Recommendation:** move the rebound, four-crossing and CFG-card results out of the Mode-I core into a "legacy instrument" appendix until G0b is captured. Keep P13's reproduction as evidence that the ledger computed what it says.

### 3.7 The CFG spread reduction is real guidance, beyond averaging (Mode-I CFG card, legacy instrument)

- In the first quarter of steps the CFG 5.5 / 1.0 spread ratio is 0.73, at the bound predicted if branch averaging alone were responsible (≥1/√2).
- After that it falls to 0.21–0.31. Guidance itself contracts the across-seed telemetry spread from roughly step 10/100 onward.
- Same caveat as §3.6: both arms were captured on the legacy instrument.

### 3.8 Lineage signature: tangent alignment (Mode-I methods, Mode-II)

- Depth-derivative alignment at matched σ separates lineage cleanly:
  - ZIT↔ZIB: 0.94 (attn) and 0.97 (mlp).
  - Mochi CFG4.5↔1.0: 0.99.
  - All 8 unrelated pairs: |cos| ≤ 0.24, every one inside the layer-permutation null.
- Shape r cannot do this: unrelated pairs reach 0.83 (Z-Image↔Mochi), which matches the generic-depth-profile baseline in the literature (Murphy et al. 2024; Cheng et al. 2025).
- SD35M↔SD35L, a same-family size pair, has tangent −0.25. That fits the ledger's "capacity breaks the microscopic law" for SD3.5. It also shows that "same family" is not "same lineage".
- **Mode-I should report tangent cosine as the lineage statistic and shape r only with its unrelated-pair baseline.**

### 3.9 LLM telemetry is a nested running statistic (Mode-I §LLM, G4)

- Capture recomputes the full context each token with no KV cache, and attention is causal. So the matrix at token n+1 is the matrix at n plus one row.
- Every logged scalar is therefore a running statistic over a nested sample. PR/n, g/g₀ and g are each exactly monotone in n inside every generation (|ρ| = 1.000, all 7 LLMs). PR grows sub-linearly (PR/n ≈ 0.33–0.37, falling), as the literature predicts when sink or massive directions cap PR.
- The per-step LLM "trajectory" carries the new token's information only as a small increment to a cumulative statistic. No existing-data observable fixes this.
- At matched n, the Qwen size-pair depth-derivative correlations stay low: 1.5B↔7B −0.12, 3B↔7B −0.06, against 1.5B↔3B +0.63. The 7B β-ladder loses a crossing under normalization, and the 1.5B and 3B ladders only ever had one.
- **Mode-I:** retire LLM per-token dynamics claims until G4's per-token or fixed-window capture. Cross-model comparisons at matched n, such as the Qwen depth-derivative result, can stay with that caveat.

### 3.10 Abliteration: telemetry localizes the edit (Mode-I §G5 pre-result)

- On identical prompt text (prefill, 120 prompts), Instruct vs ablit differ by a median |Δlog g| of 0.0013. The effect is tiny, as predicted.
- Layers 0–5 are **bit-identical**, the effect turns on at layer 6, and it is largest by 4× at the final layer. This particular ablit checkpoint evidently left the embeddings and the first six blocks untouched. That is a different edit from Arditi et al.'s all-writer projection.
- Per-prompt effect size varies 3× (0.0013–0.0038). The largest are ethics/AGI and dilemma prompts, which is suggestive of the refusal direction but not tested.
- **For G5:** use teacher-forced harmful vs benign-matched prompts and test whether the per-layer Δ profile (onset at 6, final-layer peak) scales with refusal engagement.

### 3.11 Retired or downgraded by this round

- **ZIT step-flow loop (prior "87/87").** The criterion fires in 99–100% of generations in every model. At the population level ZIT is the *least* loopy channel. Retire it.
- **φ sector information (Z-Image "cross-sector dominates", Llama-Base replication).** Once depth and σ (or log n) are in the baseline, neither φ nor φ_skip adds held-out information about β_g in any model (ΔR² ≤ 0.0014). Downgrade to "φ is redundant with [g, gap, H, depth, σ]".
- **"No universal seed collapse".** Replace with "seed share is flat after the transient; the σ≈1 elevation in ZIB and SD3.5 coincides with pooled capture".

---

## 4. Updated claim list

**Mode-I (empirical release), safe now on passive data:**
1. σ-registration: Base↔Turbo is equal-σ, per prompt.
2. Plateau onset at σ≈0.8 in all channels; step-count invariance at matched σ; late re-acceleration below σ≈0.2 in long schedules.
3. Prompt dominance of within-step variance; prompt retrieval and decoding (with regime caveats).
4. Passive seed-fate onset at σ≈0.89–0.94 in base models, with the walk-plus-jitter caveat.
5. Tangent lineage signature with its permutation null.
6. Endpoint-peaked cross-layer co-movement (K, C), with no mid-run event.
7. Abliteration weight-effect localization (prefill).
8. Methods: the three capture regimes and their measured size (§3.5); the nested LLM statistic (§3.9).

**Mode-I, pending G0/G0b:**
- Any CFG>1 vs CFG=1 absolute comparison.
- σ=1 seed and prompt shares in pooled campaigns.
- SD3.5 rebound, four-crossing topology, complex mode and CFG card.

**Mode-II (theory), worth formalizing:**
- Why the transient-to-plateau boundary sits at a fixed σ across architectures.
- How speciation timing (Biroli; Sclocchi) maps onto passive seed-fate onset.
- Why depth derivatives, not levels, carry lineage.
- How a nested-sample PR behaves as a function of n. The null is g₀ = n/(n+D), with sub-linear corrections from dominant directions.

---

## 5. Files

| File | Content |
|---|---|
| `R2_PREREGISTRATION.md`, `R2_PREREG_SHA256.txt` | Frozen predictions, amendments B1–B6, hashes |
| `R2_GATES.md` | Gate re-determination, timing, post-calculation updates (G4 reverted, G0b added, D16–18 withdrawn) |
| `R2_LITERATURE.md` | Literature pass: 10 topics, 49 references with verification marks |
| `out/E*.csv`, `out/E1_curves.pkl`, `out/E5_curves.pkl` | All numeric outputs |
| `fig/E1_sync.png`, `fig/E3_plateau_onset.png`, `fig/E5_seed_fate.png`, `fig/E13_spread_ratio.png` | Figures |
| `scripts/` | One script per experiment, plus `common.py` and `beta24.py` |
