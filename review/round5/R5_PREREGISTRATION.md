# Mode Series Round 5: preregistered predictions

Frozen 2026-10-08 before any calculation in sections Q1–Q8 was run. The SHA-256 of this file and the UTC time are in `R5_PREREG_SHA256.txt`.

**Already computed when this was frozen (disclosed):**
- Intake of the 12 new arms (`out/R5_intake.csv`), preflight output, prompt membership (26/30 frozen prompts in the core arms; every new prompt is one of the 60 Round 3 prompts, run with the same seeds 1001/1002).
- Round 4 section R scored (`out/R_section_scorecard.csv`, `out/R_section_results.json`), including the post hoc seed statistics on the core arms: VR(2/4/8) and σ50 (one fold partition) for CFG 1 cond, CFG 4.5 cond, uncond and synthetic branch-mean; the pyramid 40/100-step topology and level.
- The audit of the reviewer's v0.16 numbers (`out/A_audit_v016_table.csv`), which showed that seed-centred σ50 on the same data is 0.48 or 0.63 depending only on the fold partition, and reproduced the reviewer's topology map (+−+− at 1024 and 768 cond, +− at 512).
- The reviewer's v0.16 result files were read, including the deep-step factorial (CFG 4.5 40↔100 median |log g| 0.0042) and ZIB cond/uncond shape correlations.

Nothing in Q1–Q8 below has been computed by Claude or appears in the reviewer's files, except where a row says so.

**Rules (same as Rounds 3–4):** score every row, including NOT SCORABLE ones with the reason; no edits after results; new analyses are labelled post hoc; report every filter; σ is the clock across models; use the named estimators exactly as in `r5_common.py` (copied from Rounds 2–4).

**Design fact used below:** the new core arms reran 30 of the 60 Round 3 prompts with the same seeds, sampler, steps and resolution. ComfyUI computes cond and uncond in one batched forward, and the passive hooks only read, so the latent trajectory of a new CFG 4.5 generation is the same as the Round 3 generation with that prompt and seed, up to numerical nondeterminism. Round 3 measured it pooled (or branch-mean for ZIB); Round 5 measures it per branch. This turns the regime comparison of Round 4 P1 (different runs) into a same-trajectory comparison.

Pairing: by (prompt text, seed). Filters: generations with any nonfinite g/gap/H are dropped (one ZIB CFG 4 generation, one CFG 4.5 40-step pyramid generation); the 768 Round 3 generation whose 4 steps were not pooled is dropped from Q1d.

---

## Q1. Measurement functional on identical trajectories

| ID | Prediction | Decision rule |
|---|---|---|
| Q1a | Test-retest: the new CFG 1 run reproduces the Round 3 CFG 1 run. | SD35M 1024 CFG 1, 60 paired generations: median over all (gen, step, layer) cells of \|log(g_new/g_R3)\| ≤ 0.01. |
| Q1b | ZIB: Round 3's "branch-mean" capture is the mean of the two branches. | New ZIB CFG 4 synthetic branch-mean (g_cond+g_uncond)/2 vs Round 3 ZIB CFG 4, paired, attn and mlp: median \|log ratio\| ≤ 0.01 in both. Also report ZIB CFG 1 new vs Round 3 (test-retest). |
| Q1c | Pooling raises g on the same trajectory, by about the P1a amount, without moving the gap. | SD35M 1024 CFG 4.5, Round 3 pooled vs new synthetic branch-mean, steps ≥ 2: median log(g_pooled/g_BM) in [+0.04, +0.10]; > 0 in ≥ 90% of cells; \|median log(gap_pooled/gap_BM)\| ≤ 0.03. |
| Q1d | Same direction at 768 and 512. | Seed 1001 arms, steps ≥ 2: median log(g_pooled/g_BM) > +0.03 at both resolutions. |
| Q1e | Concavity bound. For uncentred second-moment matrices, pooling averages them, C = (C_c + C_u)/2, and tr √· is concave, so Σs_pooled ≥ (Σs_c + Σs_u)/2. With the branch Frobenius norms equal (cond and uncond g differ by 0.15%, R1h), this implies √g_pooled ≥ (√g_c + √g_u)/2. | At 1024, all steps: the inequality holds in ≥ 90% of (gen, step, layer) cells. |
| Q1f | The pooled excess tracks how different the branches are. | Population medians per (step, layer) cell at 1024: Spearman ρ between log(g_pooled/g_BM) and (1 − prompt_influence) ≥ +0.4. |

## Q2. Base→Turbo lineage on the explicit cond branch (E10 tangent, population means, all complete generations)

| ID | Prediction | Decision rule |
|---|---|---|
| Q2a | The cond branch of guided ZIB inherits ZIT's depth geometry. | tangent(ZIB CFG 4 cond, ZIT archive) ≥ 0.985, attn and mlp. |
| Q2b | The cond branch is closer to ZIT than the uncond branch. | tangent(cond, ZIT) > tangent(uncond, ZIT), attn and mlp. |
| Q2c | Guidance leaves ZIB's cond geometry intact (P2b on explicit branches). | tangent(ZIB CFG 1 new, ZIB CFG 4 cond) ≥ 0.98, attn and mlp. |

## Q3. Seed-fate estimator stability (fold partitions)

Estimator: E5 as in `r5_common.curve` (ridge α = 100, prompt-grouped 5-fold CV, optional in-fold seed centring), but with the prompt-to-fold assignment drawn at random 200 times (balanced folds). The same 200 partitions are used for every representation of one prompt set. Report median and 5–95% range of σ50 and of R²(step 0).

| ID | Prediction | Decision rule |
|---|---|---|
| Q3a | More prompts stabilise σ50. | Seed-centred σ50, CFG 1 cond-only: 5–95% range width with the Round 3 60 prompts < width with the new 30 prompts. |
| Q3b | Guidance makes passive fate onset earlier on the cond branch. | Seed-centred, 30 prompts, paired partitions: median of σ50(CFG 4.5 cond) − σ50(CFG 1 cond) > 0, and the difference is > 0 in ≥ 60% of partitions. |
| Q3c | The archive's early onset (0.89–0.94) is a property of its design (unique seeds), not of its prompt set. | SD35M archive restricted to the 30 new-core prompts, 2 runs per prompt (the first two by generation id), 200 partitions: median uncentred σ50 ≥ 0.85. |

## Q4. Persistence with unique seeds (pyramid arms: one prompt, 13–16 random seeds)

Residual = deviation of each generation from the across-seed mean (one prompt), standardised as in E5. VR(4) at 40 steps; VR(10) at 100 steps (same σ span).

| ID | Prediction | Decision rule |
|---|---|---|
| Q4a | Walk-like single-branch persistence does not need shared seeds. | CFG 1, 40 steps: VR(4) > 1.2. |
| Q4b | The core ordering reproduces with unique seeds and another prompt. | CFG 4.5, 40 steps: VR(4) cond > VR(4) branch-mean > VR(4) uncond. |
| Q4c | Persistence is a σ-span property, not a step-count property. | VR(10) at 100 steps is within ±25% of VR(4) at 40 steps, for CFG 1 cond and CFG 4.5 cond. |

## Q5. Is the 40→100-step level shift resolved?

Null: within each arm, split the seeds at random into two halves (500 draws), compute median \|log(mean_A/mean_B)\| over (step, layer), and rescale by √((1/n40 + 1/n100)/(1/n_A + 1/n_B)).

| ID | Prediction | Decision rule |
|---|---|---|
| Q5a | The CFG 1 40↔100 shift (0.0326) is within seed-sampling noise. | Observed ≤ 95th percentile of the pooled rescaled null of the two CFG 1 arms. |
| Q5b | Same at CFG 4.5 cond (0.0042). | Observed ≤ 95th percentile of the CFG 4.5 cond null. |

## Q6. What the crossing count measures

| ID | Prediction | Decision rule |
|---|---|---|
| Q6a | The extra pair of crossings in +−+− is a lobe at the noise floor. | In every new SD35M arm × functional with +−+− (1024 CFG 1; 1024 CFG 4.5 cond, uncond, BM; 768 cond, BM): the largest \|β\| of the bins strictly inside the first-to-second crossing interval, or of the two bins straddling it if none is inside, is < 2 bootstrap SE (500 prompt bootstraps, same bins). |
| Q6b | The 512 difference is prompt sampling, not branch functional. | Round 3 pooled 512, restricted to the 29 prompts shared with the new 512 arm: sequence +−. |

## Q7. Guidance contraction with unique seeds (pyramid prompt, unpaired seeds)

| ID | Prediction | Decision rule |
|---|---|---|
| Q7a | R1f replicates with unique seeds on another prompt. | Across-seed SD of layer-mean log g, CFG 4.5 cond / CFG 1, per step, median over the last three quarters of steps < 0.7, at 40 and at 100 steps. |
| Q7b | The contraction is not early. | Same ratio, median over the first quarter of steps ≥ 0.85, at 40 and at 100 steps. |

## Q8. Plateau onset on every new arm (E3, population mean)

| ID | Prediction | Decision rule |
|---|---|---|
| Q8 | Onset σ in [0.75, 0.90] on every arm. | SD35M: CFG 1 1024; CFG 4.5 cond at 1024, 768, 512; CFG 4.5 uncond 1024; pyramid CFG 1 and CFG 4.5 cond at 40 and 100 steps. ZIB: CFG 1 and CFG 4 cond, attn and mlp (report against the schedule's midpoint grid). |

---

Interpretation fixed in advance:
- If Q1a–Q1c hold, Round 4 P1 is confirmed as a pure measurement-functional effect on one trajectory, and Q1e gives it a theorem-level explanation for Mode-II.
- If Q3c fails (the archive's onset is late on the 30 core prompts too), the archive vs new difference is the prompt set or the estimator, and the unique-seed rerun (R1c–e) becomes less urgent. If Q3c holds, the rerun is the only way to settle seed fate.
- If Q6a holds, crossing count should be reported as main crossing g* plus an unresolved low-g lobe, not as 1 vs 3.
