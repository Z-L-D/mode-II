# Round 7 audit (Claude, 2026-10-08): the reviewer's Final Release Pass

Input: `ModeSeries_Round7_Final_Release_Pass_2026-10-08.zip` (SHA-256 7e4be3e5…5feec), extracted to `reviewer_r7/zip/`, plus ledger v0.20 and the two final outlines.

## Verdict

The Round 7 numbers are correct. Every value in `R7_OBSERVABLE_HIERARCHY.csv`, `R7_CONTENT_VS_DYNAMICS.csv`, `R7_ESTIMATOR_FINAL.csv`, `R7_LAYER_TOPOLOGY_NULL_QUALIFICATION.csv` and the sync line was reproduced exactly by running the shipped `r7_final_release_audit.py` against the Round 6 outputs. The script was copied to scratch and run without `python -I`, because under `-I` it cannot import `dateutil` from the user site.

Four things need correcting or relabelling before the papers use Round 7.

## R7-A1. The eight "predictions" were consistency checks

- The preregistration (SHA 9012d20f…a4e8a864b, frozen 09:02:04Z) lists only Round 6 artifacts as inputs.
- Every number P1–P6 checks was already computed and published in Round 6 (S1, S2, S3, S5, A2, A4) before the freeze.
- P7 and P8 are editorial judgements, not measurements.
- "8/8 CONFIRMED" therefore shows that the release register is internally consistent with Round 6. It is not new out-of-sample evidence.
- **In the papers:** cite the Round 6 preregistered scorecard (17 confirmed, 5 refuted) as the preregistered evidence. Mention Round 7 as the release audit only.

One number is new in Round 7: the content-versus-tangent Spearman ρ = 0.297 (p = 0.405, n = 10 pairs). It is computed from existing Round 6 tables.
- n = 10 pairs gives low power. The result is "no detectable rank coupling", not "decoupled".
- Mode-II §8 should say exactly that.

## R7-A2. Ledger v0.20 is v0.19 plus one section

- v0.20 adds the "round7-final" section, but no nav link points to it.
- The subtitle still reads v0.19.
- All other content is byte-identical to my v0.19.
- No finding was lost. v0.21 will add the nav link, this audit, and the outline corrections below.

## R7-A3. Gate relabels

`R7_FINAL_GATE_STATUS.csv` changes three labels from v0.19:

| Gate | v0.19 | R7 | Comment |
|---|---|---|---|
| Prompt protocol lock | FAILED again (R4/R5) | PENDING | The lock failed twice on real captures. "PENDING, not a release blocker" is fine for the release; the history should stay visible |
| Seed design | FAILED again | PENDING | As above |
| G1 paired-noise | finite perturbation | "pending for response/Jacobian" | Accepted. It narrows G1 to what it can license |

None of these changes the release scope. The gate table in the papers uses the v0.19 wording with the R7 release rule.

## R7-A4. Literature positioning

`R7_LITERATURE_POSITIONING.md` lists five works. All five were checked against a primary page and exist (`papers/LIT_REVIEW.md`, `papers/CITATION_KEYS.md`). Two corrections:
- "Sun et al. (ICML 2026)" is Shangwen Sun et al., a different first author from the 2024 massive-activations paper (Mingjie Sun). Its subject is language models, not diffusion transformers; the arXiv title is "The Spike, the Sparse and the Sink".
- Turri et al. (arXiv:2605.13974) describe the massive DiT channels as a prompt-conditioned carrier. A channel-removal control could therefore remove content signal along with the envelope, so the caveat in both papers says so.

## Outline corrections (final pass, applied in the drafts)

1. **Pooling theorem (Mode-II §3).** The outline says "prove the concavity bound for tr √M / the induced g change". The sharp statement is stronger than the one R5 tested.
   - Write M = Σπᵢ Mᵢ, the row concatenation of branches with weights πᵢ, and Φ(M) = (tr √M)² = D · g(M) · tr M.
   - **Φ is concave.** tr √· is concave on the PSD cone (√· is operator concave), so {Φ ≥ 1} = {tr √M ≥ 1} is convex. Φ is positive and homogeneous of degree 1, and such a function with convex superlevel sets is concave.
   - **Theorem (always true):** g_pool ≥ Σᵢ ωᵢ gᵢ, with energy weights ωᵢ = πᵢ tr Mᵢ / tr M. Pooling cannot lower g below the energy-weighted branch mean. This is the R5 caveat made exact.
   - **Equal branch energies:** g_pool ≥ mean gᵢ.
   - **Weaker corollary that R5 Q1e tested:** √g_pool ≥ mean √gᵢ (equal energies), which held in 100% of cells.
   - **Checked numerically on random matrices:** 0 of 5,000 violations of the energy-weighted bound, which is tight to 1e-11. With unequal energies, g_pool < unweighted mean g in 275 of 2,000 draws, so the weights matter.
   - The paper states the theorem with energy weights and gives the +0.040/+0.050/+0.088 shifts as the empirical size. Branch energies were not logged, so the unweighted form is an assumption in the data section (cond and uncond g differ by 0.15%, R1h).
2. **Decoding (Mode-I §5.1).** The SD3.5 L capture regime varies by domain (pooled in 9 domains, branch-mean in 4; `round2/out/E2_decoding_SD35L.csv`). A decoder on the mixed set could partly read the regime. Decoding is therefore reported within regime:
   - SD3.5 L pooled: 0.45–0.55 (chance 0.11)
   - SD3.5 L branch-mean: 0.58–0.82 (chance 0.25)
   - SD3.5 M branch-mean: 0.30–0.34 (chance 0.10)
   - The headline 0.43–0.61 (R1 §2) is kept with that qualification.
3. **Intervention hierarchy (Mode-I §8).** For the resolution arm, the clean Round 4 SD3.5 M captures (512/768/1024, CFG 4.5, explicit branches) replace the legacy SD3.5 resolution suite.
4. **Methods.** Plateau onset σ ≈ 0.81–0.84 on the resolution-corrected clock (#24), the early transient then plateau (#25), guidance contraction of seed spread 0.637 (#28, no timing), and sampling shift 3.0 (M4) all go into Mode-I.
5. **Mode-II §2.2.** g = ‖X‖*² / (D‖X‖_F²), the squared nuclear-to-Frobenius ratio over D. It is distinct from Roy–Vetterli effective rank (exp of the entropy H) and from the eigenvalue participation ratio (Σλ)²/Σλ².
6. **Mode-I §6.2 and Mode-II §9.2** use the analyzer's channel table (bootstrap rate, per-σ incidence, Surrogates A/B). They name SD3.5 L as null-generic and Mochi/ZIT attention as one-crossing aggregates.
7. **Analyzer instrument facts** go in the Mode-I appendix: the LLM-only v3 tools, attn+mlp pooling, the dropped header-less Mochi CSV, and the 4 auxiliary 100-step ZIT generations.
