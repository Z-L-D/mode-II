# Round 5 audit / v0.16 synthesis

## Formal discipline

The repaired branch instrument works, but the execution missed three frozen design constraints:
- exact prompt set: 26/30 overlap; four prompts replaced;
- unique seed pair per prompt: not executed (global 1001/1002 reused);
- R2 sample size: 15 seeds instead of >=30.

Therefore no aggregate R1/R2 hit rate is reported.

## Main new findings

1. **Branch identity works.** R1b median |log g ratio| at step 0 is 0.000453.
2. **Guidance contraction is real without pooling.** Conditional-branch seed spread ratio in the last three quarters is 0.575; quarter medians progress 1.00 -> 0.77 -> 0.55 -> 0.46.
3. **Persistence has two contributors.** VR4: CFG1 cond 1.85; CFG4.5 cond 1.25; uncond 0.59; synthetic branch mean 0.79. Guidance reduces persistence and branch mixing reduces it further.
4. **Cond/uncond output separation is largest early, while seed contraction is strongest later.** Same-time and lagged trend correlations are descriptive only because both series are autocorrelated.
5. **The four-crossing legacy card is not restored by a genuine current-passive 100-step pyramid run.** New passive pooled beta is `+-` with one crossing; 0/15 new runs are `-+-+-`, versus 30/30 legacy. The two 100-step runs use different sigma grids, so exact EMAG causation remains unidentified.
6. **Simple sigma-grid weighting does not explain B7.** Interpolating legacy onto the passive grid keeps `-+-+-`; interpolating passive onto legacy keeps `+-`.
7. **True current-passive 40->100 traversal has a small matched-sigma effect.** Median |log g ratio| = 0.0326; seed bootstrap 95% ~0.014-0.060.
8. **Passive beta crossing count is itself resolution/branch sensitive.** Corrected CFG4.5 cond is `+-` at 512, `+-+-` at 768 and `+-+-` at 1024; bootstrap modal fractions are 0.962, 0.660 and 0.882 respectively.
9. **Round-4 P1-P5 audit passes.** Prereg SHA matches; all 16 verdicts recompute; 11 confirmed / 5 refuted.
10. **D14 remains blocked.** No output identifier or quality/correctness label is present in the new telemetry.

## Paper consequence

Treat beta crossing count as a projected, intervention/resolution-sensitive diagnostic, not a universal phase invariant. The strongest stable story remains the separation between inherited normalized geometry and intervention-sensitive operating point, now augmented by an explicit measurement-functional result: stacked pooling can move PR-derived g strongly while branch-mean and cond-only scalar summaries are nearly identical.

Do not use passive future-state predictability as commitment/speciation evidence. Biroli-style clone continuations remain the required causal gate.
