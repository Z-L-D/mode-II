# R3 Part C addendum P7 (post hoc; written before computing)

Requested at 22:42Z: fold the SD1.5 data into the existing paper figures. Mode-I Fig. I-F7 and Mode-II Fig. II-F12 carry the
sigma-axis three-segment topology (R6 S4, P1), which has not been computed for SD1.5. Defined here before computing; post hoc.

P7. Clock-axis topology of the population-mean, layer-mean g (cond branch), SD1.5 attn and MLP.
- Statistic: R6 S4 `analyse()` unchanged: per-prompt mean over gens and blocks (nanmean for the one missing cell), mean over
  prompts, continuous three-segment piecewise-linear fit `fit3`, 1000 prompt bootstraps (rng seed 64).
- Clock: primary t = sigma/(1+sigma) (the SNR-matched RF clock used for P2). Sensitivity: raw VP sigma, and the
  resolution-shifted clock t' = 2t/(1+t). The sign pattern is reported for each; if they disagree, that is the result.
- Reported: topology, bootstrap fraction '+-+', modal pattern and its fraction, breakpoints, contraction (S4 `extrema`).
- No verdict is attached; the R6 statements stay scoped to the archive. Label in papers: post hoc.
