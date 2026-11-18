> **Note.** The authoritative captions are in `../main.tex`. The drafts below predate the model-identity gate (Round 12): in them, "ZIB" / "Z-Image Base" means the 25-step Turbo campaign ZIT-25, and "Base↔Turbo lineage" means the same Turbo weights under changed inference. Captions for II-F13 (real-lineage margins, Round 11 row-split tangents, caption-row energy share) and the current II-F12 are in `../main.tex` only.

# Mode-II figure captions (draft, LaTeX-ready)

Each caption body below can go straight into `\caption{...}`. Every figure is built by `scripts/fig_<id>.py` (all of them by `scripts/make_all_figures.py`). The plotted numbers are in `data/<figure>.csv`, and each row has a `source` column. "Illustration" marks simulated or schematic content. "Post hoc" marks analyses that were not preregistered (Round 6 P1–P4 and the analyzer replay A1–A4). Notation: $g=(\sum_i s_i)^2/(\sum_i s_i^2)/D$, gap $=s_1/s_2$, $H$ = entropy of $s/\sum s$, where $s_i^2$ are the eigenvalues of the uncentred $X^\top X/N$. $\beta_{\rm layer}=g(L{+}1)-g(L)$.

---

## II-F1 — `II-F1_observable_stack.pdf`

**Mathematical observable stack.** (a) Illustration: a simulated token matrix $X\in\mathbb{R}^{N\times D}$ with a non-zero token mean. The instrument keeps the mean, so it uses the uncentred second moment $C=X^\top X/N$. (b) Illustration: the amplitude spectrum $s_i=\sqrt{\lambda_i(C)}$ of the same matrix, with $g$, gap and $H$ computed from it. The leading amplitude $s_1$ follows the token-mean direction. Centring the same matrix gives a very different $g$ and gap, so $g$ and gap partly measure the ratio of mean to fluctuation. (c) The observed field $\log_{10} g(u,\sigma)$ for Z-Image Base MLP (archive, pooled CFG 4 capture, 476 generations, mean over generations). It is plotted on normalized depth $u=L/(L_{\max}-1)$ and the sampling clock $\tau=\sigma$. Panels (a) and (b) are illustrations; (c) is measured data from the raw telemetry cube.

## II-F2 — `II-F2_pooling_theorem.pdf`

**Pooling theorem and the capture functional.** Theorem: $\Phi(M)=(\operatorname{tr}M^{1/2})^2=D\,g(M)\operatorname{tr}M$ is concave and positively 1-homogeneous on the PSD cone. Hence for a pooled matrix $M=\sum_i\pi_iM_i$,
$g_{\rm pool}\ \ge\ \sum_i\omega_i g_i$, with $\omega_i=\pi_i\operatorname{tr}M_i/\operatorname{tr}M$.
That is, pooled $g$ is at least the *energy-weighted* mean of the branch values. With equal branch energies this becomes $g_{\rm pool}\ge\overline{g_i}$. The weaker corollary $\sqrt{g_{\rm pool}}\ge\overline{\sqrt{g_i}}$ is what Round 5 Q1e tested, and it held in 100% of (generation, step, layer) cells. The theorem does **not** give $g_{\rm pool}\ge$ the unweighted mean of $g_i$ in general. (a) Illustration: Monte Carlo over 20,000 random PSD branch pairs with unequal energies ($D=2$–32). Measured against the energy-weighted mean, there are 0 violations. Measured against the unweighted mean, 1,000 pairs (5%) fall below zero. (b) Identical-trajectory increments: the median $\log(g_{\rm pool}/g_{\rm BM})$ for steps $\ge2$ is $+0.040$, $+0.050$ and $+0.088$ at 1024, 768 and 512 px. This is SD3.5 Medium at CFG 4.5, with deterministic replay of the same generations (60, 29 and 29 pairs). The gap is unchanged ($|\Delta\log{\rm gap}|<3\times10^{-5}$). (c) The same increment by sampling step at 1024 px. It is largest at high $\sigma$ and decays along the schedule. The excess tracks the cond/uncond dissimilarity ($1-$prompt\_influence; Spearman 0.83, Q1f). Pooling therefore moves scalar $g$ without any change in the conditional trajectory. Source: Round 5 Q1c–Q1f (`review/round5/out/Q_results.json`); register M5/T1.

## II-F3 — `II-F3_projection_degeneracy.pdf`

**Projection degeneracy and the equivalence hierarchy.** (a,b) Illustration: two different planar fields, $F_a$ (drift plus relaxation) and $F_b$ (drift plus rotation). Their trajectories from the same start have scalar projections $P(z)=z_1$ that correlate at $0.997$, while the centred fields have cosine $0.09$ over the plotted domain. A high correlation of a one-dimensional projection does not identify a shared vector field. (c) Measured cross-model similarity at five observable levels (Round 6 S1, preregistered). Shown are the Base$\leftrightarrow$Turbo lineage pair (blue), the eight unrelated pairs (gray) and SD3.5 M$\leftrightarrow$L (open diamond). The lineage margin over the best unrelated pair rises with the observable: $-0.049$ (scalar envelope $g$), $+0.047$ ($\log g$), $+0.131$ (depth$\times\sigma$ field), $+0.152$ (registered shape) and $+0.703$ (tangent). On the scalar curve, ZIT$\leftrightarrow$SD3.5 L (0.981) exceeds ZIT$\leftrightarrow$ZIB (0.933). A scalar envelope correlation therefore cannot identify lineage when every envelope shares a monotone trend (register T2). Source: `review/round6/out/S1_hierarchy.csv`.

## II-F4 — `II-F4_closure_ladder.pdf`

**Nonautonomous closure ladder.** Values are transcribed from the findings ledger. The ledger reports ranges and medians, not per-channel tables, so ranges are drawn as intervals. (a) Held-out $R^2$ for the layer-direction flow $\beta_g$:
- depth $u$ alone: 0.70–0.95 in many attention/AR channels, but only 0.43–0.46 in Z-Image MLP;
- $u$ plus the clock $\tau$: 0.89–0.97 in diffusion;
- local state plus coordinates with whole prompts held out: 0.894–0.965 (D01).

(b) Median incremental $R^2$. The spectral state $[g,{\rm gap},H]$ adds $+0.314$ (diffusion) and $+0.146$ (fixed-context AR) over $g$ alone. Once a flexible learner has $(u,\tau)$, the full state adds only $+0.032$ (diffusion) and $+0.010$ (AR), or $+0.014$ overall. Under the frozen minimal-sufficiency rule, 7/12 channels need only $(u,\tau)$ and 5/12 need $g+u+\tau$ (D08/D09). (c) Vector-field transfer:
- within target: 0.941–0.987;
- Base$\leftrightarrow$Turbo: attention 0.700–0.746, MLP 0.722–0.892 (D03);
- residual after scaffold subtraction: Base$\leftrightarrow$Turbo 0.437 against a self ceiling of 0.579, unrelated pairs $-0.256$ against $\approx0.459$ (D15).

Transfer into Mochi and SD3.5 M/L is "approximately zero or negative" (D03). No value is tabulated for it, so it is not plotted. Together these support a model-specific nonautonomous family $\dot z=F_m^{(0)}(u,\tau)+R_m(z,u,\tau)$ and reject a universal autonomous $\beta(g)$ or a universal microscopic $F$ (register T5). Sources: ledger cards "Depth coordinate is a major explanatory variable" (script 15), "Cross-substrate CoS state…" (script 11), D01, D03, D08/D09 and D15.

## II-F5 — `II-F5_trajectory_decomposition.pdf`

**Trajectory decomposition $M_{m,p,s}(\tau)\approx\Gamma_m(\tau)+\delta_{m,p}(\tau)+\varepsilon_{m,p,s}(\tau)$.** (a) Schematic: a model backbone $\Gamma_m$, prompt modulations $\delta_{m,p}$ and seed neighbourhoods $\varepsilon_{m,p,s}$. The components are descriptive and not orthogonal. (b) Nested variance split (ledger D12). The system$\times$time scaffold holds 96.16% of normalized depth-geometry variance (theme+prompt+seed 3.84%) and 95.55% of the absolute operating point (4.45%). Most of the 96% is between-model and between-time variance. (c) Within a fixed (model, sampling-state) cell, prompt identity is the largest component. The naive between-prompt share is 0.51–0.96 across seven archive channels and all $\sigma$ (Round 2 E6; the ANOVA seed-corrected share is 0.36–0.95). ZIB's share is lowest at $\sigma=1$, where its archive capture is pooled, so part of that dip may come from the capture regime. Mochi CFG 1, with one seed per prompt, is omitted. Black (post hoc, added 2026-10-08): the SD1.5 U-Net of Mode-I, conditional branch, plotted against the rectified-flow clock at equal signal-to-noise ratio, $t=\sigma/(1+\sigma)$; its naive share is 0.67–0.75 (ANOVA 0.56–0.67), inside the archive range. Diamonds (post hoc, added 2026-10-09): the audio model ACE-Step, 0.86–0.97 (ANOVA 0.82–0.96), with no early rise (B10 refuted). State both numbers (register #14). Sources: ledger D12; `review/round2/out/E6_varcomp_by_step.csv`; `review/r3_unet/out/P3_prompt_share_by_step_naive.csv`; `review/r3_ace/out/P3_prompt_share_by_step.csv`.

## II-F6 — `II-F6_tangent_inheritance.pdf`

**Tangent-field inheritance: state correspondence is not dynamics correspondence.** (a) Registered depth-shape $r$ against the depth-derivative (tangent) cosine at matched $\sigma$ for 13 channel pairs (Round 2 E10). The open markers are Round 5 clean-branch checks:
- ZIB CFG 4 cond-branch$\leftrightarrow$ZIT: 0.992 attention, 0.992 MLP;
- ZIB CFG 1$\leftrightarrow$CFG 4: 0.991 and 0.993.

Unrelated pairs reach a shape $r$ of up to 0.83 (Z-Image$\leftrightarrow$Mochi), yet their tangent cosine stays $\le0.24$, inside each pair's permutation null (the shaded band is the median null p95, 0.29). Base$\leftrightarrow$Turbo scores 0.94–0.97 and the same-weights Mochi CFG pair 0.99. SD3.5 M$\leftrightarrow$L, a scale change within one family, scores $-0.25$. (b) Ledger D10: state-only alignment maps (Procrustes, linear, CCA) are fitted without derivative labels. After them, matched derivative correlation averages $\approx0.937$ for Base$\leftrightarrow$Turbo and $\approx0.007$ for the best map of each unrelated pair. Aligning representations does not align their dynamics. This is a partial orientation-preserving trajectory correspondence in the measured coordinates. It is weaker than formal orbit equivalence. Sources: `review/round2/out/E10_tangent.csv`, `review/round5/out/Q_results.json` (Q2a, Q2c), ledger D10.

## II-F7 — `II-F7_content_vs_dynamics.pdf`

**Content geometry and dynamics are different quotients.** (a) For each of the 10 model pairs, the first-step prompt RSA is plotted against the tangent cosine. The RSA is a Mantel $\rho$ between $117\times117$ prompt-distance matrices, with $p=0.0005$ for every pair against its own prompt-permutation null. The rank coupling is weak (Spearman $\rho=0.297$, preregistered S2g). (b) RSA at the first, middle and last step. Unrelated pairs score 0.13–0.43 and the lineage pair 0.56–0.62. Every value lies far above its permutation-null p95 ($\approx0.02$, shaded). Shared content organization exists across unrelated architectures, beyond themes and prompt text (P2), and without a shared microscopic flow. The two are distinct equivalence relations. Sources: `review/round6/out/S2_rsa.csv`, `S1_hierarchy.csv`; register #19, T3.

## II-F8 — `II-F8_two_axis_topology.pdf`

**Two different phase constructions; do not conflate them.** (a) Sampling-clock axis, Z-Image MLP only. Population-mean, layer-mean MLP $g(\sigma)$ is shown for ZIT (8 steps), ZIB (archive, pooled CFG 4) and the clean ZIB CFG 4 cond branch, recomputed from the raw cubes. The three-segment fit gives $+-+$ in 100% of 1000 prompt bootstraps in every arm (S4a–b). The contraction is $-49.7\%$ in the archive and $-50.5\%$ on the clean branch. Vertical lines mark the ZIB archive breakpoints, $\sigma=0.972$ and $0.818$. Attention follows a different curve ($++-$) in all five models (S4e, P1), so the $\sigma$-axis result is scoped to Z-Image MLP. (b) Layer axis, post hoc (analyzer replay A2–A4). The sign pattern of $\beta_{\rm layer}(g)$, binned against $g$, is shown at each sampling step. Right-hand columns give the whole-run generation-bootstrap rate of $+-+-$ and the surrogate class:
- SD3.5 M attention: 100%, beats Surrogates A and B;
- SD3.5 L attention: 100%, **null-generic** (per-layer marginals alone reproduce it);
- ZIB attention: 67%, beats A and B;
- ZIB MLP: 84%, beats A and B;
- ZIT MLP: 90%, beats A, but B (the mean field $g(L,\sigma)$) reproduces it;
- Mochi and ZIT attention: a single, null-generic crossing.

Report the layer-axis pattern per $\sigma$ or with its bootstrap rate and surrogate class, never as one crossing count. Neither panel defines a phase boundary or a separatrix. Sources: `review/round6/out/S4_topology.csv`, `A2_beta_layer_bootstrap.csv`, `A4_beta_layer_per_step.csv`, `A4_beta_layer_surrogates.csv`; raw cubes for the curves in (a); register #1.

## II-F9 — `II-F9_differencing_nulls.pdf`

**Differencing and surrogate nulls.** Let $v_k$ be i.i.d. with a common mean and covariance $\Sigma$, and let $d_k=v_{k+1}-v_k$. Then $\mathbb{E}[d_k\!\cdot\!d_{k+1}]=-\operatorname{tr}\Sigma$ and $\mathbb{E}|d_k|^2=2\operatorname{tr}\Sigma$. So the adjacent-difference cosine $\varphi$ concentrates at $-1/2$, and the stride-2 cosine at $0$, as the effective dimension $(\operatorname{tr}\Sigma)^2/\operatorname{tr}\Sigma^2$ grows. (a) Illustration: Monte Carlo distributions at $D=3072$ (filled) and $D=16$ (outline), with a power-law $\Sigma$. (b) Illustration: mean $\pm$ sd against effective dimension. At $D=3072$ (effective dimension 1390), $\varphi=-0.500\pm0.020$ and the stride-2 cosine is $0.000\pm0.027$. A raw $\varphi\approx-1/2$ is the null value, not a signal (register #9, T4). (c) Post hoc analyzer replay A4. For each channel, the plot gives the fraction of 30 draws of the 09-15 Surrogate A (per-layer marginals) and Surrogate B (mean field $g(L,\sigma)$) that reproduce the observed whole-run $\beta_{\rm layer}$ pattern. A crossing that its surrogate reproduces is null-generic and is not a phase boundary. Source: `review/round6/out/A4_beta_layer_surrogates.csv`.

## II-F10 — `II-F10_local_mode_census.pdf`

**Local linear mode census.** (a,b) The affine layer map $X_{L+1}=AX_L+b$ on $X=[g,-{\rm gap},H]$ is fitted per channel $\times$ sampling state (first, middle, last step) $\times$ semantic domain. Its eigenvalues are invariant to per-coordinate affine rescaling. This is a **re-implemented estimator**, not the ledger's script 14, recomputed from the raw archive cubes. (a) The fraction of 100 generation bootstraps in which $A$ has a complex-conjugate pair, averaged over the 10 domains. It ranges from 0 to 1 depending on architecture and state. (b) The dominant modulus $|\lambda|_{\max}$, one dot per domain. The median across-domain coefficient of variation is 1.6% here (ledger, script 14: $\approx2.2\%$). The modulus is below 1 in most cells but above 1 at the first state of Z-Image MLP. (c) Ledger, script 12: a Base$\to$Turbo time warp is chosen from depth geometry alone. Under it, the local complex-mode sequence closes for MLP (exact permutation $p=0.00196$ over all $8!$ maps) but not for attention ($p=0.259$). Rotation is a regime property; the contraction scale is the more recurrent component. This is descriptive and is no evidence of a physical oscillator or probability current. Sources: raw cubes; ledger cards "Complex local modes are regime-specific…" (script 14) and "Geometry-derived Base→Turbo warp…" (script 12).

## II-F11 — `II-F11_estimators.pdf`

**Estimator mathematics for seed predictability.** (a) $\sigma_{50}$ is a threshold crossing: the first $\sigma$ at which the cross-validated $R^2$ from state to final state passes 0.5. It therefore moves with the $R^2$ ceiling and with sample size. On nested subsets of the SD3.5 M archive (branch-mean capture), $\sigma_{50}$ goes from 0.07 at $n=60$ (undefined in 98% of partitions) to 0.94 at $n=480$. The relative onset on the same curves stays between 0.87 and 0.96. Bars are p5–p95 over fold partitions. (b) At matched $n=60$, across all seven archive channels, relative onset is 0.84–0.95 in six channels and 0.67 in ZIT MLP (8-step grid). Its range across channels is 0.28, against 0.50 for $\sigma_{50}$ (S3a–c). (c) The variance ratio VR(4) of seed-deviation residuals, against finite-length references simulated at each channel's own length (i.i.d. $\approx0.25$, AR(1) 0.8 $\approx0.72$, random walk $\approx0.82$). VR(4)/walk is 0.49–0.87 in six channels and 1.13 in ZIT attention. Seed deviations sit between AR(0.8) and a walk on the calibrated scale (S3d refuted the walk-like prediction). These panels calibrate estimators only. They make no claim about commitment or a separatrix (gate G3). Sources: `review/round5/out/H1_seedfate_vs_sample_size.csv`, `review/round6/out/S3_estimators.csv`; register #26–27.

## II-F12 — `II-F12_claim_boundary.pdf`

**Universality and falsification hierarchy: the claim boundary.** Each row is a candidate structure. Each column asks whether it survives a change of content, fixed weights (guidance, capture), model scale, distillation, unrelated architecture or data modality, and whether a causal gate is still open. The cell text gives the governing number. What survives is a hierarchy:
- the coarse scalar envelope recurs everywhere but cannot identify lineage;
- prompt-content geometry is shared across unrelated models and strongest in the lineage;
- the tangent field is preserved only under fixed weights and distillation.

The $\sigma$-axis $+-+$ is scoped to Z-Image MLP. Unrelated-architecture cells also give the SD1.5 U-Net of Mode-I where it was measured (post hoc, added 2026-10-08): its prompt geometry is shared with the transformers, while its attention envelope and its MLP envelope both rise over the whole schedule ($+++$). The audio-modality column (added 2026-10-09) is the audio flow transformer ACE-Step of Mode-I: its envelopes follow $-+-$ and its layer axis has one crossing (preregistered B6, refuted), so the layer-axis pattern is image- and video-specific; prompt geometry, tangents and local modes were not compared, because its prompts differ. The layer-axis $+-+-$ (post hoc) holds only with its bootstrap rate and surrogate class, and SD3.5 L is null-generic. A universal microscopic $F$ fails across architectures and across Qwen sizes. Separatrix or commitment statements need the committor gate G3. Not claimed anywhere: RG fixed points or universality classes, thermodynamic criticality, full-state conjugacy (register T5–T6). Per-cell evidence and sources are in `data/II-F12_claim_boundary.csv`. Statuses follow `R6C_RELEASE_CLAIM_REGISTER.md` and `R6C_GATES.md` (v0.19).

---

## Supplement

## II-S1 — `II-S1_intrinsic_progress.pdf`

**Intrinsic progress gives no decisive universal quotient (negative result).** (a,b) Illustration on data: layer-mean attention $\log g$ for the five archive models, plotted against $\sigma$ (a) and against normalized cumulative spectral arc length (b). Here arc length is re-implemented as the path length of the z-scored layer-mean $(\log g,\log{\rm gap},H)$. It shows the reparameterization only and is not the D17 computation. (c) Ledger D17: arc-length reparameterization raises unrelated and cross-channel envelope correlation by $+0.057$ on average. The preregistered paired test is not significant ($p\approx0.105$). Base/Turbo changes by $-0.004$. A universal intrinsic-progress quotient is therefore not established.

## II-S2 — `II-S2_legacy_rebound.pdf`

**Legacy rebound and local complex mode: historical, instrument-labelled.** Everything marked *legacy* comes from EMAG-schema files written by an older EMAG-guider build, whose measuring hook substitutes the stale EMA attention. It is shown as the history of the mathematical exploration and is not release evidence. (a) 24-bin $\beta_{\rm layer}(g)$ for a matched 100-step SD3.5 M study with the same prompt and schedule. The 30 legacy-schema runs give $-+-+-$ (four crossings); the 15 current passive runs give $+-$ (one crossing, $g^*=0.059$). Legacy $g$ sits about $10\times$ below passive $g$. The ledger reports the four-crossing pattern for 30/30 single legacy generations and 0/60 single passive prompts. (b) Crossing locations on the legacy resolution suite (512, 768 and 1024 px, single prompt; four crossings each) and on the passive 120-prompt SD3.5 M campaign (three crossings, $g^*=0.018/0.020/0.077$). (c) Complex-pair modulus of the local affine map:
- on the legacy rebound branch: 0.526, 0.501 and 0.400;
- in the legacy CFG $1.0\to5.5$ study, steps 0–9: 0.878 [0.862, 0.891] $\to$ 0.849 [0.836, 0.861].

Over all 100 steps of that study, CFG 1.0 has no complex pair. Sources: `review/round5/reviewer_v016/results/B7_MATCHED100_*`; ledger rebound cards with their v0.12–v0.15 annotations.
