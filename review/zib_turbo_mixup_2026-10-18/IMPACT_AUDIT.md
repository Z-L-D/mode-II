# ZIB = Turbo: claim-impact audit (2026-10-18, revised)

This is a read-only audit; no existing file was edited. Its premise comes from the coordinator and was not re-verified here. Every dataset labelled "ZIB" or "Z-Image Base" is Z-Image **Turbo**. The evidence given was that the archive "ZIB" step-0 depth profile correlates 0.99 with Turbo, against 0.79 (attn) / 0.55 (mlp) for a genuine May 2026 Base run, with a Mochi May-vs-Aug instrument control at 0.994. No genuine Base data exist in the project.

Relabelling used below:

| Label in papers | What it actually is |
|---|---|
| ZIT archive | **T8**: Turbo, 8 steps, CFG 1, cond-only |
| ZIB archive (`F_ZIB_*`, 428 + 48 abstract) | **T25p**: Turbo, 25 steps, CFG 4, pooled |
| ZIB R3/R4 (`F3_Z10/Z40`, `R5('Z10/Z40_1024_s1001')`) | **T25c1**: Turbo, 25 steps, CFG 1, cond-only. **T25e4**: Turbo, 25 steps, CFG 4, explicit branches (cond / uncond / BM) |
| "Base↔Turbo" / "lineage pair" | **T25p↔T8**: the same weights with step count (25 vs 8), guidance (4 vs 1) and capture (pooled vs cond-only) all changed. It is a fixed-weights, changed-inference pair, not a lineage. |

Classes:
- **V, VOID**: the claim rests on Base being a different model from Turbo, i.e. lineage, inheritance or distillation.
- **R, REINTERPRETABLE**: the observation is valid but describes Turbo under a 25-step schedule, CFG 4 and/or pooling. The note says what it now means.
- **U, UNAFFECTED**.

**Within-Turbo tangent decomposition.** Read with the new labels, the old "lineage tangent ladder" becomes a decomposition within one model. Values are tangent against T8, attention / MLP:

| Capture | Tangent vs T8 | Comparison it isolates |
|---|---|---|
| T25c1 | 0.999 / 0.999 | step count |
| T25e4 cond | 0.992 / 0.992 | + guidance |
| T25e4 BM | 0.988 / 0.991 | |
| T25e4 uncond | 0.981 / 0.989 | |
| T25p | 0.942 / 0.967 | + pooling |

So the archive "lineage" tangent of 0.94/0.97 is a within-model drop from guidance plus pooled capture; the 8→25 step change costs about 0.001. Caveat: prompt sets differ (T25c1 60/20 prompts, T25e4 19, T25p 117).

## Lineage claims (every sentence; all V as lineage evidence)

Each of these describes a Base↔Turbo lineage, distillation or inheritance result. Where a number is given, it is R as a T25p↔T8 fixed-weights comparison.

**Mode-I**
- I:23: title "What Recurs and What Is Inherited".
- I:33 (abstract):
  - "which are inherited within a model lineage, here one pair: Z-Image Base and Turbo, a model distilled within the same family".
  - "the best unrelated pair beats the lineage pair by $0.049$".
  - "the margin of the lineage pair over the best unrelated pair grows to $+0.047$, $+0.131$, $+0.152$ and $+0.703$".
  - "depth tangents agree at $0.94$--$0.99$ between Z-Image Base and its distilled Turbo".
  - "$0.56$--$0.62$ within the lineage".
- I:44: "Two of them form a distillation lineage … Z-Image Base is the undistilled model … By \emph{inherited} we mean shared by descent within this model lineage … it contains one lineage pair."
- I:53: research question 6 includes "distillation".
- I:62: contribution 3 "depth tangents separate the distillation lineage".
- I:77: Related work, Distillation: "We measure … how much of the depth dynamics survives."
- I:161:
  - "slightly above Base and its own distillation Turbo ($0.9325$)".
  - "Base$\leftrightarrow$Turbo $0.973$".
  - "exceeds the lineage pair ($0.933$)".
- I:171: "the lineage margin, the \BT{} value minus the best unrelated pair".
- I:175, I:181–183: Table 2 "Lineage margin by observable level", row "Base$\leftrightarrow$Turbo".
- I:188:
  - "so shape alone does not isolate the lineage. The tangent does."
  - the ladder "For \BT{} it depends on how Base was captured…".
  - "The \BT{} tangent does not depend on sample size".
  - the caption-row paragraph "bears on the lineage values only … Shared caption rows could raise the Base$\leftrightarrow$Turbo tangent … the lineage tangent is a statistic of the joint sequence".
- I:193: Fig I-F3 caption "Lineage becomes identifiable only at the level of the local depth derivative … (Base$\leftrightarrow$Turbo lineage) … beats the lineage (0.933). The tangent separates the lineage: 0.94".
- I:198: "\BT{} averages $0.937$ across maps and channels".
- I:203: Fig I-F9 caption "Only Base$\leftrightarrow$Turbo is high on both (0.98, 0.94) … Base$\leftrightarrow$Turbo keeps a derivative correlation of 0.92--0.95".
- I:247: "$0.56$--$0.62$ for \BT{}".
- I:252: Fig I-F6 caption "strongest in the lineage … The lineage cell is outlined. Base$\leftrightarrow$Turbo: 0.56--0.62".
- I:260: "strongest in the lineage at every state ($0.678$, $0.750$, $0.626$) … Theme inheritance is lineage-specific only at the first state".
- I:360: "Distillation lineage. Z-Image Base and Turbo … share registered shape ($0.97$--$0.99$) and tangents ($0.94$--$0.99$), and the geometry-derived registration closes the MLP local eigenmode sequence ($p=0.002$…)".
- I:363: "The one distillation lineage shares geometry and, especially, depth tangents; whether that comes from the distillation objective or from shared pretraining is not separable here."
- I:368: Fig I-F11 caption "distillation lineage (ZIB$\leftrightarrow$ZIT): 0.027" and "ZIB$\leftrightarrow$ZIT 0.94/0.97".
- I:462: synthesis 2 "strongest in the lineage".
- I:463: synthesis 3 "Local depth dynamics are lineage-specific in the one lineage pair available: tangents separate Z-Image Base and Turbo from every unrelated pair by $0.70$. Whether this comes from the distillation objective or from shared pretraining…".
- I:477: limitation "recomputes the lineage tangent, the margin".
- I:553: Fig S1 caption "Base$\leftrightarrow$Turbo: 0.63--0.75 at every step, the strongest … strongest in the lineage at every step and lineage-specific only at the first step".

**Mode-II**
- II:32: title "Projection, Pooling and Tangent Inheritance".
- II:42 (abstract): "local depth dynamics agree almost only within a distillation lineage" and "the lineage margin over the best unrelated pair rises from $-0.049$ … to $+0.703$".
- II:167: "exceeds the lineage pair (Turbo with Z-Image Base, $r=0.933$)".
- II:223: "High tangent agreement, such as Base$\leftrightarrow$Turbo's $0.94$--$0.99$".
- II:227–237: Data anchor "lineage margin … Base$\leftrightarrow$Turbo value" with the margin table; "$0.94$--$0.99$ for Base$\leftrightarrow$Turbo … the lineage value is unchanged on 60-generation subsets".
- II:242: Fig II-F3 caption "Base$\leftrightarrow$Turbo lineage pair (blue) … lineage margin … cannot identify lineage".
- II:269: "Base$\leftrightarrow$Turbo residual transfer … $0.437$ … Direct Base$\leftrightarrow$Turbo transfer … $0.70$--$0.75$ … $0.72$--$0.89$".
- II:272: Proposition "does not reject substantial agreement for the Base$\leftrightarrow$Turbo pair".
- II:279: Fig II-F4 caption "Base$\leftrightarrow$Turbo: attention 0.700--0.746 … Base$\leftrightarrow$Turbo 0.437".
- II:307: section "Dynamical inheritance under distillation".
- II:310: "We call the pair a distillation lineage … Base in 25 steps with guidance … read Turbo as sampling Base's $\sigma$-indexed geometry at 8 points".
- II:313: the tangent ladder "For Base$\leftrightarrow$Turbo …".
- II:316: "Base$\leftrightarrow$Turbo averages $0.937$".
- II:321: "rare within the lineage".
- II:324: "The relation between Base and Turbo is a partial, orientation-preserving correspondence … trajectory inheritance can be detected without access to the training objective."
- II:328–329: Fig II-F6 "Tangent-field inheritance … Base$\leftrightarrow$Turbo scores 0.94--0.97 … $\approx0.937$ for Base$\leftrightarrow$Turbo".
- II:338: "$0.56$--$0.62$ for Base$\leftrightarrow$Turbo".
- II:340: "strongest in the lineage, while tangent agreement is essentially confined to the lineage".
- II:345: Fig II-F7 caption "the lineage pair 0.56--0.62".
- II:385: "$\varphi$ residuals reproduce between Base and Turbo".
- II:416: "Lineage closes MLP modes. The geometry-derived Base$\to$Turbo registration…".
- II:423: Fig II-F10(c) "a Base$\to$Turbo time warp".
- II:461: universality item 3 "lineage-specific tangent dynamics … observed only under distillation".
- II:470: Fig II-F12 caption and column "distillation (Base↔Turbo)": "strongest in the lineage; the tangent field is preserved only under fixed weights and distillation".
- II:478: "Whether Base and Turbo share such a spectrum is the natural next test of dynamical inheritance".
- II:510, II:542: arc length "changes Base$\leftrightarrow$Turbo by $-0.004$".
- II:535: "a compressed learned clock between Base and Turbo".

## Claim table

| # | Claim / number (as in paper) | Where | Source script / output | Data | Class | Note (what it now means) |
|---|---|---|---|---|---|---|
| 1 | "five diffusion and flow transformers (Z-Image Turbo and Base, …)" | I:33, I:44, I:139/I:144 "all five models" | archive | T8, T25p | V (count) | Four models. "ZIB" is a second Turbo campaign (25 steps, CFG 4, pooled). |
| 2 | Lineage margins $-0.049$, $+0.047$, $+0.131$, $+0.152$, $+0.703$; Table 2 row "Base$\leftrightarrow$Turbo 0.933 0.973 0.984 0.981 0.942" | I:33, I:171–183; II:42, II:227–237; Fig I-F3a (I:193), Fig II-F3c (II:242) | `r6c_S1_hierarchy.py` S1a–c → `S1_hierarchy.csv` | T25p vs T8; unrelated pairs | V as lineage / R | Fixed-weights margin: the same Turbo weights under 8→25 steps, CFG 1→4 and cond-only→pooled beat the best unrelated pair by +0.703 on the tangent. The margin indicates model identity under changed inference, not inheritance. The unrelated-pair values (≤0.24, best 0.981 linear) are U. |
| 3 | "depth tangents agree at $0.94$--$0.99$ between Z-Image Base and its distilled Turbo"; "tangents ($0.94$--$0.99$)"; "Base$\leftrightarrow$Turbo's $0.94$--$0.99$" | I:33, I:360; II:223, II:237 | E10, R5 Q2 | T25p/T25c1/T25e4 vs T8 | V / R | Turbo vs itself across schedule, guidance and capture: 0.94 (pooled CFG 4) to 0.999 (CFG 1 cond-only, 25 vs 8 steps). |
| 4 | Tangent ladder "$0.94$/$0.97$ … pooled archive, $0.981$/$0.989$ … unconditional, $0.988$/$0.991$ … branch mean, $0.992$/$0.992$ … CFG 4, and $0.999$ … CFG 1"; Fig I-F3b; Fig II-F6 open markers | I:188, I:193; II:313, II:329 | R2 E10; R5 Q2a/b/Q2_ref (`r5_q_calcs.py`); `fig_I_F03_hierarchy.py`, `fig_II_F6_tangent_inheritance.py` | all Turbo | R | This is now a clean within-model decomposition: step count about 0.001, guidance 0.007, pooling 0.025–0.049. It supports the paper's capture-functional thesis, within Turbo. |
| 5 | "Guidance therefore moves conditional geometry by less than $1\%$, and most of the archive's shortfall from $1$ is the pooled capture (guidance moves the tangent by $0.007$, the capture by $0.025$--$0.049$)" | I:188 | R8-6, `r8_b_fast.py` | Turbo only | R (valid) | Valid as a within-Turbo statement. "Shortfall" is no longer a lineage shortfall; it is a capture effect on one model. The prompt-set difference (19 vs 117) remains a caveat. |
| 6 | "In attention that shortfall sits in the deepest third ($0.86$ …) … every third is at least $0.98$ on the conditional capture" | I:188 | R8-5 | T25p, T25e4 vs T8 | R | Within Turbo, pooling lowers the late-third attention tangent, and explicit cond restores it. |
| 7 | "\BT{} tangent does not depend on sample size … $0.937$--$0.946$ … $0.965$--$0.968$" | I:188; II:237 | R8-13 | T25p vs T8 | R | The n-stability of the T25p↔T8 tangent stands. |
| 8 | Caption-row paragraph: "Shared caption rows could raise the Base$\leftrightarrow$Turbo tangent…" | I:188 | — | — | V | Moot. Same model and same text encoder. |
| 9 | Registered shape "$0.981$ … $0.983$ … on the pooled archive, and $0.9997$ on the clean Base CFG 1 capture"; "share registered shape ($0.97$--$0.99$)" | I:188, I:360 | S1, R3 B8a | Turbo | R | The depth shape of the same weights is unchanged by step count (0.9997) and moves about 2% under CFG 4 plus pooling. |
| 10 | Scalar ordering: "$r=0.9405$ between Z-Image Base and Mochi, slightly above Base and its own distillation Turbo ($0.9325$)"; log $g$ "$0.973$ against … $0.860$"; "(Round 4, 19 prompts) … $0.989$ and $0.831$"; Fig I-F4 "(archive 0.973 vs 0.860; CFG\,1 cond-only 0.9998 vs 0.732)"; I:385 | I:161, I:166, I:385, I:515 | S1d (`r6c_S1_hierarchy.py`), `fig_I_F04_capture_regime.py` | Turbo only + Mochi | R (strengthened) | A cleaner worked example: one model's envelope ordering against Mochi flips with schedule, guidance and capture. Every substitution is now the same Turbo weights. "Distillation Turbo" wording V. |
| 11 | "unrelated pair (Turbo$\leftrightarrow$SD3.5 Large, $0.981$) exceeds the lineage pair ($0.933$)"; Prop. trend dominance data | I:161; II:167 | S1 | — | R (strengthened) | An unrelated pair beats the same model under changed inference on linear $g$. That is a stronger illustration of trend dominance. |
| 12 | D10 "\BT{} averages $0.937$"; Fig I-F9 "(0.98, 0.94)", "0.92--0.95" | I:198, I:203; II:316, II:329 | ledger D10 | T25p, T8 | R | State-only maps keep derivative agreement for the same model under changed inference. They fail for unrelated models. "rare within the lineage" (II:321) V. |
| 13 | Transfer "Base$\leftrightarrow$Turbo residual transfer $0.437$ … direct $0.70$--$0.75$ / $0.72$--$0.89$"; Proposition (no universal $F$) | II:269, II:272, II:279 | ledger D03/D15 | T25p, T8 | R | Vector fields transfer within one model across schedules. The rejection of $F_a=F_b$ for unrelated pairs is U. The "does not reject … Base$\leftrightarrow$Turbo" clause is a fixed-weights statement. |
| 14 | Interventions: "Distillation lineage" bullet and closing sentence; Fig I-F11 "distillation lineage (ZIB$\leftrightarrow$ZIT): 0.027", "ZIB$\leftrightarrow$ZIT 0.94/0.97" | I:360, I:363, I:368 | ledger, S1 | Turbo | V / R | No distillation data. The pair belongs under "Fixed weights, changed inference": shape deformation 0.027 and tangent 0.94/0.97 for 8→25 steps + CFG 1→4 + pooling. That is larger than Mochi CFG (0.9965) because the capture also changes. |
| 15 | MLP eigenmode closure "$p=0.002$" ($0.00196$) / attention "$p=0.26$"; "This test used the pooled Base archive" | I:360; II:324, II:416, II:423 | ledger script 12 | T25p, T8 | R | A geometry-derived registration between Turbo's 8- and 25-step schedules closes Turbo's own MLP mode sequence. |
| 16 | Equal-σ registration "improves … by at most $0.005$ … 99% / 100% … 117 shared prompts selects zero layer shift"; "read Turbo as sampling Base's $\sigma$-indexed geometry at 8 points"; "compressed learned clock … (replaced by equal-σ registration)" | I:134; II:310, II:535 | R1/R2 P3, Z10 | T25p, T8 | R | The same model sampled at 8 and 25 points of one σ axis registers at equal σ. This is expected for fixed weights. |
| 17 | "for Z-Image Base this is below the official value of $6.0$, and it is the setting under which Base and Turbo share a $\sigma$ axis"; "Base in 25 steps with guidance" | I:124; II:310 | — | — | V | No Base was captured. The archive "ZIB" is Turbo run at 25 steps, CFG 4, shift 3.0. |
| 18 | Table 1 row "Z-Image Base (ZIB) & 30 & 3840 & 25 & 4.0 & 476 & attn, MLP; pooled"; "ZIB, Rounds 3--4 … 25 & 1, 4 & 19--60"; Fig I-F1 rows; App A "Z-Image Base domain file (48 generations) … Base to 476"; "117 prompts are shared by Z-Image Base, Turbo and Mochi" | I:107, I:113, I:129, I:490, I:124 | `fig_I_F01_corpus.py` | Turbo | R (relabel) | Turbo campaigns: 25 steps / CFG 4 / pooled; 25 steps / CFG 1 or 4 / cond-only or explicit. |
| 19 | Fig II-F1c "$\log_{10} g(u,\sigma)$ for Z-Image Base MLP (archive, pooled CFG 4 capture, 476 generations…)" | II:106 | `fig_II_F1_observable_stack.py` | T25p | R | Turbo MLP field under 25 steps, CFG 4, pooled. |
| 20 | Backbone "all five models beat all 999 null campaigns … null … $4.9$--$14.8$"; Fig I-F2; U-Net comparison "$0.90$ and $0.68$ on SD3.5 Medium and Z-Image Base" | I:144, I:153, I:412 | backbone | T25p | R | Four models, with Turbo in two inference settings. The backbone result is U. The 0.68 is Turbo-25 MLP. |
| 21 | Plateau onset "$0.78$--$0.87$ in all eight archive channels (four architectures, CFG 1--4.5, 8--100 steps)" | I:148; II:459 | E3 | incl. T25p attn/mlp | U (label) | The channel count stands. The two "ZIB" channels are Turbo at 25 steps / CFG 4. The "four architectures" count is unchanged. |
| 22 | Step-count control "Turbo: $g$ within $1.9\%$, on a four-generation 100-step control" | I:148, Fig S2 | R2 P4 | T8 / T100 | U | Not cited in paper: T25c1↔T8 envelope 0.9998 and tangent 0.999 (S1d, Q2_ref) now give a much larger step-count control. |
| 23 | Decoding Table 4 "ZIB attn 0.489 … ZIB MLP 0.460"; Fig S5 "cinematic prompts in ZIB MLP (0.10)"; "seven transformer channels … Spearman $\rho=0.64$" | I:211, I:223–224, I:581 | `fig_I_F05_semantic.py`, `fig_I_S05_domain_confusion.py`, R9-A4 | T25p | R | Turbo-25 CFG 4 pooled decodes at 0.489/0.460 against T8 0.548/0.554. The seven channels are not seven models: two pairs are the same Turbo weights. |
| 24 | R3 matched decoding "$0.45$ (attention) and $0.35$ (MLP) for Z-Image Base at CFG 1 (cond-only)" | I:211 | `cond_only_decoding.py` | T25c1 | R | Regime-homogeneous Turbo-25 CFG 1. The control's logic still holds. |
| 25 | Mantel "$0.56$--$0.62$ for \BT{}", Fig I-F6, Fig II-F7; S2e "Shared text encoders do not explain it" | I:33, I:247, I:252; II:338, II:340, II:345 | `S2_rsa.csv` | T25p vs T8 | V as lineage / R | Prompt geometry agrees most for the same model under changed inference (0.56–0.62). Unrelated 0.13–0.43 and Z-Image↔Mochi vs SD3.5 M↔L are U. The S2e tier "lineage > family" is a fixed-weights tier. |
| 26 | Theme residuals "strongest in the lineage … $0.678$, $0.750$, $0.626$ … lineage-specific only at the first state"; Fig S1 | I:260, I:553 | `S2_theme_residual.csv` | T25p vs T8 | V as lineage / R | Same model, highest theme alignment. "Theme inheritance" V. |
| 27 | Content vs tangent Spearman "$0.297$, $p=0.41$" over ten pairs | I:260; II:340, II:345 | S2g | 10 pairs incl. T25p↔T8 | R | One of the ten pairs is a same-model pair. "tangent agreement is essentially confined to the lineage" (II:340) means confined to the fixed-weights pair. |
| 28 | Variance share "lowest values are Base at $\sigma=1$, where the pooled first step carries a large seed share"; Fig II-F5 "ZIB's share … pooled" | I:257; II:295, II:302 | R2 E6 | T25p | R | Turbo-25 pooled. Turbo cond-only (T8) is the same model, so the pooled attribution can now be checked within one model (not done here). |
| 29 | σ-axis Z-Image MLP $+,-,+$ "on the archive and on clean explicit-branch captures at CFG 1 and CFG 4"; "For the 25-step Base channel no time-order permutation fits comparably (0 of 1000 in all three captures); Turbo's eight states are too few"; Fig I-F7, Fig II-F8a ("$-49.7\%$ … $-50.5\%$", breakpoints 0.972/0.818) | I:268, I:273; II:355, II:369 | S4a–d, P3, `fig_I_F07…`, `fig_II_F8…` | T8, T25p, T25e4, T25c1 | R | All four curves are Turbo. +−+ (about 50% contraction) holds across 8/25 steps, CFG 1/4 and every capture. On the 25-step schedule it passes the time-order test (0/1000), so the "eight states too few" caveat is answered. The evidence is one model, so "Z-Image MLP" means Turbo MLP only. |
| 30 | Time-averaged MLP depth profile (peak layer 5) | I:268 | P4 | T8, T25p, T25e4 | R | Turbo only. |
| 31 | Layer-axis "$90\%$ and $84\%$ for Z-Image Turbo and Base MLP, and in $67\%$ for Base attention"; "Base MLP in 9 of 25"; Table 3 ZIB rows; "Z-Image Base attention and MLP neither surrogate reproduces"; per-step "6 of 8 in Base attention", "Base MLP (1 of 9)"; Fig I-F8, Fig II-F8b; "(in Z-Image only in Base attention)" | I:33, I:278–296, I:311; II:358, II:361, II:369 | analyzer A2/A4, `perstep_surrogate.py` | T8, T25p | R | Within Turbo, the pattern and its surrogate class change with inference. Attention: one crossing at T8 (5%), +−+− at T25p (67%, beats A and B). MLP: follows from the mean field at T8, beats both at T25p. Pattern class is therefore set by schedule, guidance and capture, not by model. "At a single noise level … every image and video architecture" is U. |
| 32 | "middle segment … $77\%$ (Base MLP)"; "Base MLP's first $+\to-$ crossing lies at higher $g^*$ inside its clock-axis contraction window" | I:306 | R8-7 | T25p | R | Turbo-25 MLP. |
| 33 | "crossing count depends on the capture regime … and on sample size"; "(20 versus 480 generations change the crossing count)" | I:306; II:364 | A2 (T25e4/T25c1 20-gen vs T25p 482) | Turbo only | R (valid) | Now a same-model comparison, though prompt sets differ. |
| 34 | ACE confound: "Z-Image Turbo … attention gives a single crossing … so distillation does not remove the sequence everywhere, though it may contribute in attention"; "the distilled Z-Image Turbo attention also gives one crossing, so modality and distillation are not yet separated" | I:456; II:358 | Table 3 | T8, T25p | R (argument changes) | The same distilled Turbo weights give +−+− in attention (67%) at 25 steps, CFG 4, pooled. The single crossing at T8 is an inference-setting effect, so distillation by itself does not produce one crossing. The MLP anisotropy point is U (both Z-Image MLPs are Turbo). |
| 35 | Synthesis 1 "… Mochi and Z-Image Base attention, … never in Turbo attention" | I:461 | as #31 | | R | "Base attention" means Turbo-25 attention. It contradicts "never in Turbo attention" as worded: same model. |
| 36 | Synthesis 3 and 4 "lineage-specific … hierarchy of invariance" | I:463, I:464 | | | V (3) / R (4) | The hierarchy separates fixed-weight pairs (Mochi CFG, Turbo 8/25) from unrelated ones. Nothing in the data bears on distillation. |
| 37 | Seed table "ZIB attn 0.934 … 0.54", "ZIB MLP 0.887 … 0.49"; R8-2 "subsampling the 25--40-step channels to the 8 states … lifts VR4/walk above one in 3 of 5 (to $1.13$--$1.35$)" | I:332–333, I:341; II:443, II:450 | S3, R8-2 | T25p | R (strengthened) | Turbo-25 attention subsampled to 8 states gives 1.13, the same as T8 attention (1.13). This confirms within one model that the outlier is the 8-state grid. |
| 38 | Sync "marginally, Base attention ($p=0.048$)"; Fig I-F12 "(ZIB and SD3.5\,M attention)" | I:377, I:394 | S5 | T25p | R | Turbo-25 attention. |
| 39 | Fig S4 "a peak of 0.64 (ZIB MLP)" | I:574 | | T25p | R | Turbo-25 MLP seed share. |
| 40 | Pooling theorem; SD3.5 M identical-trajectory sizes $+0.040$/$+0.050$/$+0.088$; R8-4 | I:94; II:111–148 | R5 Q1c–f | SD3.5 M | U | Not cited in paper: within-Turbo-25, P1b pooled vs BM (+0.265/+0.282) is now a same-model pooling size. |
| 41 | "the synthetic branch mean equals the per-branch mean to $2\times10^{-16}$" | I:490 | R5 Q1b | T25e4 | U | Instrument check; the model label is irrelevant. |
| 42 | φ "residuals reproduce between Base and Turbo"; arc length "changes Base$\leftrightarrow$Turbo by $-0.004$" | II:385, II:510, II:542 | ledger P9, D17 | T25p, T8 | R | Same model under changed inference. |
| 43 | Universality item 3; Fig II-F12 distillation column ("0.94–0.99", "0.56–0.62", "ZIT, ZIB", "90% / 84%", "MLP .002") and caption "preserved only under fixed weights and distillation" | II:461, II:470 | `fig_II_F12_claim_boundary.py` | Turbo | V / R | The distillation column has no data. Its entries belong in the fixed-weights column, so the tangent is preserved under fixed weights only. |
| 44 | Remark on rotations "such as Base$\leftrightarrow$Turbo's $0.94$--$0.99$"; Jacobian test "Whether Base and Turbo share such a spectrum" | II:223, II:478 | | | R / V | Same model, so the orthogonal-correspondence question is trivially satisfiable. The Jacobian lineage test is V. |
| 45 | Limitations "recomputes the lineage tangent"; I:56/I:478 general | I:477 | | | V (framing) | |
| 46 | Prereg scorecards (R3 B8a/B8b/B8c, R4 P1b/P2a–c, R5 Q2a–c, R6 S1c/S1d/S2a/S2c/S2e/S2g/S4a–d) | I:139, I:500–505; II:521–525 | round results | Turbo | R | Verdicts stand as computed. The rows phrased as lineage or distillation (S1c, S2a, S2c, S2e, B8c) now test fixed-weights pairs, and B8c's "distillation, not guidance" premise is V. |
| 47 | Unrelated-pair results (all ≤0.24 tangent, Mantel 0.13–0.43, unrelated best 0.981), SD3.5 M↔L, Mochi CFG pair, U-Net, ACE-Step, Qwen, Mistral, SD3.5 M resolution and pooling, seed estimators on non-Z-Image channels | throughout | | | U | Pairs that include "ZIB" are Turbo↔other model, so they remain unrelated pairs. |

Counts (47 rows): V only 4 (#1, 8, 17, 45). V as lineage / R as fixed-weights 8 (#2, 3, 14, 25, 26, 36, 43, 44). R 30 (incl. #46, whose lineage-worded rows are V in part). U 5 (#21, 22, 40, 41, 47).

## No longer supported (stand only as fixed-weights statements)
- Every lineage, inheritance or distillation result: the margin (+0.703), "tangents separate the lineage", lineage RSA/theme strength, D03/D15/D10 lineage transfer, MLP eigenmode closure as lineage, and the distillation intervention row. The archive has no lineage pair. These numbers describe Turbo vs itself.
- The paper's central contrast, "inherited vs recurring". What remains is "fixed weights vs unrelated architecture" (Mochi CFG 1↔4.5 and Turbo 8↔25 steps / CFG / capture).
- Any statement that Base and Turbo differ or agree: "Base attention" vs "Turbo attention" patterns, the ACE distillation-confound argument, Base sampling shift.

## What the Turbo data now show (R, worth keeping)
- **Step-count invariance.** Turbo at 25 vs 8 steps gives envelope r 0.9998, shape 0.9997 and tangent 0.999. This far exceeds the 4-generation 100-step control.
- **Within-model capture/guidance decomposition of the tangent:** about 0.001 (steps), 0.007 (CFG 4), 0.025–0.049 (pooling). Pooling, not the model, explains the archive's 0.94.
- **Envelope ordering against Mochi flips with inference settings of one model** (S1d). This is a cleaner capture-regime worked example.
- **σ-axis MLP +−+ holds for Turbo** at 8 and 25 steps, CFG 1 and 4, every capture. It passes time-order permutation on 25 steps.
- **Layer-axis +−+− and its surrogate class change with the inference setting of one model** (attention 5% → 67%; MLP mean-field → beats both).
- **The VR(4) outlier is the 8-state grid within one model** (T25p subsampled 1.13 = T8 1.13).
