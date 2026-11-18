# Literature review: CoS-PAFS Mode-I (empirical) and Mode-II (mathematical)

Zac Doyle (mode:LIMINAL). Compiled 2026-10-08 for the two arXiv papers on layerwise spectral telemetry of diffusion transformers during generation.

**Files.** `references_master.bib` (129 entries), `mode-I/references.bib` (110), `mode-II/references.bib` (107), `CITATION_KEYS.md` (key, use, verification URL). Keys are given below in backticks.

**Verification.** Every .bib entry was checked during this pass against a primary or index page: an arXiv abstract, a proceedings or publisher page, a Crossref or OpenAlex DOI record, an official model card or repository. Where a page from LIT_R4/R5/R6C was reused, `CITATION_KEYS.md` says so. Fields that could not be checked were left out, for example page ranges for Efron 1979 and Carlen 2010. All three .bib files compile under `plain` and `plainnat` with BibTeX 0.99d and give no warnings.

**Project rule.** External work positions, delimits or motivates. It never upgrades a blocked claim: commitment/separatrix (G3), a universal microscopic law (T5), RG universality or universality classes (T6), geometry to quality (D14), or a context-invariant AR phase (G4). Each section below says what a citation **may** and **may not** be used for.

---

## Part A. Mode-I: the empirical paper

### A1. Object, models and instrument (Mode-I §1–2)

**What prior work establishes.** The studied systems are diffusion transformers. DiT `peebles2023scalable` replaced the U-Net with a transformer acting on latent patches. SD3 `esser2024scaling` introduced the two-stream MMDiT, trained with rectified flow `liu2023flow` / flow matching `lipman2023flow`. The diffusion background is `ho2020denoising`, `song2021scorebased` and `karras2022elucidating`, and the sublayer vocabulary comes from `vaswani2017attention`.

The concrete systems are documented as follows:
- **Z-Image.** The technical report `zimageteam2025zimage` describes the single-stream S3-DiT with a Qwen3-4B text encoder (`yang2025qwen3`). Turbo is a few-step distillation of the SFT model plus reward post-training, built on Decoupled-DMD `liu2025decoupled` and DMDR `jiang2025distribution`.
- **Z-Image cards.** The Turbo card `tongyi2025zimageturbo` gives 8 NFEs and guidance 0. The Base card `tongyi2026zimage` describes an undistilled model run with CFG 3–5 for 28–50 steps.
- **Mochi 1.** `genmo2024mochi`: a 10B AsymmDiT (48 layers, visual width 3072 and text width 1536) with a single T5-XXL encoder (`raffel2020exploring`).
- **SD3.5.** The Medium card `stabilityai2024sd35medium` (MMDiT-X) and the Large card `stabilityai2024sd35large` (MMDiT with QK-norm); both use CLIP-L, OpenCLIP-bigG and T5-XXL.
- **LLM controls.** `qwen2024qwen25`, `jiang2023mistral`, and `arditi2024refusal` for the abliterated Mistral. The Llama-3.1 pair is blocked, so `grattafiori2024llama` belongs only in Limitations.
- **Capture runtime and branches.** Captures ran in ComfyUI `comfyui2023`. CFG `ho2022classifierfree` defines the cond/uncond branches, and these define the pooled, cond-only and branch-mean capture regimes.

**How this project differs.** None of these sources measures layerwise spectral statistics of the hidden states during generation. The project's observable is the spectrum of the **uncentred** second moment XᵀX/N, read out per layer, per step and per submodule. From it the project takes three numbers: g (amplitude participation ratio divided by D), the gap s1/s2, and H (the entropy of s/Σs).

**May use for.** Model descriptions, inference settings and lineage wording: "same family; Turbo is a Decoupled-DMD (+reward) distillation of a Z-Image SFT model".

**May not use for.** Claiming that the released Base checkpoint is Turbo's exact teacher. No page states it, and Turbo was released first.

**Corrections to carry into the drafts.**
- SD3.5 Medium size: the card says 2B and the Stability blog says 2.5B. Cite the card.
- ComfyUI's SD3/Z-Image sampling shift is a fixed 3.0. The official Base scheduler uses 6.0 (claim register M4).
- Encoder sharing should enter the RSA discussion in three tiers: identical Qwen3-4B (ZIT/ZIB), the full encoder set (SD3.5 M/L), and a partial T5 share (Mochi/SD3.5). This tiering is from LIT_R6C §1.

### A2. Low-dimensional organized trajectories (§3)

**What prior work establishes.**
- **Dimensionality measures.** Effective dimensionality is standard in neuroscience as an eigenvalue participation ratio: `gao2017theory`, `gao2015simplicity`, `litwinkumar2017optimal`. It has a scale-dependent generalization (`recanatesi2022scaledependent`) and is applied along depth in deep nets (`recanatesi2019dimensionality`).
- **Nonlinear intrinsic dimension along depth.** ID rises and then falls across layers (`ansuini2019intrinsic`). Similar depth profiles recur across very different transformers (`valeriani2023geometry`, `cheng2025emergence`).
- **Finite-sample bias.** The naive PR is biased at small n (`chun2026estimating`).
- **Diffusion sampling paths.** ODE sampling trajectories share a low-dimensional, content-independent shape (`chen2024trajectory`).

**How this project differs.**
- g is an **amplitude** PR, (Σs)²/Σs² divided by D, so it matches the Roy–Vetterli normalization `roy2007effective` rather than the eigenvalue PR.
- H is log erank (`roy2007effective`; `garrido2023rankme` is the deep-learning precedent; `shannon1948mathematical` for entropy).
- The PC1 / effective-rank result concerns the measured **spectral trajectory**, not the latent path or the full hidden state.

**May use for.** Defining and naming the observables, and noting that g, H and the gap are three functionals of one spectrum. Also for stating that low-dimensional paths are expected (`chen2024trajectory`).

**May not use for.** Calling g "effective rank". Implying that low dimensionality of the telemetry means low dimensionality of the network dynamics. Treating a recurring depth profile as evidence of shared computation: `valeriani2023geometry` shows the shape is generic.

### A3. Similarity is observable-dependent (§4)

**What prior work establishes.**
- **Measures.** CKA `kornblith2019similarity`, SVCCA `raghu2017svcca`, PWCCA `morcos2018insights`, and the Procrustes-type shape metrics of `williams2021generalized`. `klabunde2025similarity` surveys their invariances.
- **Pitfalls.** Different measures disagree and need statistical grounding (`ding2021grounding`), and CKA is fragile (`davari2023reliability`).
- **Trending series.** Independent trending series correlate spuriously (`yule1926why`, `granger1974spurious`).
- **Summary statistics.** Identical summary statistics can hide different structure (`matejka2017same`).

**How this project differs.** The project compares the same models at four nested levels: the scalar envelope, the registered field, the registered shape, and the tangent (depth derivative). It shows the ranking of model pairs changes with the level. On the scalar curve ZIT↔SD3.5 L (0.981) beats ZIT↔ZIT-25 (0.933; the same Turbo weights, archive files labelled ZIB). On the tangent, the same weights reach 0.94–0.999, the real distillation lineages 0.71/0.72 (Z-Image) and 0.85/0.57 (ACE-Step; the MLP value is carried by the last layers), and unrelated pairs ≤ 0.27. Closest prior art for lineage from representations: REEF (`zhang2024reef`), independence tests (`zhu2025independence`), trajectory fingerprints (`tang2026representations`) and distillation detection (`shi2025knowledge`); see CITATION_KEYS.md.

**Contemporary context.** MTA `chi2026mta` aligns depth trajectories in LLM distillation and treats them as distinct from fixed-layer similarity.

**May use for.** Motivating an observable hierarchy, and explaining why scalar-envelope correlations are uninformative.

**May not use for.** Using MTA or CKA results as evidence that the Base↔Turbo tangent match reflects a shared mechanism. MTA is LLM distillation with an alignment loss, not passive telemetry.

### A4. Content geometry (§5)

**What prior work establishes.**
- **RSA and the Mantel test.** RSA compares representational dissimilarity matrices, so systems of different width can be compared (`kriegeskorte2008representational`). The Mantel test is a permutation test on distance matrices (`mantel1967detection`).
- **Caveats on Mantel tests.** They test a hypothesis about distances (`legendre2010comparison`). They are biased under autocorrelation, and partial Mantel does not repair this (`guillot2013dismantling`).
- **Input structure.** The structure of the input population inflates RSA/CKA even for random networks (`cui2022deconfounded`).
- **Convergence across models.** The Platonic hypothesis `huh2024platonic` predicts convergence across models. `groger2026revisiting` shows that after permutation calibration, global convergence largely disappears while local-neighbourhood agreement survives.
- **Diffusion-specific ID.** Intrinsic dimension in Stable Diffusion depends on prompt and layer (`kvinge2023exploring`).
- **Rank correlation.** `spearman1904proof` and `kendall1938new`.

**How this project differs.** The project uses within-theme permutations and a text-only (TF-IDF) partial control, which is the remedy `cui2022deconfounded` proposes. Under these controls it finds prompt-relational structure shared across five diffusion models (Mantel ρ 0.13–0.43 for unrelated pairs, 0.56–0.62 for Base↔Turbo). For decoding, the ordered depth profile beats layer-averaged scalars and the prompt text.

**May use for.** Methods (RSA, Mantel, Spearman), and the reasons for the within-theme and text-partial controls. Also for positioning the cross-model result relative to the Platonic hypothesis, with Gröger et al.'s calibration as the caution.

**May not use for.** Saying the Platonic hypothesis is "confirmed". Treating a partial-Mantel p-value as decisive. Treating encoder sharing as ruled out: Mochi and SD3.5 share T5.

### A5. Phase and topology observations (§6)

**What prior work establishes.** Theory finds symmetry breaking and speciation in reverse diffusion:
- `biroli2023generative` and `biroli2024dynamical`;
- `raya2023spontaneous`: an early linear phase, then attractor dynamics;
- `sclocchi2025phase` and `sclocchi2025probing`: class features flip at a threshold while low-level features change smoothly;
- `li2024critical`: critical windows.

All of these describe the **data distribution, the score or the outputs**, not activation statistics. An interior peak in a representation statistic versus noise can arise without any phase transition (`li2025understanding`). In DiTs, depth and timestep roles interact (`chen2024deltadit`), and massive activations are modulated by the timestep embedding (`gan2025massive`), so a σ-locked envelope is expected from the architecture alone.

**How this project differs.** It keeps two constructions apart:
- **Along σ:** in Z-Image MLP, the layer-mean g goes + − +.
- **Along depth:** the analyzer's β_layer shows a + − + − sign topology at fixed σ. It is reported with bootstrap incidence (`efron1979bootstrap`) and tested against marginal and mean-field surrogates.

The post hoc depth profile uses a piecewise-linear fit (`muggeo2003estimating`).

**May use for.** Positioning, plus the statement that output-level transitions exist in theory and in other measurements.

**May not use for.** Calling the sign changes a "phase transition", "critical point", "fixed point" or "universality class". Equating any crossing with speciation time. SD3.5 L is null-generic, and crossing ratios are not invariants.

### A6. Seed persistence without commitment language (§7)

**What prior work establishes.**
- **Clone-based commitment.** Commitment has been measured with clones or forward–backward continuations (`biroli2024dynamical`, `sclocchi2025probing`, `li2024critical`), and recently through class-conditional entropy (`handke2026entropic`, on EDM2-XS and SD1.5).
- **Distilled students.** They fix structure at the first step (`gandikota2026distilling`, `wu2026diversity`).
- **Initial noise.** Initial-noise patches anticipate object positions (`ban2025crystal`).
- **The committor.** Formally it is defined by transition path theory (`e2006towards`, `e2010transitionpath`) and estimated by shooting many continuations (`bolhuis2002transition`).
- **Variance ratio.** The test is `lo1988stock`. Its finite-sample size and power are in `lo1989size`, and its downward small-sample bias under the null is in `kim1991mean`.
- **Cross-validation.** Error bars are large at n ≈ 100 (`varoquaux2018cross`).

**How this project differs.** Its seed observables are passive predictability from telemetry. Raw σ50 is sample-size and fold sensitive. The relative onset at matched n is more stable. VR4 is calibrated against a simulated walk, AR(0.8) and iid at matched length.

**May use for.** Explaining why sample size must be matched and why VR needs calibration. Also for stating that a clone/committor experiment (G3) is the right test and is not done here.

**May not use for.** Any "commitment time", "fate boundary" or "separatrix" wording. Reading Gandikota & Bau or Handke et al. as evidence that our telemetry measured commitment.

### A7. Intervention hierarchy and distillation (§8)

**What prior work establishes.**
- **Distillation families.** Knowledge distillation (`hinton2015distilling`); diffusion distillation families: progressive (`salimans2022progressive`), consistency (`song2023consistency`), DMD/DMD2 (`yin2024onestep`, `yin2024improved`), and adversarial (`sauer2024adversarial`, `sauer2024fast`).
- **Students keep teacher structure.** Distilled diffusion students keep the base model's concept directions (`gandikota2026distilling`) and SAE features (`surkov2025onestep`), and DMD students copy the teacher's noise-to-data pairings (`li2026why`, a non-archival workshop poster).
- **Counterpoint.** LLM students can restructure their internal circuits (`haskins2025distilled`).
- **Guidance and resolution.** Guidance acts mostly mid-chain (`kynkaanniemi2024applying`). Resolution changes effective SNR at a fixed nominal σ (`hoogeboom2023simple`, `esser2024scaling` §5.3.2).

**How this project differs.** The prior evidence is about features and outputs. This project measures the **tangent of layerwise spectral geometry at equal σ**. It finds that Base↔Turbo preserves tangents far more than any unrelated pair, while scaling (Qwen sizes, SD3.5 M↔L) does not preserve microscopic dynamics.

**May use for.** "A strong Base↔Turbo correspondence is consistent with prior evidence that DMD-family students retain teacher structure."

**May not use for.**
- Calling the inheritance surprising or novel in kind.
- Attributing differences to distillation alone: the RL stage and a possible checkpoint mismatch are confounds.
- Claiming that the Mochi CFG or SD3.5 resolution results isolate guidance or resolution: the arms also differ in steps and sample design.

### A8. Falsification results (§9)

**What prior work establishes.**
- **Surrogate testing.** `theiler1992testing`, `schreiber1996improved`, `schreiber2000surrogate`, `lancaster2018surrogate`. Phase-randomized and circular-shift surrogates assume stationarity.
- **Trend-preserving surrogates.** `nakamura2005small`.
- **Slow drift.** Under slow drift, circular-shift and phase nulls still give false positives, and permuting replication units works (`harris2020nonsense`).
- **Multiple comparisons.** Max-statistic permutation control (`westfall1993resampling`, `nichols2002nonparametric`).
- **Forking paths and analytic flexibility.** `gelman2014statistical`, `simmons2011false`; preregistration practice in `nosek2018preregistration`.

**How this project differs.** The synchronization peak R reproduces numerically. It fails the trend/warp null as a universal event, because a straight line beats it in 7/7 channels. The Qwen β ladder is demoted because v1 LLM telemetry is a nested running statistic.

**May use for.** Justifying the nulls and the preregistration and correction trail, which the paper presents as a contribution.

**May not use for.** Claiming that the frozen preregistration covers analyses added after the data were seen. Label those post hoc.

### A9. Limitations and future tests (§11)

**Massive activations.**
- **LLMs.** A few huge, input-independent activations act as biases (`sun2024massive`). They co-occur with attention sinks (`xiao2024efficient`) and resemble high-norm register tokens (`darcet2024vision`). They give near-constant representations across layers (`sun2026anatomy`).
- **DiTs.** Massive activations sit in fixed channels and are AdaLN/timestep-linked (`gan2025unleashing`, `gan2025massive`). They are also a sparse, prompt-conditioned carrier whose removal collapses generation (`turri2026few`).

Because the project's matrix is uncentred, s1 and the gap may be dominated by these channels. This stays a caveat (claim register T7) until a centering or channel-removal control is run.

**G3.** `e2006towards`, `bolhuis2002transition` and `handke2026entropic` motivate the clone experiment.

**G4.** Context-invariant AR telemetry; no external citation upgrades it.

---

## Part B. Mode-II: the mathematical paper

### B1. Problem statement and observables (§1–2)

**Prior work.**
- **Depth as time.** `e2017proposal`, `lu2018beyond`, `chen2018neural`, `lu2019understanding`, and the interacting-particle view of transformers in `geshkovski2025mathematical`.
- **Layerwise readouts.** `belrose2023eliciting` and `nostalgebraist2020interpreting`.
- **Equivalence notions.** Conjugacy and orbit equivalence are textbook notions (`katok1995introduction`).

**Difference.** The state is z = (g, gap, H), observed through a fixed spectral projection of the hidden state. Two clocks are kept separate: depth u and the sampling clock τ = σ.

**May use for.** Defining the framework, and stating precisely the equivalences that are **not** claimed: full-state conjugacy and orbit equivalence (`katok1995introduction`).

**May not use for.** Importing particle-system clustering results (`geshkovski2025mathematical`) as statements about these trained diffusion models.

### B2. The pooling theorem (§3)

**Prior work.**
- **Operator concavity.** t ↦ t^{1/2} is operator monotone and hence operator concave (Löwner–Heinz; `bhatia1997matrix`, Ch. V).
- **Trace concavity.** For a concave f, A ↦ Tr f(A) is concave, so M ↦ tr M^{1/2} is concave on PSD matrices (`carlen2010trace`).
- **Related trace results.** Classic trace-concavity results (`lieb1973convex`, `ando1979concavity`). The concavity of Φ_{p,1}(A₁,…,A_m) = Tr(Σ A_j^p)^{1/p} for 0 < p ≤ 1 (`carlen2008minkowski`) is related but is not the statement used.
- **Homogeneity step.** A degree-1 homogeneous function with convex superlevel sets is concave (`boyd2004convex` background; no exercise number is cited).

**Difference.** The result itself is the project's own: pooled g ≥ the **trace-weighted** mean of the branch g values, with equality when A ∝ B. LIT_R5 found 4,232 of 20,000 random cases where pooled g fell below the **unweighted** branch mean.

**May use for.** The proof ingredients.

**May not use for.** Stating "pooled ≥ unweighted branch mean". Claiming any guarantee for s1/s2.

### B3. Projection degeneracy and the equivalence hierarchy (§4)

**Prior work.**
- Spurious correlation of trending series (`yule1926why`, `granger1974spurious`).
- Identical summary statistics with different structure (`matejka2017same`).
- Metric families and their invariances (`williams2021generalized`, `kornblith2019similarity`, `klabunde2025similarity`).

**Difference.** The paper proves that a high corr(PΓ_a, PΓ_b) cannot identify a shared vector field when all projected envelopes share a monotone trend. It anchors this on data: the scalar lineage margin is below 0, and the field, shape and tangent margins rise monotonically.

**May use for.** Motivating the hierarchy envelope < field < shape < tangent.

**May not use for.** Presenting the hierarchy as a known ordering of these metric families. It is defined here.

### B4. Nonautonomous flow family (§5)

**Prior work.** The framework for explicitly time-dependent flows is `kloeden2011nonautonomous`, with the depth-as-time views listed in B1.

**Difference.** Here the explicitly time-dependent decomposition dz/du = F⁽⁰⁾_m(u, τ) + R_m(z, u, τ) is fitted to data. Cross-architecture transfer fails (D03, D15).

**May use for.** Terminology: process, nonautonomous, pullback.

**May not use for.** Any "universal F" claim (T5 is blocked).

**Added 2026-10-09 (positioning, Zac's request 22:24Z).** `clark2026lyapunov` derives the full Lyapunov spectrum of the random tanh recurrent network at large N and stresses that participation-ratio dimensions depend on the choice of variables while Lyapunov-derived dimensions do not (its Fig. 2: PR rises with coupling, attractor dimension saturates then falls). `shen2025covariance` and `clark2026linear` give the linear-equivalence result: a chaotic nonlinear random network and a stable noise-driven linear network with the same couplings share the covariance matrix. **May use for:** the dimension-measures remark (§2), the state-vs-dynamics proposition (§7) as a theoretical instance of aligned second moments with different stability, and the open-mathematics item naming a finite-depth Jacobian amplification spectrum as the next level (not Lyapunov exponents: no invariant measure in a finite non-autonomous composition). **May not use for:** criticality, edge-of-chaos or separatrix claims; the models studied are untrained autonomous recurrent networks, not transformers.

### B5. Trajectory family, inheritance and the content quotient (§6–8)

**Prior work.**
- **Low-dimensional sampling paths.** `chen2024trajectory`.
- **Distilled students keep structure.** `gandikota2026distilling`, `surkov2025onestep`, `li2026why`.
- **Trajectory alignment as its own object.** `chi2026mta`.
- **RSA and the calibration critique.** `kriegeskorte2008representational`, `mantel1967detection`, `huh2024platonic`, `groger2026revisiting`, with `spearman1904proof` for the rank coupling.

**Difference.** Equal-σ registration rather than a learned clock. The tangent cosine is the lineage statistic. D10 shows that state alignment can raise the similarity of unrelated representations without aligning their derivatives. Content geometry is weakly rank-coupled to tangent similarity (ρ ≈ 0.297), so these are distinct equivalence relations.

**May use for.** "Partial orientation-preserving trajectory correspondence in the measured coordinates."

**May not use for.** "Orbit equivalence". "Shared mechanism".

### B6. Phase constructions and kinematic nulls (§9–10)

**Prior work.** Surrogate theory (`theiler1992testing`, `schreiber1996improved`, `schreiber2000surrogate`, `lancaster2018surrogate`, `nakamura2005small`, `harris2020nonsense`). Bootstrap (`efron1979bootstrap`). The diffusion phase theory listed in A5.

**Difference.** The E[φ] → −1/2 result for adjacent differences and the stride-2 expectation of 0 are elementary derivations made here. The cloud-β surrogate A/B logic is the project's own.

**May use for.** Explaining why a null-generic crossing is not a phase boundary.

**May not use for.** Linking crossings to speciation or RG fixed points (T6).

### B7. Local linearization (§11)

**Prior work.** DMD (`schmid2010dynamic`, `kutz2016dynamic`). EDMD as a Koopman approximation (`williams2015datadriven`). Linearization of trained network dynamics around slow points (`sussillo2013opening`).

**Difference.** The fit is a local affine model X_{u+du} = A X_u + b along depth on the telemetry coordinates. Contraction moduli are more stable than rotations. The geometry-derived Base→Turbo map closes the MLP local eigenmodes but not the attention ones.

**May use for.** Methodology and terminology: modes, moduli, complex pairs.

**May not use for.** Koopman-spectrum claims about the full network.

### B8. Estimator theory for seed predictability (§12)

**Prior work.** The VR test (`lo1988stock`). Its finite-sample behaviour (`lo1989size`). Its downward small-sample bias under the null (`kim1991mean`). Exact rank/sign VR tests (`wright2000alternative`). CV error bars at small n (`varoquaux2018cross`). PR bias at small n (`chun2026estimating`).

**Difference.** Threshold-crossing σ50 depends on the ceiling and the sample size. The paper defines a relative onset and calibrates VR(q) against finite-length walk, AR and iid references.

**May use for.** Estimator calibration.

**May not use for.** Commitment claims. G3 needs committor estimation (`e2006towards`, `bolhuis2002transition`).

### B9. Kinds of universality and open causal mathematics (§13–14)

**Prior work.**
- **Generic structure.** The coarse depth structure recurs generically (`valeriani2023geometry`, `cheng2025emergence`).
- **Calibrated convergence.** Cross-model convergence is mostly local once calibrated (`groger2026revisiting`).
- **Massive activations.** They can dominate uncentred statistics (`sun2024massive`, `gan2025massive`, `sun2026anatomy`, `turri2026few`).
- **Commitment.** A proper commitment test needs clones (`biroli2024dynamical`, `handke2026entropic`, `e2010transitionpath`).

**May use for.** Stating which coarser universality notions remain possible (ordering/topology, quotient geometry), and listing the conditions a stronger claim would have to meet.

**May not use for.** Any upgrade of T5/T6, G3 or D14.

---

## Part C. The reviewer's five 2026 references (R7_LITERATURE_POSITIONING.md)

| # | As cited by the reviewer | Status | Verified metadata and notes |
|---|---|---|---|
| 1 | Roy & Vetterli (EUSIPCO 2007), "The Effective Rank: A Measure of Effective Dimensionality" | **VERIFIED** | Olivier Roy, Martin Vetterli, 15th EUSIPCO, Poznań, 2007. EPFL Infoscience record 110188, and the EURASIP/Zenodo record (DOI 10.5281/zenodo.40328, published 2007-09-03). Page numbers are not shown on any page reached, so they are omitted. The reviewer's point holds: erank = exp(entropy of normalized singular values), which is distinct from g. Key `roy2007effective`. |
| 2 | Handke et al. (ICML 2026), "The Entropic Signature of Class Speciation in Diffusion Models" | **VERIFIED** | Florian Handke, Dejan Stančević, Felix Koulischer, Thomas Demeester, Luca Ambrogioni. ICML 2026 poster (icml.cc/virtual/2026/poster/61282); arXiv:2602.09651 (2026-02-10). Validated on EDM2-XS and Stable Diffusion 1.5. The reviewer's framing (motivation for G3, not validation) is correct. Key `handke2026entropic`. |
| 3 | Chi et al. (ACL 2026), "MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation" | **VERIFIED** | Pham Khanh Chi, Quoc Phong Dao, Thuat Nguyen, Linh Ngo Van, Trung Le, Thanh Hong Nguyen. Proc. 64th ACL (Vol. 1: Long Papers), San Diego, July 2026, pp. 32669–32684, doi:10.18653/v1/2026.acl-long.1507; arXiv:2605.01374. The ACL Anthology files it under "Chi" (BibTeX key `chi-etal-2026-mta`), so "Chi et al." matches the Anthology. The abstract says existing methods ignore "how representations evolve across depth", which fits the reviewer's description. Key `chi2026mta`. |
| 4 | Sun et al. (ICML 2026), "Anatomy of Massive Activations and Attention Sinks" | **VERIFIED, with a naming caution** | ICML 2026 poster with exactly this title (icml.cc/virtual/2026/poster/63610). Authors: **Shangwen** Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu. The arXiv version (2603.05498) is titled "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks". Do not confuse it with Sun **Mingjie** et al. 2024 (`sun2024massive`). The abstract concerns language models: a few tokens with outliers in a few channels produce near-constant representations across layers, acting as implicit parameters. The reviewer's "few extreme channels with persistent global effects" is a fair paraphrase, but the paper is not about DiTs. Key `sun2026anatomy`. |
| 5 | Turri et al. (arXiv 2026), "Few Channels Draw The Whole Picture: Revealing Massive Activations in Diffusion Transformers" | **VERIFIED** | Evelyn Turri, Davide Bucciarelli, Sara Sarto, Lorenzo Baraldi, Marcella Cornia; arXiv:2605.13974 (v1 2026-05-13, v2 2026-07-07). The abstract names no specific checkpoints. Zeroing the massive channels causes "a sharp collapse in generation quality", and the channels act as "a sparse prompt-conditioned carrier subspace". Note: this means the channels carry prompt information, not only a bias, so a channel-removal control could also remove content signal. Key `turri2026few`. |

None of the five is invented or materially mis-cited. Item 4 needs the author clarification above.

---

## Part D. Unverified or dropped

**Left out of the .bib (could not be verified on a primary or index page in this pass):**
- **Gelman & Loken 2013**, "The garden of forking paths" (unpublished manuscript). The year is not shown on the author page. Cite the 2014 American Scientist piece (`gelman2014statistical`) instead.
- **Efron & Tibshirani 1993**, *An Introduction to the Bootstrap*: not checked. `efron1979bootstrap` is used instead.
- **Koopman 1931** (PNAS): not checked. Koopman theory is cited through `williams2015datadriven`.
- **Albergo & Vanden-Eijnden 2023**, stochastic interpolants: not checked; not needed.
- **Haber & Ruthotto 2017** (Inverse Problems): not checked; `e2017proposal` and `lu2018beyond` cover depth-as-time.
- **Chow & Denning 1993** (J. Econometrics): seen only on a secondary R package page (LIT_R5).
- **Krstajic et al. 2014; Charles & Darné 2009**: unverified in LIT_R5 and not needed.
- **Geissler, Dellago & Chandler 1999; Poterba & Summers 1988**: only partially verified in LIT_R4.
- **Sucholutsky et al. 2023**: verified in LIT_R6C but dropped as non-essential (very long author list).
- **Kang, Trizio & Parrinello 2024** (committor): not needed beyond `e2006towards` and `bolhuis2002transition`.

**Partially verified fields (the entry is kept and the unverified part is left out):**
- **Efron 1979.** Volume and issue verified; the page range is omitted (Crossref gave none).
- **Spearman 1904.** The start page 72 was verified but the end page was not, so the page field is omitted.
- **Carlen 2010.** Volume 529 and the MR number were verified; the page range is omitted.
- **Bhatia 1997.** The chapter is cited, not a theorem number.
- **Boyd & Vandenberghe 2004.** Cited without an exercise number.
- **Mantel 1967.** The AACR article page resolves at vol. 27, issue 2 (Part 1), first page 209, and its abstract matches. Volume, issue and pages 209–220 also come from the PySAL reference list. No DOI is included.
- **Kendall 1938.** Checked on a bibliographic index page that gives the DOI; Crossref returned 429 twice.
- **Katok & Hasselblatt 1995.** Series (EMA 54), publisher and year come from the title of a Cambridge review page seen in search results. The book page itself was not opened.
- **Harris 2020** (bioRxiv): the metadata is reused from LIT_R6C 5.8, because bioRxiv rate-limited this pass.
- **LADD** (`sauer2024fast`): the SIGGRAPH Asia venue is not verified, so it is cited as arXiv.
- **Classifier-free guidance** (`ho2022classifierfree`): the NeurIPS 2021 workshop venue is not verified, so it is cited as arXiv.
- **Model-card years.** Z-Image Base is dated 2026 from the GitHub release date (LIT_R6C 1.4). SD3.5 is dated 2024 from the Stability release post and press coverage dated October 2024.
- **NeurIPS volume numbers** (30–38): these follow the series numbering; the volume was seen directly only for 37 and 38.

**Pages that could not be read in this pass:**
- Europe PMC REST API (HTTP 429; not retried).
- Springer article page for Carlen & Lieb 2008 (429). It was verified through the Princeton record instead.
- One Crossref record (Ando 1979), rate-limited. Verified through OpenAlex instead.
- bioRxiv (429).
- An OpenAlex search query (429).
- ScienceDirect (blocked by robots.txt).
- Wiley, PNAS, CVF and Project Euclid publisher pages (403).
- OpenReview (browser check).
- The GitHub API (403).

**Novelty statement.** Across LIT_R4–R6C and this pass, no paper was found that measures PR, effective rank or the singular-value entropy of DiT hidden states per layer and per sampling step across Z-Image, Mochi and SD3.5, or that compares teacher and student through matched-σ tangents. Phrase this as "to our knowledge".
