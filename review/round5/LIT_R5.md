# LIT_R5: Literature check for the diffusion internal-telemetry review (round 5)

Scope: six questions raised by Round-5 measurements (pooled vs branch-mean g under CFG; resolution dependence at matched σ; where cond/uncond differ; CV fold-partition instability of the fate-onset estimator; small-sample VR bias; shared-noise layout and commitment in flow models). Sources already in `round4/LIT_R4.md` are not repeated unless a new page adds something specific. Tags: `[verified: url]` = a primary or authoritative page was fetched and the claim checked on it; `[unverified: reason]` = not checked on a primary page. Search date: 2026-10-08.

Notation: M = XᵀX/n (uncentred), s = sqrt(eig M), so Σs = tr M^{1/2} = ‖X‖_*/√n and Σs² = tr M = ‖X‖_F²/n. Our g = (tr M^{1/2})² / (D · tr M).

---

## 1. Concavity of trace functions; effective rank of pooled conditions

**Wirth, M. (n.d.). *Trace Inequalities and Quantum Entropies* (lecture notes), Univ. Leipzig.**
Theorem 6.3: "If f is convex, then so is A ↦ Tr f(A)" on self-adjoint matrices, and if f is monotone increasing then so is A ↦ Tr f(A). With f(t) = −√t on t ≥ 0 this gives concavity of A ↦ tr A^{1/2} on PSD matrices, so tr((A+B)/2)^{1/2} ≥ (tr A^{1/2} + tr B^{1/2})/2. The notes do not give equality conditions. [verified: https://www.math.uni-leipzig.de/~wirth/notes_of_trace_inequalities.pdf]

**Wikipedia, "Trace inequality" (secondary; points to Carlen's notes).**
States that Tr f is convex for convex f and "strictly convex if f is strictly convex," citing Carlen's introductory course for proof and discussion. It also discusses the Löwner–Heinz theorem and operator concavity, again citing Carlen. Use it to find the primary citation, not as the citation itself. [verified: https://en.wikipedia.org/wiki/Trace_inequality]

**Carlen, E. A. (2010). Trace inequalities and quantum entropy: an introductory course. In *Entropy and the Quantum*, Contemporary Mathematics 529, AMS.**
This is the standard citable source for convexity of trace functions, including the strict form (strictly concave f gives a strictly concave Tr f, hence equality iff A = B), and for an elementary proof of Löwner–Heinz. [unverified: the PDF host failed (TLS/robots); the volume and number are from memory, and the content was confirmed only through the Wikipedia page above]

**Bhatia, R. (1997). *Matrix Analysis*. Graduate Texts in Mathematics 169, Springer.**
The usual reference for Löwner–Heinz: t^r is operator monotone, and hence operator concave, on [0, ∞) for 0 ≤ r ≤ 1. This gives the stronger Loewner-order statement ((A+B)/2)^{1/2} ⪰ (A^{1/2}+B^{1/2})/2, and taking traces gives the scalar inequality. [unverified: book not fetched; the theorem number is not confirmed, so cite the chapter (V) without a number]

**Carlen, E. A., Lieb, E. H. (2008). A Minkowski type trace inequality and strong subadditivity of quantum entropy. arXiv:math/0701352.**
Proves joint concavity of Φ_p(A₁,…,A_n) = Tr(Σ_j A_j^p)^{1/p} for 0 < p ≤ 1. It is related to our inequality but is not the statement we need. Cite Carlen 2010 or Bhatia instead. [verified: https://arxiv.org/abs/math/0701352]

**Roy, O., Vetterli, M. (2007). The effective rank: a measure of effective dimensionality. EUSIPCO 2007, Poznań.**
Defines erank(A) = exp(H(p)) with p_k = σ_k/Σσ_k. This is exactly exp of our "entropy of s/Σs" when applied to X/√n. Properties: 1 ≤ erank ≤ rank, invariance to scale and to unitary transformations, and **subadditivity for PSD sums: erank(A+B) ≤ erank(A) + erank(B)**. That subadditivity is the only mixture/pooling property it states, an upper bound with no lower bound. [verified: https://infoscience.epfl.ch/record/111484?ln=en (record and venue); https://infoscience.epfl.ch/record/111484/files/Roy07.pdf?version=1 (slides with the definitions and properties)]

**Own derivation (not literature; numerically checked).** Proof of the requested statement with equality: by Löwner–Heinz, P := ((A+B)/2)^{1/2} − (A^{1/2}+B^{1/2})/2 ⪰ 0. If the traces are equal, tr P = 0, so P = 0. Squaring gives (A+B)/2 = (A + B + A^{1/2}B^{1/2} + B^{1/2}A^{1/2})/4, which is (A^{1/2} − B^{1/2})² = 0, so A = B. **The statement about g is weaker than "pooled ≥ branch-mean":** N(M) := (tr M^{1/2})² is positively homogeneous of degree 1 and has convex superlevel sets, so it is concave. Hence
  g_pooled ≥ (tr A · g_A + tr B · g_B) / (tr A + tr B), the **trace-(energy)-weighted** mean of the branch g values,
with equality when A ∝ B. It does **not** imply g_pooled ≥ (g_A + g_B)/2. In 20,000 random PSD pairs (D = 2–11, unequal energies) there were 0 violations of the tr-sqrt inequality and 0 of the energy-weighted bound, but 4,232 (21%) cases where pooled g fell below the unweighted branch-mean. The gap from the unweighted mean is a covariance term ∝ (tr A − tr B)(g_A − g_B). s1/s2 has no such guarantee. Unchanged s1/s2 under pooling means the top two directions are shared, which says nothing about the spectral tail.

---

## 2. Resolution dependence of the noise schedule at matched σ; ComfyUI shift

**Esser, P., et al. (2024). Scaling rectified flow transformers for high-resolution image synthesis (SD3). ICML 2024. arXiv:2403.03206.**
§5.3.2 "Resolution-dependent shifting of timestep schedules": treating a constant image as n observations of one value, the uncertainty is σ(t, n) = (t/(1−t))·√(1/n), so "doubling the width and height leads to half the uncertainty at any given time". Matching σ(t_n, n) = σ(t_m, m) gives t_m = α t_n / (1 + (α−1) t_n) with α = √(m/n) (Eq. 23), where m and n are pixel or token counts. SD3 uses α = **3.0** for training and sampling at 1024², chosen by a human preference study ("strong preference for samples with shifts greater than 1.5"). The constant-image model is called "not realistic". It is a heuristic. [verified: https://arxiv.org/pdf/2403.03206 (§5.3.2)]

**Hoogeboom, E., Heek, J., Salimans, T. (2023). Simple diffusion: end-to-end diffusion for high resolution images. ICML 2023 (PMLR 202). arXiv:2301.11093.**
"For high resolutions not enough noise is added". Average-pooling an s×s window multiplies α_t/σ_t by s, "and thus the SNR by s²". They propose a shifted schedule SNR^{d×d}(t) = SNR^{64×64}(t)·(64/d)², a constant shift of 2·log(64/d) in log-SNR. This is the cleanest statement that **the same nominal noise level is a different effective (low-frequency) SNR at a different resolution.** [verified: https://arxiv.org/html/2301.11093v2]

**Chen, T. (2023). On the importance of noise scheduling for diffusion models. arXiv:2301.10972.**
"The optimal one depends on the task (e.g., image sizes)". For larger images "the optimal noise scheduling shifts towards a noisier one", attributed to redundancy between neighbouring pixels. Scaling the input by b is equivalent to shifting log-SNR by log b. [verified: https://arxiv.org/abs/2301.10972v3 (no peer-reviewed venue listed)]

**ComfyUI source, `comfy_extras/nodes_model_advanced.py` (master).**
`ModelSamplingSD3` takes only `model` and `shift` (default 3.0, range 0–100) and calls `set_parameters(shift=shift, multiplier=1000)`. It **reads no width or height.** `ModelSamplingFlux`, by contrast, computes shift linearly in token count, `(width*height/(8*8*2*2))*mm + b`, between base_shift 0.5 at 256 tokens and max_shift 1.15 at 4096. [verified: https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_model_advanced.py]

**ComfyUI source, `comfy/model_sampling.py` (master).**
`time_snr_shift(alpha, t) = alpha*t/(1+(alpha-1)*t)` (the identity if alpha = 1). `ModelSamplingDiscreteFlow.sigma(timestep) = time_snr_shift(self.shift, timestep/self.multiplier)`. This is the same functional form as SD3 Eq. 23, applied with a **fixed** α. [verified: https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_sampling.py]

**ComfyUI source, `comfy/supported_models.py` (master).**
Model-class defaults when no ModelSampling node is used: `SD3` has `shift: 3.0`; there is no separate SD35 class, so SD3.5-M loads as SD3. `ZImage(Lumina2)` has `multiplier: 1.0, shift: 3.0`, while `Lumina2` itself has shift 6.0. None of these is resolution-dependent. [verified: https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/supported_models.py]

**ComfyUI docs, ModelSamplingSD3 node.** Default shift 3.0 "together with a fixed internal multiplier of 1000". The docs say nothing about resolution. [verified: https://docs.comfy.org/built-in-nodes/ModelSamplingSD3.md]

**Tongyi-MAI, Z-Image and Z-Image-Turbo diffusers scheduler configs (Hugging Face).**
Both use `FlowMatchEulerDiscreteScheduler` with `use_dynamic_shifting: false`. **Z-Image (Base): shift 6.0; Z-Image-Turbo: shift 3.0.** ComfyUI's ZImage class default (3.0) therefore matches Turbo but **not** the official Base config, unless the workflow sets shift explicitly. [verified: https://huggingface.co/Tongyi-MAI/Z-Image/raw/main/scheduler/scheduler_config.json ; https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/raw/main/scheduler/scheduler_config.json]

**Stability AI, SD3.5-Medium diffusers scheduler config.** [unverified: gated repo, HTTP 401. The ComfyUI default of 3.0 is verified; the official diffusers value was not checked]

Implication (from SD3 Eq. 23 composed with the shift; our arithmetic). Shifts compose multiplicatively in noise odds t/(1−t). A 512² run at shift 3 has 4× fewer tokens than 1024², so α = 2 between the resolutions, and the run is equivalent in SD3's heuristic to a 1024² run at **shift ≈ 6**. At matched σ, a 512² state corresponds to a 1024² state with noise odds ×2. "Matched σ" across 512 and 1024 is therefore not matched effective SNR.

---

## 3. Where cond and uncond differ, and where guidance acts (beyond Kynkäänniemi, CADS, Wang in LIT_R4)

**Zhu, S., Peng, Q., Hu, Y., et al. (2025–26). RAAG: Ratio Aware Adaptive Guidance; retitled in v4 "MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance". arXiv:2508.03442.**
Defines RATIO = ‖v_c − v_u‖₂ / ‖v_u‖₂ on velocity predictions in **rectified-flow models including SD3.5**, Lumina-Next, Qwen-Image and WAN2.1. RATIO "starts high", with a sharp spike at the first steps, and "rapidly decays within a few steps and stabilizes". The earliest steps are "strikingly sensitive to the guidance scale". They argue the spike is "a dataset-level property rather than models", and say it does not hold as cleanly for ε-diffusion models. Their initial-RATIO values are internally inconsistent: 0.1–0.3 in the text vs ~0.5 in a figure. This is the **closest published match to our "cond/uncond differ most at the first steps"**, measured on SD3.5. [verified: https://arxiv.org/html/2508.03442v2 ; https://arxiv.org/abs/2508.03442 (no venue)]

**Gao, J., Zheng, T., Zou, J., et al. (2026). C²FG: Control classifier-free guidance via score discrepancy analysis. CVPR 2026, pp. 34398–34407. arXiv:2603.08155.**
Bounds the conditional–unconditional **score** discrepancy for VP/VE SDEs. It is "strictly monotonically decreasing in the forward process", so in sampling the scores are similar at high t and diverge toward t → 0, with an MSE plot growing toward t = 0. **Apparent conflict with RAAG and with our observation**: it is in the score parameterization, where a difference in x₀-predictions is divided by σ_t, so it blows up as σ → 0 by construction. Velocity or hidden-state differences need not follow it. The bounds are singular at t → 0. No flow-matching or SD3 measurement. [verified: https://openaccess.thecvf.com/content/CVPR2026/html/Gao_C2FG_Control_Classifier-Free_Guidance_via_Score_Discrepancy_Analysis_CVPR_2026_paper.html ; https://arxiv.org/html/2603.08155v1]

**Jin, C., Shi, Q., Gu, Y. (2025–26). Stage-wise dynamics of classifier-free guidance in diffusion models. arXiv:2509.22007 (listed as an ICLR 2026 poster).**
Three stages: (i) Direction Shift (early), where "guidance accelerates movement toward the weighted mean"; (ii) Mode Separation, where "the inherited bias suppresses weaker modes", reducing global diversity; (iii) Concentration (late), where "guidance amplifies within-mode contraction, diminishing fine-grained variability". Early strong guidance erodes global diversity, and late strong guidance suppresses fine-grained variation. **This reconciles our late contraction with the early-guidance literature: late contraction of across-seed spread may be within-mode contraction (stage iii), not mode loss.** It is theory in a mixture model plus experiments, with no cond/uncond gap measurement. [verified: https://arxiv.org/abs/2509.22007 ; ICLR 2026 venue only from the search listing https://iclr.cc/virtual/2026/poster/10008240, not fetched]

**Bradley, A., Nakkiran, P. (2025). Classifier-free guidance is a predictor-corrector. TMLR 2025. arXiv:2408.09000.**
In the SDE limit, CFG equals a DDIM predictor for the conditional distribution plus a Langevin corrector on a γ-powered distribution. CFG does **not** sample the γ-powered (sharpened) distribution. "The DDPM and DDIM variants of CFG can generate different distributions", and "the DDIM variant of CFG is exponentially sharper". Our ComfyUI Euler-ODE sampling is the deterministic, sharper variant. It gives no time-resolved cond/uncond analysis beyond noting that the guided and true conditionals differ at high noise. [verified: https://arxiv.org/html/2408.09000v2 ; venue https://mlanthology.org/tmlr/2025/bradley2025tmlr-classifierfree]

**Chung, H., Kim, J., Park, G. Y., Nam, H., Ye, J. C. (2025). CFG++: Manifold-constrained classifier free guidance for diffusion models. ICLR 2025. arXiv:2406.08070.**
Attributes CFG failures (mode collapse at high scale, non-invertible DDIM) to an "off-manifold phenomenon". The fix is to renoise with the unconditional rather than the guided noise estimate. The abstract makes no explicit time-resolved claim. Relevant mainly as an alternative mechanism: off-manifold intermediate states could themselves change hidden-state geometry (g) under CFG. [verified: https://arxiv.org/abs/2406.08070 ; venue https://mlanthology.org/iclr/2025/chung2025iclr-cfg/]

**Yi, M., Li, A., Xin, Y., Li, Z. (2024). Towards understanding the working mechanism of text-to-image diffusion model. NeurIPS 2024. arXiv:2405.15330.**
Shape forms in the first few steps, and details come later. Early on "the image is mostly decided by the special token" ([EOS]). Later "the diffusion model completes the details … by information from themselves". Text guidance can be dropped in late steps, giving ≥25% speed-up. Consistent with cond/uncond mattering most early. (SD-family U-Nets, not DiT.) [verified: https://arxiv.org/abs/2405.15330 ; venue https://neurips.cc/virtual/2024/poster/92954]

**Cai, J.-F., Su, Z., Wang, C. (2026). Particle dynamics of flow matching and classifier-free guidance from a stagewise geometry perspective. arXiv:2609.06947.**
Flow-matching trajectories are attracted successively to the global mean, the data convex hull, then a local cluster. Under CFG this becomes "an extrapolated mean, an inflated conditional convex hull". This is theory only and the abstract makes no timing claim about diversity. [verified: https://arxiv.org/abs/2609.06947 (abstract only)]

**Wang, B., Vastola, J. (2023). Diffusion models generate images like painters: an analytical theory of outline first, details later. arXiv:2303.02490.**
High-variance features such as layout emerge earlier than low-variance details, and "early perturbations affect image content more than later ones". Supports early commitment of layout. [verified: https://binxuw.scholars.harvard.edu/publications/diffusion-models-generate-images-painters-analytical-theory-outline-first ; venue not stated there]

---

## 4. Cross-validation instability with small samples and fold partition

**Varoquaux, G. (2018). Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage* 180, 68–77. arXiv:1706.07581.**
Binomial best case: about ±15% at n = 30 and ±10% at n = 100 (Table 1; ±6% at 300). "The standard error across folds strongly underestimates" the true error: SEM bounds were too narrow by ×0.73 for leave-one-out and ×0.26 for 50 repeated 20% splits. Repeated splits are preferred, but they "will not fix the problem of large error bars". With 30 prompts our onset estimate is in this regime. [verified: https://ar5iv.labs.arxiv.org/html/1706.07581 ; NeuroImage venue via the ScienceDirect PII S1053811917305311 linked from https://gael-varoquaux.info/science/our-research-in-2017-personal-scientific-highlights.html. Volume and pages are from memory]

**Varoquaux, G., Raamana, P. R., Engemann, D., Hoyos-Idrobo, A., Schwartz, Y., Thirion, B. (2017). Assessing and tuning brain decoders: cross-validation, caveats, and guidelines. *NeuroImage* (DOI 10.1016/j.neuroimage.2016.10.038). arXiv:1606.05201.**
Leave-one-out gives unstable and biased estimates, "a repeated random splits method should be preferred". Typical CV confidence intervals are about 10%. [verified: https://arxiv.org/abs/1606.05201]

**Bengio, Y., Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *JMLR* 5, 1089–1105.**
There is "no universal (valid under all distributions) unbiased estimator of the variance" of K-fold CV. Naive estimators ignore correlations from overlapping training sets and underestimate. The spread across fold assignments therefore cannot be summarized by a formula from one partition. It must be sampled. [verified: https://www.jmlr.org/papers/v5/grandvalet04a.html]

**Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. MLSys 2021. arXiv:2103.03098.**
"Variance due to data sampling, parameter initialization and hyperparameter choice impact markedly the results". Also "adding more sources of variation to an imperfect estimator approaches better the ideal estimator". For us this means randomizing the prompt-to-fold assignment (and seeds) across many repeats rather than fixing one partition. [verified: https://arxiv.org/abs/2103.03098 ; venue https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html (seen in search listing, not fetched)]

**Krstajic, D., et al. (2014). Cross-validation pitfalls when selecting and assessing regression and classification models. *J. Cheminformatics* 6:10.** [unverified: fetch was rate-limited (HTTP 429) and not retried. Usually cited for repeated grid-search CV because single-split CV results vary with the random fold split]

---

## 5. Small-sample behaviour of the variance-ratio test (T ≈ 40)

**Lo, A. W., MacKinlay, A. C. (1989). The size and power of the variance ratio test in finite samples: a Monte Carlo investigation. *Journal of Econometrics* 40, 203–238.**
Monte Carlo study of finite-sample size and power. Under iid Gaussian the size is comparable to Dickey–Fuller and Box–Pierce. Under a heteroskedastic random walk VR is "more reliable", and it has competitive power against AR(1) and unit-root alternatives. Per Chen & Deo (below), it reports that VR statistics are "severely biased and right skewed for large k". [verified (abstract): https://web.mit.edu/~alo/www/Papers/lo-mackinlay-89.html. The bias/skew detail was seen only via Chen & Deo]

**Kim, M. J., Nelson, C. R., Startz, R. (1988/1991). Mean reversion in stock prices? A reappraisal of the empirical evidence. NBER WP 2795 (published *Review of Economic Studies* 58, 1991).**
**Under the null of serial independence the sample VR(K) is biased downward**, because finite-sample autocorrelations have negative expectation, and the bias grows with K relative to T. They give E[VR(K)] under the null (their eq. 3) and divide by it. Bartlett-type SEs understate dispersion at long lags and the distribution is skewed, so they use randomization (permutation) to get significance. After this correction the evidence for mean reversion is "substantially weaker than reported". **Directly relevant: with T = 39, VR(k) < 1 at large k is partly mechanical.** [verified: https://nber.org/system/files/working_papers/w2795/w2795.pdf ; RES venue from memory, unverified]

**Chen, W. W., Deo, R. S. (2004 WP; *Econometric Theory* 2006). The variance ratio statistic at large horizons.**
Asymptotic theory "provides a poor approximation to the small sample distribution". The normal approximation is good "only when k is not too large relative to n", and was poor at k/n = 0.25 even after their power transformation. At large k the statistic is a weighted sum of few exponentials, hence right-skewed. Joint Wald tests are "blind to the inherent one sided nature of a mean reverting alternative". [verified: https://econwpa.ub.uni-muenchen.de/econ-wp/em/papers/0501/0501003.pdf ; ET 2006 venue inferred from the Cambridge DOI S0266466606060099 seen in search, not fetched]

**Chow, K. V., Denning, K. C. (1993). A simple multiple variance ratio test. *Journal of Econometrics* 58, 385–401.**
A joint test over a vector of horizons (CD1 for iid, CD2 heteroskedasticity-robust) that controls size when several k are examined. Our VR at several k should be tested jointly, not k-by-k. [verified (secondary): https://search.r-project.org/CRAN/refmans/vrtest/html/Chow.Denning.html (R `vrtest` reference page). Primary article not fetched]

**Wright, J. H. (2000). Alternative variance-ratio tests using ranks and signs. *Journal of Business & Economic Statistics* 18(1), 1–9.**
Rank- and sign-based VR tests of the martingale-difference null. "Unlike conventional variance-ratio tests, these tests can be exact", and in Monte Carlo they are often more powerful. Exactness is the attraction at T ≈ 40. [verified: https://ideas.repec.org/a/bes/jnlbes/v18y2000i1p1-9.html]

**Charles, A., Darné, O. (2009). Variance-ratio tests of random walk: an overview. *Journal of Economic Surveys* 23(3), 503–527.** [unverified: HAL page blocked by a bot filter. Only the RePEc listing title was seen in search results. A useful survey of small-sample corrections, bootstrap, and wild bootstrap VR]

---

## 6. Shared initial noise across prompts; commitment time in flow-matching models

**Ban, Y., Wang, R., Zhou, T., Gong, B., Hsieh, C.-J., Cheng, M. (2024/2025). The Crystal Ball hypothesis in diffusion models: anticipating object positions from initial noise. arXiv:2406.01970 (ICLR 2025).**
Identifies "trigger patches" in the initial noise that induce objects at their location. These patches are "'universal' and can be generalized across various positions, seeds, and prompts", and transplanting them into another noise moves objects to the targeted areas. This is stronger evidence than Mao+ 2023 for **noise-determined layout shared across prompts**. The models are not named in the abstract. [verified: https://arxiv.org/abs/2406.01970 ; ICLR 2025 venue from https://proceedings.iclr.cc/paper_files/paper/2025/hash/54e13b23fa2f399cea6e67acf9063c40-Abstract-Conference.html (search listing, not fetched)]

**Wang, R., Huang, H., Zhu, Y., Russakovsky, O., Wu, Y. (2025). The Silent Assistant: NoiseQuery as implicit guidance for goal-driven image generation. arXiv:2412.05101 (ICCV 2025).**
Selects an initial noise from a pre-built library as "implicit guidance". It helps high-level semantics and "low-level visual attributes" across "multiple T2I models" and architectures. This is indirect evidence that noise carries prompt-independent structure. [verified: https://arxiv.org/abs/2412.05101 ; ICCV 2025 from the openaccess supplemental URL in search, not fetched]

**Yan, S., Zhai, W., et al. (2025–26). Determinism of randomness: prompt-residual seed shaping for diffusion generation. arXiv:2511.07756.**
"Changing only the random seed can lead to large differences in prompt faithfulness, composition, and visual quality", attributed to semantic anisotropy of the initial noise. [verified: https://arxiv.org/abs/2511.07756 (models not named in abstract)]

**Handke, F., Stančević, D., Koulischer, F., Demeester, T., Ambrogioni, L. (2026). The entropic signature of class speciation in diffusion models. arXiv:2602.09651.**
A 2026 follow-up to Biroli/Raya–Ambrogioni. It tracks the class-conditional entropy of a latent semantic variable given x_t as a signature of commitment, and "the entropy rate concentrates on the same logarithmic time scale as the speciation symmetry-breaking instability". It quantifies how guidance redistributes semantic information over time. **Validated on EDM2-XS and Stable Diffusion 1.5 (VP diffusion), not SD3 or flow matching. Uses no clones.** [verified: https://arxiv.org/abs/2602.09651]

**Albrychiewicz, E., Franco Valiente, A., Chen, L.-C. (2026). Dynamical regimes of multimodal diffusion models. arXiv:2602.04780.**
Coupled OU theory with a "synchronization gap", in which eigenmodes stabilize at different times. Experiments use MNIST and exact-score samplers only. [verified: https://arxiv.org/abs/2602.04780]

**Li, M., Karan, A., Chen, S. (2025). Blink of an eye: a simple theory for feature localization in generative models. arXiv:2502.00921.**
Generalizes critical windows to autoregressive and diffusion models. Empirical validation is on LLMs, with no SD3 or flow measurement. [verified: https://arxiv.org/abs/2502.00921]

---

## What this changes for us

- **Pooled vs branch-mean g:** concavity of tr M^{1/2} gives a citable inequality (Löwner–Heinz/Carlen). What the math guarantees for g is only **pooled g ≥ energy-weighted mean of the branch g values**, not the unweighted mean. Report the energy-weighted mean alongside branch-mean. The excess over it measures cond/uncond second-moment non-proportionality, which is a quantity in its own right. The 30% excess over branch-mean on Z-Image suggests strongly non-proportional branches (recompute it against the energy-weighted mean). Check whether text tokens are included in X for the single-stream S3-DiT: cond and uncond text blocks differ by construction.
- **Stop treating pooled g as a capture artefact to be removed.** It is a deterministic function of the two branch matrices. Pick one estimand (branch-wise, preferably cond-only) and state it.
- **Resolution:** ComfyUI applies a fixed shift (SD3.5: 3.0; Z-Image: 3.0) with no resolution term. By SD3's own Eq. 23, 512² at shift 3 ≈ 1024² at shift 6. Re-index the β-vs-g topology on a resolution-corrected time (odds × √(m/n)) before concluding that the topology changes with resolution.
- **Z-Image Base was likely run off-spec:** the official diffusers config uses shift 6.0 and ComfyUI defaults to 3.0. Confirm the shift actually used, and flag Base-vs-Turbo comparisons where Base used 3.0.
- **Cond/uncond divergence at the first steps is consistent with RAAG/MAMBO-G on SD3.5** (velocity ratio spikes early, then decays). C²FG's opposite trend is a score-parameterization effect. Our late across-seed contraction matches Jin+ stage (iii) within-mode contraction, so test the "guidance in first quarter only" vs "last three quarters only" split from LIT_R4.
- **Fate-onset estimator:** a 0.48–0.63 swing from fold assignment alone is expected at n = 30 groups (Varoquaux: ±10–15%; no unbiased CV variance estimator). Report the distribution over ≥50 random prompt-grouped partitions (median and IQR), not one partition, and do not interpret onset differences smaller than that spread.
- **VR at T = 39:** VR(k) is biased below 1 and right-skewed under the null, increasingly with k/T. Calibrate against a simulated or permutation null at the same T and k (Kim–Nelson–Startz), test across k jointly (Chow–Denning), or use Wright's exact rank/sign VR. Limit k/T to ≲ 0.25.
- **Shared-noise layout claim:** now has stronger support (Crystal Ball: trigger patches universal across prompts). There is still no flow-matching or SD3 clone-based commitment-time measurement, so our clone/re-noise test on SD3.5/Z-Image would be new.

## Not found

- A primary paper stating the equality case tr((A+B)/2)^{1/2} = mean ⇔ A = B in exactly this form. It follows from strict concavity (Carlen 2010, unverified) or from our Löwner–Heinz argument above.
- Any paper on participation ratio, or nuclear-norm-based g, of **pooled CFG branches**, or pooled-vs-within PR beyond Recanatesi+ 2022 and Gao+ 2017 (already in LIT_R4).
- Any paper measuring internal **hidden-state** (rather than output-prediction) cond/uncond divergence over time in SD3/DiT models.
- The official SD3.5-Medium diffusers shift value (gated repo).
- A closed-form small-sample bias for overlapping VR(k) checked on a primary page. Kim–Nelson–Startz eq. 3 exists, but the formula was not transcribed.
- Any 2025–2026 clone or re-noise commitment-time measurement on SD3, SD3.5, FLUX, Z-Image or other flow-matching T2I models. The closest is Handke+ 2026 (SD1.5, entropy-based).
