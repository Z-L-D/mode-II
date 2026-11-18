# LIT_LENSES: literature lenses for a coarse spectral observable g

*Literature survey for CoS-PAFS, 2026-10-08. Author: literature agent. Nobody else edits this file.*

## How the citations were checked

- Every paper cited below, and every entry in the BibTeX block, was confirmed to exist by fetching its arXiv abstract page or a publisher or repository page during this session. Summaries are based on those fetched abstracts, plus full text where the text says so.
- Pages I could not read, and what that means:
  - The ScienceDirect page for Coifman and Lafon, "Diffusion maps" (ACHA 2006), was blocked by robots.txt, and doi.org returned 429. The paper is therefore **not cited**.
  - nature.com returned 429. Stringer et al. 2019 was verified through UCL Discovery instead.
  - zenodo.org/records/19696125 ("Coxeter-theoretic identities near α⁻¹", the only web hit for the phrase "Coxeter ladder") returned 429. I could not see its authors or content (see D).
  - PMC (captcha) and HAL (access denied) could not be read. Their papers are verified through other pages.
- **Definitions follow the capture code, not the task prompt.** The prompt defines g from the eigenvalues s of XᵀX/N. The capture code (`capture_code_r2/cos_pafs_llm_measure.py` l.671–681, quoted in `LENS_PREREGISTRATION.md`) uses singular values: σᵢ = √eig(XᵀX/N), pᵢ = σᵢ/Σσ, and g·D = (Σσ)²/Σσ². These give different numbers (see §0.3). **The papers must say which one they use.**

---

## 0. What kind of object g is, and why that decides which literature applies

### 0.1 g and the singular-value entropy are two effective ranks of the same distribution

With pᵢ = σᵢ/Σσ:
- g·D = 1/Σpᵢ² = exp H₂(p), the Rényi-2 entropy.
- sv_entropy = H₁(p), the Shannon entropy.
- exp H₁(p) is exactly the **effective rank of Roy & Vetterli (2007)**. This is the standard definition: exp of the Shannon entropy of the ℓ₁-normalised singular values. The fetched landing page did not show the formula.
- The same erank is used by RankMe (Garrido et al.) and, in an entropy form, by Diff-eRank (Wei et al.).
- If the eigenvalues λ = σ² were used instead, g·D would be the **participation ratio** (Σλ)²/Σλ² of Gao et al. (2017) and Recanatesi et al. (2019).

Safe sentence: "g is the Rényi-2 effective rank of the singular-value distribution, normalised by D. The singular-value entropy is its Rényi-1 counterpart (exp H₁ is the effective rank of Roy & Vetterli), so H₁ ≥ H₂ identically."

### 0.2 "Coarse-grained observable": the wording matters

- In the RG sense, a coarse-grained observable is a function of **block variables** produced by integrating out short-scale degrees of freedom (Kadanoff 1966).
- g is not that. It is a **low-order spectral summary**: a moment ratio of one empirical distribution. Every layer of these isotropic DiTs keeps the token count N and the width D fixed, so depth decimates nothing.
- **Safe wording:** "g is a coarse (low-dimensional, rotation-invariant) summary of the activation spectrum."
- **Unsafe wording:** "g is a coarse-grained variable under an RG transformation".

### 0.3 Finite-sample null (Marchenko–Pastur)

White Gaussian X (N tokens × D) with q = D/N gives a value of g that is not 1. The values below come from the Marchenko–Pastur density by numerical integration in this session; a simulation at D = 1000 agrees to ±0.001:

| q = D/N | g, singular-value definition (code) | g, eigenvalue definition (prompt) | 1/(1+q) |
|---|---|---|---|
| 0.10 | 0.975 | 0.909 | 0.909 |
| 0.25 | 0.936 | 0.800 | 0.800 |
| 0.375 | 0.904 | 0.727 | 0.727 |
| 0.50 | 0.870 | 0.667 | 0.667 |
| 0.75 | 0.800 | 0.571 | 0.571 |
| 1.00 | 0.721 | 0.500 | 0.500 |

- For the eigenvalue definition the null is exactly 1/(1+q), because E[λ²] = 1+q for MP.
- For the singular-value definition it is (E√λ)²/Eλ, which at q = 1 is (8/3π)² ≈ 0.72.
- **Recommendation:** any absolute statement about g ("high"/"low"/"near isotropic") should be reported relative to g_MP(D/N) for that channel's N and D. The LLM captures have N < D: there the rank is N, and g ≤ N/D trivially.

### 0.4 Uncentred statistic, outliers and symmetries

- XᵀX/N includes the mean direction μμᵀ.
- **Massive activations:** a few coordinates whose values are ~10⁵× typical and nearly input-independent. They are documented in LLMs and ViTs (Sun et al. 2024) and in DiTs, where they are tied to AdaLN (Gan et al. 2025).
- **Anisotropy:** representations do not spread evenly across directions (Ethayarajh 2019 for BERT, ELMo and GPT-2), and in decoders the anisotropy is bell-shaped in depth (Razzhigaev et al. 2023).
- **Consequence:** p₁ (hence the spectral gap and much of g's variation) can be dominated by one outlier direction. Its depth or step dependence would then be "outlier dynamics", not "bulk geometry". The papers should either show a centred companion, or report g with the top-k singular values removed for a sample of captures.
- **Symmetry blindness:** all three statistics are invariant under X → XO for orthogonal O. They are therefore blind, in the mean, to any Z2 or O(D) symmetry breaking (already noted in the preregistration, L4).

---

## A. Renormalization group, ML, and representation dimensionality

### A1. RG ↔ deep learning (foundational)

- **Kadanoff 1966**, "Scaling laws for Ising models near Tc" (Physics Physique Fizika 2, 263; doi:10.1103/PhysicsPhysiqueFizika.2.263, verified via INSPIRE).
  - Introduces the block-spin construction: group spins into blocks, define block variables, and argue that the couplings flow under this rescaling. The scaling laws follow from that flow.
  - This is the template for "coarse-graining = fewer degrees of freedom at a larger scale".
- **Bény 2013**, "Deep learning and the renormalization group" (arXiv:1301.3124).
  - Compares RG with deep learning, with depth playing the role of scale.
  - Adapts MERA into a learning algorithm built on a generative hierarchical Bayesian network.
- **Mehta & Schwab 2014**, "An exact mapping between the variational renormalization group and deep learning" (arXiv:1410.3831).
  - Constructs an exact correspondence between variational RG and stacked RBMs, illustrated on 1D and 2D Ising models.
  - Suggests deep networks may implement an RG-like feature extraction.
- **Lin, Tegmark & Rolnick 2017**, "Why does deep and cheap learning work so well?" (arXiv:1608.08225; J. Stat. Phys., doi:10.1007/s10955-017-1836-5).
  - Argues that the functions of interest are cheap because of physics: symmetry, locality, compositional and hierarchical structure.
  - Relates depth to hierarchical generative processes, and proves no-flattening results for some computations.
- **Koch-Janusz & Ringel 2018**, "Mutual information, neural networks and the renormalization group" (arXiv:1704.06279; Nature Physics, doi:10.1038/s41567-018-0081-4).
  - A network learns a real-space RG step by maximising mutual information between a block and its environment.
  - Reproduces RG flow and extracts the Ising critical exponent.

**Relevance to g:**
- In all of these, RG is defined by (i) a map that *removes* degrees of freedom and (ii) a flow of couplings under repeated application of the *same* map.
- Neither holds automatically across transformer depth. N and D are constant, and each layer has its own weights, so the depth "map" is not one repeated map.
- **Safe claim:** none, beyond background. If the papers mention RG at all, it should be to say that β_layer is a depth finite difference and is not interpreted as an RG beta function.
- **Test suggested:** the autonomy test (preregistration R1), see T4.

### A2. Effective field theory of deep networks: RG-like depth recursions

- **Roberts, Yaida & Hanin 2022**, *The Principles of Deep Learning Theory* (arXiv:2106.10165; CUP, doi:10.1017/9781009023405).
  - An effective-theory treatment of deep networks built from layer-to-layer recursions of preactivation statistics. The book calls this a "representation group flow".
  - Its main results:
    - near-Gaussianity, with corrections controlled by the depth-to-width ratio;
    - criticality tuning against vanishing and exploding gradients;
    - universality classes of activation functions.
- **Halverson, Maiti & Stoner 2021**, "Neural networks and quantum field theory" (arXiv:2008.08601; MLST, doi:10.1088/2632-2153/abeca3).
  - Networks near their Gaussian-process limit behave as free fields, and finite-width effects act as interactions, organised by Wilsonian EFT and Feynman diagrams.
- **Erbin, Lahoche & Samary 2022**, "Nonperturbative renormalization for the neural network–QFT correspondence" (arXiv:2108.01403; MLST, doi:10.1088/2632-2153/ac4f69).
  - Applies the Wetterich–Morris functional RG to the NN–QFT picture.
  - Changing the weight-distribution variance is read as an RG flow among networks.

**Relevance to g:**
- Roberts et al. is the closest legitimate analogue. Their recursion acts on the **kernel**, the 2-point Gram statistic, which is exactly the object whose spectrum g summarises.
- Their recursion is **autonomous**: the same map at each layer, because layers are i.i.d. at initialisation. Trained networks break this.
- **Safe claim:** "β_layer(g) is a depth finite difference of a spectral summary of the Gram statistic, in the spirit of (but not equivalent to) depth recursions of kernel statistics in effective theories of deep networks [Roberts et al.]. Those recursions are defined for networks at initialisation."
- **Test suggested:** whether g_{L+1} is a function of g_L alone, i.e. whether the flow is autonomous (preregistration R1; T4).
- The NN-QFT papers (Halverson et al.; Erbin et al.) describe flows in *width* or *weight variance*, not in data or depth. They are not relevant to telemetry from trained models. Do not cite them in the papers.

### A3. Diffusion models as (inverse) RG and as coarse-to-fine generators

- **Cotler & Rezchikov 2023**, "Renormalizing diffusion models" (arXiv:2308.12355).
  - Diffusion models are trained to invert specified RG schemes for lattice field theories.
  - Nonperturbative RG schemes are cast as diffusion processes over field space.
  - Gives a physical interpretation of multiscale diffusion models.
- **Masuki & Ashida 2025**, "Generative diffusion model with inverse renormalization group flows" (arXiv:2501.09064).
  - Defines a forward flow that progressively removes fine-scale detail (RG-like), and generates by reversing it, coarse to fine.
  - Reports quality gains and up to ~10× faster sampling on protein and image tasks.
- **Rissanen, Heinonen & Solin 2023**, "Generative modelling with inverse heat dissipation" (arXiv:2206.13397; ICLR 2023).
  - Generates by stochastically reversing the heat equation, which removes fine scales first.
  - A spectral analysis of natural images argues that standard diffusion models carry an implicit coarse-to-fine bias.
- **Hoogeboom & Salimans 2023**, "Blurring diffusion models" (arXiv:2209.05557).
  - Shows that heat-dissipation blurring equals Gaussian diffusion with non-isotropic (frequency-dependent) noise.
  - Unifies the two in one model family.
- **Falck et al. 2025**, "A Fourier space perspective on diffusion models" (arXiv:2505.11278).
  - Under DDPM, high frequencies lose SNR earlier because data Fourier variance decays with frequency, so the reverse process builds low frequencies first.
  - An equal-SNR forward process removes this hierarchy, which helps on high-frequency-dominant data.
- **Wang & Vastola 2023**, "The hidden linear structure in score-based models and its application" (arXiv:2311.10892).
  - At high noise, the learned score is well approximated by the linear score of a Gaussian, and early trajectories have a closed form.

**Relevance to g:**
- The **noise level σ, not depth, is the axis on which a scale interpretation is literature-backed**. "Coarse-to-fine over σ" is the established reading (Rissanen; Hoogeboom; Falck). Masuki & Ashida and Cotler & Rezchikov make the RG analogy explicit *along diffusion time*.
- **Safe claim:** "the σ-trajectory of g is consistent with the coarse-to-fine character of reverse diffusion documented in the frequency domain [Rissanen; Hoogeboom; Falck]".
- **Unsafe claim:** "g implements RG", unless g is shown to follow a specific scheme.
- **Crucial confound:** at the input, a rectified-flow latent x_t = (1−t)x₀ + tε has second moment (1−t)²M₀ + t²I, where M₀ is the uncentred data moment.
  - The input spectrum at each step is therefore an exact affine deformation λᵢ → (1−t)²λᵢ + t². In normalised form this is precisely the preregistration's "HK" additive-isotropic family (L3).
  - Part of any σ-trajectory of g is inherited from the forward process.
  - The informative quantity is the activation trajectory *relative to* this input-level baseline (T3).
- Wang & Vastola imply that early (high-σ) steps should be near-linear-Gaussian. A linear-Gaussian prediction of early-step g is a natural null.

### A4. Hierarchical data, phase transitions and critical windows in diffusion

- **Sclocchi, Favero & Wyart 2024/25**, "A phase transition in diffusion models reveals the hierarchical nature of data" (arXiv:2402.16991).
  - For hierarchical generative data, noising then denoising shows a phase transition at a threshold time. Past it, high-level (class) features are lost abruptly, while low-level features change smoothly.
  - Validated on ImageNet diffusion models.
- **Sclocchi, Favero, Levi & Wyart 2025**, "Probing the latent hierarchical structure of data via diffusion models" (arXiv:2410.13770).
  - Changes under noise-then-denoise occur in correlated chunks whose size grows near the transition noise level.
  - Confirmed on text and image diffusion models.
- **Li & Chen 2024**, "Critical windows: non-asymptotic theory for feature emergence in diffusion models" (arXiv:2403.01633).
  - For mixtures of log-concave densities, proves bounds on narrow time windows in which specific features are decided.
  - Views diffusion as a hierarchical sampler.

**Relevance to g:**
- These papers predict a **specific σ*** around which across-generation statistics should change sharply.
- **Safe claim:** none about g from the current telemetry alone, because g is rotation-invariant and need not respond to which class is chosen (see C).
- **Test suggested:** whether the within-prompt (across-seed) variance of g, or its step derivative, has an interior extremum near a model-specific σ* (T6/T7). A real test of the Sclocchi prediction requires new noise-then-denoise runs.

### A5. Spectra of weights and activations (RMT, heavy tails, power laws)

- **Martin & Mahoney 2021**, "Implicit self-regularization in deep neural networks: evidence from random matrix theory and implications for learning" (arXiv:1810.01075; JMLR 22(165):1–73).
  - Empirical spectral densities of *weight* matrices show training phases, from MP-like to heavy-tailed self-organisation, and batch size moves a model through them.
- **Martin, Peng & Mahoney 2021**, "Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data" (arXiv:2002.06716; Nat. Commun., doi:10.1038/s41467-021-24025-8).
  - Power-law metrics of weight spectra predict quality across hundreds of public models better than norm-based metrics do.
- **Stringer et al. 2019**, "High-dimensional geometry of population responses in visual cortex" (Nature 571:361–365, doi:10.1038/s41586-019-1346-5; verified via UCL Discovery).
  - The nth PC variance of mouse V1 responses scales as 1/n.
  - Theory ties the exponent to smoothness of the code; lower-dimensional stimuli give larger exponents.
- **Agrawal et al. 2022**, "α-ReQ: assessing representation quality in self-supervised learning by measuring eigenspectrum decay" (NeurIPS 2022).
  - The power-law decay coefficient α of the feature-covariance spectrum predicts linear-probe generalisation.
- **Ettori 2026**, "Spectral geometry for deep learning: compression and hallucination detection via random matrix theory" (arXiv:2601.17357, a thesis).
  - Spectral features of hidden activations, and their temporal dynamics, are used for hallucination and OOD detection and for compression.

**Relevance to g:**
- g is a single number. Many spectral shapes share the same g.
- For a pure power law λₙ ∝ n^(−α) on D modes (eigenvalue definition), g·D behaves as follows:
  - α < ½: g → a constant, (1−2α)/(1−α)².
  - ½ < α < 1: g ~ D^(1−2α) → 0.
  - α > 1: g ~ 1/D.

  The same exponent therefore gives different g at different widths. **Cross-model comparison of raw g across D (1536–3840) conflates width with spectral shape.**
- **Safe claim:** "g is a coarse proxy for spectral concentration; we do not infer spectral shape from it."
- Martin–Mahoney and α-ReQ are precedent that spectral shape *can* track quality. The papers make no geometry→quality claim and should not borrow one.
- **Test suggested:** fit a spike-plus-power-law to (H₁, H₂, gap) jointly (preregistration P1b/P1c). Report α and its D-dependence, not g alone (T8).

### A6. Dimensionality across depth: effective rank, PR and intrinsic dimension

- **Roy & Vetterli 2007**, "The effective rank: a measure of effective dimensionality" (EUSIPCO 2007; Zenodo doi:10.5281/zenodo.40328).
  - A real-valued extension of rank, defined from the entropy of the normalised singular values.
- **Gao, Trautmann, Yu, Santhanam, Ryu, Shenoy & Ganguli 2017**, "A theory of multineuronal dimensionality, dynamics and measurement" (bioRxiv, doi:10.1101/214262).
  - Uses the participation ratio as "a natural continuous measure of dimensionality": it equals 1 when all variance lies in one dimension and M when it is spread evenly over M dimensions.
  - Shows that neural task complexity upper-bounds PR dimensionality.
- **Recanatesi et al. 2019**, "Dimensionality compression and expansion in deep neural networks" (arXiv:1906.00443).
  - Early layers expand dimensionality and later layers compress it.
  - SGD noise acts as an implicit regulariser of dimensionality.
- **Ansuini, Laio, Macke & Zoccolan 2019**, "Intrinsic dimension of data representations in deep neural networks" (arXiv:1905.12784; NeurIPS 2019).
  - Nonlinear intrinsic dimension (ID) first rises then falls across layers (a "hunchback" profile), and last-layer ID predicts test accuracy.
  - **The pattern is not seen with linear (PCA) estimates.**
- **Valeriani et al. 2023**, "The geometry of hidden representations of large transformer models" (arXiv:2302.00294).
  - In protein-LM and image transformers, ID rises early, drops sharply in intermediate layers, then plateaus or forms a second small peak.
  - Semantic content is best at the end of the first peak.
- **Razzhigaev et al. 2024**, "The shape of learning: anisotropy and intrinsic dimensions in transformer-based models" (arXiv:2311.05928).
  - In decoders, anisotropy is bell-shaped in depth, peaking mid-depth; encoders are flatter.
  - ID grows and then shrinks over training.
- **Skean et al. 2025**, "Layer by layer: uncovering hidden representations in language models" (arXiv:2502.02013).
  - Information-theoretic and geometric metrics, including matrix entropy, show mid-depth layers often beat the final layer across 32 embedding tasks.
- **Dong, Cordonnier & Loukas 2021**, "Attention is not all you need: pure attention loses rank doubly exponentially with depth" (arXiv:2103.03404).
  - Pure self-attention collapses token representations to rank one doubly exponentially; skip connections and MLPs counteract this.
- **Noci et al. 2022**, "Signal propagation in transformers: theoretical perspectives and the role of rank collapse" (arXiv:2206.03126).
  - Rank collapse at initialisation makes query and key gradients vanish.
  - Depth-scaled residual branches prevent it.
- **Garrido et al. 2023**, "RankMe" (arXiv:2210.02885), and **Wei et al. 2024**, "Diff-eRank" (arXiv:2401.17139).
  - Effective rank of representations used as a label-free quality signal, for SSL and for LLMs respectively.

**Relevance to g:**
- This is **the literature the papers should actually cite** for "a recurring depth shape of g".
- **Safe claim:** "non-monotone depth profiles of representation dimensionality and anisotropy are well documented in transformers [Valeriani; Razzhigaev; Skean]. We report the analogous profile for a linear, rotation-invariant effective-rank statistic in diffusion transformers."
- **Must also say:** linear (PCA-type) and nonlinear ID estimates can disagree [Ansuini], so g is not an intrinsic dimension.
- Rank-collapse theory [Dong; Noci] supplies a mechanism for low g in deep attention stacks. Residual and MLP paths counteract it, which is a mechanistic hypothesis for where g recovers.
- **Test suggested:** location of the g extremum in *relative* depth L/L_max across the 5 diffusion channels and the LLMs, with prompt-bootstrap CIs, compared to the published ID/anisotropy peak positions (T9).

---

## B. Heat kernels and spectral geometry

**Papers:** Rissanen et al. 2023; Hoogeboom & Salimans 2023; Falck et al. 2025 (summarised in A3), plus the following.

- **Tsitsulin et al. 2020**, "The shape of data: intrinsic distance for data distributions" (IMD; arXiv:1905.11141; ICLR 2020).
  - Compares data manifolds through a lower bound on a spectral Gromov–Wasserstein distance.
  - The multi-scale descriptor is the **heat kernel trace** of kNN-graph Laplacians. Full text: "The HKT contains all the information in the graph's spectrum, both local and global."
  - Used to compare representations and evaluate generative models.
- **Khandait & Gerken 2026**, "From layers to networks: comparing neural representations via diffusion geometry" (arXiv:2605.15901).
  - Rewrites representational-similarity measures as Markov (diffusion) operators on layer activations.
  - Gets multi-scale CKA and distance correlation from powers of the operator, and fuses layers by alternating diffusion.
- **Connes spectral triples in ML:** not found.
  - Searches returned only one applied use of the spectral-triple formalism: Najem, Mrad & Elsayed 2025, "Geometric features of higher-order networks via the spectral triplet" (arXiv:2509.04311). It computes dimension, curvature and distance on **simplicial complexes in network science, not neural networks**.
  - I found no paper applying spectral triples to neural representations or diffusion models.
- **Weyl law / spectral dimension of neural representations:** no verified ML paper found. Do not cite.

**Relevance to g:**

1. **Exact but modest identity.** For the operator M = XᵀX/N with heat trace Z(t) = Σ e^(−tλᵢ):
   - Z′(0) = −Σλ and Z″(0) = Σλ², so (eigenvalue-definition) g = Z′(0)² / (D·Z″(0)).
   - With λ → σ the same holds for the code's singular-value g.
   - So g is the lowest-order heat-trace invariant of the activation second-moment operator. This is a Taylor identity: it imports no heat-equation dynamics and is safe to state only as a remark.
2. The **heat equation in these models acts on pixels and latents** (frequency space), not on the activation spectrum.
   - What is legitimately "heat-kernel-like" in the telemetry is the forward-process deformation of the *input* spectrum (A3 confound). Under Gaussian noising this deformation is additive-isotropic: the preregistration's HK family.
   - **Safe claim:** "under the forward process the input second-moment spectrum deforms exactly as λ → a_t²λ + b_t². Departures of activation spectra from this deformation are model-generated."
3. True spectral-geometry statistics (heat trace or Weyl counting on a token kNN graph, IMD) are **nonlinear and multi-scale**. They would need stored activations, not the (g, gap, entropy) telemetry. They are a future extension only.

---

## C. Statistical mechanics, spin glasses and associative memory in diffusion

- **Biroli & Mézard 2023**, "Generative diffusion in very large dimensions" (arXiv:2306.03518; J. Stat. Mech. 093402, doi:10.1088/1742-5468/acf8ba).
  - Gaussian and Curie–Weiss analyses: capturing the relative weights of low-temperature states needs n ≫ d.
  - Derives scaling laws for efficient generation.
- **Biroli, Bonnaire, de Bortoli & Mézard 2024**, "Dynamical regimes of diffusion models" (arXiv:2402.18491; Nat. Commun., doi:10.1038/s41467-024-54281-3).
  - Three regimes of the backward process: before speciation, speciation (symmetry-breaking-like emergence of coarse structure), and collapse onto training points (glass-like condensation).
  - Verified from the full text: the forward process is OU, x = a e^(−t) + z√Δ_t with Δ_t = 1 − e^(−2t). The **speciation time is defined by Λ e^(−2t_S) = 1, i.e. t_S = ½ log Λ, where Λ is the largest eigenvalue of the data covariance**.
  - The collapse time t_C is where the excess entropy density crosses zero; for Gaussian mixtures t_C = ½ log(1 + σ²/(n^(2/d) − 1)).
- **Raya & Ambrogioni 2023**, "Spontaneous symmetry breaking in generative diffusion models" (arXiv:2305.19693; NeurIPS 2023).
  - Generation has a linear phase around a central fixed point, then an attractor phase after the fixed point loses stability. Diversity is decided in that instability window.
  - Gaussian late initialisation improves FID up to 3× with fast samplers.
- **Ambrogioni 2023/24**, "The statistical thermodynamics of generative diffusion models: phase transitions, symmetry breaking and critical instability" (arXiv:2310.17467).
  - An equilibrium statistical-mechanics reading: second-order, symmetry-breaking phase transitions of mean-field class, arising from a self-consistency condition.
  - Generation viewed as a stochastic adiabatic process.
- **Ventura, Achilli, Silvestri, Lucibello & Ambrogioni 2024/25**, "Manifolds, random matrices and spectral gaps: the geometric phases of generative diffusion" (arXiv:2410.05898).
  - **Spectral gaps of the score Jacobian** reveal sub-manifolds and their dimensions.
  - Three phases: trivial; manifold coverage; consolidation, in which the score becomes orthogonal to the manifold. Different geometric features form on different timescales.
- **Achilli et al. 2024/26**, "Losing dimensions: geometric memorization in generative diffusion" (arXiv:2410.08727).
  - As data become scarce, the latent dimensionality measured from the score field shrinks gradually; salient features are lost first.
  - This is a "geometric memorization" regime between generalisation and copying.
- **Hoover et al. 2023**, "Memory in plain sight" (arXiv:2309.16750); **Ambrogioni 2023**, "In search of dispersed memories: generative diffusion models are associative memory networks" (arXiv:2309.17290); **Pham et al. 2025**, "Memorization to generalization: emergence of diffusion models from associative memory" (arXiv:2505.21777).
  - Diffusion seen as dense-associative-memory retrieval.
  - Pham et al. describe a transition from memorisation (one attractor per sample) through a spurious-state phase to generalisation.
- **Bonnaire, Urfin, Biroli & Mézard 2025**, "Why diffusion models don't memorize: the role of implicit dynamical regularization in training" (arXiv:2505.17638).
  - Two training timescales: generalisation time, which is roughly n-independent, and memorisation time, which grows with n.
- **Kadkhodaie, Guth, Simoncelli & Mallat 2024**, "Generalization in diffusion models arises from geometry-adaptive harmonic representations" (arXiv:2310.02557).
  - Networks trained on disjoint halves of the data learn nearly the same score, and denoise by shrinkage in geometry-adaptive harmonic bases.
- **Li & Chen 2024** (A4): critical windows.

**Relevance to g:**

1. **Speciation-time prediction.** This is the most concrete, falsifiable import.
   - OU condition: Λe^(−2t) = 1, signal variance on the top PC equal to the unit noise scale.
   - Transferring it heuristically to rectified flow (x_t = (1−t)x₀ + tε) by requiring top-PC SNR = 1 gives (1−t)²Λ = t², i.e. **t_S = √Λ / (1 + √Λ)**.
   - Λ is the largest eigenvalue of the (scaled) latent data covariance.
   - **This transfer is my derivation, not a result of the paper. It holds only to the extent the high-d, optimal-score analysis carries over.**
   - SD3.5 and Z-Image use shifted schedules, so the comparison must use the recorded σ, not step index.
2. **Z2 / symmetry breaking is structurally invisible to g in the mean.**
   - X → −X, and any O(D) rotation, leaves the triple unchanged. In Raya–Ambrogioni's symmetric picture the two branches have identical spectra, so even the **across-seed variance of g need not rise at the symmetry-breaking time**.
   - Only *asymmetric* speciation, where branches with different spectral statistics separate (e.g. distinct content classes), can show up as an across-seed variance peak.
   - This supports the preregistration's L4 prediction. The papers should not suggest that g probes symmetry breaking.
3. **Ventura et al.** is the closest published object to the spectral gap telemetry: gaps in the spectrum of the denoiser Jacobian, tracked over noise level.
   - But the Jacobian ∂ε̂/∂x is not the activation second moment.
   - **Safe claim:** "analogous in spirit to timescale-separated spectral gap formation in score Jacobians [Ventura et al.]"; nothing stronger.
4. **Replica overlaps.** Parisi overlaps are defined between configurations, such as two generated samples or two activation fields.
   - The preregistration L5 overlap of *g-fields* (correlation of summary-statistic fields over step × layer) is a descriptive similarity, not a replica overlap. Bimodality or ultrametricity there should be interpreted as structure of the prompt set (data hierarchy), as the preregistration already states.
   - Real overlaps need stored activations or outputs per seed.
5. **Memorisation and collapse** (Biroli t_C; Achilli; Pham; Bonnaire) concern proximity to *training* samples. They are untestable on this telemetry without a training-set reference. They are not relevant to the papers.

---

## D. Fibonacci / golden-mean universality, "Coxeter ladder", E10, PGL(2,Z)

**Verified papers:**

- **Popkov, Schadschneider, Schmidt & Schütz 2015**, "Fibonacci family of dynamical universality classes" (arXiv:1505.04461; PNAS 112(41):12645–12650, doi:10.1073/pnas.1512261112; volume/pages verified via PMC4611608 metadata).
  - **Setting:** one-dimensional systems far from equilibrium with n conserved quantities, analysed with nonlinear fluctuating hydrodynamics and mode coupling. The simulated model has three conserved particle species.
  - **Verified claims:**
    - the dynamical exponents z (time ~ length^z) are ratios of neighbouring Fibonacci numbers;
    - the family starts at z = 2 (diffusive) or z = 3/2 (KPZ);
    - when neither a diffusive nor a KPZ mode is present, all modes have z = golden mean (1+√5)/2;
    - the scaling functions are asymmetric Lévy distributions fixed by the macroscopic currents and the compressibility matrix;
    - the paper calls these exponents "Kepler ratios", truncations of the golden mean's continued fraction.
- **Schütz 2017**, "On the Fibonacci universality classes in nonlinear fluctuating hydrodynamics" (arXiv:1710.09121).
  - States the exponents as "the Kepler ratios z_i = F_{i+1}/F_i".
  - Gives criteria on macroscopic currents for non-KPZ classes, and a lattice gas expected to show modified KPZ without fine-tuning.
- **Popkov & Schütz 2024**, "Quest for the golden ratio universality class" (arXiv:2310.19116; PRE, doi:10.1103/PhysRevE.109.044111).
  - Gives closed-form mode-coupling conditions for the possible classes.
  - Rules out some families of microscopic models for the golden-ratio class, which requires antisymmetric currents and correlated densities.
  - Shows that mode-coupling predictions are exact for a noisy harmonic chain.

**"Ruiz" + "Coxeter ladder": not found in the literature.**
- Searches tried: "Ruiz Coxeter ladder", "Coxeter ladder", Ruiz with Coxeter number, E8, golden ratio, universality or neural network.
- None returned a paper by an author named Ruiz on a Coxeter-number ladder in physics or ML.
- The only exact-phrase hit for "Coxeter ladder" was a Zenodo record, "Coxeter-theoretic identities near α⁻¹: a structural enumeration" (zenodo.org/records/19696125). It could not be opened (HTTP 429), so its authors, content and any link to "Ruiz" are **unknown**. It should not be cited without being read.
- The preregistration (L6/Part B) says the ladder is defined from project records. It appears to be a project-internal construct, not an external result.
- The only verified Coxeter–ML intersection is **Chen, Dechant, He, Heyes, Hirst & Riabchenko 2023**, "Machine learning Clifford invariants of ADE Coxeter elements" (arXiv:2310.00041). It applies ML *to* Coxeter data (A8, D8, E8); it does not find Coxeter structure *in* ML models.

**E10 / hyperbolic Kac–Moody:**
- **Damour, Henneaux & Nicolai 2002**, "E10 and a 'small tension expansion' of M theory" (arXiv:hep-th/0207267; PRL 89, 221601, doi:10.1103/PhysRevLett.89.221601).
  - 11D supergravity near a spacelike singularity matches null geodesic motion on E10/K(E10) up to 30th order.
- **Damour, Henneaux & Nicolai 2003**, "Cosmological billiards" (arXiv:hep-th/0212256).
  - Near a spacelike singularity, the dynamics become billiard motion in a region of hyperbolic space. In many cases the billiard is the Weyl chamber of a Kac–Moody algebra.
- **Damour 2007**, "Chaos and symmetry in string cosmology" (arXiv:0704.1457; doi:10.1142/9789812834300_0003).
  - The abstract states that the billiard chambers are "identical to the Weyl chambers of the Lorentzian Kac-Moody algebras AE_d, or E_{10}".
- **PGL(2,Z):** I could not verify from fetched text the commonly quoted statement that the 4D pure-gravity billiard group is PGL(2,Z) (the Weyl group of AE3). It is therefore **not asserted here**.
- **No ML or dynamical-universality link** to E10 or PGL(2,Z) was found.
  - The one Kac–Moody hit with "ML" in the query (Kleinschmidt, Malcha & Nicolai, JHEP 2025, 124) is pure mathematical physics.
  - The preregistration's R3 "PGL(2,ℝ)" test is a *Möbius-reparametrisation invariance* check. It is unrelated to E10 billiards (see the note in T5).

**Relevance to g:**
- **No mechanism links transformer telemetry to 1D driven conserved-mode hydrodynamics.** There are no conserved quantities transported along depth or σ, and no mode coupling.
- **Safe claim for the papers:** none. Do not mention.
- **If a project exponent is compared to Kepler ratios, the look-elsewhere problem is severe.**
  - The ratios 2, 3/2, 5/3, 8/5, 13/8, 21/13, … accumulate at φ ≈ 1.618.
  - Any value in roughly [1.58, 1.67] lies within ~1–2% of some member.
  - A chance-coverage null is mandatory (the preregistration's Poisson-binomial log-uniform design is appropriate); see T10.

---

## E. Salem and Pisot numbers in dynamics (brief)

- **McMullen 2002**, "Coxeter groups, Salem numbers and the Hilbert metric" (Publ. Math. IHÉS 95:151–183, doi:10.1007/s102400200001; full text read on Numdam).
  - Theorem 1.1: the spectral radius λ(w) of a Coxeter element is either 1 or ≥ λ_Lehmer ≈ 1.17628.
  - Quote: "λ_Lehmer = λ(w) for the Coxeter elements of Y_{2,3,7}".
  - §7 identifies "E_10 = Y_{2,3,7}", and inf β(W,S) = λ_Lehmer.
  - **So Lehmer's number is the spectral radius of the E10 Coxeter element, and the minimum over Coxeter systems: verified.**
- **Hironaka**, "What is … Lehmer's number?" (expository; PDF hosted at math.fsu.edu).
  - States that Lehmer's number is "the spectral radius of the Coxeter element of E10", and the dilatation of the (−2,3,7)-pretzel-knot monodromy.
  - The venue was not shown on the fetched pages. Cite McMullen instead.
- **McMullen 2002**, "Dynamics on K3 surfaces: Salem numbers and Siegel disks" (J. reine angew. Math. 545:201–233, doi:10.1515/crll.2002.036).
  - Constructs K3 automorphisms whose leading eigenvalue on H²(X) is a Salem number of degree 22.
  - Standard background, not re-verified here: the topological entropy of such an automorphism is log of that spectral radius, by Gromov–Yomdin.
- **Oguiso 2009**, "The third smallest Salem number in automorphisms of K3 surfaces" (arXiv:0905.2396).
  - Realisation of small Salem numbers as K3 automorphism entropies.
  - Shows log(Lehmer's number) is **not** realisable as the entropy of an Enriques surface automorphism.
- **Akiyama, Barge, Berthé, Lee & Siegel**, "On the Pisot substitution conjecture" (survey chapter; PDF at irif.fr; the book venue was not confirmed from fetched pages).
  - Conjecture: an irreducible substitution with Pisot expansion factor has pure discrete spectrum, for both the symbolic system and the tiling of the line.

**Relevance:**
- Salem and Pisot numbers arise as **leading eigenvalues of integer matrices**: Coxeter elements, lattice isometries, substitution matrices. That is why they are algebraic integers.
- Activation spectra are empirical real-valued statistics of trained floating-point weights. There is no integer-matrix structure that would force algebraic-integer eigenvalues.
- An earlier project round (preregistration L6) found no Lehmer match.
- **Not for the papers.**

---

## F. What belongs in the papers

**Cite, with the suggested wording:**
1. **Definitions:** Roy & Vetterli (erank = exp H₁); Gao et al. (PR). Add one sentence on Rényi-1 vs Rényi-2 and on singular-value vs eigenvalue normalisation.
2. **Depth shape:**
   - Valeriani et al., Razzhigaev et al. and Skean et al. for documented non-monotone depth profiles.
   - Ansuini et al. for the caveat that linear and nonlinear dimension estimates disagree.
   - Dong et al. and Noci et al. for rank-collapse mechanisms.
3. **Outliers and anisotropy:** Sun et al. (massive activations), Gan et al. (massive activations in DiTs, via AdaLN) and Ethayarajh, as the main caveat on uncentred statistics.
4. **σ-trajectory:** Rissanen et al., Hoogeboom & Salimans and Falck et al. (coarse-to-fine, frequency-wise SNR), plus Wang & Vastola (linear-Gaussian high-noise regime).
5. **Optional context, one sentence:** Biroli et al., Raya & Ambrogioni and Sclocchi et al. for the existence of sharp regimes or critical windows in reverse diffusion. State explicitly that g is rotation-invariant and is not an order parameter for them.
6. **RG, one sentence at most:** "β_layer denotes a depth finite difference; we do not interpret it as a renormalization-group beta function," optionally citing Roberts et al. for the contrast with autonomous depth recursions at initialisation.

**Do not include:** Mehta–Schwab, Koch-Janusz–Ringel, NN-QFT, the Fibonacci classes, E10/billiards, Salem/Pisot or spectral triples. None of them is supported by a mechanism or test on this data.

---

## Testable on existing telemetry (g, gap, entropy per layer × step × generation)

Each test is listed with its source literature, the statistic, the null and the decision rule. "Existing" means it needs only the cached g, gap and sv_entropy plus σ and prompt. Tests marked **[+data]** need small extra inputs, and they are flagged as such.

### T0. Estimator sanity (Rényi ordering)
- **Source:** Roy–Vetterli; the Rényi identities.
- **Check:** H₁ ≥ H₂, and ln gap ≥ 0, on every row (= preregistration L0).
- **Null:** none; this is an identity. Any violation indicates a capture or rounding bug.

### T1. Finite-sample isotropy null
- **Source:** RMT (§0.3).
- **Statistic:** report g / g_MP(D/N) per channel, with g_MP from the singular-value MP formula using that channel's actual N (tokens per capture) and D.
- **Decision:** treat statements like "near-isotropic" as valid only if g is within the 95% band of g_MP, estimated by simulating Gaussian X with the same N and D.

### T2. Outlier dominance (spike-plus-bulk) [model-based on existing telemetry; exact with +data]
- **Sources:** Sun et al.; Gan et al.; Ethayarajh.
- **Statistic:** from (H₂, ln gap), infer the spike strength under the spike-plus-power-law model (preregistration P1b, SPL). Report the fraction of g's depth and step variance explained by the spike parameter alone versus the bulk exponent.
- **Null:** a spike-free power-law model.
- **Decision:** if the spike explains > 50% of the variance of g across layers, the "depth shape" should be described as outlier-driven.
- **[+data]:** recompute g on a handful of stored activations with the mean removed and with the top-1 or top-3 directions projected out.

### T3. Forward-process baseline for the σ-trajectory [+data: latent second-moment spectrum of ~1k encoded images per model; or use the patch-embed or first-layer g as a proxy]
- **Sources:** Rissanen, Hoogeboom, Falck, Wang–Vastola, and the HK identity in §B.
- **Statistic:** predicted input g(t) from λᵢ → (1−t)²λᵢ + t² (rectified flow), against observed g at each layer.
- **Null:** observed = input-level prediction, i.e. the network merely propagates the noise mixture.
- **Decision:** report per layer the step range where |g_obs − g_input| exceeds 3 prompt-bootstrap SE. Only that excess is a "model-generated" coarse-to-fine trajectory.
- **Connection to the preregistration:** this sharpens L3. HK is the *exact* input-level deformation, so HK beating TH at early layers is expected under the null and is not evidence of heat-kernel physics.

### T4. Autonomy of the depth recursion (is there a one-coupling flow?)
- **Sources:** Roberts–Yaida–Hanin; the RG background (Kadanoff, Mehta–Schwab).
- **Statistic:** = preregistration R1. Out-of-sample R² of β = f(g) versus f(g) + layer effect, grouped by prompt.
- **Null:** autonomy, ΔR² < 0.05.
- **Decision:** if autonomy fails (as preregistered), "crossings" are zeros of a layer-pooled average. They must not be called fixed points.
- **Additional check:** repeat within early, mid and late σ bins. Autonomy might hold only at high σ, where the denoiser is near-linear (Wang–Vastola).

### T5. Reparametrisation-invariant content of crossings
- **Sources:** RG (fixed-point eigenvalues are invariant under smooth coupling redefinitions); dynamical systems (F′(g*) is invariant under smooth conjugacy of a 1D map).
- **Statistic:** at each crossing, the multiplier m = 1 + dβ/dg (the layer map's derivative), with generation-bootstrap SEs (preregistration R2).
- **Note on R3:** cross-ratios are invariant only under Möbius maps. H₂ = ln(gD) is not a Möbius image of g, so CR agreement in g but not in H₂ would be an artefact of the parametrisation. Report CR in both g and H₂.
- **Null:** multipliers differ across families by > 2 SE.
- **Decision:** a "universal" crossing requires matching m (not only matching g*) in ≥ 2 families, *and* T4 autonomy.

### T6. Speciation-time prediction [+data: largest eigenvalue Λ of the scaled latent data covariance per model]
- **Source:** Biroli et al. 2024.
- **Prediction:** t_S = √Λ/(1+√Λ) in rectified-flow t (my transfer of Λe^(−2t_S) = 1; see C1).
- **Statistic:** the step of maximal |d ln gap / d ln σ| (layer-averaged), and of the within-prompt variance χ_w(σ) (preregistration L4), compared to σ_S.
- **Null:** the feature location is uniform over the σ grid. The p-value is the probability of falling within ±1 step of σ_S by chance, corrected for the number of features examined.
- **Decision:** a match in ≥ 2 families. A single-channel match is anecdotal.

### T7. Susceptibility (across-seed variance) peak
- **Sources:** Raya–Ambrogioni; Ambrogioni (critical instability); Sclocchi et al.; Li–Chen.
- **Statistic:** = preregistration L4 χ_w. Within-prompt, across-seed variance of layer-mean g (and of gap and H₁) as a function of σ.
- **Null:** a generation-label permutation within prompt, plus a bounded-statistic null. Variance of a statistic confined to [1/D, 1] can peak mid-trajectory simply because the mean passes through mid-range; normalise by the binomial-like m(1−m), or use logit(g).
- **Decision:** an interior peak that survives normalisation in ≥ 2 families.
- **Interpretive constraint (C2):** symmetric branch selection cannot produce a peak. A peak indicates *asymmetric* (content-dependent) branching.

### T8. Spectral shape versus g across widths
- **Sources:** Martin–Mahoney; Stringer et al.; α-ReQ.
- **Statistic:** the SPL bulk exponent α per channel (preregistration P1b/P1c), and the predicted g(α, D), against observed g.
- **Null:** the same α across families, with g differing only through D.
- **Decision:** if the differences in g across models are explained by D at fixed α, cross-model statements must be about α, not g.

### T9. Depth-profile placement versus prior representation-geometry results
- **Sources:** Ansuini; Valeriani; Razzhigaev; Skean; Dong/Noci.
- **Statistic:** relative depth of the g extremum (and of inflections) per channel at the final step, with prompt-bootstrap CI. Compare to the published peak locations (qualitatively; those used other estimators).
- **Null:** extremum positions uniformly distributed in relative depth across the 5 channels and 7 LLMs.
- **Decision:** call the shape "recurring" only if the CIs overlap across families beyond the uniform null (e.g. a KS test on relative positions).

### T10. Ladder numerology guard (Fibonacci / Kepler, Coxeter, Salem)
- **Sources:** Popkov et al.; Schütz; McMullen.
- **Statistic:** each dimensionless exponent with bootstrap SE (preregistration L6), matched to a *frozen* target set.
- **Null:** a log-uniform chance-coverage, Poisson-binomial calculation over the actual target density (high near φ for the Kepler set).
- **Decision:** a "ladder" requires p < 0.05 after look-elsewhere correction, cadence invariance (every-second-step subsampling), recurrence in ≥ 2 families, *and* a stated mechanism. No mechanism is available in the literature, so a positive result would be reported as a coincidence to be replicated, not as a universality class.

### Not testable on the current telemetry (would need stored activations, outputs or training data)
- Replica overlaps of activations or samples, and ultrametricity of real configurations (Parisi). The preregistration L5 on g-fields is descriptive only.
- Collapse or memorisation times (Biroli t_C, Achilli, Pham, Bonnaire): need nearest-training-sample distances.
- Score-Jacobian spectral gaps (Ventura): need Jacobian-vector products.
- Heat-trace / IMD / diffusion-geometry multi-scale descriptors (Tsitsulin; Khandait & Gerken): need token-level activations.
- Mutual-information RG (Koch-Janusz & Ringel) and noise-then-denoise hierarchy probes (Sclocchi et al. 2025): need new runs.

---

## BibTeX (verified entries only)

```bibtex
@article{kadanoff1966scaling,
  author  = {Kadanoff, Leo P.},
  title   = {Scaling laws for {Ising} models near {$T_c$}},
  journal = {Physics Physique Fizika},
  volume  = {2},
  pages   = {263--272},
  year    = {1966},
  doi     = {10.1103/PhysicsPhysiqueFizika.2.263}
}

@misc{beny2013deep,
  author        = {B{\'e}ny, C{\'e}dric},
  title         = {Deep learning and the renormalization group},
  year          = {2013},
  eprint        = {1301.3124},
  archivePrefix = {arXiv}
}

@misc{mehta2014exact,
  author        = {Mehta, Pankaj and Schwab, David J.},
  title         = {An exact mapping between the Variational Renormalization Group and Deep Learning},
  year          = {2014},
  eprint        = {1410.3831},
  archivePrefix = {arXiv}
}

@article{lin2017deep,
  author  = {Lin, Henry W. and Tegmark, Max and Rolnick, David},
  title   = {Why does deep and cheap learning work so well?},
  journal = {Journal of Statistical Physics},
  year    = {2017},
  doi     = {10.1007/s10955-017-1836-5},
  eprint  = {1608.08225},
  archivePrefix = {arXiv}
}

@article{kochjanusz2018mutual,
  author  = {Koch-Janusz, Maciej and Ringel, Zohar},
  title   = {Mutual information, neural networks and the renormalization group},
  journal = {Nature Physics},
  year    = {2018},
  doi     = {10.1038/s41567-018-0081-4},
  eprint  = {1704.06279},
  archivePrefix = {arXiv}
}

@book{roberts2022principles,
  author    = {Roberts, Daniel A. and Yaida, Sho and Hanin, Boris},
  title     = {The Principles of Deep Learning Theory},
  publisher = {Cambridge University Press},
  year      = {2022},
  doi       = {10.1017/9781009023405},
  eprint    = {2106.10165},
  archivePrefix = {arXiv}
}

@article{halverson2021neural,
  author  = {Halverson, James and Maiti, Anindita and Stoner, Keegan},
  title   = {Neural networks and quantum field theory},
  journal = {Machine Learning: Science and Technology},
  year    = {2021},
  doi     = {10.1088/2632-2153/abeca3},
  eprint  = {2008.08601},
  archivePrefix = {arXiv}
}

@article{erbin2022nonperturbative,
  author  = {Erbin, Harold and Lahoche, Vincent and Samary, Dine Ousmane},
  title   = {Nonperturbative renormalization for the neural network-{QFT} correspondence},
  journal = {Machine Learning: Science and Technology},
  year    = {2022},
  doi     = {10.1088/2632-2153/ac4f69},
  eprint  = {2108.01403},
  archivePrefix = {arXiv}
}

@misc{cotler2023renormalizing,
  author        = {Cotler, Jordan and Rezchikov, Semon},
  title         = {Renormalizing Diffusion Models},
  year          = {2023},
  eprint        = {2308.12355},
  archivePrefix = {arXiv}
}

@misc{masuki2025generative,
  author        = {Masuki, Kanta and Ashida, Yuto},
  title         = {Generative diffusion model with inverse renormalization group flows},
  year          = {2025},
  eprint        = {2501.09064},
  archivePrefix = {arXiv}
}

@inproceedings{rissanen2023generative,
  author    = {Rissanen, Severi and Heinonen, Markus and Solin, Arno},
  title     = {Generative Modelling With Inverse Heat Dissipation},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  eprint    = {2206.13397},
  archivePrefix = {arXiv}
}

@misc{hoogeboom2022blurring,
  author        = {Hoogeboom, Emiel and Salimans, Tim},
  title         = {Blurring Diffusion Models},
  year          = {2022},
  eprint        = {2209.05557},
  archivePrefix = {arXiv}
}

@misc{falck2025fourier,
  author        = {Falck, Fabian and Pandeva, Teodora and Zahirnia, Kiarash and Lawrence, Rachel and Turner, Richard and Meeds, Edward and Zazo, Javier and Karmalkar, Sushrut},
  title         = {A Fourier Space Perspective on Diffusion Models},
  year          = {2025},
  eprint        = {2505.11278},
  archivePrefix = {arXiv}
}

@misc{wang2023hidden,
  author        = {Wang, Binxu and Vastola, John J.},
  title         = {The Hidden Linear Structure in Score-Based Models and its Application},
  year          = {2023},
  eprint        = {2311.10892},
  archivePrefix = {arXiv}
}

@misc{sclocchi2024phase,
  author        = {Sclocchi, Antonio and Favero, Alessandro and Wyart, Matthieu},
  title         = {A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data},
  year          = {2024},
  eprint        = {2402.16991},
  archivePrefix = {arXiv}
}

@misc{sclocchi2025probing,
  author        = {Sclocchi, Antonio and Favero, Alessandro and Levi, Noam Itzhak and Wyart, Matthieu},
  title         = {Probing the Latent Hierarchical Structure of Data via Diffusion Models},
  year          = {2024},
  eprint        = {2410.13770},
  archivePrefix = {arXiv}
}

@misc{li2024critical,
  author        = {Li, Marvin and Chen, Sitan},
  title         = {Critical windows: non-asymptotic theory for feature emergence in diffusion models},
  year          = {2024},
  eprint        = {2403.01633},
  archivePrefix = {arXiv}
}

@article{martin2021implicit,
  author  = {Martin, Charles H. and Mahoney, Michael W.},
  title   = {Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  number  = {165},
  pages   = {1--73},
  year    = {2021},
  eprint  = {1810.01075},
  archivePrefix = {arXiv}
}

@article{martin2021predicting,
  author  = {Martin, Charles H. and Peng, Tongsu and Mahoney, Michael W.},
  title   = {Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data},
  journal = {Nature Communications},
  year    = {2021},
  doi     = {10.1038/s41467-021-24025-8},
  eprint  = {2002.06716},
  archivePrefix = {arXiv}
}

@article{stringer2019high,
  author  = {Stringer, Carsen and Pachitariu, Marius and Steinmetz, Nicholas and Carandini, Matteo and Harris, Kenneth D.},
  title   = {High-dimensional geometry of population responses in visual cortex},
  journal = {Nature},
  volume  = {571},
  pages   = {361--365},
  year    = {2019},
  doi     = {10.1038/s41586-019-1346-5}
}

@inproceedings{agrawal2022alphareq,
  author    = {Agrawal, Kumar K. and Mondal, Arnab Kumar and Ghosh, Arna and Richards, Blake},
  title     = {$\alpha$-{ReQ}: Assessing Representation Quality in Self-Supervised Learning by measuring eigenspectrum decay},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  year      = {2022}
}

@misc{ettori2026spectral,
  author        = {Ettori, Davide},
  title         = {Spectral Geometry for Deep Learning: Compression and Hallucination Detection via Random Matrix Theory},
  year          = {2026},
  eprint        = {2601.17357},
  archivePrefix = {arXiv}
}

@inproceedings{roy2007effective,
  author    = {Roy, Olivier and Vetterli, Martin},
  title     = {The Effective Rank: a Measure of Effective Dimensionality},
  booktitle = {European Signal Processing Conference (EUSIPCO)},
  address   = {Poznan, Poland},
  year      = {2007},
  doi       = {10.5281/zenodo.40328}
}

@misc{gao2017theory,
  author = {Gao, Peiran and Trautmann, Eric and Yu, Byron and Santhanam, Gopal and Ryu, Stephen and Shenoy, Krishna and Ganguli, Surya},
  title  = {A theory of multineuronal dimensionality, dynamics and measurement},
  year   = {2017},
  howpublished = {bioRxiv},
  doi    = {10.1101/214262}
}

@misc{recanatesi2019dimensionality,
  author        = {Recanatesi, Stefano and Farrell, Matthew and Advani, Madhu and Moore, Timothy and Lajoie, Guillaume and Shea-Brown, Eric},
  title         = {Dimensionality compression and expansion in Deep Neural Networks},
  year          = {2019},
  eprint        = {1906.00443},
  archivePrefix = {arXiv}
}

@inproceedings{ansuini2019intrinsic,
  author    = {Ansuini, Alessio and Laio, Alessandro and Macke, Jakob H. and Zoccolan, Davide},
  title     = {Intrinsic dimension of data representations in deep neural networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
  eprint    = {1905.12784},
  archivePrefix = {arXiv}
}

@misc{valeriani2023geometry,
  author        = {Valeriani, Lucrezia and Doimo, Diego and Cuturello, Francesca and Laio, Alessandro and Ansuini, Alessio and Cazzaniga, Alberto},
  title         = {The geometry of hidden representations of large transformer models},
  year          = {2023},
  eprint        = {2302.00294},
  archivePrefix = {arXiv}
}

@misc{razzhigaev2023shape,
  author        = {Razzhigaev, Anton and Mikhalchuk, Matvey and Goncharova, Elizaveta and Oseledets, Ivan and Dimitrov, Denis and Kuznetsov, Andrey},
  title         = {The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models},
  year          = {2023},
  eprint        = {2311.05928},
  archivePrefix = {arXiv}
}

@misc{skean2025layer,
  author        = {Skean, Oscar and Arefin, Md Rifat and Zhao, Dan and Patel, Niket and Naghiyev, Jalal and LeCun, Yann and Shwartz-Ziv, Ravid},
  title         = {Layer by Layer: Uncovering Hidden Representations in Language Models},
  year          = {2025},
  eprint        = {2502.02013},
  archivePrefix = {arXiv}
}

@misc{dong2021attention,
  author        = {Dong, Yihe and Cordonnier, Jean-Baptiste and Loukas, Andreas},
  title         = {Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth},
  year          = {2021},
  eprint        = {2103.03404},
  archivePrefix = {arXiv}
}

@misc{noci2022signal,
  author        = {Noci, Lorenzo and Anagnostidis, Sotiris and Biggio, Luca and Orvieto, Antonio and Singh, Sidak Pal and Lucchi, Aurelien},
  title         = {Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse},
  year          = {2022},
  eprint        = {2206.03126},
  archivePrefix = {arXiv}
}

@misc{garrido2022rankme,
  author        = {Garrido, Quentin and Balestriero, Randall and Najman, Laurent and LeCun, Yann},
  title         = {{RankMe}: Assessing the downstream performance of pretrained self-supervised representations by their rank},
  year          = {2022},
  eprint        = {2210.02885},
  archivePrefix = {arXiv}
}

@misc{wei2024differank,
  author        = {Wei, Lai and Tan, Zhiquan and Li, Chenghai and Wang, Jindong and Huang, Weiran},
  title         = {{Diff-eRank}: A Novel Rank-Based Metric for Evaluating Large Language Models},
  year          = {2024},
  eprint        = {2401.17139},
  archivePrefix = {arXiv}
}

@misc{sun2024massive,
  author        = {Sun, Mingjie and Chen, Xinlei and Kolter, J. Zico and Liu, Zhuang},
  title         = {Massive Activations in Large Language Models},
  year          = {2024},
  eprint        = {2402.17762},
  archivePrefix = {arXiv}
}

@misc{gan2025unleashing,
  author        = {Gan, Chaofan and Tu, Yuanpeng and Chen, Xi and Chen, Tieyuan and Li, Yuxi and Harandi, Mehrtash and Lin, Weiyao},
  title         = {Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations},
  year          = {2025},
  eprint        = {2505.18584},
  archivePrefix = {arXiv}
}

@misc{ethayarajh2019contextual,
  author        = {Ethayarajh, Kawin},
  title         = {How Contextual are Contextualized Word Representations? Comparing the Geometry of {BERT}, {ELMo}, and {GPT-2} Embeddings},
  year          = {2019},
  eprint        = {1909.00512},
  archivePrefix = {arXiv}
}

@inproceedings{tsitsulin2020shape,
  author    = {Tsitsulin, Anton and Munkhoeva, Marina and Mottin, Davide and Karras, Panagiotis and Bronstein, Alex and Oseledets, Ivan and M{\"u}ller, Emmanuel},
  title     = {The Shape of Data: Intrinsic Distance for Data Distributions},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  eprint    = {1905.11141},
  archivePrefix = {arXiv}
}

@misc{khandait2026layers,
  author        = {Khandait, Atharva and Gerken, Jan E.},
  title         = {From Layers to Networks: Comparing Neural Representations via Diffusion Geometry},
  year          = {2026},
  eprint        = {2605.15901},
  archivePrefix = {arXiv}
}

@misc{najem2025geometric,
  author        = {Najem, Sara and Mrad, Dima and Elsayed, Mohammad},
  title         = {Geometric Features of Higher-Order Networks via the Spectral Triplet},
  year          = {2025},
  eprint        = {2509.04311},
  archivePrefix = {arXiv}
}

@article{biroli2023generative,
  author  = {Biroli, Giulio and M{\'e}zard, Marc},
  title   = {Generative diffusion in very large dimensions},
  journal = {Journal of Statistical Mechanics: Theory and Experiment},
  pages   = {093402},
  year    = {2023},
  doi     = {10.1088/1742-5468/acf8ba},
  eprint  = {2306.03518},
  archivePrefix = {arXiv}
}

@article{biroli2024dynamical,
  author  = {Biroli, Giulio and Bonnaire, Tony and de Bortoli, Valentin and M{\'e}zard, Marc},
  title   = {Dynamical regimes of diffusion models},
  journal = {Nature Communications},
  year    = {2024},
  doi     = {10.1038/s41467-024-54281-3},
  eprint  = {2402.18491},
  archivePrefix = {arXiv}
}

@inproceedings{raya2023spontaneous,
  author    = {Raya, Gabriel and Ambrogioni, Luca},
  title     = {Spontaneous Symmetry Breaking in Generative Diffusion Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2305.19693},
  archivePrefix = {arXiv}
}

@misc{ambrogioni2023statistical,
  author        = {Ambrogioni, Luca},
  title         = {The statistical thermodynamics of generative diffusion models: Phase transitions, symmetry breaking and critical instability},
  year          = {2023},
  eprint        = {2310.17467},
  archivePrefix = {arXiv}
}

@misc{ventura2024manifolds,
  author        = {Ventura, Enrico and Achilli, Beatrice and Silvestri, Gianluigi and Lucibello, Carlo and Ambrogioni, Luca},
  title         = {Manifolds, Random Matrices and Spectral Gaps: The geometric phases of generative diffusion},
  year          = {2024},
  eprint        = {2410.05898},
  archivePrefix = {arXiv}
}

@misc{achilli2024losing,
  author        = {Achilli, Beatrice and Ventura, Enrico and Silvestri, Gianluigi and Pham, Bao and Raya, Gabriel and Krotov, Dmitry and Lucibello, Carlo and Ambrogioni, Luca},
  title         = {Losing dimensions: Geometric memorization in generative diffusion},
  year          = {2024},
  eprint        = {2410.08727},
  archivePrefix = {arXiv}
}

@misc{hoover2023memory,
  author        = {Hoover, Benjamin and Strobelt, Hendrik and Krotov, Dmitry and Hoffman, Judy and Kira, Zsolt and Chau, Duen Horng},
  title         = {Memory in Plain Sight: Surveying the Uncanny Resemblances of Associative Memories and Diffusion Models},
  year          = {2023},
  eprint        = {2309.16750},
  archivePrefix = {arXiv}
}

@misc{ambrogioni2023search,
  author        = {Ambrogioni, Luca},
  title         = {In search of dispersed memories: Generative diffusion models are associative memory networks},
  year          = {2023},
  eprint        = {2309.17290},
  archivePrefix = {arXiv}
}

@misc{pham2025memorization,
  author        = {Pham, Bao and Raya, Gabriel and Negri, Matteo and Zaki, Mohammed J. and Ambrogioni, Luca and Krotov, Dmitry},
  title         = {Memorization to Generalization: Emergence of Diffusion Models from Associative Memory},
  year          = {2025},
  eprint        = {2505.21777},
  archivePrefix = {arXiv}
}

@misc{bonnaire2025why,
  author        = {Bonnaire, Tony and Urfin, Rapha{\"e}l and Biroli, Giulio and M{\'e}zard, Marc},
  title         = {Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training},
  year          = {2025},
  eprint        = {2505.17638},
  archivePrefix = {arXiv}
}

@misc{kadkhodaie2023generalization,
  author        = {Kadkhodaie, Zahra and Guth, Florentin and Simoncelli, Eero P. and Mallat, St{\'e}phane},
  title         = {Generalization in diffusion models arises from geometry-adaptive harmonic representations},
  year          = {2023},
  eprint        = {2310.02557},
  archivePrefix = {arXiv}
}

@article{popkov2015fibonacci,
  author  = {Popkov, Vladislav and Schadschneider, Andreas and Schmidt, Johannes and Sch{\"u}tz, Gunter M.},
  title   = {Fibonacci family of dynamical universality classes},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {112},
  number  = {41},
  pages   = {12645--12650},
  year    = {2015},
  doi     = {10.1073/pnas.1512261112},
  eprint  = {1505.04461},
  archivePrefix = {arXiv}
}

@misc{schutz2017fibonacci,
  author        = {Sch{\"u}tz, Gunter M.},
  title         = {On the {Fibonacci} universality classes in nonlinear fluctuating hydrodynamics},
  year          = {2017},
  eprint        = {1710.09121},
  archivePrefix = {arXiv}
}

@article{popkov2024quest,
  author  = {Popkov, V. and Sch{\"u}tz, G. M.},
  title   = {Quest for the golden ratio universality class},
  journal = {Physical Review E},
  volume  = {109},
  pages   = {044111},
  year    = {2024},
  doi     = {10.1103/PhysRevE.109.044111},
  eprint  = {2310.19116},
  archivePrefix = {arXiv}
}

@misc{chen2023machine,
  author        = {Chen, Siqi and Dechant, Pierre-Philippe and He, Yang-Hui and Heyes, Elli and Hirst, Edward and Riabchenko, Dmitrii},
  title         = {Machine Learning {Clifford} invariants of {ADE} {Coxeter} elements},
  year          = {2023},
  eprint        = {2310.00041},
  archivePrefix = {arXiv}
}

@article{damour2002e10,
  author  = {Damour, Thibault and Henneaux, Marc and Nicolai, Hermann},
  title   = {{E}10 and a ``small tension expansion'' of {M} theory},
  journal = {Physical Review Letters},
  volume  = {89},
  pages   = {221601},
  year    = {2002},
  doi     = {10.1103/PhysRevLett.89.221601},
  eprint  = {hep-th/0207267},
  archivePrefix = {arXiv}
}

@misc{damour2003cosmological,
  author        = {Damour, Thibault and Henneaux, Marc and Nicolai, Hermann},
  title         = {Cosmological Billiards},
  year          = {2002},
  eprint        = {hep-th/0212256},
  archivePrefix = {arXiv}
}

@incollection{damour2007chaos,
  author        = {Damour, Thibault},
  title         = {Chaos and Symmetry in String Cosmology},
  year          = {2007},
  doi           = {10.1142/9789812834300_0003},
  eprint        = {0704.1457},
  archivePrefix = {arXiv}
}

@misc{kleinschmidt2025tensor,
  author = {Kleinschmidt, Axel and Malcha, Hannes and Nicolai, Hermann},
  title  = {From tensor algebras to hyperbolic {Kac-Moody} algebras},
  howpublished = {Journal of High Energy Physics 2025, article 124},
  year   = {2025},
  doi    = {10.1007/JHEP12(2025)124}
}

@article{mcmullen2002coxeter,
  author  = {McMullen, Curtis T.},
  title   = {Coxeter groups, {Salem} numbers and the {Hilbert} metric},
  journal = {Publications Math{\'e}matiques de l'IH{\'E}S},
  volume  = {95},
  pages   = {151--183},
  year    = {2002},
  doi     = {10.1007/s102400200001}
}

@article{mcmullen2002k3,
  author  = {McMullen, Curtis T.},
  title   = {Dynamics on {K3} surfaces: {Salem} numbers and {Siegel} disks},
  journal = {Journal f{\"u}r die reine und angewandte Mathematik},
  volume  = {545},
  pages   = {201--233},
  year    = {2002},
  doi     = {10.1515/crll.2002.036}
}

@misc{oguiso2009third,
  author        = {Oguiso, Keiji},
  title         = {The third smallest {Salem} number in automorphisms of {K3} surfaces},
  year          = {2009},
  eprint        = {0905.2396},
  archivePrefix = {arXiv}
}

@misc{hironaka_lehmer,
  author = {Hironaka, Eriko},
  title  = {What is ... {Lehmer}'s number?},
  note   = {Expository note; PDF at https://www.math.fsu.edu/~aluffi/archive/paper355.pdf (venue not verified)}
}

@misc{akiyama_pisot,
  author = {Akiyama, Shigeki and Barge, Marcy and Berth{\'e}, Val{\'e}rie and Lee, Jeong-Yup and Siegel, Anne},
  title  = {On the {Pisot} substitution conjecture},
  note   = {Survey chapter; PDF at https://www.irif.fr/~berthe/Articles/AkiyamaBargeBertheLeeSiegel.pdf (book venue not verified)}
}
```
