# Citation keys: CoS-PAFS Mode-I and Mode-II

Generated 2026-10-08 with `references_master.bib`. 129 entries: 110 in `mode-I/references.bib`, 107 in `mode-II/references.bib` (overlap allowed).

Columns: **key** | **short reference** | **paper(s) and section(s)** (outline numbering in `_sources/MODE_I_FINAL_OUTLINE.md` and `MODE_II_FINAL_OUTLINE.md`) | **what it may support** | **verification URL** (the page checked; "LIT_R6C x.y" means a page verified in that earlier round and reused).

Project rule: external work positions, delimits or motivates. It never upgrades a blocked claim (commitment/separatrix, a universal microscopic law, RG universality, geometry to quality).

Key renames relative to the current `main.tex` drafts: `zimage2025` should be `zimageteam2025zimage`; `qwen2025qwen25` should be `qwen2024qwen25` (arXiv v1 is 2024-12-19); `turri2026fewchannels` should be `turri2026few`. All other keys cited in the drafts exist here.

## 1. Models, samplers and generative formulations

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `zimageteam2025zimage` | Z-Image Team et al. 2025, Z-Image tech report (arXiv:2511.22699) | I, II: I 2.2; II 1 | Z-Image S3-DiT architecture; Qwen3-4B text encoder; Turbo = few-step distillation of the SFT model plus reward post-training. Do not write that Turbo is the student of the released Base checkpoint. | https://arxiv.org/abs/2511.22699 |
| `tongyi2025zimageturbo` | Tongyi-MAI 2025, Z-Image-Turbo model card | I, II: I 2.2; II 7 | 8 NFEs (num_inference_steps=9), guidance_scale 0, CFG not used, RL post-training. Cite for the ZIT inference settings. | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| `tongyi2026zimage` | Tongyi-MAI 2026, Z-Image (Base) model card | I, II: I 2.2; II 7 | Undistilled 6B base model, 28-50 steps, guidance 3-5, full CFG. Official scheduler shift is 6.0; the captures used ComfyUI shift 3.0 (claim register M4). | https://huggingface.co/Tongyi-MAI/Z-Image ; release date from https://github.com/Tongyi-MAI/Z-Image (LIT_R6C 1.4) |
| `liu2025decoupled` | Liu D. et al. 2025, Decoupled DMD (arXiv:2511.22677) | I, II: I 2.2, 8; II 7 | The distillation algorithm behind ZIT (CFG augmentation as engine, distribution matching as regularizer). Explains why ZIT runs without CFG. | https://arxiv.org/abs/2511.22677 |
| `jiang2025distribution` | Jiang D. et al. 2025, DMDR (arXiv:2511.13649) | I, II: I 2.2, 8; II 7 | DMD + RL post-training named by Z-Image. No page says ZIT used exactly DMDR; write 'reward post-training' for the RL step. | https://arxiv.org/abs/2511.13649 |
| `yang2025qwen3` | Yang A. et al. 2025, Qwen3 Technical Report (arXiv:2505.09388) | I: I 2.2 | Qwen3-4B is the shared, byte-identical text encoder of ZIT and ZIB (LIT_R6C 1.5). | https://arxiv.org/abs/2505.09388 |
| `genmo2024mochi` | Genmo Team 2024, Mochi 1 (model card and code) | I, II: I 2.2; II 1 | 10B AsymmDiT, 48 layers, visual width 3072 / text width 1536, single T5-XXL encoder, Apache-2.0. | https://huggingface.co/genmo/mochi-1-preview ; https://github.com/genmoai/mochi |
| `raffel2020exploring` | Raffel et al. 2020, T5 (JMLR 21) | I: I 2.2 | T5 text encoder family (T5-v1.1-XXL is shared by Mochi and SD3.5). | https://mlanthology.org/jmlr/2020/raffel2020jmlr-exploring/ ; https://arxiv.org/abs/1910.10683 |
| `esser2024scaling` | Esser et al. 2024, Scaling Rectified Flow Transformers (SD3, ICML) | I, II: I 2.2, 2.3, 8; II 1 | MMDiT and rectified-flow formulation of SD3/SD3.5; resolution-dependent timestep shift (Sec. 5.3.2) for the resolution caveat. | https://proceedings.mlr.press/v235/esser24a.html ; https://arxiv.org/abs/2403.03206 |
| `stabilityai2024sd35medium` | Stability AI 2024, SD3.5 Medium model card | I: I 2.2 | MMDiT-X, three text encoders (CLIP-L, OpenCLIP-bigG, T5-XXL). Card says 2B params, the release blog says 2.5B: quote the card and name the source. | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium ; https://stability.ai/news/introducing-stable-diffusion-3-5 |
| `stabilityai2024sd35large` | Stability AI 2024, SD3.5 Large model card | I: I 2.2 | 8B MMDiT with QK-normalization; same three text encoders. | https://huggingface.co/stabilityai/stable-diffusion-3.5-large |
| `peebles2023scalable` | Peebles & Xie 2023, DiT (ICCV) | I, II: I 1; II 1 | Diffusion transformer backbone; defines the object of study. | https://mlanthology.org/iccv/2023/peebles2023iccv-scalable/ ; https://arxiv.org/abs/2212.09748 |
| `vaswani2017attention` | Vaswani et al. 2017, Transformer (NeurIPS) | I, II: I 2.1; II 2 | Attention / MLP sublayer terminology for the capture hooks. | https://mlanthology.org/neurips/2017/vaswani2017neurips-attention/ ; https://arxiv.org/abs/1706.03762 |
| `ho2020denoising` | Ho, Jain & Abbeel 2020, DDPM (NeurIPS) | I, II: I 1; II 1 | Diffusion background. | https://mlanthology.org/neurips/2020/ho2020neurips-denoising/ ; https://arxiv.org/abs/2006.11239 |
| `song2021scorebased` | Song et al. 2021, Score SDE (ICLR) | I, II: I 2.3; II 1, 5 | Probability-flow ODE view; justifies treating sigma as the sampling clock. | https://mlanthology.org/iclr/2021/song2021iclr-scorebased/ ; https://arxiv.org/abs/2011.13456 |
| `karras2022elucidating` | Karras et al. 2022, EDM (NeurIPS) | I, II: I 2.3; II 1 | sigma parameterization and step-size/curvature effects; supports equal-sigma registration as a coordinate choice. | https://mlanthology.org/neurips/2022/karras2022neurips-elucidating/ ; https://arxiv.org/abs/2206.00364 |
| `lipman2023flow` | Lipman et al. 2023, Flow Matching (ICLR) | I, II: I 2.3; II 1 | Flow-matching formulation used by the studied models. | https://mlanthology.org/iclr/2023/lipman2023iclr-flow/ ; https://arxiv.org/abs/2210.02747 |
| `liu2023flow` | Liu, Gong & Liu 2023, Rectified Flow (ICLR) | I, II: I 2.3; II 1 | Rectified-flow formulation (straight paths; un-reflowed models are not exactly straight). | https://mlanthology.org/iclr/2023/liu2023iclr-flow/ ; https://arxiv.org/abs/2209.03003 |
| `ho2022classifierfree` | Ho & Salimans 2022, Classifier-Free Guidance (arXiv) | I, II: I 2.1, 6.1, 8; II 2.3 | Defines cond/uncond branches, hence the pooled / cond-only / branch-mean capture regimes. | https://arxiv.org/abs/2207.12598 |
| `kynkaanniemi2024applying` | Kynkaanniemi et al. 2024, Guidance interval (NeurIPS) | I: I 6.1, 8 | Guidance is harmful early, useful mid-chain; positions the CFG 1 vs 4 controls. Do not use for guidance-timing claims (Mode-III rider). | https://mlanthology.org/neurips/2024/kynkaanniemi2024neurips-applying/ ; https://arxiv.org/abs/2404.07724 |
| `hoogeboom2023simple` | Hoogeboom, Heek & Salimans 2023, simple diffusion (ICML) | I: I 8 | Same nominal noise level is a different effective SNR at another resolution; qualifies the SD3.5 512/768 resolution evidence. | https://proceedings.mlr.press/v202/hoogeboom23a.html |
| `comfyui2023` | comfyanonymous / Comfy-Org 2023, ComfyUI (software) | I: I 2.2, 2.4 | Capture runtime; cite with the exact commit hash. Its ModelSamplingSD3 applies a fixed shift (3.0) with no resolution term (LIT_R5 Sec. 2). | https://github.com/comfyanonymous/ComfyUI ; initial commit date from https://blog.comfy.org/p/comfyui-turns-2-a-journey-and-call |

## 2. Distillation and representation preservation

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `hinton2015distilling` | Hinton, Vinyals & Dean 2015, Knowledge distillation (arXiv) | I, II: I 8; II 7 | General definition of teacher-student distillation. | https://arxiv.org/abs/1503.02531 |
| `salimans2022progressive` | Salimans & Ho 2022, Progressive Distillation (ICLR) | I, II: I 8; II 7 | Trajectory-matching distillation, a contrast to DMD-family students. | https://mlanthology.org/iclr/2022/salimans2022iclr-progressive/ ; https://arxiv.org/abs/2202.00512 |
| `song2023consistency` | Song et al. 2023, Consistency Models (ICML) | I, II: I 8; II 7 | Few-step distillation family; context only. | https://proceedings.mlr.press/v202/song23a.html |
| `yin2024onestep` | Yin et al. 2024, DMD (CVPR) | I, II: I 2.2, 8; II 7 | Distribution Matching Distillation, the base of Decoupled-DMD. | https://mlanthology.org/cvpr/2024/yin2024cvpr-onestep/ ; https://arxiv.org/abs/2311.18828 |
| `yin2024improved` | Yin et al. 2024, DMD2 (NeurIPS) | I, II: I 8; II 7 | DMD2 drops the teacher-pair regression loss, so the student is less tied to teacher trajectories. | https://mlanthology.org/neurips/2024/yin2024neurips-improved/ ; https://arxiv.org/abs/2405.14867 |
| `sauer2024adversarial` | Sauer et al. 2024, Adversarial Diffusion Distillation (ECCV) | I, II: I 8; II 7 | Adversarial few-step distillation; context only. | https://mlanthology.org/eccv/2024/sauer2024eccv-adversarial/ ; https://arxiv.org/abs/2311.17042 |
| `sauer2024fast` | Sauer et al. 2024, LADD (arXiv:2403.12015) | I, II: I 8; II 7 | Latent adversarial distillation used for SD3-family turbo models; context only. Venue not verified, cite as arXiv. | https://arxiv.org/abs/2403.12015 |
| `gandikota2026distilling` | Gandikota & Bau 2026, Distilling Diversity and Control (WACV) | I, II: I 7, 8; II 7 | Distilled students keep base concept representations, yet commit to structure at the first step. Prior for strong Base-Turbo inheritance. Must not be used to claim our telemetry measured commitment. | https://openaccess.thecvf.com/content/WACV2026/html/Gandikota_Distilling_Diversity_and_Control_in_Diffusion_Models_WACV_2026_paper.html ; https://arxiv.org/abs/2503.10637 |
| `surkov2025onestep` | Surkov et al. 2025, One-step SAEs transfer to multi-step SDXL (NeurIPS) | I, II: I 4.3, 8; II 7 | Feature-level evidence that student and teacher share representations. | https://papers.neurips.cc/paper_files/paper/2025/hash/90291c8d685db65a8a1e32fd2b66f8bb-Abstract-Conference.html |
| `li2026why` | Li, Jones, Tong & Bronstein 2026, DMD students copy teacher pairings (ICML workshop) | I, II: I 8; II 7 | DMD students reproduce the teacher's noise-to-data pairings. Non-archival workshop poster: cite as such. | https://icml.cc/virtual/2026/70197 |
| `wu2026diversity` | Wu et al. 2026, Diversity-preserved DMD (arXiv) | I: I 7 | Locates DMD diversity loss at the first step. | https://arxiv.org/abs/2602.03139 |
| `haskins2025distilled` | Haskins & Adams 2025, Distilled Circuits (arXiv) | I, II: I 8; II 7 | Counterpoint: LLM students can restructure internal computation. Keeps the inheritance claim from being read as generic. | https://arxiv.org/abs/2505.10822 |
| `chi2026mta` | Chi et al. 2026, MTA trajectory alignment for LLM distillation (ACL) | I, II: I 4.3; II 7 | Contemporary work treating depth-trajectory alignment as distinct from fixed-layer similarity. Context only; it is LLM distillation, not diffusion. | https://aclanthology.org/2026.acl-long.1507/ ; https://arxiv.org/abs/2605.01374 |

## 3. Language-model controls

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `qwen2024qwen25` | Qwen Team 2024, Qwen2.5 Technical Report | I, II: I 2.2, 8, 9.4; II 5 | Qwen2.5 1.5B/3B/7B scaling controls. | https://arxiv.org/abs/2412.15115 |
| `jiang2023mistral` | Jiang A. Q. et al. 2023, Mistral 7B | I: I 2.2, 8 | Mistral 7B Instruct fixed-weight control. | https://arxiv.org/abs/2310.06825 |
| `arditi2024refusal` | Arditi et al. 2024, Refusal direction / abliteration (NeurIPS) | I: I 8 | Rank-1 orthogonalization behind the 'ablit' checkpoint; explains why the edit is small. | https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html |
| `grattafiori2024llama` | Grattafiori et al. 2024, The Llama 3 Herd of Models | I: I 11 | Only if the blocked Llama-3.1-8B Base/Instruct pair is mentioned in Limitations. | https://arxiv.org/abs/2407.21783 |

## 4. Spectral observables and dimensionality

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `roy2007effective` | Roy & Vetterli 2007, Effective rank (EUSIPCO) | I, II: I 2.1; II 2.2 | erank = exp(entropy of normalized singular values). Our H is log erank on the same amplitude-normalized spectrum. g (amplitude PR / D) is a different functional; do not call g 'effective rank'. | https://infoscience.epfl.ch/record/110188 ; https://zenodo.org/records/40328 |
| `shannon1948mathematical` | Shannon 1948, A Mathematical Theory of Communication | I, II: I 2.1; II 2.2 | Entropy underlying the spectral entropy H. | https://api.openalex.org/works/doi:10.1002/j.1538-7305.1948.tb01338.x |
| `gao2017theory` | Gao et al. 2017, Multineuronal dimensionality (bioRxiv) | I, II: I 2.1, 3; II 2.2 | Participation ratio in neuroscience, defined on eigenvalues. Cite as 'related (eigenvalue) PR', not as our definition. | https://www.biorxiv.org/content/10.1101/214262v2.article-info |
| `gao2015simplicity` | Gao & Ganguli 2015, Simplicity and complexity (arXiv) | I: I 3 | Review motivating dimensionality measures of population activity. | https://arxiv.org/abs/1503.08779 |
| `recanatesi2022scaledependent` | Recanatesi et al. 2022, Scale-dependent dimensionality (Patterns) | I, II: I 3; II 2.3 | Global PR mixes local and between-cluster structure; relevant to pooled vs branch statistics. | https://biophysics.princeton.edu/publications/scale-dependent-measure-system-dimensionality |
| `recanatesi2019dimensionality` | Recanatesi et al. 2019, Dimensionality compression and expansion in DNNs (arXiv) | I, II: I 3, 6.2; II 9.2 | PR-type dimension changes along depth in deep nets. | https://arxiv.org/abs/1906.00443 |
| `litwinkumar2017optimal` | Litwin-Kumar et al. 2017, Optimal degrees of synaptic connectivity (Neuron) | I, II: I 2.1; II 2.2 | Standard use of a PR-type representation dimension. | https://cris.huji.ac.il/en/publications/optimal-degrees-of-synaptic-connectivity/ |
| `ansuini2019intrinsic` | Ansuini et al. 2019, Intrinsic dimension of DNN representations (NeurIPS) | I, II: I 3, 6.2; II 9.2 | Nonlinear ID rises then falls along depth; linear PCA measures do not show it. Our g is linear. | https://mlanthology.org/neurips/2019/ansuini2019neurips-intrinsic/ ; https://arxiv.org/abs/1905.12784 |
| `valeriani2023geometry` | Valeriani et al. 2023, Geometry of hidden representations (NeurIPS) | I, II: I 6.2, 10; II 9.2, 13 | Depth profiles of ID recur across very different transformers: recurrence of coarse depth shape is expected, not evidence of shared computation. | https://mlanthology.org/neurips/2023/valeriani2023neurips-geometry/ ; https://arxiv.org/abs/2302.00294 |
| `cheng2025emergence` | Cheng et al. 2025, High-dimensional abstraction phase (ICLR) | I, II: I 6.2; II 9.2 | Generic mid-depth high-ID phase in LMs; same caution as Valeriani. | https://mlanthology.org/iclr/2025/cheng2025iclr-emergence/ |
| `chun2026estimating` | Chun et al. 2026, PR from finite samples (ICLR) | I, II: I 3; II 2.2, 12 | Naive PR is biased at small sample size; g depends on n relative to D. Supports reporting n with every g. | https://iclr.cc/virtual/2026/poster/10007970 ; https://arxiv.org/abs/2509.26560 |
| `garrido2023rankme` | Garrido et al. 2023, RankMe (ICML) | I, II: I 2.1; II 2.2 | Precedent for entropy effective rank (Roy-Vetterli) as a representation diagnostic in deep nets. | https://proceedings.mlr.press/v202/garrido23a.html |
| `kvinge2023exploring` | Kvinge, Brown & Godfrey 2023, ID of Stable Diffusion representations (arXiv) | I: I 5.1 | Prompt- and layer-dependent ID in a diffusion model; no timestep analysis. | https://arxiv.org/abs/2302.09301 |

## 5. Massive activations and attention sinks (caveat for uncentred statistics)

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `sun2024massive` | Sun M. et al. 2024, Massive Activations in LLMs (COLM) | I, II: I 11; II 2.1, 14 | A few input-independent huge activations act as biases and would dominate s1 of an uncentred matrix. Caveat only (claim register T7); not tested here. | https://arxiv.org/abs/2402.17762 ; venue line in https://arxiv.org/html/2402.17762v2 |
| `xiao2024efficient` | Xiao et al. 2024, Attention sinks (ICLR) | I, II: I 11; II 14 | Attention-sink tokens; context for token-level outliers. | https://mlanthology.org/iclr/2024/xiao2024iclr-efficient/ |
| `darcet2024vision` | Darcet et al. 2024, ViTs need registers (ICLR) | I, II: I 11; II 14 | High-norm outlier tokens in vision transformers. | https://mlanthology.org/iclr/2024/darcet2024iclr-vision/ |
| `gan2025unleashing` | Gan et al. 2025, Massive activations in DiTs for correspondence (NeurIPS) | I, II: I 11; II 2.1, 14 | DiT massive activations in fixed dimensions across tokens, linked to AdaLN. | https://papers.nips.cc/paper_files/paper/2025/hash/a654298de44ec95dec28aa37e8d0af51-Abstract-Conference.html ; https://arxiv.org/abs/2505.18584 |
| `gan2025massive` | Gan et al. 2025, Massive activations and local detail in DiTs (arXiv) | I, II: I 11; II 2.1, 14 | DiT massive activations are modulated by the timestep embedding (SD3, SD3.5, Flux): a sigma-locked envelope is expected from architecture alone. | https://arxiv.org/abs/2510.11538 |
| `sun2026anatomy` | Sun S. et al. 2026, Anatomy of Massive Activations and Attention Sinks (ICML) | I, II: I 11; II 14 | LMs: massive activations give near-constant representations across layers (implicit parameters); co-occurrence with sinks is a pre-norm artifact. Caveat only. | https://icml.cc/virtual/2026/poster/63610 ; https://arxiv.org/abs/2603.05498 |
| `turri2026few` | Turri et al. 2026, Few Channels Draw the Whole Picture (arXiv) | I, II: I 11; II 14 | DiT massive channels are a prompt-conditioned carrier; zeroing them collapses generation. Motivates the channel-removal / centering control before interpreting s1-dominated statistics. | https://arxiv.org/abs/2605.13974 |

## 6. Representational similarity

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `kriegeskorte2008representational` | Kriegeskorte, Mur & Bandettini 2008, RSA | I, II: I 5.2; II 8 | Compare RDMs, not units: allows cross-model comparison at different widths. | https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full |
| `mantel1967detection` | Mantel 1967, Mantel test (Cancer Research) | I, II: I 5.2; II 8 | Permutation test of association between distance matrices. | https://aacrjournals.org/cancerres/article/27/2_Part_1/209/476508/The-Detection-of-Disease-Clustering-and-a ; citation fields from https://pysal.org/pointpats/references.html |
| `legendre2010comparison` | Legendre & Fortin 2010, Mantel test vs alternatives | I: I 5.2 | Mantel tests answer a distance-matrix question and have lower power than raw-data tests. | https://api.openalex.org/works/doi:10.1111/j.1755-0998.2010.02866.x ; LIT_R6C 2.6 (Crossref) |
| `guillot2013dismantling` | Guillot & Rousset 2013, Dismantling the Mantel tests | I, II: I 5.2; II 8 | Mantel and partial Mantel tests are biased under autocorrelation; supports within-theme permutation and caution with partial Mantel. | https://esploro.umontpellier.fr/esploro/outputs/journalArticle/Dismantling-the-Mantel-tests/99247307709311 ; https://arxiv.org/abs/1112.0651 |
| `kornblith2019similarity` | Kornblith et al. 2019, CKA (ICML) | I, II: I 4; II 4 | Standard representation-similarity comparator; CCA-type measures fail when D > n. | https://proceedings.mlr.press/v97/kornblith19a.html |
| `raghu2017svcca` | Raghu et al. 2017, SVCCA (NeurIPS) | I, II: I 4; II 4 | CCA-based comparator. | https://mlanthology.org/neurips/2017/raghu2017neurips-svcca/ |
| `morcos2018insights` | Morcos, Raghu & Bengio 2018, PWCCA (NeurIPS) | I, II: I 4; II 4 | Projection-weighted CCA. | https://mlanthology.org/neurips/2018/morcos2018neurips-insights/ |
| `williams2021generalized` | Williams et al. 2021, Generalized shape metrics (NeurIPS) | I, II: I 4.2; II 4, 7 | Procrustes-type shape metrics; frames 'registered shape' as a metric equivalence weaker than tangent correspondence. | https://mlanthology.org/neurips/2021/williams2021neurips-generalized/ ; https://arxiv.org/abs/2110.14739 |
| `huh2024platonic` | Huh et al. 2024, Platonic Representation Hypothesis (ICML) | I, II: I 5.2, 10; II 13 | Cross-model convergence hypothesis our cross-family RSA speaks to. Cite with the calibration critique. | https://proceedings.mlr.press/v235/huh24a.html |
| `groger2026revisiting` | Groger, Wen & Brbic 2026, Platonic hypothesis revisited (arXiv) | I, II: I 5.2, 10; II 13 | Similarity metrics are confounded by scale; after permutation calibration global convergence largely disappears, local neighbourhood agreement survives. | https://arxiv.org/abs/2602.14486 |
| `cui2022deconfounded` | Cui et al. 2022, Deconfounded representation similarity (NeurIPS) | I, II: I 5.2; II 8 | Input population structure inflates RSA/CKA; motivates the text-partial (TF-IDF) control. | https://mlanthology.org/neurips/2022/cui2022neurips-deconfounded/ |
| `klabunde2025similarity` | Klabunde et al. 2025, Survey of similarity measures (ACM CSUR 57(9)) | I, II: I 4; II 4 | Survey of measures and invariances for the methods section. | https://arxiv.org/abs/2305.06329 ; https://api.openalex.org/works/doi:10.1145%2F3728458 |
| `ding2021grounding` | Ding, Denain & Steinhardt 2021, Grounding representation similarity (NeurIPS) | I: I 4 | Similarity measures disagree and need functional grounding / statistical testing. | https://mlanthology.org/neurips/2021/ding2021neurips-grounding/ |
| `davari2023reliability` | Davari et al. 2023, Reliability of CKA (ICLR) | I: I 4 | CKA can be manipulated and is outlier-sensitive. | https://mlanthology.org/iclr/2023/davari2023iclr-reliability/ |
| `matejka2017same` | Matejka & Fitzmaurice 2017, Same Stats, Different Graphs (CHI) | I, II: I 4.1; II 4 | Constructive example that identical summary statistics do not imply shared structure (projection degeneracy). | https://www.research.autodesk.com/publications/same-stats-different-graphs/ ; https://api.openalex.org/works/doi:10.1145/3025453.3025912 |

## 7. Diffusion phases, commitment and sampling dynamics

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `biroli2023generative` | Biroli & Mezard 2023, Generative diffusion in very large dimensions (JSTAT) | I, II: I 6.1; II 9, 14 | Symmetry breaking in reverse diffusion; positions sigma-axis observations. Not evidence that activation statistics show a phase transition. | https://arxiv.org/abs/2306.03518 |
| `biroli2024dynamical` | Biroli et al. 2024, Dynamical regimes of diffusion models (Nat. Commun.) | I, II: I 6.1, 7; II 9, 14 | Speciation and collapse times; clone-based commitment measurement that our telemetry does not perform (G3). | https://www.nature.com/articles/s41467-024-54281-3 |
| `raya2023spontaneous` | Raya & Ambrogioni 2023, Spontaneous symmetry breaking (NeurIPS) | I, II: I 6.1; II 9, 11 | Early linear phase around a fixed point, then attractor dynamics; context for the early high-speed transient. | https://mlanthology.org/neurips/2023/raya2023neurips-spontaneous/ |
| `sclocchi2025phase` | Sclocchi, Favero & Wyart 2025, Phase transition in diffusion (PNAS) | I, II: I 6.1, 7; II 9 | Class features flip at a noise threshold while low-level features change smoothly: averaged telemetry can look gradual. | https://discovery.ucl.ac.uk/id/eprint/10206005 ; https://arxiv.org/abs/2402.16991 |
| `sclocchi2025probing` | Sclocchi et al. 2025, Probing latent hierarchical structure (ICLR) | I, II: I 7; II 9 | Forward-backward experiments with diverging correlation length at the transition (output-level, not activation-level). | https://proceedings.iclr.cc/paper_files/paper/2025/hash/b503810343f1c7e80fab37b400dbc7a8-Abstract-Conference.html |
| `li2024critical` | Li & Chen 2024, Critical windows (ICML) | I, II: I 7; II 12, 14 | Narrow windows in which features are decided, defined by forward-reverse continuations. Our sigma50/onset estimators are not critical windows. | https://proceedings.mlr.press/v235/li24g.html |
| `handke2026entropic` | Handke et al. 2026, Entropic signature of class speciation (ICML) | I, II: I 7, 11; II 14 | Class-conditional entropy as a commitment signature (EDM2-XS, SD1.5). Motivation for G3 only. | https://icml.cc/virtual/2026/poster/61282 ; https://arxiv.org/abs/2602.09651 |
| `li2025understanding` | Li X. et al. 2025, Representation dynamics of diffusion models (arXiv) | I, II: I 6.1; II 9.3 | An interior peak in representation quality vs noise can arise without a phase transition. | https://arxiv.org/abs/2502.05743 |
| `chen2024trajectory` | Chen D. et al. 2024, Trajectory regularity of ODE diffusion sampling (ICML) | I, II: I 3; II 6 | Sampling trajectories share a low-dimensional content-independent shape; prior for the low-dimensional backbone (about the latent trajectory, not our spectral telemetry). | https://proceedings.mlr.press/v235/chen24bm.html |
| `ban2025crystal` | Ban et al. 2025, Crystal Ball hypothesis (ICLR) | I: I 7 | Initial-noise trigger patches fix object positions; context for seed persistence. | https://proceedings.iclr.cc/paper_files/paper/2025/hash/54e13b23fa2f399cea6e67acf9063c40-Abstract-Conference.html |
| `chen2024deltadit` | Chen P. et al. 2024, Delta-DiT (arXiv) | I: I 6 | Empirical depth x timestep role structure in DiTs (front blocks outline, rear blocks detail). | https://arxiv.org/abs/2406.01125 |

## 8. Committor and transition path theory (open gate G3 only)

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `e2006towards` | E & Vanden-Eijnden 2006, Towards a theory of transition paths (J. Stat. Phys.) | I, II: I 7, 11; II 12, 14 | Defines the committor; states what a separatrix claim would require. | https://collaborate.princeton.edu/en/publications/towards-a-theory-of-transition-paths/ |
| `e2010transitionpath` | E & Vanden-Eijnden 2010, Transition-path theory review (Annu. Rev. Phys. Chem.) | I, II: I 11; II 14 | Review of TPT and committor estimation. | https://collaborate.princeton.edu/en/publications/transition-path-theory-and-path-finding-algorithms-for-the-study-/ |
| `bolhuis2002transition` | Bolhuis et al. 2002, Transition path sampling (Annu. Rev. Phys. Chem.) | I, II: I 11; II 14 | Committor by shooting many continuations from one configuration: the clone experiment G3 needs. | https://api.openalex.org/works/doi:10.1146/annurev.physchem.53.082301.113146 |

## 9. Dynamical systems, Koopman/DMD and depth-as-time

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `kloeden2011nonautonomous` | Kloeden & Rasmussen 2011, Nonautonomous Dynamical Systems (AMS) | II: II 5 | Framework for explicitly time-dependent flows dz/du = F(z,u,tau). | https://bookstore.ams.org/SURV/176 |
| `katok1995introduction` | Katok & Hasselblatt 1995, Modern theory of dynamical systems (CUP) | II: II 1, 7 | Standard definitions of conjugacy and orbit equivalence, which the paper states it does NOT claim. | https://www.cambridge.org/core/journals/ergodic-theory-and-dynamical-systems/article/introduction-to-the-modern-theory-of-dynamical-systems-by-anatole-katok-and-boris-hasselblatt-encyclopedia-of-mathematics-and-its-applications-54-cambridge-university-press-1995-822-pp/61D98DE209D926E0CB28A3F500211832 (publisher review page whose title gives series, volume, publisher and year; seen in search listing) |
| `schmid2010dynamic` | Schmid 2010, Dynamic mode decomposition (JFM) | II: II 11 | Local linear operator fit X_{u+1} = A X_u; eigenvalue moduli and rotations. | https://www.cambridge.org/core/product/AA4C763B525515AD4521A6CC5E10DBD4 |
| `williams2015datadriven` | Williams, Kevrekidis & Rowley 2015, EDMD (J. Nonlinear Sci.) | II: II 11 | Extended DMD as a Koopman approximation; positions the affine local model. | https://collaborate.princeton.edu/en/publications/a-datadriven-approximation-of-the-koopman-operator-extending-dyna/ |
| `kutz2016dynamic` | Kutz et al. 2016, Dynamic Mode Decomposition (SIAM book) | II: II 11 | Textbook reference for DMD practice. | https://amath.washington.edu/research/publications/dynamic-mode-decomposition-data-driven-modeling-complex-system |
| `sussillo2013opening` | Sussillo & Barak 2013, Opening the black box (Neural Comput.) | II: II 11 | Local linearization of network dynamics around slow points. | https://api.crossref.org/works/10.1162/NECO_a_00409 |
| `e2017proposal` | E 2017, ML via dynamical systems (Commun. Math. Stat.) | II: II 5 | Depth as time in residual networks. | https://collaborate.princeton.edu/en/publications/a-proposal-on-machine-learning-via-dynamical-systems/ |
| `lu2018beyond` | Lu Y. et al. 2018, Deep architectures as ODE discretizations (ICML) | II: II 5 | Residual layers as numerical ODE steps. | https://proceedings.mlr.press/v80/lu18d.html |
| `lu2019understanding` | Lu Y. et al. 2019, Transformer as multi-particle dynamical system (arXiv) | II: II 5 | Transformer layers as a particle ODE discretization. | https://arxiv.org/abs/1906.02762 |
| `chen2018neural` | Chen R. T. Q. et al. 2018, Neural ODEs (NeurIPS) | II: II 5 | Continuous-depth models. | https://mlanthology.org/neurips/2018/chen2018neurips-neural/ |
| `geshkovski2025mathematical` | Geshkovski et al. 2025, Mathematical perspective on Transformers (Bull. AMS 62(3)) | II: II 5, 13 | Transformers as interacting-particle flows (clustering). | https://arxiv.org/abs/2312.10794 ; https://par.nsf.gov/biblio/10633837 ; year from the reference list of https://arxiv.org/pdf/2512.01868 |
| `belrose2023eliciting` | Belrose et al. 2023, Tuned lens (arXiv) | I, II: I 1; II 5 | Layerwise readouts as a depth-resolved view of transformer computation. | https://arxiv.org/abs/2303.08112 |
| `nostalgebraist2020interpreting` | nostalgebraist 2020, The logit lens (blog) | I, II: I 1; II 5 | Original logit-lens post; informal source. | https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens |

## 10. Matrix analysis for the pooling theorem

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `bhatia1997matrix` | Bhatia 1997, Matrix Analysis (GTM 169) | II: II 3 | Loewner-Heinz / operator concavity of t^{1/2} (Ch. V; cite chapter, theorem number not checked). | https://link.springer.com/978-0-387-94846-1 |
| `carlen2010trace` | Carlen 2010, Trace inequalities and quantum entropy (Contemp. Math. 529) | II: II 3 | Tr f(A) is concave for concave f; with f = sqrt gives concavity of tr M^{1/2}. | https://www.ams.org/books/conm/529/ |
| `carlen2008minkowski` | Carlen & Lieb 2008, Minkowski-type trace inequality II (Lett. Math. Phys.) | II: II 3 | Concavity of Phi_{p,1} = Tr(sum A_j^p)^{1/p} for 0<p<=1; related, not the exact statement. | https://collaborate.princeton.edu/en/publications/a-minkowski-type-trace-inequality-and-strong-subadditivity-of-qua/ ; https://arxiv.org/abs/0710.4167 |
| `lieb1973convex` | Lieb 1973, Convex trace functions (Adv. Math.) | II: II 3 | Classical trace-concavity theorem; background. | https://api.crossref.org/works/10.1016/0001-8708(73)90011-X |
| `ando1979concavity` | Ando 1979, Concavity of maps on PD matrices (Linear Algebra Appl.) | II: II 3 | Concavity of operator maps; background. | https://api.openalex.org/works/doi:10.1016/0024-3795(79)90179-4 ; reference list of https://www.jams.jp/scm/contents/e-2010-5/2010-47.pdf |
| `boyd2004convex` | Boyd & Vandenberghe 2004, Convex Optimization (CUP) | II: II 3 | Convexity/quasiconcavity background for the step 'degree-1 homogeneous with convex superlevel sets implies concave'. No exercise number checked. | https://www.seas.ucla.edu/~vandenbe/cvxbook.html ; https://search.worldcat.org/oclc/53331084 |

## 11. Statistics: surrogates, trends, variance ratio, resampling, preregistration

| key | short reference | paper(s) / section(s) | supports (use for) | verification URL |
|---|---|---|---|---|
| `theiler1992testing` | Theiler et al. 1992, Surrogate data (Physica D) | I, II: I 3, 6.2; II 10 | Surrogate-data hypothesis testing; phase-randomized surrogates assume stationarity. | https://api.crossref.org/works/10.1016/0167-2789(92)90102-S |
| `schreiber1996improved` | Schreiber & Schmitz 1996, IAAFT surrogates (PRL) | I, II: I 6.2; II 10 | Amplitude-adjusted surrogates. | https://api.crossref.org/works/10.1103/PhysRevLett.77.635 |
| `schreiber2000surrogate` | Schreiber & Schmitz 2000, Surrogate time series (Physica D) | I, II: I 6.2, 9.1; II 10 | Review; rejection does not say which null assumption failed. | https://api.crossref.org/works/10.1016/S0167-2789(00)00043-9 ; https://arxiv.org/abs/chao-dyn/9909037 |
| `nakamura2005small` | Nakamura & Small 2005, Small-shuffle surrogates (PRE) | I, II: I 9.1; II 10 | Trend-preserving surrogates. | https://ira.lib.polyu.edu.hk/handle/10397/4826 |
| `lancaster2018surrogate` | Lancaster et al. 2018, Surrogate data review (Phys. Rep.) | I, II: I 9.1; II 10 | Comprehensive surrogate review incl. synchronization tests. | https://research.lancaster-university.uk/en/publications/surrogate-data-for-hypothesis-testing-of-physical-systems/ |
| `yule1926why` | Yule 1926, Nonsense correlations (JRSS) | I, II: I 4.1, 9.1; II 4 | Trending series correlate spuriously: high scalar-envelope correlations are expected. | https://api.openalex.org/works/doi:10.2307/2341482 |
| `granger1974spurious` | Granger & Newbold 1974, Spurious regressions | I, II: I 4.1, 9.1; II 4 | Spurious significance between independent trending series. | https://api.openalex.org/works/doi:10.1016/0304-4076(74)90034-7 |
| `harris2020nonsense` | Harris 2020, Nonsense correlations in neuroscience (bioRxiv) | I, II: I 9.1; II 10 | Circular-shift and phase nulls fail under slow drift; permute replication units instead. | https://www.biorxiv.org/content/10.1101/2020.11.29.402719v1 (LIT_R6C 5.8) |
| `lo1988stock` | Lo & MacKinlay 1988, Variance ratio test (RFS) | I, II: I 7; II 12 | Defines VR(q). Cite for the test only. | https://web.mit.edu/Alo/www/Papers/lo-mackinlay-88.html ; https://api.crossref.org/works/10.1093/rfs/1.1.41 |
| `lo1989size` | Lo & MacKinlay 1989, VR size and power in finite samples (J. Econometrics) | I, II: I 7; II 12 | Finite-sample behaviour of VR. | https://web.mit.edu/~alo/www/Papers/lo-mackinlay-89.html ; https://api.crossref.org/works/10.1016/0304-4076(89)90083-3 |
| `kim1991mean` | Kim, Nelson & Startz 1991, Mean reversion reappraisal (RES) | II: II 12 | Small-sample VR is biased below 1 under the null; justifies calibrating VR against simulated walk/AR/iid at matched length. | https://www.nber.org/papers/w2795 |
| `wright2000alternative` | Wright 2000, Rank/sign VR tests (JBES) | II: II 12 | Exact rank/sign VR tests for short series. | https://ideas.repec.org/a/bes/jnlbes/v18y2000i1p1-9.html |
| `efron1979bootstrap` | Efron 1979, Bootstrap (Ann. Stat.) | I, II: I 6; II 9.1 | Bootstrap incidence of sign topologies. | https://api.crossref.org/works/10.1214/aos/1176344552 |
| `muggeo2003estimating` | Muggeo 2003, Segmented regression (Stat. Med.) | I, II: I 6.1 (post hoc P4 profile); II 9.1 | Piecewise-linear fits with unknown breakpoints. | https://api.crossref.org/works/10.1002/sim.1545 |
| `kendall1938new` | Kendall 1938, Rank correlation (Biometrika) | I, II: I 5.2; II 8 | Rank correlation (if Kendall tau is reported). | https://academicnewsletter.sufe.edu.cn/info/450507 (index record with DOI) |
| `spearman1904proof` | Spearman 1904, Rank correlation (Am. J. Psychol.) | I, II: I 5.2; II 8 | Spearman rho (Mantel rho; tangent-RSA coupling rho ~ 0.297). | https://api.crossref.org/works/10.2307/1412159 |
| `westfall1993resampling` | Westfall & Young 1993, Resampling-based multiple testing (Wiley) | I: I 2.4, 9 | Max-T/min-P familywise control over scanned cells. | https://ci.nii.ac.jp/ncid/BA19597310 (LIT_R6C 6.1) |
| `nichols2002nonparametric` | Nichols & Holmes 2002, Permutation tests primer (Hum. Brain Mapp.) | I: I 2.4, 9 | Max-statistic permutation recipe for correlated tests. | https://api.openalex.org/works/doi:10.1002/hbm.1058 ; https://deepblue.lib.umich.edu/handle/2027.42/35194 |
| `nosek2018preregistration` | Nosek et al. 2018, The preregistration revolution (PNAS) | I, II: I 2.4; II appendix | Preregistration practice behind the frozen gates. | https://api.crossref.org/works/10.1073/pnas.1708274114 |
| `simmons2011false` | Simmons, Nelson & Simonsohn 2011, False-positive psychology | I: I 2.4 | Undisclosed analytic flexibility; motivates SHA-stamped preregistration. | https://api.openalex.org/works/doi:10.1177/0956797611417632 |
| `gelman2014statistical` | Gelman & Loken 2014, The statistical crisis in science | I: I 2.4, 9 | Forking paths: label post hoc observables as such. | https://api.crossref.org/works/10.1511/2014.111.460 (LIT_R6C 6.4) |
| `varoquaux2018cross` | Varoquaux 2018, Cross-validation failure (NeuroImage) | I, II: I 5.1, 7; II 12 | Large CV error bars at n ~ 100; supports matched-n comparisons and the fold-sensitivity of sigma50. | https://api.crossref.org/works/10.1016/j.neuroimage.2017.06.061 |

## Added 2026-10-08 (origin caveat, both papers)
- `falck2025fourier`: Falck et al., "A Fourier Space Perspective on Diffusion Models", arXiv:2505.11278 (verified on arXiv abs page). Noise schedule orders generation coarse-to-fine in frequency.
- `kang2025videophysics`: Kang et al., "How Far is Video Generation from World Model: A Physical Law Perspective", ICML 2025, arXiv:2411.02385 (verified). Video models generalise case-based, not by abstracting physical laws.

## Added 2026-10-08 (U-Net out-of-family check: Mode-I §11, Mode-II §flow and §universality)
- `rombach2022high`: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022, pp. 10684–10695, arXiv:2112.10752. Verified on arXiv and on the CVF open-access page. The Stable Diffusion 1.5 architecture.
- `ronneberger2015unet`: Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015, LNCS 9351, pp. 234–241, arXiv:1505.04597. Verified on arXiv and Crossref (doi 10.1007/978-3-319-24574-4_28).
- `marchenko1967distribution`: Marčenko & Pastur, "Distribution of Eigenvalues for Some Sets of Random Matrices", Math. USSR-Sb. 1(4):457–483, 1967. Verified on Crossref (doi 10.1070/SM1967v001n04ABEH001994). The isotropic baseline g_MP.

## Added 2026-10-09 (rotation-invariance lemma: Mode-II §hierarchy)
- `ashkboos2024slicegpt`: Ashkboos, Croci, do Nascimento, Hoefler & Hensman, "SliceGPT: Compress Large Language Models by Deleting Rows and Columns", ICLR 2024, arXiv:2401.15024. Verified on arXiv (abstract page and v2 PDF, whose header reads "Published as a conference paper at ICLR 2024"). Theorem 1 (§3.1) states computational invariance: in an RMSNorm residual network an orthogonal Q applied to the embedding, block input/output weights and head leaves the output unchanged; §3.2 converts LayerNorm networks by absorbing the mean subtraction and scales into the weights. Supports only the statement that the residual-stream rotation is a function-preserving symmetry of RMSNorm networks.

## Added 2026-10-09 (positioning only: Mode-II §observables remark, §tangent, §open)
- `clark2026lyapunov`: Clark, "Lyapunov spectrum of random neural networks", arXiv:2610.12426 (cond-mat.dis-nn, submitted 2026-10-08, 36 pp). Verified on arXiv and from the PDF. Supports: participation-ratio dimensions depend on the choice of variables while Lyapunov-derived dimensions are coordinate-invariant (§1.2, §1.5); PR and attractor dimension move differently with coupling (Fig. 2); depth Jacobian spectra in a feed-forward network are not Lyapunov exponents (no invariant measure). Never cited for criticality or "edge of chaos".
- `clark2026linear`: Clark, "Linear equivalence of nonlinear recurrent neural networks", arXiv:2604.23489 (v3 2026-06-08). Verified on arXiv. Supports: a chaotic nonlinear random network and a stable linear network with the same couplings driven by independent noise share the covariance matrix, while their Lyapunov behaviour differs.
- `shen2025covariance`: Shen & Hu, "Covariance spectrum in nonlinear recurrent neural networks", arXiv:2508.05288 (v2 2026-09-05). Verified on arXiv. The source of the linear-equivalence ansatz that Clark derives. Cited only alongside clark2026linear.
- `skean2025layer`: Skean et al., "Layer by Layer: Uncovering Hidden Representations in Language Models", arXiv:2502.02013 (v2 2025-06-15). Verified on arXiv 2026-11-05. Supports: layerwise entropy and effective rank of LLM representations; intermediate layers.
- `zhang2024reef`: Zhang et al., "REEF: Representation Encoding Fingerprints for Large Language Models", arXiv:2410.14273. Verified on arXiv 2026-11-05. Supports: CKA on representations detects models derived from a victim model.
- `zhu2025independence`: Zhu, Ahmed, Kuditipudi, Liang, "Independence Tests for Language Models", arXiv:2502.12292 (v2 2025-03-12). Verified on arXiv 2026-11-05. Supports: p-values for whether two LMs were trained independently, from weights or activations.
- `shi2025knowledge`: Shi, Zheng, Song, Yeh, "Knowledge Distillation Detection for Open-weights Models", arXiv:2510.02302. Verified on arXiv 2026-11-05. Supports: detecting distillation from student weights and teacher access, including text-to-image models.
- `meshram2026diffusion`: Meshram & Chandrasekaran, "Diffusion Model Attribution via Spectral Coupling of Denoiser Responses", arXiv:2606.28092. Verified on arXiv 2026-11-05. Supports: spectral denoiser signatures identify diffusion models (so supervised identification from spectra is possible; our trend-dominance result concerns correlation).
- `tang2026representations`: Tang, Wu, Zhang, Vasilakos, "From Representations to Computational Trajectories: Intrinsic Fingerprinting for LLM Provenance Verification", arXiv:2610.34675 (2026-10-26, cs.CR). Verified on arXiv 2026-11-05 (abstract). Supports: depth-trajectory fingerprints distinguish related from unrelated LLMs; does not address distillation or diffusion models.
