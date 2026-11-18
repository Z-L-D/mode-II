# LIT_R4: Literature check for diffusion internal-telemetry review (round 4)

**Summary (verification status)**
1. Verified (title, authors, venue/ID, URL all checked on a primary page): Biroli+ 2024, Raya & Ambrogioni 2023, Sclocchi+ 2025, Li & Chen 2024, Bolhuis+ 2002, Mao+ 2023, Li+ 2025 (seeds), Qi+ 2024, Zhou+ 2025, Hertz+ 2023, Kynkäänniemi+ 2024, Chidambaram+ 2024, Sadat+ 2024 (CADS), Wang+ 2024, Lo & MacKinlay 1988, Karras+ 2022, Chen+ 2024, Liu+ 2022, Ning+ 2024, Kornblith+ 2019, Z-Image report 2025, Gandikota & Bau 2025, Surkov+ 2025, Gao+ 2017, Recanatesi+ 2022, Litwin-Kumar+ 2017.
2. Partially verified: Geissler, Dellago & Chandler 1999 (seen only as a reference entry in another paper, not on its own page); Poterba & Summers 1988 (NBER page and journal PDF filename only); Esser+ 2024 SD3 (arXiv checked, ICML venue from memory, step-count claim not found in the text).
3. Not found: no diffusion-distillation paper reporting layer-wise CKA/representation similarity between teacher and student. No paper explicitly stating that intermediate states at a matched noise level are step-count invariant. No primary source stating that pooling two conditions always raises PR (and it does not: see topic 8).
4. Tensions with our assumptions: (a) the CFG literature puts the diversity-reducing effect of guidance at **high noise (early)**, not late (topic 4); (b) pooling can **lower** PR when between-condition mean shift dominates (topic 8); (c) Lo & MacKinlay's own finding was VR > 1 (positive autocorrelation), not mean reversion (topic 5).
5. Commitment measured by clone or re-noise continuations: Biroli+ (two clones from the same x_t, P(same class)); Sclocchi+ (re-noise to t then regenerate, P(class preserved) plus low-level feature overlap); Li & Chen (forward to t then reverse, TV to a sub-mixture; feature agreement in SD 2.1). Raya & Ambrogioni do **not** use continuations.

---

## 1. Commitment / speciation / symmetry breaking during sampling

- **Biroli, G., Bonnaire, T., de Bortoli, V., Mézard, M. (2024). Dynamical regimes of diffusion models. *Nature Communications* 15, 9957. arXiv:2402.18491.**
  URL: https://arxiv.org/abs/2402.18491 (full text: https://arxiv.org/html/2402.18491)
  Identifies three regimes: a speciation transition at t_S = ½ log Λ (Λ is the largest eigenvalue of the data covariance), followed by a collapse transition. **Uses clone continuations:** "We generate two independent trajectories at time t of the backward process, and numerically estimate the probability φ(t) that they end up in the same class." t_S is defined by finite-size-scaling crossing, approximately φ(t_S) ≈ 0.775. This is the closest analogue of a committor estimate from launching continuations. — *verified*

- **Raya, G., Ambrogioni, L. (2023). Spontaneous symmetry breaking in generative diffusion models. NeurIPS 2023. arXiv:2305.19693.**
  URL: https://arxiv.org/abs/2305.19693 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/d0da30e312b75a3fffd9e9191f8bc1b0-Abstract-Conference.html
  Generation splits into an early linear phase around a central fixed point and a later attractor phase after a stability change (a bifurcation). "Early fluctuations are reverted to the central fixed point", which motivates a Gaussian late-initialization scheme. **No re-noising or clone continuations.** The statistic is analytical/potential-based. — *verified*

- **Sclocchi, A., Favero, A., Wyart, M. (2025). A phase transition in diffusion models reveals the hierarchical nature of data. *PNAS* (DOI 10.1073/pnas.2408799121). arXiv:2402.16991.**
  URL: https://arxiv.org/abs/2402.16991
  **Re-noising (forward-backward) experiment:** they noise a sample to time t and regenerate it. The probability of keeping the class drops sharply at a threshold time, a phase transition. Low-level features change smoothly over the whole process. The test uses class-unconditional ImageNet models. Statistic: class-retention probability plus overlap of low-level features. — *verified* (PNAS DOI found on the Nature Index page; arXiv content checked)

- **Li, M., Chen, S. (2024). Critical windows: non-asymptotic theory for feature emergence in diffusion models. ICML 2024 (PMLR v235). arXiv:2403.01633.**
  URL: https://arxiv.org/abs/2403.01633 ; https://proceedings.mlr.press/v235/li24g.html
  Defines critical windows with a **forward-then-reverse continuation**: "run the forward process for time t starting from x … then run the reverse process for time t". It asks when the output distribution is close in **total variation** to a target sub-mixture. Bounds are in terms of inter- and intra-group separation. A Stable Diffusion 2.1 experiment (250 car images, CLIP-labelled color, background and size) finds sharp drops in feature agreement within specific timestep windows. — *verified*

## 2. Committor methodology (transition path sampling)

- **Bolhuis, P. G., Chandler, D., Dellago, C., Geissler, P. L. (2002). Transition path sampling: throwing ropes over rough mountain passes, in the dark. *Annual Review of Physical Chemistry* 53, 291–318. DOI 10.1146/annurev.physchem.53.082301.113146.**
  URL: https://www.annualreviews.org/content/journals/10.1146/annurev.physchem.53.082301.113146
  Standard review of transition path sampling. It covers committor ("commitment probability") analysis: launch many independent trajectories with randomized momenta from one configuration and record the fraction that reaches B before A. — *verified* (bibliographic details checked. The committor content is standard but could not be quoted because the full text is paywalled.)

- **Geissler, P. L., Dellago, C., Chandler, D. (1999). Kinetic pathways of ion pair dissociation in water. *J. Phys. Chem. B* 103, 3706.**
  URL: (seen as ref. 12 in https://arxiv.org/abs/2110.05646. The publisher page was not reached.)
  Classic use of committor histograms built from many shooting continuations per configuration. It showed that a plausible-looking coordinate (ion distance) is not a sufficient reaction coordinate. This is a useful analogue: a telemetry statistic that tracks progress is not necessarily a committor. — *partially verified*

- **Kang, P., Trizio, E., Parrinello, M. (2024). Computing the committor with the committor: an anatomy of the transition state ensemble. arXiv:2401.05279.**
  URL: https://arxiv.org/abs/2401.05279
  Gives a modern statement of the definition: q(x) = P(reach B before A | start at x), with q ≈ ½ marking the transition-state ensemble. It notes that "committor analysis" by shooting is expensive and depends on the basin criteria. That caveat applies to us: the "committed" threshold is a choice. — *verified*

## 3. The initial noise seed fixes layout or composition across prompts

- **Mao, J., Wang, X., Aizawa, K. (2023). Guided image synthesis via initial image editing in diffusion model. ACM Multimedia 2023. arXiv:2305.03382.**
  URL: https://arxiv.org/abs/2305.03382
  Closest primary source for our claim: "When the initial images are the same, the same categories under different prompts are highly similar in position and visual appearance". Pixel blocks of the initial latent have content preferences, and editing part of the noise changes only the corresponding region. — *verified*

- **Li, S., Le, H., Xu, J., Salzmann, M. (2025). Enhancing compositional text-to-image generation with reliable random seeds. ICLR 2025. arXiv:2411.18810.**
  URL: https://arxiv.org/abs/2411.18810
  "Different initial random seeds tend to guide the model to place objects in distinct image areas, potentially adhering to specific patterns of camera angles and image composition associated with the seed." They mine reliable seeds to improve counting and spatial composition. — *verified*

- **Qi, Z., Bai, L., Xiong, H., Xie, Z. (2024). Not all noises are created equally: diffusion noise selection and optimization. arXiv:2407.14041.**
  URL: https://arxiv.org/abs/2407.14041
  Generation quality depends on the noise, measured through its inversion stability. They cite Mao et al. for the layout effect, so this paper is **secondary** for the cross-prompt layout claim. — *verified*

- **Zhou, Z., Shao, S., Bai, L., Zhang, S., Xu, Z., Han, B., Xie, Z. (2025). Golden noise for diffusion models: a learning framework. ICCV 2025. arXiv:2411.09502.**
  URL: https://arxiv.org/abs/2411.09502
  Learns a prompt-conditioned noise perturbation ("noise prompt") that improves text-image alignment, so some noises are better than others for a given prompt. It does not directly show layout persistence across prompts. — *verified*

- **Caveat: Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D. (2023). Prompt-to-Prompt image editing with cross attention control. ICLR 2023. arXiv:2208.01626.**
  URL: https://arxiv.org/abs/2208.01626
  Counterpoint: with a fixed seed, "even a small modification of the text prompt often leads to a completely different outcome" (SD/Imagen era). Seed-fixes-layout is therefore a tendency, not a law. It should be measured for SD3.5, Z-Image and Mochi rather than assumed. — *verified*

## 4. Time dependence of classifier-free guidance and its effect on diversity

- **Kynkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., Lehtinen, J. (2024). Applying guidance in a limited interval improves sample and distribution quality in diffusion models. NeurIPS 2024. arXiv:2404.07724.**
  URL: https://arxiv.org/abs/2404.07724
  Guidance is "clearly harmful toward the beginning of the chain" (high noise), helpful in the middle, and "largely unnecessary toward the end". At high noise it "drastically reduces the variation in the results", leading toward a handful of "template images" per prompt (mode dropping in a 1D toy model). **Implication:** the diversity loss is *caused* by guidance at high noise. — *verified*

- **Chidambaram, M., Gatmiry, K., Chen, S., Lee, H., Lu, J. (2024). What does guidance do? A fine-grained analysis in a simple setting. NeurIPS 2024. arXiv:2409.13074.**
  URL: https://arxiv.org/abs/2409.13074
  Proves that guidance does not sample the tilted distribution. As the guidance weight grows, samples concentrate on the boundary of the conditional support, which reduces diversity. With any score error, large guidance pushes samples off-support. No explicit time-resolved analysis. — *verified*

- **Sadat, S., Buhmann, J., Bradley, D., Hilliges, O., Weber, R. M. (2024). CADS: Unleashing the diversity of diffusion models through condition-annealed sampling. ICLR 2024 (spotlight). arXiv:2310.17347.**
  URL: https://arxiv.org/abs/2310.17347
  High CFG reduces output diversity. CADS adds strong noise to the condition **early** in sampling and anneals it to zero by τ₁ using a piecewise-linear γ(t). The rationale is that diversity is set early. — *verified*

- **Wang, X., Dufour, N., Andreou, N., Cani, M.-P., Fernandez Abrevaya, V., Picard, D., Kalogeiton, V. (2024). Analysis of classifier-free guidance weight schedulers. TMLR. arXiv:2404.13040.**
  URL: https://arxiv.org/abs/2404.13040
  Monotonically **increasing** schedules (low guidance early, high late) consistently improve the quality/diversity trade-off. This is consistent with early guidance being the diversity-destroying part. — *verified*

- **Related (topic 7): Gandikota & Bau 2025** report that base SDXL settles image structure within about the first 30% of steps.

> **Relation to our observation** (across-seed spread under CFG 4.5 contracting relative to CFG 1 mainly in the later three quarters): the literature says the *cause* of the diversity loss is early or high-noise guidance. The two can be reconciled if the early guided drift is small in activation-PR terms and only shows up as reduced spread later (a delayed readout). A direct test: run guidance only in the first quarter versus only in the last three quarters, as in Kynkäänniemi+, and see which reproduces the contraction. The two schedules may also not transfer directly to flow-matching models with SD3-style timestep shift.

## 5. Variance-ratio test

- **Lo, A. W., MacKinlay, A. C. (1988). Stock market prices do not follow random walks: evidence from a simple specification test. *Review of Financial Studies* 1(1), 41–66.**
  URL: https://web.mit.edu/Alo/www/Papers/lo-mackinlay-88.html
  Introduces the variance-ratio specification test, VR(q) = Var(q-period increment) / (q · Var(1-period increment)). Under a random walk VR = 1. Includes homoskedastic and heteroskedasticity-robust z-statistics. Note: their empirical result was VR > 1 (positive autocorrelation), and they say the findings "do not support a mean-reverting model". Cite them for the test, not for mean reversion. — *verified*

- **Poterba, J. M., Summers, L. H. (1988). Mean reversion in stock prices: evidence and implications. *Journal of Financial Economics* 22(1), 27–59. (NBER WP 2343).**
  URL: https://nber.org/papers/w2343
  Standard companion citation for interpreting VR < 1 as mean reversion. — *partially verified* (NBER page and JFE PDF filename only)

## 6. Step-count / discretization effects on ODE trajectories

- **Karras, T., Aittala, M., Aila, T., Laine, S. (2022). Elucidating the design space of diffusion-based generative models. NeurIPS 2022. arXiv:2206.00364.**
  URL: https://arxiv.org/abs/2206.00364
  Local truncation error grows faster than linearly with step size: O(h²) per step for Euler, O(h³) for Heun. ρ = 3 nearly equalizes per-step truncation error, but ρ = 5–10 (more steps at low σ) works better. Trajectories are nearly linear at large and small σ, with substantial curvature only in between. Implication: states at matched σ converge as NFE grows, and step-count sensitivity should be concentrated in the curved middle region. — *verified*

- **Chen, D., Zhou, Z., Wang, C., Shen, C., Lyu, S. (2024). On the trajectory regularity of ODE-based diffusion sampling. ICML 2024. arXiv:2405.11326.**
  URL: https://arxiv.org/abs/2405.11326
  Sampling trajectories share a content-independent linear–nonlinear–linear ("boomerang") shape and are about 85% explained by 2–3 principal components. The shared shape justifies one time schedule for all samples (a dynamic-programming schedule that helps at 5–10 NFE). This supports approximate step-count invariance at matched noise for moderate NFE, but it is not a direct test. — *verified*

- **Liu, X., Gong, C., Liu, Q. (2022). Flow straight and fast: learning to generate and transfer data with rectified flow. arXiv:2209.03003 (ICLR 2023).**
  URL: https://arxiv.org/abs/2209.03003
  Straight paths "can be simulated exactly without time discretization". Un-reflowed (1-rectified) flows are not straight. SD3.5 and Mochi are flow-matching models without reflow, so some discretization error at matched t is expected. — *verified* (ICLR venue from memory)

- **Ning, M., Li, M., Su, J., Salah, A. A., Ertugrul, I. O. (2024). Elucidating the exposure bias in diffusion models. ICLR 2024. arXiv:2308.15321.**
  URL: https://arxiv.org/abs/2308.15321
  Sampled x_t has inflated variance relative to training x_t, and the sampled ε has a larger norm than in training. The mismatch accumulates toward the end of sampling and differs between 20- and 50-step runs (DDPM/DDIM, CIFAR-10). **Evidence against exact step-count invariance:** statistics of the state at a matched t depend somewhat on step count, especially late. — *verified*

- **Esser, P., et al. (2024). Scaling rectified flow transformers for high-resolution image synthesis (SD3). arXiv:2403.03206 (ICML 2024).**
  URL: https://arxiv.org/abs/2403.03206
  This is the SD3/SD3.5 formulation (rectified flow with logit-normal timestep sampling and timestep shift). I could not find a quoted step-count invariance result in the text. — *partially verified*

- **Not found:** a paper that directly measures internal activations at matched σ across different step counts. This appears to be a gap our telemetry could fill.

## 7. Distilled students and teacher representations; Z-Image distillation

- **Kornblith, S., Norouzi, M., Lee, H., Hinton, G. (2019). Similarity of neural network representations revisited. ICML 2019. arXiv:1905.00414.**
  URL: https://arxiv.org/abs/1905.00414
  Introduces linear and RBF CKA. CKA reliably finds corresponding layers across networks trained from different initializations, which CCA-type measures cannot do when dimension exceeds the sample count. This is the standard tool for a teacher/student layer comparison. — *verified*

- **Gandikota, R., Bau, D. (2025). Distilling diversity and control in diffusion models. arXiv:2503.10637.**
  URL: https://arxiv.org/abs/2503.10637
  Distilled SDXL variants (Turbo, Lightning, LCM, DMD2) keep the base model's concept representations: Concept Sliders and LoRAs trained on the base transfer "seamlessly". Distilled models "commit to their final image structure almost immediately at the first timestep", while base SDXL needs about 30% of steps. Using the base model for step 1 restores diversity. Directly relevant to Z-Image Base vs Turbo commitment timing. — *verified* (arXiv preprint; no venue found)

- **Surkov, V., Wendler, C., Mari, A., Terekhov, M., Deschenaux, J., West, R., Gulcehre, C., Bau, D. (2025). One-step is enough: sparse autoencoders for text-to-image diffusion models. NeurIPS 2025. arXiv:2410.22366.**
  URL: https://papers.neurips.cc/paper_files/paper/2025/hash/90291c8d685db65a8a1e32fd2b66f8bb-Abstract-Conference.html
  SAEs trained on 1-step SDXL-Turbo transformer blocks "generalize to 4-step SDXL Turbo and even to the multi-step SDXL base model … without additional training". This is feature-level evidence that student and teacher share representations. — *verified*

- **Z-Image Team, Tongyi Lab (Alibaba) (2025). Z-Image: an efficient image generation foundation model with single-stream diffusion transformer. arXiv:2511.22699.**
  URL: https://arxiv.org/abs/2511.22699 ; model cards https://huggingface.co/Tongyi-MAI/Z-Image-Turbo , https://huggingface.co/Tongyi-MAI/Z-Image
  S3-DiT: 6.15B parameters, 30 layers, hidden size 3840, 32 heads. Pipeline: low-resolution pretraining → omni-pretraining → SFT → few-step distillation → RLHF (DPO + GRPO). **Turbo is distilled from the SFT checkpoint** with Decoupled-DMD (arXiv:2511.22677; CFG-augmentation as the main engine and distribution matching as regularizer) and DMDR (arXiv:2511.13649; RL plus DMD). It runs at 8 NFE with guidance 0. The Z-Image (base) card says "undistilled", CFG 3–5, 28–50 steps, RL ❌ vs Turbo RL ✅, and diversity High vs Low. **Caveat:** neither the card nor the report says the released Z-Image base weights are exactly the teacher used for Turbo, and Turbo also went through RL. Base-vs-Turbo representation differences therefore mix distillation with RL and possibly a checkpoint mismatch. The report does not discuss teacher–student representation similarity. — *verified*

- **Not found:** a diffusion-distillation paper reporting layer-wise CKA between teacher and student.

## 8. Participation ratio as effective dimensionality

- **Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K., Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and measurement. bioRxiv 214262.**
  URL: https://www.biorxiv.org/content/10.1101/214262v2.full-text
  Defines PR = (Σλ)² / Σλ², which runs from 1 to M. Main result: measured dimensionality is bounded by min(M, neural task complexity), and task complexity grows with the number and range of task parameters. More conditions allow higher measured dimension, which is the closest primary support for "pooling conditions can raise PR". — *verified* (preprint)

- **Recanatesi, S., Bradde, S., Balasubramanian, V., Steinmetz, N. A., Shea-Brown, E. (2022). A scale-dependent measure of system dimensionality. *Patterns* 3(8), 100555. DOI 10.1016/j.patter.2022.100555.**
  URL: https://biophysics.princeton.edu/publications/scale-dependent-measure-system-dimensionality
  Generalizes PR to a scale-dependent version (local, intermediate, global). The global PR mixes local structure with between-cluster structure. Useful for separating within-condition from pooled dimension. — *verified*

- **Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H., Abbott, L. F. (2017). Optimal degrees of synaptic connectivity. *Neuron* 93(5), 1153–1164. DOI 10.1016/j.neuron.2017.01.030.**
  URL: https://cris.huji.ac.il/en/publications/optimal-degrees-of-synaptic-connectivity/
  Widely cited use of PR as the "dimension" of population representations. — *verified* (bibliographic details only. The PR definition inside the paper was not re-read.)

- **Pooling note (no primary citation found; our own check).** By the law of total covariance, pooled covariance = mean within-condition covariance + between-condition covariance (rank ≤ k−1 for k conditions). PR can therefore go **either way**. In a quick numerical check (d = 50, isotropic within-condition): mean shifts of 0, 3, 10 and 30 SD along one axis gave pooled PR of 49.4, 45.4, 7.7 and 1.5, against 48.8 within-condition. Two anisotropic conditions with *different* dominant axes went from 2.2 each to 4.3 pooled. Pooling raises PR when the conditions occupy different subspaces. It lowers PR when one large mean-difference direction dominates. Any claim that "mixing seeds/prompts inflates PR" should state which regime applies.
