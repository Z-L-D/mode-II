# R9 addendum: text control for R9-B6 (preregistration)

Frozen before computation. Parent: R9_PREREGISTRATION.md (SHA 09c1f170). Trigger: Zac (09:30Z) approved all five R9 candidates. He read item 5 (shared prompt RSA across LLMs) as the same kind of result as Mochi↔Z-Image in diffusion.

**Why a control is needed.** In the diffusion models, the measured tokens are image latents and the prompt enters only through conditioning. In the v1 LLM captures, the measured matrix contains the prompt tokens themselves: at k = 0 it is nothing but them. A shared RSA could therefore come from the shared text alone (which words, how many). The diffusion RSA was already shown to persist after partialling out prompt-text overlap (Round 6). This addendum applies the same control to the LLMs.

**R9-B6b.** Same pairs, token indices and n-residualised features as R9-B6. Partial Mantel on ranks with two controls, residualising the rank vectors of both RDMs on [1, rank C_len, rank C_text]:
- C_len = |Δ mean log n|;
- C_text = TF-IDF cosine distance of the prompt texts (Round 6 S2f vectoriser, unchanged).

Permutations: 10,000, rng 2026100908.

Also reported:
- each model's own Mantel ρ with C_text (text RSA);
- the same doubly-controlled statistic for the diffusion unrelated pairs at the first state (S2 features), as the like-for-like comparison.

**Rule.**
- **Survives the text control:** for a pair, the doubly-controlled partial p ≤ 0.01 at ≥ 2 of 3 k, and the partial is ≥ 0.5 × the R9-B6 partial at those k.
- **Mostly text:** the partial falls below 0.5 × in ≥ 2 of 3 k.
- The rule applies to the Qwen pairs and to Mistral↔Llama separately.
