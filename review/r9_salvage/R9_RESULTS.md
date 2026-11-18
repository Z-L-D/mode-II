# Round 9: Codex missed correlations and LLM salvage (results)

**Preregistration:** `R9_PREREGISTRATION.md`, SHA-256 `09c1f170644d883e6508d0d91b57328a9858517716b8f64fd4f10a9a3362610e`, frozen 2026-10-09T09:07:54Z.
**Computed:** 2026-10-09 09:08–09:18Z.

**Scripts:**
- `scripts/r9_a_codex.py` (Part A);
- `scripts/r9_b_llm.py` (Part B).

Outputs are in `out/`. No PSTF or pstf_llama data was used.

There are 11 decision rules:
- **5 held:** A4, B1, B2, B3, B6.
- **6 tested, not supported:** A2, A3, A5, B4, B5, B8.

A1 and A6 are tabulations and B7 is descriptive.

## Scorecard

| Test | Result | Verdict |
|---|---|---|
| A1 [tab] | The Codex memo is reproduced exactly: RSA~theme ρ −0.405 / 0.810 / 0.976 (exact p 0.33 / 0.022 / 0.0004); RSA~tangent −0.714 (middle, p 0.058) | tabulation |
| A2 held-out replication on 10 SD1.5↔transformer pairs | RSA~theme_r at the last state: attn 0.80 (p 0.13), MLP 0.50. At the middle state: 0.0 in both | **tested, not supported**: the memo's strongest correlation does not replicate on new pairs |
| A3 three-crossing incidence below vs above onset (6 ch.) | higher below onset in 1/6 (SD3.5 L). SD3.5 M 0.00 vs 0.77, ZIT MLP 0.00 vs 0.75, ZIB attn 0.20 vs 0.50 | **tested, not supported**. Where it differs, three crossings sit *before* the onset, unlike ZIB MLP in R8-7 |
| A4 first-state prompt share vs first-state decoding (7 ch.) | ρ 0.64, LOO 0.54–0.77 | **associated**. Descriptive: mean share vs mean middle-state RSA over 5 models, ρ 0.90 |
| A5 anisotropy vs backbone PC1 (9 ch.) | ρ −0.45; LOO fails | **tested, not supported** |
| A6 [desc] capture shift vs tangent loss (ZIB) | the ordering is the same in both submodules (BM < uncond < pooled; ρ 1.0, n = 3). The pooled shift is 0.28 / 0.32 in log g | descriptive |
| B1 model-free nested i.i.d. capture | \|ρ(n, g)\| = 1.000, \|ρ(n, PR/n)\| = 1.000 and \|ρ(n, H)\| = 1.000 in all 6 populations (D 1536/4096; isotropic, power law, power law + spike) | **the retirement statistic is model-free** |
| B2 Qwen derivative r with uncertainty | 1.5B↔3B **0.631** (bootstrap 0.614–0.649; layer-permutation null 95% 0.382); 1.5B↔7B −0.122 (−0.160 to −0.081); 3B↔7B −0.056 (−0.102 to −0.009) | **1.5B↔3B is above its null and separated from the other two pairs** |
| B3 Mistral ablation tangent at prefill | cosine 0.99980 (bootstrap 5–95% 0.99976–0.99983); log-g derivative r 0.99998 | **the edit preserves the tangent** |
| B4 divergence position | divergence at generated token median 7 (mean 8.5; 10–90% 1–17; max 47; 13% at the first token). ρ with prefill \|Δlog g\| −0.12 (p 0.19) | **tested, not supported** |
| B5 localisation along the shared prefix (905 tokens) | layers 0–5 identical in 99.8% of tokens; the final layer is the argmax in 56% (layer 31: 509, 26: 160, 20: 51) | **tested, not supported**. The prefill "final layer 4×" does not hold token by token |
| B6 prompt RSA across LLMs at matched context, n-residualised, partial on \|Δlog n\| | Qwen 1.5↔3: 0.79–0.87; 1.5↔7: 0.74–0.85; 3↔7: 0.72–0.89; Mistral↔Llama 3.1 instruct: 0.45–0.49; Mistral instruct↔ablit: 0.999 / 0.64 / 0.67. Every p = 0.0001 | **shared** in all 5 pairs at all 3 token indices |
| B7 [desc] derivative r vs context length | rises monotonically with n in all three Qwen pairs (Spearman 1.0 over 8 bins). 1.5↔3: 0.53 → 0.65 | descriptive |
| B8 Qwen "terminal collapse" | 1.5B and 3B: none (the final-layer step is never ≤ −0.8). 7B: final-layer g_L/g_{L−1} − 1 lies between −0.70 and −0.87 in **every** row (median −0.84), with ≤ −0.8 in 78.6%. The rate on final EOS tokens (0.61) is *below* that on other tokens (0.79) | **tested, not supported** as EOS and narrowly below the preregistered 80% readout threshold. Post hoc reading: a final-layer property of Qwen 7B on every token, not an EOS signature |

## Fact-check of the premises

**GLM**
- The rank ceiling is correct: max n = 389 < D, so g ≤ n/D.
- But g sits at 0.33–0.37 of the ceiling, not on it. A ceiling-normalised PR/n is still perfectly monotone, and B1 shows that any nested sample does that, so the retirement was right for a broader reason than the ceiling.
- N12/N15 cannot be run, because only scalars were logged. The isotropic finite-N reference it was meant to supply was already done in R8-4.
- N11's comparison with the diffusion ≤ 0.24 band mixes statistics. Against its own null, 1.5B↔3B does stand out (B2).
- The "~token 22" divergence in ledger v0.27 is not reproduced: by layer-0 identity the texts diverge at a median of generated token 7.
- **The chat-template flag:** the prefill length is identical across the three Qwen sizes for all 24 prompts (checked), so the scaling comparison is within one template regime. Which template that regime was is not determined. The tokenizer was not reachable, and the v1 LLM script set every tokenizer's chat template to Llama-3.1's if that line was live (`capture_code_r2/README_R2.md`). *Correction, 2026-10-09 11:35Z:* an earlier version of this line inferred Qwen's own template. The papers say only "one chat-template regime".

**DeepSeek**
- The increment-test premise is wrong: E7's derivative is across depth, not n.
- The MP crossing at n = D is never reached (n ≤ 389).
- The abliteration depth profile is already in Mode-I §8.

**Qwen 3.8**
- The terminal collapse is not EOS (B8).
- Prefill domain decoding is impossible because the LLM prompts carry no domain labels.
- "Almost certainly", "proves" and "thermodynamics of termination" are unsupported wording.
- Most generations stop at the 256-token cap, not on EOS.

**Codex**
- The memo's numbers are exact (A1), but its strongest correlation does not replicate on held-out pairs (A2).
- The salvage reply's numbers from outside the archive (rebound census, 0.15 R², D08/D09) were not verified.
- Its context-residualised decoding would be trivial (the prompt rows are shared across seeds). The RSA form held (B6).

## Reading
1. **The retirement of per-token LLM dynamics is now proven to be an instrument property.** Any nested running sample gives |ρ| = 1 for g, PR/n and H, with no model at all (B1).
2. **What the clean LLM data still says, at matched context:**
   - Qwen 1.5B↔3B depth-derivative agreement (0.63) is real and above its null, while both pairs with 7B sit slightly below zero, inside a tight interval. Agreement rises with context length in every pair.
   - The Mistral abliteration preserves the depth tangent at prefill (0.9998), which completes the intervention grid's ablation cell.
   - Prompt-relational geometry is shared across the LLMs (n-controlled partial Mantel 0.45–0.89).
3. **Caveat on B6.** The LLMs read the same token text directly, so a shared RSA is weaker evidence than in diffusion. No text-similarity control was preregistered.
4. **What does not hold:**
   - the Codex RSA~theme relation on new pairs;
   - three-crossing concentration on the plateau;
   - anisotropy vs backbone;
   - divergence position vs prefill deformation;
   - token-by-token final-layer localisation;
   - an EOS reading of the Qwen collapse.

## Candidate paper changes (not applied; awaiting Zac)
1. **Mode-I §8, scaling bullet.** Add intervals: +0.63 (bootstrap 0.61–0.65; layer-permutation null 95% 0.38), −0.12 (−0.16 to −0.08), −0.06 (−0.10 to −0.01). Add that agreement rises with context length in all three pairs. Add one clause that the 1.5B↔3B value is above its null, so scaling can preserve part of the local derivative.
2. **Mode-I §8, Mistral bullet.** The abliteration keeps the prefill depth tangent at 0.9998, and the texts diverge at a median of generated token 7. (Neither paper states a divergence token.)
3. **Mode-I instrument/limits and Mode-II.** A model-free nested i.i.d. sample reproduces |ρ(n, g)| = |ρ(n, PR/n)| = 1, so the LLM retirement is an instrument property, independent of the model.
4. **Mode-I §6.** First-state prompt share tracks first-state decoding across channels (ρ 0.64, LOO-stable).
5. **Optional, with the caveat above.** Prompt-relational geometry is shared among LLMs at matched context (partial Mantel 0.45–0.89, n-controlled).
6. **Ledger only:** A2, A3, A5, B4, B5 and B8.

## New capture worth making (from GLM; recommendation only)
A truly distilled LLM teacher→student pair, captured with the v2 windowed instrument (G4). This would test whether tangent inheritance extends beyond diffusion distillation.

## Addendum B6b: text control for the LLM RSA
Preregistered as `R9_ADDENDUM_B6b.md` (SHA cdb6f994, frozen 09:30:44Z) after Zac approved all five candidates (09:30Z).

**Method.** A partial Mantel on the n-residualised profiles, with two controls: context-length distance and TF-IDF text distance. Script: `scripts/r9_c_textcontrol.py`; output: `out/R9_B6b_*`.

**Results:**
- Qwen 1.5↔3: 0.79–0.87.
- Qwen 1.5↔7: 0.74–0.85.
- Qwen 3↔7: 0.72–0.89.
- Mistral↔Llama 3.1 instruct: 0.44–0.48.
- Every p = 0.0001.
- The text control removes 0.3–3%.
- Each model's own text-distance RSA is only 0.11–0.23.

**Diffusion comparison under the same two controls:** the unrelated pairs at the first state give 0.23–0.38, with 2–5% removed.

**Verdict:** **survives the text control** in all four pairs.

## Paper status (09:50Z)
All five candidates are in both papers. Zac approved them at 09:30Z. Both PDFs build cleanly (Mode-I 40 pp, Mode-II 33 pp), and the prereg tables carry a Round 9 row.
