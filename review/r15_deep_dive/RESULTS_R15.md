# Round 15 results (prereg PREREG_R15.md, sha 3acbb027, frozen 2026-11-05T21:23:24Z; scored 21:25-21:40Z)

Scoring script: `scripts/score_r15.py` (SCR = cube dir). LLM cubes: `scripts/build_llm_cubes.py` -> `out/llm_cubes/`.
Verdicts: `out/R15_verdicts.csv`. Six of seven supported; R15-3 refuted.

| Test | Verdict | Key numbers |
|---|---|---|
| R15-1 text-to-latent handoff on held-out captures | SUPPORTED (4/5) | cond-uncond prompt RSA at step 0 / step ceil(0.1T) / final: SD3.5 M attn 0.01 / 0.68 / 0.998; ZIT-25 attn 0.08 / 0.83 / 0.93; ZIT-25 MLP 0.00 / 0.81 / 0.797 (final misses 0.80); U-Net attn 0.12 / 0.85 / 0.998; U-Net MLP 0.05 / 0.67 / 0.988 |
| R15-2 guidance does not raise seed share | SUPPORTED | SD3.5 M cond, 30 prompts x 2 seeds: CFG 4.5 <= CFG 1 + 0.02 at 100% of steps; equal at step 0 (0.232, as it must be), then CFG 4.5 0.23-0.28 vs CFG 1 0.30-0.66 |
| R15-3 lineage content geometry in ACE-Step | REFUTED | ACE Base vs Turbo prompt RSA 0.05-0.07 (attn, p 0.09-0.14), -0.04 to 0.02 (MLP) |
| R15-4 block-restricted layer null | SUPPORTED | margins over block p95: same weights 0.60/0.70; Z-Image lineage 0.47/0.48; ACE attn 0.44; ACE MLP 0.19 (< 0.20, as predicted); unrelated max -0.015 |
| R15-5 lineage discrimination | SUPPORTED | AUC 1.000 for all three contrasts (attn lineage>unrelated, attn same>lineage, MLP lineage>unrelated) |
| R15-6 LLM cross-family tangents | SUPPORTED | attn Q35-Q25 -0.22, Q35-G3 -0.23, Q25-G3 -0.12 (p95 0.30-0.37); MLP full: Q35-Q25 0.656 (p95 0.32), others <= -0.08; MLP without last step: Q35-Q25 0.44 (p95 0.31), Q35-G3 0.28 (0.25), Q25-G3 0.11 (0.30) |
| R15-7 LLM and DiT prompt geometry | SUPPORTED | Qwen2.5-1.5B vs Z-Image Base first-state RSA 0.09 (attn) / 0.12 (MLP); Qwen3.5-0.8B 0.14 / 0.13; all p = 0.001 (120 prompts) |

Readings and caveats
- R15-1: prompt-specific geometry reaches the unconditional branch within the first 10% of steps in SD3.5 M, Z-Image Turbo
  and the SD1.5 U-Net. SD3.5 M and ZIT-25 use one global seed per arm, so every prompt starts from the same noise there:
  the uncond branch can only differ across prompts through the latent the guided sampler has written. This rules out
  shared seed noise as the explanation (a confound in the exploratory X2 on Z1 and ACE, which use random seeds).
- R15-2: guidance halves seed share in SD3.5 M. The exploratory X3 contrast (Base at CFG 4 shows 2-15x the seed share of
  Turbo at CFG 1) therefore runs against the guidance effect, not with it. A same-CFG Base/Turbo capture is still the clean test.
- R15-3: the exploratory X1b result (Z-Image Base<->Turbo prompt RSA at the same-weights level) does not replicate in
  ACE-Step. ACE Base is seed-dominated (exploratory seed share 0.5-0.8), and with 2 seeds per prompt its prompt means are
  noisy, so this may be a power problem; as preregistered, content-geometry inheritance is a Z-Image result only.
- R15-4: a conservative null that keeps the coarse depth order (permutation within quarters) leaves every tier intact.
  ACE MLP is above that null (margin 0.19) but below the 0.20 used for the other lineage channels, consistent with the
  exploratory trim (0.572 -> 0.384 without the last step; 0.158 without the first and last two).
- R15-5: AUC 1.000 is a statement about these pairs; draws are 30-generation subsets of the same captures, so the
  independent units are the 6 same-weights, 6 lineage and 19 unrelated channel pairs, not the 200 draws.
- R15-6: no attention lineage-like similarity between three LLM families. Qwen3.5 vs Qwen2.5 MLP is above its null
  (0.656 full; 0.44 without the last step), unlike either Qwen vs Gemma: a within-vendor MLP resemblance that is not a
  distillation lineage. Worth a preregistered follow-up with known Qwen derivatives.
- R15-7: LLMs reading the image prompts share a weak but reliable prompt geometry with Z-Image Base (0.09-0.14), below
  DiT-DiT unrelated pairs (0.21-0.39 exploratory). Neither LLM is Z-Image's text encoder (Qwen3-4B).
