# Round 12 preregistration, addendum 1: mixed 1024 / 1008 px width in the Base primary (declared before any Round 12 data was seen)

Written 2026-10-24T18:33:56Z. PREREG_R12.md (SHA-256 ad06a1bb…) is unchanged.

Zac reported at 2026-10-24 18:32Z that about half of the Z-Image Base primary generations are 1024×1024 and about half are 1008×1024. At 1008 px there are 63 × 64 = 4032 image rows instead of 4096 (1.6% fewer; still a multiple of 32, so no pad rows). Each CSV row's `seq_len` identifies the resolution.

Expected size of the effect, from the existing SD3.5 M resolution arms (Round 3, seed 1001, 60 prompts):
- 768 vs 1024 (Δlog N = −0.58): tangent 0.988, mean log g shift +0.015.
- 512 vs 1024 (Δlog N = −1.39): tangent 0.966, mean log g shift −0.084.

1008 vs 1024 is Δlog N = −0.016, about 36 times smaller than 768 vs 1024. That extrapolates to a tangent loss of about 0.0003 and a log g shift of about 0.001, against a prompt-to-prompt SD of about 0.16.

Rules, fixed now:
1. **Pooling check, run first, internal to Z1:** the 1008-px half against the 1024-px half, attn and MLP.
   - Pass if the tangent is ≥ 0.99 and the matched-σ mean log g difference is |Δ| ≤ 0.05.
   - On a pass, the primary analysis uses all Z1 generations, and every R12 test is also reported on the 1024-px half alone as a robustness line.
   - On a fail, the primary analysis uses the 1024-px half only, and the 1008-px half is reported descriptively.
2. Analyses at the prompt or domain level (decoding, RSA, prompt share) include resolution as a covariate, or are stratified by it, whenever the split is unbalanced across domains.
3. The identity gate and the preflight run per resolution: `--img-rows 4096` for the 1024-px files and `--img-rows 4032` for the 1008-px files.
4. If the 1024-px half holds fewer than 2 generations for some prompts, the robustness line uses only the prompts that have them, and says so.
