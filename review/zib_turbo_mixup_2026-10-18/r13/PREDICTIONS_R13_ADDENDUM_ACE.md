# Round 13 addendum: ACE Turbo on the image prompts (frozen 2026-10-27T22:15:54Z, before any HOLDOUT ACE number is computed)

Zac (22:14Z): how does AS15T on the matched prompts compare with the image models on the same prompts? Same split (SPLIT.md), exploration log explore/explore_ace.log.
Statistics: Round 6 S1 tangent(); per-prompt level = mean over gens, steps and layers of log g; ACE audio reference = the Round 3 ACE Turbo audio run (all 120 gens).
Image models: Z1 (Base cond), Z3 (Turbo matched), ZIT archive, and for attn SD3.5 M, SD3.5 L, Mochi archives.

| ID | Prediction | Supported if (HOLDOUT) |
|---|---|---|
| A1 | ACE's depth fingerprint is unrelated to every image model, even on identical prompts | tangent < 0.40 against every image model, attn and MLP |
| A2 | Prompts that raise or lower g in the image models do not do the same in ACE | abs(Pearson r) of per-prompt level < 0.30 against every image model, attn and MLP |
| A3 | ACE's weights, not the prompt type, set its attention fingerprint; MLP is more content-sensitive | ACE image-prompt vs ACE audio-prompt tangent >= 0.85 in attn, and MLP tangent < attn tangent |
| A4 | ACE's level carries little image-domain information | domain eta^2 of per-prompt level <= 0.30 (attn and MLP) |
