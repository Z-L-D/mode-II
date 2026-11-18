# Addendum (post hoc), 2026-10-08T21:58:36Z

Written after C1–C3 and P1–P4 had been computed, and before P5 and P6 were run. Both are post hoc. They were added because the Mode-I content-geometry claims (domain decoding and cross-model prompt RSA) are headline results that are well-defined for a U-Net.

- **P5. Domain decoding.** Ten-way decoding with the Mode-I I-F5 decoder, unchanged:
  - StandardScaler + LogisticRegression(C=0.1), 5-fold GroupKFold by prompt;
  - features log|[g, gap, H]| per block at the first state (cond);
  - controls: layer mean, prompt length, TF-IDF on the prompt text, and 99 depth scrambles.
- **P6. Prompt RSA.** The Round 6 S2 method, unchanged:
  - prompt-mean depth profiles interpolated to 24 normalised-depth points;
  - RDM = 1 − correlation;
  - Mantel test on ranks with 2000 permutations.
  - SD1.5 (attn, and MLP) is compared against each of the five archive models at the first, middle and last step, on the prompts shared by all six. The archive pairs are recomputed on the same prompt set as a reference.
