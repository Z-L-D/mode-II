# Round 13 predictions (frozen 2026-10-27T22:14:41Z, before any HOLDOUT number is computed)

Split: SPLIT.md (sha 3973063e…). Exploration used only EXPLORE prompts (logs in explore/). Every test below runs only on HOLDOUT prompts.
Arms: Base spectral = Z2 (CFG 1, 25 steps), Turbo spectral = ZIT CFG 1 (8 steps), matched by nearest sigma as in explore2.py.
Base primary = Z1 cond / uncond, Turbo matched = Z3 cond (identical settings). Statistic: mean over generations of log g (layer by layer).

What exploration showed (EXPLORE half): Base is more concentrated (lower g) than Turbo almost everywhere, but the difference largely disappears once
the mean direction is removed; Base image tokens carry a larger shared mean direction at every layer; the Base-Turbo difference in image rows is
large at the start and end of the network and small in the middle; classifier-free guidance changes attention g in opposite directions at the ends and in the middle.

| ID | Prediction | Supported if (HOLDOUT) |
|---|---|---|
| H1 | Base image tokens have a larger shared mean direction than Turbo | median mean_energy_share (over gens and steps) Base > Turbo in >= 27 of 30 layers, in attn and in MLP |
| H2 | Removing the mean direction removes most of the Base-Turbo level difference | mean over layers of abs(Base - Turbo) is at least 2x larger in V1 (image rows) than in V3 (centred), in attn and in MLP |
| H3 | Base is more concentrated than Turbo | (a) V1: Base mean log g < Turbo in >= 27 of 30 layers, attn and MLP; (b) Z1 vs Z3 all rows: Base < Turbo at every one of the 25 steps (layer-averaged), attn and MLP |
| H4 | Distillation changed the ends of the network more than the middle | V1 image rows, abs(Base - Turbo) averaged over layers 17-22 is < 0.5x the average over layers 0-5 and < 0.5x the average over layers 24-29, attn and MLP |
| H5 | Guidance signature in Base attention | Z1 cond - uncond mean log g (attn): mean over layers 0-2 > 0, mean over layers 25-29 > 0, mean over layers 6-21 < 0; and the layer-averaged difference is < 0 at every step from 2 to 15 |
| H6 | The Base all-rows statistic is set by the prompt from the first step | Z1 cond attn: prompt share (eta^2 over 4 seeds per prompt) of layer-mean log g at step 0 >= 0.90 |

Not predicted (descriptive only): where caption dominance begins by layer; top-1 channel shares.
