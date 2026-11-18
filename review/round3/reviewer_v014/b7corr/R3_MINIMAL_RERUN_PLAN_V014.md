# R3 minimal rerun plan v0.14 — corrected B7 priority

## Correction to v0.13

The v0.13 rerun plan incorrectly treated the three-crossing 40-step passive result as sufficient evidence that the historical four-crossing 100-step card was not robust to passive capture. That attribution was confounded by actual denoising-step count/traversal and prompt composition.

## B7 decisive rerun

If the four-crossing/rebound result is retained in Mode-I or Mode-II, run this **before paper freeze**:

1. SD3.5-Medium, CFG 1.0, current Passive instrument.
2. **100 denoising steps**, matching the historical suite.
3. Historical pyramid-construction prompt.
4. 1024 resolution first; >=30 seeds.
5. Add 512/768 only if the resolution-invariance claim remains important.
6. Best case: run the historical Diagnostic-EMAG build and current Passive build on the same prompt/seeds/100-step schedule.

The existing-data re-audit shows that merely thinning the historical 100-step observations to the nearest 40 sigma locations does **not** erase the fourth crossing at 512, 768, or 1024. Truncating the old curve to the new study's sigma range also leaves all four crossings. Therefore neither observation density nor omission of the very-late sigma tail is sufficient. The actual 100-vs-40 sampler traversal remains a confound because a true 40-step run follows a different numerical path.

## Other R3 reruns

All other v0.13 gate determinations remain unchanged unless separately noted in `R3_GATE_STATUS_V014.csv`.
