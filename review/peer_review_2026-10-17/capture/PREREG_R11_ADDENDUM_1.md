# Round 11 preregistration, addendum 1 (protocol deviation, declared before any data was seen)

Written 2026-10-18T05:08:15Z. PREREG_R11.md (SHA-256 3834abb0…) is unchanged.

Zac reported at 2026-10-18 05:07Z that the Round 11 captures used the 120 primary-campaign prompts, not the 60-prompt Round 3 file. The 60 preregistered prompts are a subset of the 120: all 60 appear in the ZIT, SD3.5 M, SD3.5 L and Mochi archive prompt sets, and 57 of 60 in the ZIB archive.

- Primary analysis: R11-1 to R11-7 run exactly as frozen, restricted to the 60 preregistered prompts.
- Secondary analysis, declared now: the same tests on all prompts captured. These are reported as robustness only and cannot overturn or rescue a primary result.
- Gate 4 (bit-identity) runs on the overlapping 60 prompts. It applies only where the run used CFG 1, seed 1001 and 1024 px; the settings actually used are recorded on receipt.
