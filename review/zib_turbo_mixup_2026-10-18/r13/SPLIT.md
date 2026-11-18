# Round 13 split (frozen 2026-10-27T22:13:06Z, before any Round 13 look at the data)

Zac (2026-10-27 22:12Z): look for further patterns in the new data; when one shows up, stop and write a prediction before calculating.

Design: split by **prompt**. SPLIT.tsv assigns each of the 120 primary prompts to EXPLORE or HOLDOUT: within each domain, prompts are ranked by sha256(prompt text) and alternated, separately inside the 60 Round 3 prompts and the other 60. So each domain has 6 + 6, and the 60 spectral-arm prompts split 30 + 30 (3 + 3 per domain).

Rules:
1. Exploration uses only EXPLORE prompts, in every arm (Z1 cond/uncond, Z3, Z2, ZIT spectral, SD3.5 M spectral, ACE Turbo matched) and in the archive cubes.
2. Any pattern seen there becomes a written prediction in PREDICTIONS_R13.md (timestamped, SHA-256 recorded) **before** any HOLDOUT number is computed.
3. Predictions are scored only on HOLDOUT prompts. Exploration numbers are never reported as evidence.
4. Already-seen whole-data results (RESULTS_R11_R12.md, posthoc_text_vs_image_rows.csv) cannot be the basis of a "new" prediction without saying so.
