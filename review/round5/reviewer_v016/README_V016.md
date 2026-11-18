# CoS-PAFS v0.16 repaired-branch / deepstep audit

This directory contains the Oct 7, 2026 audit of the repaired per-branch diffusion captures and the 40/100-step pyramid captures.

## Execution order

1. `INDEPENDENT_NEW_CAPTURE_PREREG.md` — frozen before numerical inspection.
2. `01_new_capture_intake.py` — archive/schema/instrument/design intake.
3. `scripts/02_R1_per_branch_core.py` — R1a/b/f/g/h and branch geometry.
4. `scripts/03_R2_deepstep.py` — deepstep R2 topology and matched-sigma calculations.
5. `scripts/04_R1_seedcentred_posthoc.py` through `16_protocol_and_output_linkage_audit.py` — explicitly post-hoc robustness/mechanism/process audits.
6. `scripts/06_audit_round4_scorecard.py` — independent replay of supplied Round-4 P1-P5 scorecard.
7. `figure_scripts/*.py` — paper/audit figures.
8. `scripts/18_update_ledger_v016.py` — append-only update of the supplied v0.15 ledger.

## Formal scoring discipline

Do not report an aggregate R1/R2 hit rate. The new instrument works, but the execution substituted 4/30 prompts, reused global seeds 1001/1002 across prompts, and the 100-step CFG1 pyramid arm has only 15 seeds rather than the preregistered >=30.

`results/R1_R2_AUDIT_SCORECARD.csv` distinguishes formal verdicts, criterion-level evidence, and post-hoc results.

## Known input dependencies

Scripts expect the conversation-mounted inputs under `/mnt/data`, including:
- `uncond-cond+deepstep-telemetry.zip`
- `TELEMETRY.zip`
- `matched_prompts_30(2).tsv`
- the extracted supplied Round-4 package at `/mnt/data/r4_pkg/round4` for scorecard replay

## Reproducibility map

`CLAIM_SCRIPT_MANIFEST_V016.csv` maps every new quantitative claim/figure to its Python reproducer and outputs.
