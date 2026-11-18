#!/bin/sh
# Build the derived cubes (Rounds 11-14) from the raw captures listed in DATA.md.
# Run from the repository root. Output: $COSPAFS_CACHE (default cache/). Rules: review/zib_turbo_mixup_2026-10-18/r12/scripts/build_cubes.py header.
set -e
C=${COSPAFS_CACHE:-cache}; R=data/r12_captures; A=review/ace_base_r14/data
B=review/zib_turbo_mixup_2026-10-18/r12/scripts/build_cubes.py
BA=review/ace_base_r14/scripts/build_cubes.py          # same builder; domain parse guarded for ACE Base file names
mkdir -p "$C"
python3 $B Z1 cond 4 $C $R/Z1_ZIB_CFG4_primary/*.csv      # real Z-Image Base, CFG 4, cond branch, 4 gens per prompt
python3 $B Z1U uncond 4 $C $R/Z1_ZIB_CFG4_primary/*.csv   # same run, uncond branch
python3 $B Z2 cond 0 $C $R/Z2_ZIB_CFG1_variants/*.csv     # real Base, CFG 1, row-split variants (V_Z2_*.pkl)
python3 $B Z3 cond 0 $C $R/Z3_ZIT_CFG4_matched/*.csv      # Turbo at Z1 settings
python3 $B ZITSV cond 0 $C $R/R11_ZIT_CFG1_variants/*.csv # Turbo 8 steps, CFG 1, row-split variants
python3 $B SD3SV cond 0 $C $R/R11_SD35M_CFG1_variants/*.csv
python3 $B ACET cond 0 $C review/r3_ace/data/*.csv        # ACE-Step turbo (R3 Part B raw CSVs)
python3 $BA ACEB cond 0 $C $A/ACE15B_180s_CFG4_gpu0.csv $A/ACE15B_180s_CFG4_gpu1.csv
python3 $BA ACEBU uncond 0 $C $A/ACE15B_180s_CFG4_gpu0.csv $A/ACE15B_180s_CFG4_gpu1.csv
python3 $BA ACEB0 cond 0 $C $A/ACE15B_180s_CFG4_gpu0.csv
python3 $BA ACEB1 cond 0 $C $A/ACE15B_180s_CFG4_gpu1.csv
# Round 13 holdout splits (Z1H, Z1UH, Z2H, Z3H, ZITSVH) are written by review/zib_turbo_mixup_2026-10-18/r13/scripts/.
