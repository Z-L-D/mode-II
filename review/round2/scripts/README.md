# Round 2 scripts

Run from the scratchpad working directory. It must contain:
- `pq/*.parquet`, built by `review/scripts/load.py`;
- `F_*.pkl`, built by `review/scripts/build_fields.py`;
- `common.py` and `beta24.py`.

The CFG-study script reads `/mnt/project-files/data/sd35m_cfg1.0_vs_5.5/`. The prior 8/16/32 script runs inside the Drive `cos-dm/dm_telemetry` folder.

| Script | Experiment |
|---|---|
| e1_sync.py | P1 |
| e2_regime.py | P2 |
| e3_sigma_registration.py | P3 |
| e4_stepcount.py, e4_prior_8_16_32.py, e4b_mochi_speed.py | P4 |
| e5_seed_fate.py, e5b_persistence.py | P5, plus the post hoc variance-ratio test |
| e6_varcomp.py | P6 |
| e7_llm_context.py | P7 |
| e8_ablit.py | P8 |
| e9_llama_sectors.py, e9b_zimage_calibration.py | P9 |
| e10_tangent.py | P10 |
| e11_loop.py | P11 |
| e12_beta_topology.py | P12 |
| e13_cfg_study.py | P13 |
| figures.py | Figures |
