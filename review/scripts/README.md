Run order (from a work dir): load.py (unzip TELEMETRY.zip + nested DM zips first; fixes the headerless Mochi file) -> build_fields.py -> any analysis script.
common.py: loaders, completeness filter (exact steps x layers x submodules), syncR (ledger definition), depthnorm, shape_r.
sync_artifact.py, increment_coherence.py, slow_flow.py, registration_and_shape.py, seq_len_audit.py, sem.py (domain decoding + controls), seed_fate.py, llm_checks.py.
