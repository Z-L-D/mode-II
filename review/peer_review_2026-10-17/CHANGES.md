# Mode-I revisions from the 2026-10-17 peer reviews

Zac accepted these changes on 2026-10-17 at 11:46Z. The pre-edit copy of the paper is `main_before_edits_2026-10-17.tex`. Numbers refer to Zac's list.

| # | Change | Where |
|---|---|---|
| 1, 23 | Caption-row sentence: both Z-Image hooks include 64 caption rows from the same text encoder. These rows could raise the lineage tangent but cannot lower the unrelated values. The lineage tangent is a statistic of the joint sequence. | §5.2, end of the tangent paragraph |
| 2, 21 | "Inherited" defined once ("shared by descent within this model lineage"). Base is a sibling of Turbo's teacher. One lineage pair. The objective cannot be separated from shared pretraining. | §1, lineage paragraph |
| 2, 21 | "Distillation preserves" → "the one distillation lineage shares … not separable". The bullet is renamed "Distillation lineage". | §9 bullet and closing sentence; synthesis item 3 |
| 3 | "one pair" / "one lineage pair". | Abstract (first sentence; "distillation pair" → "lineage pair"); synthesis item 3 |
| 4, 19 | Surrogate class carried into the depth-sign sentence. | Abstract; synthesis item 1 |
| 5, 23 | Scope statement: every statistic is of the uncentred second moment, the archive stores only (g, gap, H), and results are scoped to it. | §3.1 |
| 6, 20 | "was not observed in the one audio model tested; following the preregistered consequence we do not extend it beyond image and video". | Abstract; §12 closing paragraph; synthesis item 6 |
| 7, 18 | Abliteration moved to its own bullet, "Fixed architecture, targeted weight edit". The resolution text moved into the fixed-weights bullet. Fig. 11 legend now has a "weight edit" marker and "distillation lineage"; the caption is updated. | §9; Fig. 11 script and caption |
| 8 | Footnote: the prefill value is a single statistic of the prompt matrix, not a running statistic. | §6.2 |
| 9, 24 | Footnote with the threshold sensitivity (1.5×: 0.57–0.81; 3×: 0.79–0.93; window ±0.1 moves the range by ≤ 0.09). | §4 |
| 10 | New Fig. S5: per-domain recall and pooled confusion. Script `scripts/fig_I_S05_domain_confusion.py`; data `data/I-S5_*.csv`. Reproduces Table 3 accuracies exactly. | Appendix F; cited in §6.1 |
| 11 | TF-IDF details (sklearn defaults, no stop words removed, ℓ2, 1 − cosine). | §6.2 |
| 12 | Definition of the corrected share (one-way random-effects ANOVA with n₀). | §6.3 |
| 13 | The cause of the Qwen 7B break is untested; depth does not explain it (1.5B and 7B both have 28 layers). | §9 scaling bullet |
| 14 | Checkpoints / ComfyUI commit: **not added** (Zac's call). | none |
| 15, 22 | ZIT separates part of the audio confound: its MLP anisotropy is in the image range, and its MLP keeps +−+−. | §12 closing paragraph |
| 16 | Cond-only decoding on the matched Round 3 captures (`out/cond_only_decoding.csv`): ZIB CFG 1 attn 0.45 / MLP 0.35, SD3.5 M CFG 1 0.32; TF-IDF 0.28; p = 0.01 against 99 scrambles. | §6.1 |
| 17 | Operating-point shift A defined in the Fig. 11 caption. Definition taken from the ledger's "manifold, operating point and clock" table; the original computation script is not in the project. | Fig. 11 caption |
| 25 | Tests and verification clause. | Code and data |
| 4 (optional) | Per-step surrogate run (`scripts/perstep_surrogate.py`, `out/perstep_surrogate_final.csv`; per-step observed patterns match Round 6 A4 exactly). The pattern beats a per-step mean-field surrogate (≤1/20 draws) at 6/13 steps (SD3.5 M), 12/24 (SD3.5 L), 10/18 (Mochi) and 6/8 (ZIB attn), binomial p < 0.001 each; ZIB MLP 1/9, ZIT MLP 1/3, ZIT attn 0/2. New paragraph. | §7.2, after the surrogate paragraph |
| — | Massive-activation limitation rewritten to name the recapture (see `capture/CAPTURE_SPEC.md`). | §14 |

Build: pdflatex + bibtex, clean (no undefined references or errors), 43 pages: `build/mode-I_revised_2026-10-17.pdf`. Not rebuilt: `papers/dist/` and the release zips. Mode-II not touched.
