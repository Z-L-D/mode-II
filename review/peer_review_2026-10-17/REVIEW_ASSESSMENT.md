# Mode-I peer reviews (DeepSeek, Qwen): point-by-point assessment

Date: 2026-10-17. Paper state: the main.tex attached on 2026-10-17 (identical to `papers/mode-I/main.tex`). No paper file was edited.

Verdicts:
- **Valid**: the paper should change.
- **Wording**: the result stands, but the text invites the misreading.
- **Already addressed**: the paper already does or says this.
- **Misreading / not applicable**: the reviewer misread the paper, or the request can't be done with this instrument.
- **Needs capture**: settling it needs new telemetry.

## Summary

Four points are real and need changes to the text.
1. The abliteration comparison is filed under "fixed weights", but abliteration edits weights.
2. The synthesis states the depth sign sequence without its surrogate qualification.
3. The audio conclusion is worded as modality specificity.
4. "Distillation preserves" is causal, but Turbo was distilled from the SFT model, not from Base.

Both reviewers' top technical concern is the uncentred spectrum (massive activations). For Z-Image there is also the text-row issue. Neither can be tested from the archive, because the capture stores only g, gap and H, computed on the whole uncentred matrix. One small new capture would settle both (see the end).

Several minor complaints come from DeepSeek reading a broken PDF-to-text extraction: the "igure 1" typo, a duplicated paragraph, "ModeI", merged table columns. None of these is in the tex. Qwen's CKA request cannot be done with this instrument.

## DeepSeek: major points

**1. Z-Image hooks include 64 text rows. Valid as a scoping point; settling it needs a capture.**
- The capture flattens the whole joint sequence and computes the uncentred spectrum on it (`capture_code_r3/nodes.py` ~1505–1520). The text rows are 64 of 4160 rows per branch (1.5%; ZIT `seq` = 4160, ZIB pooled 8320). They cannot be removed afterwards.
- Domain decoding does not need text rows. The image-only channels decode as well or better (Mochi 0.61, SD3.5 L 0.55, against Z-Image 0.46–0.55).
- Unrelated-pair results are unaffected in the direction the reviewer fears. The Z-Image↔Mochi RSA (0.39) has text rows on one side only, and every unrelated tangent stays at or below 0.24 either way.
- Where text rows could matter is the lineage itself: the Base↔Turbo tangent (0.94–0.99) and the lineage RSA (0.56–0.62). Both models put the same text-encoder rows into the matrix. Uncentred spectra can be moved by a few high-norm rows, so 1.5% of rows is not automatically negligible.
- The paper already discloses the text rows (§3.3), but never connects them to the lineage result.
- Fix now: one sentence in §5.2 noting this. Settle it with the capture below.

**2. "Inherited" is causal. Valid, and stronger than the reviewer says.**
- The intro says Turbo was distilled from the Z-Image SFT model, not from Base. Base and Turbo are therefore siblings that share pretraining. The design cannot separate what distillation did from what the shared lineage already had.
- §9 says "Distillation preserves geometry", and §1/§2 frame the comparison as distillation.
- Proposal: keep "inherited" in the title only if it is defined once as "shared by descent within a model lineage" (a reasonable reading). Change "distillation preserves" to "the Base/Turbo lineage shares" in §9 and the synthesis.
- The title is your call.

**3. One lineage pair, a selected baseline, no margin CI. Mostly already addressed.**
- The "best unrelated pair" is the maximum over all eight unrelated pairs. That choice works against the lineage claim, not for it.
- The tangent already has a layer-permutation null (p95 0.24/0.20) and a subset interval for the lineage value (0.937–0.946).
- The single-lineage point is fair, but the paper already reports one lineage pair, not distillation in general.
- Fix: add "one lineage pair" to the abstract or synthesis. A prompt-bootstrap CI on the margins is cheap to add if you want it, but a 0.70 margin will not change.

**4. Multiplicity, and "at some noise level". Partly valid.**
- Table 4 gives surrogate classes for the pooled pattern only. The per-step counts ("SD3.5 L 24/28 steps") have no per-step surrogate.
- SD3.5 L is null-generic, so its 24/28 shows that a mean-field surrogate also produces the pattern at many steps.
- Synthesis item 1 ("the depth sign sequence at some noise level in every model") drops this qualification. That is the real problem, and Qwen flags it too.
- Fix: carry the surrogate class into the abstract sentence and synthesis item 1. A per-step surrogate run with the analyzer is optional work.

**5. Uncentred spectra and massive activations. Valid; needs a capture. Already listed as a limitation.**
- Only scalars are stored, so no post hoc centring or channel removal is possible.
- One existing piece of evidence: the lens round found that s1/s2 is smaller than a spike-plus-power-law model implies. The top of the spectrum is a plateau of several large directions, not a single dominant spike (`review/lenses/LENS_RESULTS.md`, P1b). This argues against "one or two channels drive everything", but it is not the control.
- Fix: move this from limitations into §3.1 as an explicit scope statement. Settle it with the capture below.

**6. The audio conclusion is overclaimed. Valid (wording).**
- §12 and synthesis item 6 say the depth sign sequence "is specific to image and video data in this corpus". One distilled six-step model cannot show specificity.
- Fix: "was not observed in the one audio model tested; following the preregistered consequence we do not extend it beyond image and video."
- "Strong anisotropy is not a fixed property of denoisers" can stay. One counterexample does refute "fixed property".

**7. Abliteration under "fixed weights". Valid.**
- Abliteration orthogonalises weights. Layers 0–5 are bit-identical, so the edit sits in later layers.
- Fix: relabel as "fixed architecture, targeted weight edit" in §9 and the Fig. 11 caption, or make it its own bullet. The ordering of the hierarchy does not change.

**8. LM RSA on running statistics. Mostly already addressed.**
- The prefill comparison is a one-shot statistic of the prompt matrix, not a nested running one. The token-128/255 comparisons are at matched context length and residualised on log length, which removes the n-clock the reviewer worries about.
- Optional: one clause saying the prefill comparison is unaffected by the running-statistic issue.

## DeepSeek: minor points

| # | Point | Verdict |
|---|---|---|
| 1–4 | Merged tables, "igure 1", duplicated §8 paragraph, "ModeI" | Misreading: extraction artefacts. The tex has one copy of the paragraph and a consistent title. Check the PDF once. |
| 5 | Plateau-onset threshold is arbitrary | **Valid.** Run today (`out/plateau_onset_threshold_sensitivity.txt`): the default rule reproduces 0.78–0.87. A 1.5× threshold gives 0.57–0.81 and 3× gives 0.79–0.93. The plateau window (σ<0.5/0.6/0.7) matters less. The location is a property of the rule, so give the rule's name and this range in a footnote or appendix. |
| 6 | Confusion matrix / per-domain accuracy | Fair, cheap. A supplementary figure from the existing decoder. |
| 7 | TF-IDF details | Valid, one clause: sklearn defaults (no stop-word removal, l2-normalised), distance = 1 − cosine (`review/round6/scripts/r6c_S2_content.py`:45). |
| 8 | Corrected prompt share undefined | Valid, one sentence defining the seed-replication correction. |
| 9 | Qwen 1.5B/3B vs 7B jump | Wording. One sentence saying the cause is untested; depth does not explain it (1.5B and 7B both have 28 layers). |
| 10 | Unit tests | Already exists: `capture_code_r3/tests/` and the repo verify scripts. Say so in Code and data. |
| 11 | Checkpoints and ComfyUI hashes | Valid if you have them. The appendix has the telemetry SHA but no checkpoint file names or ComfyUI commit. |

## Qwen

| Point | Verdict |
|---|---|
| A. Massive activations / centring | Same as DeepSeek 5. Needs a capture. "Reframe as telemetry of massive activations" goes too far, but the control is the right ask. |
| B. Separate distillation from modality using ZIT | **Valid, and already answerable from existing numbers.** Z-Image MLP (including distilled 8-step cond-only Turbo) has median g/g_MP 0.0014–0.0019, 150–200× more anisotropic than ACE MLP (0.30). Distillation and few steps do not produce the near-isotropy. For the sign sequence, ZIT MLP shows +−+− in 90% of bootstraps while ZIT attention has one crossing, so distillation does not remove it everywhere. Worth two sentences in §12. |
| C. Decoding and variance shares on explicit cond branches | Partly already addressed. ZIT is cond-only (0.55), and regime-homogeneous SD3.5 subsets are reported. The Round 3–4 cond captures have about 6 prompts per domain and one seed, so decoding there would be weak. Optional. |
| D. CKA vs tangent | **Not applicable.** The instrument stores per-layer scalars, not activations, so CKA cannot be computed. The paper never claims the tangent beats CKA; §2 only notes that similarity measures disagree. |
| Synthesis vs null-generic SD3.5 L | Valid; same as DeepSeek 4. |
| Define the operating-point shift A | Valid; the Fig. 11 caption says only "log-RMS". |
| Fig. 8c readability, Table 2 caption | Cosmetic; optional. |

## Proposed text changes (none made)

1. §9 and Fig. 11: relabel the abliteration comparison.
2. Abstract sentence and synthesis item 1: give the depth sign sequence with its surrogate class.
3. §12 and synthesis item 6: "not observed in the one audio model".
4. §9 and synthesis: "lineage shares", not "distillation preserves". Optionally define "inherited" once.
5. §12: two sentences using ZIT to separate distillation from modality (Qwen B).
6. §5.2: one sentence on Z-Image text rows and the lineage tangent. §3.1: scope statement on the uncentred spectrum.
7. §4: plateau-onset threshold sensitivity (footnote or appendix row).
8. Methods clauses: TF-IDF, corrected share, operating point A, tests, checkpoints.

## One capture that answers both top concerns

ZIB and ZIT at CFG 1 (cond-only), the 30 frozen prompts, with the node computing three variants per hook:
- all rows, as now;
- image rows only (drop the 64 text rows);
- row-centred, and with the top-k channels by mean |activation| removed.

Then recompute the Base↔Turbo tangent, one unrelated tangent (against the existing SD3.5 M or Mochi cubes), the backbone PC1, and the ZIB MLP sign sequence under each variant. This is a node change of a few lines.
