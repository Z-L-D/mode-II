# Matched capture study for tonight (replaces the single-prompt 100-step reruns)

**Recommendation:** run the resolution and CFG studies at the campaign settings, not as 100-step single-prompt reruns. Three reasons:
- The single-prompt suite was the biggest caveat on the SD3.5 rebound cards.
- On the 120-prompt campaign the β topology already looked different (Round 2, E12).
- Matching the campaign lets every new row sit next to the archive rows.

The 100-step suite's one unique asset is the region below σ≈0.2. The archive's ZIT and Mochi 100-step runs already cover that, so the legacy suite can go to an appendix.

---

## Shared settings

**Guider:** use the patched Passive Telemetry Guider from `capture_code_r2/nodes.py` with `split_cfg_branches = on`.
- Every row carries `branch` (cond / uncond), `prompt_influence` and `seed`.
- `hook_mode` reads `passive-branch`, so the new instrument is identifiable.

**Prompts:** `matched_prompts_30.tsv`. That is 3 prompts per theme across the 10 themes, taken from the archive in archive order. All 30 prompts are also in the ZIB archive, now that the abstract_conceptual file has been supplied (Oct 5), so ZIB arms can use all 10 themes. Paste the same text into `prompt_text`.

**Seeds:** use the same seeds in every arm, e.g. 1001 and 1002.
- Each CFG pair (SD3.5-M 1.0 vs 4.5, ZIB 1.0 vs 4.0) then runs on identical initial noise, so it doubles as a paired-noise test (G1) at no extra cost.

**Everything else:** sampler, scheduler, text encoders (3TE for SD3.5), resolution and step count exactly as in the archive campaign unless the table says otherwise.

---

## Tier 1 (tonight)

| Arm | Model | Res | Steps | CFG | Gens per theme | Total | Answers |
|---|---|---|---|---|---|---|---|
| 1 | SD3.5-M | 1024 | 40 | 4.5 | 3 prompts × 2 seeds = **6** | 60 | Anchor: per-branch re-capture of the campaign (G0); 1024 point of the resolution card; 4.5 point of the CFG card |
| 2 | SD3.5-M | 1024 | 40 | **1.0** | 3 × 2 = **6** | 60 | CFG card: 1.0 vs 4.5 on paired noise; seed share with no guidance |
| 3 | SD3.5-M | **512** | 40 | 4.5 | 3 × 1 = **3** | 30 | Resolution card |
| 4 | SD3.5-M | **768** | 40 | 4.5 | 3 × 1 = **3** | 30 | Resolution card |
| 5 | ZIB | 1024 | 25 | 4.0 | 3 × 1 = **3** (10 themes; 9 if already running) | 30 (27) | Per-branch ZIB; ZIB↔ZIT on the cond branch |
| 6 | ZIB | 1024 | 25 | **1.0** | 3 × 1 = **3** | 30 (27) | ZIB with the same capture as ZIT, which isolates distillation from guidance |

**Tier 1 total: 234 generations.** Arm 2 at CFG 1.0 skips the uncond pass, so it is about half the cost of the others.

**Order if time runs short:** 1 → 2 → 3/4 → 5/6. Arms 1–2 alone re-establish the SD3.5 CFG card on the campaign instrument. Arms 3–4 add the resolution card.

## Tier 2 (after the papers, or if Tier 1 finishes early)

| Arm | Model | Res | Steps | CFG | Gens per theme | Total | Answers |
|---|---|---|---|---|---|---|---|
| 7 | SD3.5-L | 1024 | 28 | 3.5 | 3 × 1 | 30 | Per-branch SD35L (the domain-dependent regime) |
| 8 | SD3.5-M | 1024 | 40 | 5.5 | 3 × 1 | 30 | Third CFG point, continuous with the ledger's 5.5 |
| 9 | Second seed for arms 3–6 | | | | +3 per theme | ~114 | Seed-share and seed-fate estimates in every arm |
| A/B | Old Diagnostic build vs patched Passive, res-suite settings | 512 | 100 | 1.0 | 3 seeds total | 6 | Whether the legacy rebound files can be cited at all |

---

## What I will run as soon as the CSVs arrive

All of these are already scripted in `review/round2/scripts`.
- E12 β(g) topology and TRI per resolution and CFG arm, on the cond branch.
- E13 eigenpair and spread ratio, now on paired noise.
- E2 regime check: cond vs uncond vs pooled-equivalent.
- E6 variance components.
- E5 seed fate (arms 1–2).
- E3 plateau onset per arm.
- E10 tangent cosine for ZIB(1.0)↔ZIT and SD35M↔SD35L.
- **New:** per-layer prompt_influence(σ), which is the passive conditioning-influence observable.
