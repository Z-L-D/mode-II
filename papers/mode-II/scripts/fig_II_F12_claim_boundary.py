"""II-F12 Universality / falsification hierarchy: the final mathematical claim boundary.
A status matrix, not a plot of new numbers. Each cell records whether a structure survives a change of content, fixed weights
(guidance / capture), model scale, distillation, unrelated architecture, and whether a causal gate is still required.
Every cell carries its evidence and source in data/II-F12_claim_boundary.csv; statuses follow
papers/_sources/R6C_RELEASE_CLAIM_REGISTER.md and review/round6/R6C_GATES.md (v0.19), and the numbers are those of the cited outputs.
Unrelated-architecture cells also carry the SD1.5 U-Net where it was measured (post hoc, review/r3_unet/, ledger v0.25-v0.26).
The audio-modality column is ACE-Step 1.5 turbo (review/r3_ace/, ledger v0.27) and base (Round 14, review/ace_base_r14/).
The distillation column uses the real Z-Image Base (Rounds 11-13) and ACE-Step Base (Round 14); comparisons against the
ZIT-25 campaign (archive files labelled ZIB) are same Turbo weights (ZIT-25 <-> ZIT) and sit in the fixed-weights column.
"""
from m2common import *
from matplotlib.patches import Rectangle

FIG = 'II-F12_claim_boundary'
COLS = ['content\n(prompts, themes)', 'fixed weights\n(CFG, capture)', 'model scale\n(SD3.5, Qwen)', 'distillation\n(Base↔Turbo)', 'unrelated\narchitectures', 'audio modality\n(ACE-Step)', 'causal\ngate']
H, S, F, U, B = 'holds', 'scoped', 'fails', 'not tested', 'gate required'
R6o, R2o, ACEo = 'review/round6/out/', 'review/round2/out/', 'review/r3_ace/out/'
R12o, R14o, R11o = 'review/zib_turbo_mixup_2026-10-18/r12/out/', 'review/ace_base_r14/out/', 'review/peer_review_2026-10-17/r11/out/'
CELLS = [
    ('scalar envelope\n(attn $g$ rises, falls late)', [
        (H, '92–100%', R6o + 'posthoc_R6C.json P1 (prompt bootstraps; ZIT 92%)'),
        (H, '++− all arms', R6o + 'S4_topology.csv (R3/R4 ZIT-25 attn CFG 1, CFG 4 cond/uncond)'),
        (H, 'M, L ++−', R6o + 'posthoc_R6C.json P1'),
        (H, 'Base ++−', R14o + 'posthoc_backbone_clock.csv Z1 attn ++− (post hoc); ACE Base attn −++ (R14 post hoc)'),
        (S, 'r .73–.98; U-Net +++', R6o + 'S1_hierarchy.csv L0_D1 (S1b); recurs but cannot identify lineage (register T2). '
            'SD1.5 U-Net attn (post hoc, cond, t clock): +++ in 100% of bootstraps, no late fall (review/r3_unet/out/P7_clock_topology.csv)'),
        (F, 'T −+− / B −++', ACEo + 'P7_clock_topology.csv ACE-attn: −+− in 100% of 1000 prompt bootstraps, max at sigma=1, 6 steps (post hoc, cond-only); ' + R14o + 'posthoc_backbone_clock.csv ACEB attn −++'),
        (U, 'descriptive', 'register #23 (post hoc, descriptive)')]),
    ('content geometry\n(prompt RSA)', [
        (H, '29/30 within-theme', R6o + 'P2_rsa_robustness.csv / R6C_RESULTS P2 (within-theme p<0.05 in 29 of 30 cells)'),
        (H, '0.56–0.62', R6o + 'S2_rsa.csv ZIT-ZIB = ZIT-25 vs ZIT, same Turbo weights'),
        (H, '0.31–0.43', R6o + 'S2_rsa.csv SD35M-SD35L'),
        (U, 'not on real Base', 'not recomputed on the real Base captures'),
        (H, '.13–.43; U-Net .11–.35', R6o + 'S2_rsa.csv unrelated pairs, all p 0.0005 (register #19); SD1.5 U-Net attn vs each DiT, 116 prompts, '
            'all p 0.0005 (post hoc, review/r3_unet/out/P6_rsa.csv)'),
        (U, 'other prompts', 'ACE-Step used 29 music prompts; no prompts shared with the image archive'),
        (U, '—', 'no intervention on content geometry')]),
    ('tangent field\n(depth derivative)', [
        (H, 'R² 0.905–0.956', 'ledger D04 (held-out-theme vector-field R² within model)'),
        (H, '0.94–0.999', R2o + 'E10_tangent.csv Mochi CFG 1↔4.5 0.992, ZIT-25↔ZIT 0.943/0.968; review/round5/out/Q_results.json Q2a-c; ' + R12o + 'r12_tangents.csv Z3↔ZIT 0.990/0.992'),
        (F, '−0.25', R6o + 'S1_hierarchy.csv SD35M-SD35L L3_tangent'),
        (S, 'middle tier .71/.72', R12o + 'r12_tangents.csv Z1↔ZIT 0.707/0.719 (R12-1..5 preregistered); ' + R14o + 'r14_tangents.csv ACE Base↔Turbo 0.850/0.572 (R12-6)'),
        (F, '≤ 0.28', R6o + 'S1_hierarchy.csv unrelated L3_tangent ≤0.24, inside permutation null (E10); ' + R12o + 'r12_unrelated_attn.json ≤0.21; ' + R14o + 'posthoc_ace_vs_image_tangents.csv ≤0.27'),
        (S, 'ACE tier .85/.57', R14o + 'r14_tangents.csv: the lineage tier recurs inside audio (R12-6); ACE vs image models −0.11 to 0.27 (post hoc)'),
        (U, 'G1 response', 'R6C_GATES.md G1: geometry resolved; response/Jacobian not measured')]),
    ('σ-axis MLP + − +\n(Z-Image)', [
        (H, '100%', R6o + 'S4_topology.csv frac_pmp (S4a)'),
        (H, 'CFG 1/4, clean', R6o + 'S4_topology.csv (S4b–d; contraction −49.7% vs −50.5%)'),
        (U, 'no MLP hooks', 'R6C_GATES.md "MLP hooks on other architectures" (Mode-III)'),
        (F, 'Base +++', R11o + 'r11_clock.csv Base MLP +++ in every variant (R11-6 refuted); ' + R14o + 'posthoc_backbone_clock.csv Z1 mlp +++'),
        (F, 'U-Net MLP +++', R6o + 'S4_results.json S4e refuted; MLP not captured in Mochi/SD3.5. SD1.5 U-Net MLP (post hoc, cond): +++ in 100% '
            'of bootstraps on t, sigma_VP and resolution-shifted clocks, no contraction (review/r3_unet/out/P7_clock_topology.csv)'),
        (F, 'T −+− / B ++−', ACEo + 'P7_clock_topology.csv ACE-mlp: −+− in 100% of bootstraps, no +−+ (post hoc); ' + R14o + 'posthoc_backbone_clock.csv ACEB mlp ++−'),
        (U, '—', 'descriptive')]),
    ('layer-axis $\\beta_{\\rm layer}$\n+ − + − (post hoc)', [
        (H, 'SD3.5 M, 10 dom.', R2o + 'E12_beta_topology.csv (10 domains, +−+−, bootstrap 0.635–1.0)'),
        (S, 'CFG1 74%; CFG4 4%', R6o + 'A2_beta_layer_bootstrap.csv ZIT-25 R4 cond MLP'),
        (S, 'L null-generic', R6o + 'A2 (M, L 100%) + A4_beta_layer_surrogates.csv (SD3.5 L reproduced by A and B)'),
        (F, 'Base 1 crossing', R12o + 'r12_8_crossings.csv Z1 attn modal 1 (0.50; 3 in 0.11), MLP 1 in 1.00 (R12-8); ACE Base 1 (R12-7)'),
        (S, 'per σ, all 4', 'R6C_ANALYZER_REPLAY.md A3; A4_beta_layer_per_step.csv'),
        (F, '1 crossing, both', ACEo + 'B_scores.csv B6 refuted: attn +− at g*=0.090, 1 crossing in 99/100 prompt bootstraps; ' + R14o + 'r14_B6_boot.csv base 1 crossing in 100/100 (R12-7: modality rule; Z-Image Base also 1)'),
        (U, '—', 'no intervention')]),
    ('local linear modes\n(modulus, complex pairs)', [
        (S, 'CV ≈2.2%; cx 0–1', 'ledger card "Complex local modes are regime-specific" (script 14)'),
        (S, 'MLP .002; attn .26', 'ledger card script 12 (p=0.00196 MLP, p=0.259 attention); the 25->8-step registration is within Turbo (ZIT-25 -> ZIT)'),
        (U, '—', 'not tested'),
        (U, 'not on real Base', 'not recomputed on the real Base captures'),
        (F, 'complex pairs < null', 'Mode-I RC23D: complex pairs 17% of cells vs 67% under layer permutation; modulus above null in 95% = smoothness'),
        (U, '—', 'not tested'),
        (B, 'response / Jacobian', 'ledger "Next gate — rotational current versus representation-space rotation" (open)')]),
    ('microscopic field $F$\n(universal)', [
        (H, 'R² 0.894–0.965', 'ledger D01 (prompt-held), D04'),
        (H, 'R² 0.70–0.89', 'ledger D03 control (ZIT-25↔ZIT, same weights); D15 residual 0.437 vs ceiling 0.579'),
        (F, 'Qwen sizes −0.06', 'ledger D13 (LLM population derivative, 3B↔7B −0.064; 1.5B↔7B −0.062)'),
        (U, 'not on real Base', 'field transfer not refitted on the real Base captures'),
        (F, '≈0 / −0.256', 'ledger D03 (approx. zero or negative), D15 (−0.256); register T5 DO NOT USE'),
        (S, 'B4: T yes, B no', 'not tested across modality; B4 within ACE: turbo dR2 0.60/0.77 (confirmed), base 0.254/0.671 (attn below 0.30, refuted; ' + R14o + 'r14_B_scores.csv)'),
        (U, '—', 'not claimed')]),
    ('separatrix /\ncommitment', [
        (U, 'gradual onset', R6o + 'S3_estimators.csv (relative onset 0.84–0.95; no sharp step)'),
        (U, '—', ''), (U, '—', ''), (U, '—', ''), (U, '—', ''),
        (U, '—', ''),
        (B, 'G3 committor', 'R6C_GATES.md G3: clone continuations required before any commitment wording')]),
]
STYLE = {H: dict(fc=C[0], ec=C[0]), S: dict(fc='half', ec=C[0]), F: dict(fc='x', ec=C[1]), U: dict(fc='none', ec=LIGHT), B: dict(fc='sq', ec=INK2)}

nr, nc = len(CELLS), len(COLS)
import textwrap
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 5.6))
ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr + 0.35, -1.05); ax.axis('off')
for j, c in enumerate(COLS):
    ax.text(j, -0.62, c, ha='center', va='center', fontsize=6.0, color=INK)
r_ = 0.13
for i, (row, cells) in enumerate(CELLS):
    ax.text(-0.62, i, row, ha='right', va='center', fontsize=7.0, color=INK)
    if i % 2 == 0:
        ax.add_patch(Rectangle((-0.5, i - 0.5), nc, 1, color='#f6f5f2', lw=0, zorder=0))
    for j, (st, txt, src) in enumerate(cells):
        cx, cy = j, i - 0.16
        stl = STYLE[st]
        if st == H:
            ax.plot(cx, cy, 'o', ms=7, color=stl['fc'])
        elif st == S:
            ax.plot(cx, cy, 'o', ms=7, mfc='white', mec=stl['ec'], mew=2.0); ax.plot(cx, cy, 'o', ms=2.6, color=stl['ec'])
        elif st == F:
            ax.plot(cx, cy, 'x', ms=6.5, mew=1.6, color=stl['ec'])
        elif st == B:
            ax.plot(cx, cy, 's', ms=6.5, mfc='white', mec=stl['ec'], mew=1.2)
        else:
            ax.plot(cx, cy, 'o', ms=6.5, mfc='white', mec=GRAY, mew=0.8)
        ax.text(cx, i + 0.06, textwrap.fill(txt, 15), ha='center', va='top', fontsize=5.9, color=INK2, linespacing=1.0)
leg = [(H, 'holds'), (S, 'holds with scope'), (F, 'fails'), (U, 'not tested / not applicable'), (B, 'causal gate required')]
XL = [-0.35, 0.5, 1.85, 2.65, 4.75]
for k, (st, t_) in enumerate(leg):
    xx = XL[k]; yy = nr + 0.05
    if st == H: ax.plot(xx, yy, 'o', ms=6, color=C[0])
    elif st == S: ax.plot(xx, yy, 'o', ms=6, mfc='white', mec=C[0], mew=1.8); ax.plot(xx, yy, 'o', ms=2.2, color=C[0])
    elif st == F: ax.plot(xx, yy, 'x', ms=5.5, mew=1.5, color=C[1])
    elif st == B: ax.plot(xx, yy, 's', ms=5.5, mfc='white', mec=INK2, mew=1.1)
    else: ax.plot(xx, yy, 'o', ms=5.5, mfc='white', mec=GRAY, mew=0.8)
    ax.text(xx + 0.1, yy, t_, va='center', ha='left', fontsize=6.8, color=INK)

rows = []
for i, (row, cells) in enumerate(CELLS):
    for j, (st, txt, src) in enumerate(cells):
        rows.append(dict(panel='matrix', series=row.replace('\n', ' '), x=COLS[j].replace('\n', ' '), status=st, label=txt,
                         source=src or 'not tested'))
write_data(FIG, [pd.DataFrame(rows)])
save(fig, FIG)
print(FIG, 'ok')
