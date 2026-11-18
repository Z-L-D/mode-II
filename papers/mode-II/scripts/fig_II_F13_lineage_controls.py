"""II-F13  The real distillation lineage under the uncentred-statistic controls.
(a) Post hoc: margin over the best unrelated pair at the five comparison levels of Definition 2 (Round 6 S1 functions,
    unchanged), for the same-weights pair ZIT-25<->ZIT (Round 6 S1, preregistered) and for the real
    Z-Image Base (Z1, cond branch) <-> ZIT (post hoc, review/zib_turbo_mixup_2026-10-18/r12/scripts/posthoc_hierarchy_realbase.py).
(b) Round 11 (preregistered): Base<->Turbo tangent cosine per row-split variant (V0 all rows, V1 image rows, V3 centred,
    V4 top-8 channels removed; V2 caption rows is post hoc), attention and MLP, with the layer-permutation null p95, and the
    largest unrelated value (Base or Turbo vs SD3.5 M, attention).
(c) Round 11 R11-7 (preregistered diagnostic): median share of the uncentred second-moment energy carried by the 32-64
    caption rows of the joint Z-Image sequence, by layer (Base Z2 and Turbo ZITSV, 60 prompts, cond branch).
Sources: review/round6/out/S1_results.json; review/zib_turbo_mixup_2026-10-18/r12/out/posthoc_hierarchy_margins.csv;
         review/peer_review_2026-10-17/r11/out/r11_tangents.csv, posthoc_text_vs_image_rows.csv; V_Z2_*.pkl and V_ZITSV_*.pkl
         row-split cubes (column 14 = caption-row energy share), restricted to the 60 preregistered prompts.
"""
import json
from m2common import *

FIG = 'II-F13_lineage_controls'
R11 = f'{REV}/peer_review_2026-10-17/r11/out'; R12 = f'{REV}/zib_turbo_mixup_2026-10-18/r12/out'
S1 = json.load(open(f'{R6}/S1_results.json'))
PM = pd.read_csv(f'{R12}/posthoc_hierarchy_margins.csv').set_index('pair')
LEV = ['L0_D1', 'L0_D2', 'L1_field', 'L2_shape', 'L3_tangent']; LAB = ['envelope $g$', 'envelope $\\log g$', 'field', 'shape', 'tangent']
rows = []
for k, l in zip(LEV, LAB):
    rows.append(dict(panel='a', series='same weights ZIT-25<->ZIT (R6 S1, preregistered)', x=l, y=S1['margin'][k], source='review/round6/out/S1_results.json margin'))
    rows.append(dict(panel='a', series='real lineage Z1<->ZIT (post hoc)', x=l, y=PM.loc['Z1-ZIT', k], source='r12/out/posthoc_hierarchy_margins.csv'))
T = pd.read_csv(f'{R11}/r11_tangents.csv'); PT = pd.read_csv(f'{R11}/posthoc_text_vs_image_rows.csv').set_index(['var', 'sub'])
VAR = [('V0', 'all rows'), ('V1', 'image rows'), ('V2', 'caption rows*'), ('V3', 'centred'), ('V4', 'top-8 removed')]
for v, vl in VAR:
    for sub in ('attn', 'mlp'):
        if v == 'V2':
            r = PT.loc[('V2_caption', sub)]; tg, p95, src = r.tangent, r.null_p95, 'r11/out/posthoc_text_vs_image_rows.csv (post hoc)'
        else:
            r = T[(T.a == 'Base') & (T.b == 'Turbo') & (T['sub'] == sub) & (T['var'] == v)].iloc[0]; tg, p95, src = r.tangent, r.null_p95, 'r11/out/r11_tangents.csv'
        rows.append(dict(panel='b', series=f'Base<->Turbo {sub}', x=vl, y=tg, null_p95=p95, source=src))
    if v != 'V2':
        u = T[(T.b == 'SD35M') & (T['var'] == v)].tangent.max()
        rows.append(dict(panel='b', series='max unrelated (vs SD3.5 M, attn)', x=vl, y=u, source='r11/out/r11_tangents.csv'))
P60 = [l.strip() for l in open(f'{REV}/peer_review_2026-10-17/capture/prompts_r3_cond60.txt') if l.strip()]
for arm, cname in (('Base', 'Z2'), ('Turbo', 'ZITSV')):
    for sub in ('attn', 'mlp'):
        V = pickle.load(open(os.path.join(CUBES, f'V_{cname}_{sub}.pkl'), 'rb'))
        keep = V['meta'].pt.astype(str).str.strip().isin(P60).values; W = V['V'][keep][..., 14]
        med = np.median(W, (0, 1))
        for l, m in enumerate(med):
            rows.append(dict(panel='c', series=f'{arm} {sub}', x=l, y=float(m), source=f'V_{cname}_{sub}.pkl column 14, median over gens and steps (60 prereg prompts)'))
D = pd.DataFrame(rows)

fig = plt.figure(figsize=(TEXTWIDTH, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.62)
ax = fig.add_subplot(gs[0]); a = D[D.panel == 'a']; x = np.arange(len(LEV))
for (s, col, mk) in [('same weights ZIT-25<->ZIT (R6 S1, preregistered)', GRAY, 's'), ('real lineage Z1<->ZIT (post hoc)', C[0], 'o')]:
    d = a[a.series == s]; ax.plot(x, d.y, marker=mk, color=col, lw=1.1, ms=4, label='same weights (ZIT-25↔ZIT)' if 'same' in s else 'Base↔Turbo lineage')
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(LAB, rotation=55, ha='right', rotation_mode='anchor', fontsize=6.4)
ax.set_ylabel('margin over best unrelated pair'); ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.62), fontsize=6.0, borderaxespad=0.1)
panel_label(ax, 'a')

ax = fig.add_subplot(gs[1]); b = D[D.panel == 'b']; xv = np.arange(len(VAR))
for sub, mk, fc, dx in (('attn', 'o', C[0], -0.12), ('mlp', 's', 'white', 0.12)):
    d = b[b.series == f'Base<->Turbo {sub}'].reset_index(drop=True)
    ax.plot(xv + dx, d.y, mk, ms=4.5, mfc=fc, mec=C[0], ls='none', label=f'Base↔Turbo {"attn" if sub == "attn" else "MLP"}')
    for j, r in d.iterrows(): ax.plot([j + dx - 0.13, j + dx + 0.13], [r.null_p95] * 2, color=INK2, lw=1.6)
u = b[b.series.str.startswith('max unrelated')]; xu = [ [vl for _, vl in VAR].index(t) for t in u.x]
ax.plot(xu, u.y, 'x', ms=4.5, color=INK2, ls='none', label='max unrelated (attn)')
ax.plot([], [], color=INK2, lw=1.6, label='null p95')
ax.axhline(0, color=INK2, lw=0.5); ax.set_ylim(-0.3, 1.05)
ax.set_xticks(xv); ax.set_xticklabels([vl for _, vl in VAR], rotation=55, ha='right', rotation_mode='anchor', fontsize=6.4)
ax.set_ylabel('tangent cosine'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.36), fontsize=5.8, borderaxespad=0.1, ncol=2, columnspacing=0.5, handletextpad=0.2)
panel_label(ax, 'b')

ax = fig.add_subplot(gs[2]); c = D[D.panel == 'c']
for s, col, ls in (('Base attn', C[1], '-'), ('Base mlp', C[1], (0, (3, 1.5))), ('Turbo attn', C[0], '-'), ('Turbo mlp', C[0], (0, (3, 1.5)))):
    d = c[c.series == s]; ax.plot(d.x, d.y * 100, color=col, ls=ls, lw=1.2, label=s.replace('mlp', 'MLP'))
ax.set_xlabel('Z-Image layer'); ax.set_ylabel('caption-row energy share (%)', labelpad=1); ax.set_ylim(0, 135); ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(loc='upper left', fontsize=5.8, borderaxespad=0.1, handlelength=1.6)
panel_label(ax, 'c')
write_data(FIG, [D])
save(fig, FIG)
print(FIG, 'ok')
