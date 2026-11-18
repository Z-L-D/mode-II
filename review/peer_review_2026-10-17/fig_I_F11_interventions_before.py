"""I-F11  Intervention hierarchy (register #30 USE WITH WORDING; outline section 8).
(a) Normalized depth-shape deformation 1 - r for fixed-weight, distillation and capacity-scaling comparisons
    (ledger intervention table). SD3.5 resolution rows: single prompt, CFG 1, legacy EMAG-schema instrument (hollow).
(b) Local-dynamics alignment for the comparisons where it was measured: matched-sigma depth-derivative (tangent) cosine
    (R2 E10) for diffusion; prefill depth-tangent cosine for the Mistral abliteration (R9 B3, preregistered); median
    depth-derivative r at matched context length for Qwen with prompt-bootstrap 5-95% bars and the layer-permutation
    null p95 (R9 B2, preregistered; the point estimates reproduce R2 E7c exactly).
(c) Shape deformation against absolute operating-point displacement (A log-RMS shift) for the fixed-weight comparisons
    (ledger "Approximate separation of manifold, operating point and clock" table).
Sources: papers/_sources/ledger_v020_text.txt (intervention-hierarchy table; M/A/tau table); review/round2/out/E10_tangent.csv,
E7_qwen_pairs.csv; review/r9_salvage/out/R9_B_results.json
"""
import io, json
from common_I import *

TAB = """key,comparison,klass,shape_r,one_minus_r,clock_control,A_log_rms,legacy_instrument
mistral,Mistral Instruct↔ablit,fixed weights,0.9998,0.0002,same token clock,0.0013,0
mochi,Mochi CFG 1↔4.5,fixed weights,0.9965,0.0035,same sigma,0.0559,0
res512_768,SD3.5 M res 512↔768,fixed weights,0.9960,0.0040,same sigma,0.1398,1
res768_1024,SD3.5 M res 768↔1024,fixed weights,0.9939,0.0061,same sigma,0.1557,1
res512_1024,SD3.5 M res 512↔1024,fixed weights,0.9891,0.0109,same sigma,0.2215,1
zitzib,ZIB↔ZIT,distillation,0.9730,0.0270,"25→8 steps, equal σ",,0
qwen15_3,Qwen 1.5B↔3B,scaling,0.6280,0.3720,same seq_len,,0
qwen15_7,Qwen 1.5B↔7B,scaling,0.5560,0.4440,same seq_len,,0
qwen3_7,Qwen 3B↔7B,scaling,0.4510,0.5490,same seq_len,,0
sd35ml,SD3.5 M↔L,scaling,0.2330,0.7670,matched sigma,,0"""
ledger_has('Mistral ablation fixed weights/checkpoint ablation 0.9998 0.0002 same token clock',
           'SD3.5 resolution 512↔1024 fixed weights/input token load 0.9891 0.0109 same sigma',
           'ZIB↔ZIT distillation + step compression 0.9730 0.0270', 'Qwen 3↔7B capacity scaling 0.4510 0.5490 same seq_len',
           'SD3.5 M↔L capacity/architecture/training + CFG 0.2330 0.7670 matched sigma',
           'Mochi CFG all 0.9965 0.0035 0.0559 10 SD3.5 resolution 512_1024 0.9891 0.0109 0.2215 100 SD3.5 resolution 512_768 0.9960 0.0040 0.1398 100 SD3.5 resolution 768_1024 0.9939 0.0061 0.1557 100 Mistral ablation all 0.9998 0.0002 0.0013 256')
D = pd.read_csv(io.StringIO(TAB))
D['source'] = 'ledger_v020_text.txt: "Intervention hierarchy" table (shape r, 1-r); "Approximate separation of manifold, operating point and clock" table (A log-RMS)'
E10 = pd.read_csv(f'{R2}/E10_tangent.csv').set_index(['a', 'b']); QP = pd.read_csv(f'{R2}/E7_qwen_pairs.csv')
QP = QP[QP.obs == 'g_micro'].set_index('pair')
R9B = json.load(open(f'{REVIEW}/r9_salvage/out/R9_B_results.json'))
B2, B3 = R9B['R9-B2']['pairs'], R9B['R9-B3'][1]
dyn = {'mistral': (B3['tangent_cos'], np.nan, 'prefill depth-tangent cosine, R9 B3 (preregistered)'),
       'mochi': (E10.loc[('MOCHI-attn', 'MOCHI1-attn'), 'tangent_cos'], np.nan, 'tangent cosine, R2 E10'),
       'zitzib': (E10.loc[('ZIT-attn', 'ZIB-attn'), 'tangent_cos'], E10.loc[('ZIT-mlp', 'ZIB-mlp'), 'tangent_cos'], 'tangent cosine attn / MLP, R2 E10'),
       'sd35ml': (E10.loc[('SD35M-attn', 'SD35L-attn'), 'tangent_cos'], np.nan, 'tangent cosine, R2 E10'),
       'qwen15_3': (B2['qwen25_1_5b~qwen25_3b']['r'], np.nan, 'median depth-derivative r at matched n, R9 B2 (R2 E7c reproduced)'),
       'qwen15_7': (B2['qwen25_1_5b~qwen25_7b']['r'], np.nan, 'median depth-derivative r at matched n, R9 B2 (R2 E7c reproduced)'),
       'qwen3_7': (B2['qwen25_3b~qwen25_7b']['r'], np.nan, 'median depth-derivative r at matched n, R9 B2 (R2 E7c reproduced)')}
D['dyn_align'] = D.key.map(lambda k: dyn[k][0] if k in dyn else np.nan)
D['dyn_align_mlp'] = D.key.map(lambda k: dyn[k][1] if k in dyn else np.nan)
QK = {'qwen15_3': 'qwen25_1_5b~qwen25_3b', 'qwen15_7': 'qwen25_1_5b~qwen25_7b', 'qwen3_7': 'qwen25_3b~qwen25_7b'}
for col, fld in [('dyn_boot_p5', 'boot_p5'), ('dyn_boot_p95', 'boot_p95'), ('dyn_null_p95', 'null_p95')]:
    D[col] = D.key.map(lambda k: B2[QK[k]][fld] if k in QK else np.nan)
D['dyn_source'] = D.key.map(lambda k: dyn[k][2] if k in dyn else 'not measured')
write_csv(D, 'I-F11_intervention_hierarchy')

STY = {'fixed weights': ('o', INK), 'distillation': ('D', MODEL_COLOR['ZIT']), 'scaling': ('s', MODEL_COLOR['SD35M'])}
fig = plt.figure(figsize=(TEXTWIDTH, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.75, 0.95], wspace=0.12)
y = np.arange(len(D))


def mark(ax, x, j, r, s=24):
    mk, col = STY[r.klass]
    if r.legacy_instrument:
        ax.scatter(x, j, marker=mk, s=s, facecolor='white', edgecolor=col, lw=0.9, zorder=3)
    else:
        ax.scatter(x, j, marker=mk, s=s, color=col, lw=0, zorder=3)


ax = fig.add_subplot(gs[0])
for j, r in D.iterrows(): mark(ax, r.one_minus_r, j, r)
ax.set_xscale('log'); ax.set_xlim(1e-4, 1.5)
ax.set_yticks(y, D.comparison, fontsize=7); ax.set_ylim(len(D) - 0.5, -0.6)
ax.set_xlabel('shape deformation 1 − r')
for k, (mk, col) in STY.items(): ax.scatter([], [], marker=mk, s=22, color=col, label=k)
ax.scatter([], [], marker='o', s=22, facecolor='white', edgecolor=INK, label='legacy instrument')
ax.legend(loc='lower left', fontsize=6.2, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax, 'a')

ax = fig.add_subplot(gs[1])
for j, r in D.iterrows():
    if np.isfinite(r.dyn_align):
        if np.isfinite(r.dyn_null_p95):
            ax.plot([r.dyn_null_p95] * 2, [j - 0.3, j + 0.3], color=GRAY, lw=1.0, zorder=2)
            ax.plot([r.dyn_boot_p5, r.dyn_boot_p95], [j, j], color=STY[r.klass][1], lw=1.2, zorder=2)
        mark(ax, r.dyn_align, j, r)
        if np.isfinite(r.dyn_align_mlp):
            mk, col = STY[r.klass]; ax.scatter(r.dyn_align_mlp, j + 0.32, marker=mk, s=18, facecolor='white', edgecolor=col, lw=0.9, zorder=3)
            ax.text(r.dyn_align - 0.07, j, 'attn', fontsize=6, color=INK2, ha='right', va='center')
            ax.text(r.dyn_align_mlp - 0.07, j + 0.32, 'MLP', fontsize=6, color=INK2, ha='right', va='center')
    else:
        ax.text(0.0, j, 'not measured', fontsize=6.2, color=GRAY, ha='center', va='center')
ax.axvline(0, color=LIGHT, lw=0.8, zorder=1)
ax.set_xlim(-0.5, 1.15); ax.set_yticks(y, [''] * len(D)); ax.set_ylim(len(D) - 0.5, -0.6)
ax.set_xlabel('derivative alignment')
ax.plot([], [], color=GRAY, lw=1.0, label='null p95'); ax.legend(loc='lower right', fontsize=6.2, handlelength=0.8, borderaxespad=0.1)
panel_label(ax, 'b')

ax = fig.add_subplot(gs[2])
p = ax.get_position(); ax.set_position([p.x0 + 0.07, p.y0, p.width - 0.07, p.height])
F = D[D.klass == 'fixed weights']
for _, r in F.iterrows():
    mark(ax, r.A_log_rms, r.one_minus_r, r, s=28)
OFF = {'mistral': (6, 0), 'mochi': (-6, 0), 'res512_768': (5, -7), 'res768_1024': (6, 3), 'res512_1024': (-4, 7)}
HA = {'mistral': 'left', 'mochi': 'right', 'res512_768': 'left', 'res768_1024': 'left', 'res512_1024': 'right'}
NAME = {'mistral': 'Mistral ablit', 'mochi': 'Mochi CFG', 'res512_768': '512↔768', 'res768_1024': '768↔1024', 'res512_1024': '512↔1024'}
for _, r in F.iterrows():
    ax.annotate(NAME[r.key], (r.A_log_rms, r.one_minus_r), xytext=OFF[r.key], textcoords='offset points', fontsize=6.3, color=INK, ha=HA[r.key], va='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(5e-4, 1.0); ax.set_ylim(1e-4, 0.05); ax.set_xticks([1e-3, 1e-2, 1e-1])
ax.set_xlabel('operating-point shift A'); ax.set_ylabel('shape deformation 1 − r', labelpad=1)
panel_label(ax, 'c')
save(fig, 'I-F11_intervention_hierarchy')
print('I-F11 done')
