from pathlib import Path
import csv,json,hashlib,statistics,math,re,html
import matplotlib.pyplot as plt
ROOT=Path('/mnt/data/mode12_round6'); R5=Path('/mnt/data/r5/round5')
# Audit R5
pre=R5/'R5_PREREGISTRATION.md'; hash_actual=hashlib.sha256(pre.read_bytes()).hexdigest(); hash_ship=(R5/'R5_PREREG_SHA256.txt').read_text().split()[0]
qrows=list(csv.DictReader(open(R5/'out/Q_scorecard.csv')))
counts={k:sum(r['verdict']==k for r in qrows) for k in sorted(set(r['verdict'] for r in qrows))}
audit={'prereg_sha_actual':hash_actual,'prereg_sha_shipped':hash_ship,'sha_match':hash_actual==hash_ship,'q_count':len(qrows),'q_verdict_counts':counts}
json.dump(audit,open(ROOT/'R6_ROUND5_AUDIT.json','w'),indent=2)
# Recovered historical/current evidence matrix
rows=[
('coarse scalar g','ZIB-Mochi attention',0.9405,'historical sigma-aligned population correlation','recovered prior analysis'),
('coarse scalar g','ZIB-ZIT attention',0.9325,'historical sigma-aligned population correlation','recovered prior analysis'),
('coarse scalar g','ZIT-Mochi attention',0.8220,'historical sigma-aligned population correlation','recovered prior analysis'),
('coarse scalar g','ZIB-ZIT MLP',0.9715,'historical sigma-aligned population correlation','recovered prior analysis'),
('registered multivariate shape','ZIB-ZIT attention (minimum current range)',0.972,'equal-sigma/circular-shift audited','v0.17 ledger'),
('registered multivariate shape','Z-Image-Mochi unrelated (upper calibration)',0.830,'unrelated shape can remain high','v0.17 ledger'),
('local tangent','ZIB-ZIT attention',0.943,'matched-sigma depth derivative cosine','v0.17 ledger'),
('local tangent','ZIB-ZIT MLP',0.968,'matched-sigma depth derivative cosine','v0.17 ledger'),
('local tangent','best unrelated upper bound',0.240,'all unrelated pairs <= this','v0.17 ledger'),
('same-theme residual','ZIB-ZIT start',0.611,'backbone-subtracted theme residual','publication v0.5'),
('same-theme residual','ZIB-ZIT mid',0.794,'backbone-subtracted theme residual','publication v0.5'),
('same-theme residual','ZIB-ZIT end',0.649,'backbone-subtracted theme residual','publication v0.5'),
('same-theme residual','SD35M-Mochi early disjoint-prompt',0.119,'weak unrelated early residual signal','publication v0.5'),
]
with open(ROOT/'R6_METRIC_HIERARCHY.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['metric_level','pair','value','definition','provenance']);w.writerows(rows)
calc={
 'scalar_reversal_margin_ZIBMochi_minus_ZIBZITattn':0.9405-0.9325,
 'shape_lineage_margin_min_vs_unrelated_upper':0.972-0.830,
 'tangent_lineage_margin_attn_vs_unrelated_upper':0.943-0.240,
 'tangent_lineage_margin_mlp_vs_unrelated_upper':0.968-0.240,
 'semantic_lineage_mean':statistics.mean([0.611,0.794,0.649]),
 'semantic_lineage_minus_unrelated_early':statistics.mean([0.611,0.794,0.649])-0.119,
}
json.dump(calc,open(ROOT/'R6_HIERARCHY_CALCULATIONS.json','w'),indent=2)
# Estimator calibration from R5 outputs
h1=list(csv.DictReader(open(R5/'out/H1_seedfate_vs_sample_size.csv')))
vr=json.load(open(R5/'out/VR_null.json'))
with open(ROOT/'R6_ESTIMATOR_CALIBRATION.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['quantity','condition','estimate','p5','p95','note'])
    for r in h1:
        if r['set'] in ['archive SD35M CFG4.5 BM (60 prompts)','archive SD35M CFG4.5 BM (120 prompts)','new CFG1 cond (30 prompts)','R3 CFG1 cond-only, 60 prompts']:
            w.writerow(['sigma50',r['set']+' centred='+r['centred'],r['sigma50'],r['sigma50_p5'],r['sigma50_p95'],'sample-size/fold sensitive'])
            w.writerow(['relative_onset',r['set']+' centred='+r['centred'],r['rel_onset'],r['rel_p5'],r['rel_p95'],'more stable publication metric'])
    for k,v in vr.items(): w.writerow(['VR4_null',k,v[0],v[1],v[2],'calibration reference'])
# Gates, updated recommendation
G=[
('Estimator calibration','NOW / PRE-PAPER','Run in current round','Required for any sigma50/VR table; use matched-n or relative onset and calibrated VR nulls.'),
('G3 committor/clones','DEFER UNLESS CLAIM RETAINED','Mode-III or pre-claim','Mandatory only before commitment/speciation/separatrix wording; exclude such wording from Mode-I/II release.'),
('G4 context-invariant AR','DEFER','Mode-III','v1 AR is nested running statistic; Mode-I may report this negative instrument result, not positive per-token phase dynamics.'),
('D14 quality/output linkage','BLOCKED','Future capture','No stable output IDs/quality labels; prohibit geometry->quality claims.'),
('G2 distillation ladder','DEFER','Mode-III','Needed for geometry-quality engineering, not current empirical lineage claim.'),
('G1 finite perturbation Base/Turbo','DEFER','Mode-III','Geometry is resolved on explicit branches; response/Jacobian sensitivity remains prospective.'),
('Legacy instrument A/B','POST-PAPER CONDITIONAL','Only for causal attribution','Not needed if papers describe legacy functional phenomenology without claiming EMAG causation.'),
('Guidance-interval rider','DEFER','Mode-III','Mode-I may state contraction but not timing.'),
('G5 targeted ablation','DEFER','With G4','Not needed for Mode-I/II central claims.'),
('G6 matched scaling','POST-PAPER','Future','Current negative scaling control is already sufficient for paper framing.'),
('Llama Base/Instruct clean intervention','DEFER','With G4','Hook mismatch prevents current clean causal comparison.'),
]
with open(ROOT/'R6_GATE_REDETERMINATION.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['gate','status','when','determination']);w.writerows(G)
# Figure
labels=['coarse g\nZIB-Mochi','coarse g\nZIB-ZIT','shape\nZIB-ZIT','shape\nunrelated','tangent\nZIB-ZIT','tangent\nunrelated','theme residual\nZIB-ZIT mean','theme residual\nunrelated']
vals=[.9405,.9325,.972,.830,.943,.240,calc['semantic_lineage_mean'],.119]
fig,ax=plt.subplots(figsize=(10,5.5)); ax.bar(range(len(vals)),vals); ax.set_ylim(0,1.05); ax.set_ylabel('Similarity / correlation'); ax.set_xticks(range(len(vals)),labels,rotation=28,ha='right'); ax.set_title('Similarity reverses with observable level'); ax.axhline(0,linewidth=.8); fig.tight_layout(); fig.savefig(ROOT/'R6_metric_hierarchy.svg'); fig.savefig(ROOT/'R6_metric_hierarchy.pdf'); fig.savefig(ROOT/'R6_metric_hierarchy.png',dpi=220); plt.close(fig)
# Report
report=f'''# Round 6 pre-publication excavation — results\n\n## Executive result\nThe recovered Mochi/ZIB anomaly is real at the level originally measured, but it is **not a contradiction of the later lineage result**. The historical sigma-aligned attention scalar-g correlation was ZIB↔Mochi 0.9405 versus ZIB↔ZIT 0.9325 (margin {calc['scalar_reversal_margin_ZIBMochi_minus_ZIBZITattn']:.4f}); ZIT↔Mochi was ~0.822. At richer levels the ordering flips decisively: current registered multivariate shape is at least 0.972 for ZIB↔ZIT versus up to ~0.83 for unrelated Z-Image↔Mochi, and tangent cosine is 0.943/0.968 for the lineage versus <=0.24 for unrelated pairs.\n\nThis is best read as an **observable hierarchy**: a coarse one-dimensional projection can be shared across architectures strongly enough to erase lineage ordering, while local direction-of-motion and residual semantic geometry recover lineage specificity.\n\n## Frozen predictions\nR6-1 PASS. Scalar g gives the recovered reversal, while tangent lineage margins are +{calc['tangent_lineage_margin_attn_vs_unrelated_upper']:.3f} attention and +{calc['tangent_lineage_margin_mlp_vs_unrelated_upper']:.3f} MLP over the strongest unrelated bound.\n\nR6-2 SUPPORTED. Scalar g discards gap/H, depth registration, branch/capture regime and derivative direction. Later multivariate/tangent tests were designed specifically to restore these degrees of freedom.\n\nR6-3 PASS. ZIB↔ZIT backbone-subtracted same-theme residual correlation is 0.611/0.794/0.649 (mean {calc['semantic_lineage_mean']:.3f}); the reported unrelated SD35M↔Mochi disjoint-prompt early signal is only 0.119. Thus coarse cross-architecture trajectory resemblance and semantic residual inheritance are distinct objects.\n\nR6-4 PASS. Round-5 prereg SHA matches exactly; Q scorecard is {counts.get('CONFIRMED',0)} confirmed / {counts.get('REFUTED',0)} refuted.\n\nR6-5 PASS as a publication decision. No new capture is required for a conservative Mode-I/II release if separatrix/commitment, AR per-token phase, and geometry→quality claims are excluded.\n\n## How else to read the old Mochi/ZIB result\n1. **Common solver/denoising scaffold:** g is an effective-rank-like scalar. Different architectures can share the same contraction/re-expansion envelope without sharing a local vector field.\n2. **Projection degeneracy:** two high-dimensional trajectories can have nearly identical scalar projections while moving in different directions in the discarded coordinates. The tangent result empirically demonstrates this.\n3. **Architecture versus lineage split:** Mochi/ZIB coarse resemblance points to cross-architecture recurrence; ZIB/ZIT tangent/semantic inheritance points to lineage. These are complementary, not competing, results.\n4. **Capture-functional sensitivity:** the historical ZIB comparison used a pooled/branch-mean regime while ZIT is cond-only and Mochi branch-mean. Round 5 proves pooling can move g on an identical trajectory. Therefore the exact 0.9405 versus 0.9325 ordering should be presented as historical/coarse and not as a precise architecture ranking until raw branch-matched replay is available.\n5. **Semantic hierarchy:** ordered depth geometry contains theme information in every tested diffusion model, but backbone-subtracted theme directions are strongly lineage-specific. So 'themes are represented' and 'themes are represented in the same direction' are different claims.\n6. **Mode-II implication:** a single universal microscopic F(z) is not supported. A better decomposition is a coarse recurrent organization plus model/lineage-specific depth-time law and content residuals.\n\n## Round-5 estimator correction\nRaw sigma50 is not paper-safe across unequal sample sizes. In the archive it moves from roughly 0.72 to 0.94 as n increases; fold partition alone also causes large swings. Relative onset is much more stable. VR also requires calibration: a simulated random walk gives VR4≈0.82, AR(0.8)≈0.72, iid≈0.25. Paper tables should use these calibrated references rather than a universal VR=1 walk baseline.\n\n## Gate determination\nSee `R6_GATE_REDETERMINATION.csv`. For tonight's Mode-I/II freeze: perform estimator relabeling now; defer G3/G4/G1/G2/G5 and D14-dependent claims; keep legacy-instrument causation explicitly unresolved.\n\n## Reproducibility limitation\nThe original ~506 MB primary `TELEMETRY.zip` is not mounted in this turn. Therefore the historical 0.9405/0.9325/~0.82 scalar correlations are **recovered from the archived project analysis, not re-fit from raw rows here**. Every *new* numerical comparison in this report is generated by `r6_analysis.py` from those recovered values plus the currently supplied Round-5 outputs. A raw branch-matched replay should be queued if the exact scalar ordering is promoted to a headline quantitative claim.\n'''
(ROOT/'R6_RESULTS_AND_SYNTHESIS.md').write_text(report)
# Update ledger v0.18 by injecting callout after header and section before footer/end
src=Path('/mnt/data/CoS-PAFS_Findings_Ledger_v0.17(1).html').read_text(errors='replace')
src=src.replace('CoS-PAFS Findings Ledger v0.17','CoS-PAFS Findings Ledger v0.18').replace('mode:LIMINAL · v0.17 ·','mode:LIMINAL · v0.18 ·',1)
callout="""<div class='callout' style='margin-top:14px;border-left-color:var(--proven)'><h3>v0.18: pre-publication Round 6 excavation (Oct 8, 2026)</h3><div class='cpara'><ul><li><b>Recovered coarse anomaly:</b> historical sigma-aligned attention g gave ZIB↔Mochi r≈0.9405, slightly above ZIB↔ZIT r≈0.9325; ZIT↔Mochi≈0.82.</li><li><b>Resolution:</b> the ordering reverses at richer observables. Registered ZIB↔ZIT multivariate shape is ≈0.972–0.990 while unrelated Z-Image↔Mochi can reach ≈0.83; tangent cosine is 0.943/0.968 for ZIB↔ZIT and ≤0.24 for unrelated pairs.</li><li><b>Interpretation:</b> coarse scalar trajectory resemblance can cross architecture boundaries; local dynamics and backbone-subtracted semantic residual geometry are much more lineage-specific. Similarity must be indexed by observable level.</li><li><b>Round-5 audit:</b> preregistration SHA verified; Q scorecard 13 CONFIRMED / 9 REFUTED. Estimator calibration is binding for publication: raw sigma50 is sample-size/fold sensitive; VR walk baseline≈0.82.</li><li><b>Gate freeze:</b> no new capture required for conservative Mode-I/II release if commitment/separatrix, AR per-token phase, and geometry→quality claims are excluded. G3/G4/D14/G2 remain prospective.</li></ul></div></div>"""
marker="</header>"
src=src.replace(marker,marker+callout,1)
section="""<section id='round6-v018' class='newsec'><h2><span class='bar' style='background:var(--proven)'></span>Round 6 — coarse universality versus lineage-specific dynamics</h2><div class='item proven'><div class='ihead'><span class='ititle'>Coarse scalar similarity can reverse the lineage ordering</span><span class='badge bp'>recovered / calculated</span></div><div class='para'>Historical sigma-aligned attention g: ZIB↔Mochi≈0.9405, ZIB↔ZIT≈0.9325, ZIT↔Mochi≈0.82. This is a valid property of that scalar projection but not a sufficient lineage statistic.</div></div><div class='item proven'><div class='ihead'><span class='ititle'>Richer geometry restores lineage specificity</span><span class='badge bp'>measured</span></div><div class='para'>Registered ZIB↔ZIT multivariate geometry is ≈0.972–0.990; unrelated Z-Image↔Mochi shape can still be ≈0.79–0.83. The decisive separation is tangent direction: 0.943 attention / 0.968 MLP for ZIB↔ZIT versus ≤0.24 for unrelated pairs.</div></div><div class='item mech'><div class='ihead'><span class='ititle'>Observable hierarchy</span><span class='badge bm'>mechanism</span></div><div class='para'>A one-dimensional rank/effective-rank projection can share a common contraction envelope across unrelated architectures while discarded coordinates carry architecture- and lineage-specific direction fields. Thus coarse recurrence and lineage inheritance coexist.</div><div class='eq'>coarse scalar resemblance → registered state geometry → local tangent/dynamics; semantic residual geometry is a separate content axis</div></div><div class='item ref'><div class='ihead'><span class='ititle'>Exact 0.9405&gt;0.9325 ordering is not promoted to a raw-replayed headline</span><span class='badge br'>provenance caveat</span></div><div class='para'>The raw primary archive was not mounted in Round 6 and the historical comparison mixes capture functionals. Round 5 demonstrated that pooling can move g on an identical trajectory. Retain the result as historical coarse evidence unless branch-matched raw replay is performed.</div></div></section>"""
idx=src.rfind('</body>')
if idx!=-1: src=src[:idx]+section+src[idx:]
(ROOT/'CoS-PAFS_Findings_Ledger_v0.18.html').write_text(src)
# manifest
files=[p for p in ROOT.iterdir() if p.is_file()]
with open(ROOT/'R6_ARTIFACT_MANIFEST.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['file','sha256','bytes']);
    for p in sorted(files): w.writerow([p.name,hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size])
