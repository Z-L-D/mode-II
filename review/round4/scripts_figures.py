import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 4 figures (light mode, static PNG for the report and ledger).
import json, pickle, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,f'{_COSPAFS_ROOT}/review/round2/scripts'); from common import load as load_archive
FIG=f'{_COSPAFS_ROOT}/review/round4/fig'; P=json.load(open(f'{_COSPAFS_ROOT}/review/round4/out/P_results.json'))
S1,S2,S3='#2a78d6','#eb6834','#1baf7a'; INK,INK2,GRID,SURF='#0b0b0b','#52514e','#e4e3df','#fcfcfb'
plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6})
def F(n): return pickle.load(open(n,'rb'))
# Fig 1: what the old ZIB<->ZIT gap was made of
fig,ax=plt.subplots(1,2,figsize=(10,3.2))
lab=['ZIB CFG 1 (cond only)','ZIB CFG 4, branch-mean','ZIB CFG 4, pooled (archive)']
off=[P['P2c_offset_ZIB1_ZIT_attn'],P['P2c_offset_ZIB4BM_ZIT_attn'],P['P2c_offset_ZIB4pooled_ZIT_attn']]
tan=[P['ref_tangent_ZIB1_ZIT_attn'],P['P2a_tangent_ZIB4BM_ZIT_attn'],P['ref_tangent_ZIB4pooled_ZIT_attn']]
y=np.arange(3)[::-1]
ax[0].barh(y,off,color=S1,height=0.5); ax[0].set_yticks(y,lab); ax[0].set_xlabel('median |Δ mean log g| vs ZIT (attn)')
for yi,v in zip(y,off): ax[0].text(v+0.006,yi,f'{v:.3f}',va='center',color=INK)
ax[0].set_title('Absolute-g offset from ZIT',loc='left',fontsize=11); ax[0].set_xlim(0,0.43)
ax[1].barh(y,[1-t for t in tan],color=S1,height=0.5); ax[1].set_yticks(y,['']*3); ax[1].set_xlabel('1 − tangent cosine vs ZIT (attn)')
for yi,t in zip(y,tan): ax[1].text(1-t+0.0015,yi,f'cos {t:.3f}',va='center',color=INK)
ax[1].set_title('Depth-geometry gap from ZIT',loc='left',fontsize=11); ax[1].set_xlim(0,0.075)
fig.tight_layout(); fig.savefig(f'{FIG}/R4_fig1_zib_zit_decomposition.png',dpi=160); plt.close()
# Fig 2: legacy vs passive layer-mean g over sigma (SD3.5-M, CFG 1)
dr,mr,_,_=load_archive('SDRES'); d=dr[dr.gid.isin(mr.index[mr.res==1024])]; leg=d.groupby('token_idx').agg(s=('sigma','first'),g=('g_micro','mean'))
c=pd.read_csv(f'{_COSPAFS_ROOT}/data/sd35m_cfg1.0_vs_5.5/sd-3.5-medium-CFG1.0-30.1.csv'); cs=c.groupby('token_idx').agg(s=('sigma','first'),g=('g_micro','mean'))
pa=[F(f'F3_S10_1024_s100{k}_attn.pkl') for k in (1,2)]; X=np.concatenate([p[2] for p in pa]); ps=pa[0][3]
fig,ax=plt.subplots(figsize=(7,3.6))
ax.plot(ps,X[...,0].mean(0).mean(1),color=S1,lw=2,label='passive, 1024 px, 40 steps (R3)')
ax.plot(leg.s,leg.g,color=S2,lw=2,label='legacy schema, res suite 1024 px, 100 steps')
ax.plot(cs.s,cs.g,color=S3,lw=2,label='legacy schema, CFG study 256 px, 100 steps')
ax.set_yscale('log'); ax.invert_xaxis(); ax.set_xlabel('σ (noise level, sampling runs left to right)'); ax.set_ylabel('layer-mean g')
ax.legend(frameon=False,fontsize=9,loc='lower left'); ax.set_title('SD3.5-M at CFG 1: three capture files, two behaviours',loc='left',fontsize=11)
fig.tight_layout(); fig.savefig(f'{FIG}/R4_fig2_legacy_vs_passive_g.png',dpi=160); plt.close()
# Fig 3: seed-fate R2 curves, uncentred vs seed-centred (SD3.5-M CFG 1, R3)
fig,ax=plt.subplots(figsize=(7,3.4))
ax.plot(ps,P['P3_R2curve_uncentred_CFG1'],color=S2,lw=2,label='uncentred (shared seeds 1001/1002)')
ax.plot(ps,P['P3_R2curve_centred_CFG1'],color=S1,lw=2,label='seed-centred inside each CV fold')
ax.axhline(0.5,color=INK2,lw=1,ls='--'); ax.text(0.62,0.44,'R² = 0.5',color=INK2,fontsize=9)
ax.invert_xaxis(); ax.set_xlabel('σ'); ax.set_ylabel('held-out-prompt R² for the final state'); ax.set_ylim(-0.05,1.02)
ax.legend(frameon=False,fontsize=9,loc='lower right'); ax.set_title('Future-state predictability, SD3.5-M 1024, CFG 1',loc='left',fontsize=11)
fig.tight_layout(); fig.savefig(f'{FIG}/R4_fig3_seed_fate_centring.png',dpi=160); plt.close()
# Fig 4: VR(4) by capture regime
H=json.load(open(f'{_COSPAFS_ROOT}/review/round4/out/posthoc_results.json'))
lab=['CFG 1, cond only (R3)','CFG 4.5, pooled (R3)','CFG 4.5, branch-mean (archive, 2 runs/prompt)']
v=[P['P3_VR_uncentred_CFG1']['4'],P['P3_VR_uncentred_CFG45']['4'],H['H3_archive_SD35M_2runs_VR4_median']]
fig,ax=plt.subplots(figsize=(8,2.8)); y=np.arange(3)[::-1]
ax.barh(y,v,color=S1,height=0.5); ax.set_yticks(y,lab); ax.axvline(1,color=INK2,lw=1,ls='--'); ax.text(1.02,2.3,'random walk = 1',color=INK2,fontsize=9)
for yi,x in zip(y,v): ax.text(x+0.03,yi,f'{x:.2f}',va='center')
ax.set_xlabel('variance ratio VR(4) of seed deviations'); ax.set_xlim(0,2.1); fig.suptitle('SD3.5-M 1024: seed-deviation persistence by arm',x=0.02,ha='left',fontsize=11)
fig.tight_layout(); fig.savefig(f'{FIG}/R4_fig4_vr_by_regime.png',dpi=160); plt.close()
print('ok')
