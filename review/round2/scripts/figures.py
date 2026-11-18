import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 2 figures from saved outputs.
import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O=f'{_COSPAFS_ROOT}/review/round2/out/'; F=f'{_COSPAFS_ROOT}/review/round2/fig/'
E1=pickle.load(open(O+'E1_curves.pkl','rb')); E5=pickle.load(open(O+'E5_curves.pkl','rb')); S=pd.read_csv(O+'E1_sync.csv')
fig,ax=plt.subplots(2,7,figsize=(24,6.5))
for i,(k,v) in enumerate(E1.items()):
    T=len(v['R']); t=np.arange(T)/(T-1); r=S.set_index('ch').loc[k]
    ax[0,i].plot(t,v['R'],'k-',label='R (z-scored)'); ax[0,i].axhline(r.null_a1_p95,color='r',ls='--',label='warp null p95'); ax[0,i].set_title(f"{k}\nR_max p={r.p_a1:.3f}",fontsize=9)
    tk=np.arange(T-1)/(T-2); ax[1,i].errorbar(tk,v['K'],v['Kse'],color='C0',label='K(t)'); a2=ax[1,i].twinx(); a2.errorbar(tk,v['C'],v['Cse'],color='C1',label='C(t)')
    ax[1,i].set_xlabel('normalized step'); 
ax[0,0].legend(fontsize=7); ax[1,0].set_ylabel('K (blue) / C (orange, right)')
fig.suptitle('E1: z-scored sync R sits at the monotone-warp null; centering-free K and C peak at endpoints, never mid-run'); fig.tight_layout(); fig.savefig(F+'E1_sync.png',dpi=110)
fig,ax=plt.subplots(1,7,figsize=(24,3.6)); D5=pd.read_csv(O+'E5_seed_fate.csv').set_index('ch')
for i,(k,v) in enumerate(E5.items()):
    s=v['sigma']; ax[i].plot(s,v['obs'],'k-o',ms=3,label='observed'); ax[i].plot(s,v['sur'],'r--',label='diffusive surrogate'); ax[i].invert_xaxis(); ax[i].axhline(.5,color='gray',lw=.5)
    ax[i].set_title(f"{k}\nσ(R²=0.5)={D5.loc[k,'sigma50']:.2f}",fontsize=9); ax[i].set_xlabel('σ')
ax[0].set_ylabel('seed-fate R²'); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(F+'E5_seed_fate.png',dpi=110)
O3=pd.read_csv(O+'E3_plateau_onset.csv'); fig,ax=plt.subplots(1,2,figsize=(9,3.5))
ax[0].scatter(O3.onset_stepfrac,range(len(O3))); ax[0].set_yticks(range(len(O3))); ax[0].set_yticklabels(O3.ch,fontsize=8); ax[0].set_xlim(0,1); ax[0].set_xlabel('plateau onset, step fraction')
ax[1].scatter(O3.onset_sigma,range(len(O3))); ax[1].set_yticks([]); ax[1].set_xlim(0,1); ax[1].set_xlabel('plateau onset, σ'); fig.suptitle('E3: plateau onset clusters at σ≈0.78–0.87, not at a step fraction'); fig.tight_layout(); fig.savefig(F+'E3_plateau_onset.png',dpi=110)
E13=pd.read_csv(O+'E13_spread_by_step.csv'); fig,ax=plt.subplots(figsize=(6,3.5)); ax.plot(E13.step,E13.ratio,'k-'); ax.axhline(1/np.sqrt(2),color='r',ls='--',label='branch-averaging bound 1/√2'); ax.set_xlabel('step (of 100)'); ax.set_ylabel('spread ratio CFG5.5 / CFG1.0'); ax.legend(); ax.set_title('E13: spread reduction exceeds branch averaging after the first ~10 steps',fontsize=9); fig.tight_layout(); fig.savefig(F+'E13_spread_ratio.png',dpi=110)
print('ok')
