#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE=Path('/mnt/data/r4_r5_audit_v016')
D=pd.read_csv(BASE/'results/R1f_cfg_spread_ratio_cells.csv')
# prompt-level distribution at each step
G=D.groupby(['step','sigma'])['sd_ratio_cfg45_over_cfg1']
Q=G.quantile([.25,.5,.75]).unstack().reset_index()
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.plot(Q['sigma'],Q[0.5],marker='o',markersize=3.5,linewidth=1.6,label='Median across prompts')
ax.fill_between(Q['sigma'],Q[0.25],Q[0.75],alpha=.18,label='Interquartile range')
ax.axhline(.7,linestyle='--',linewidth=1.1,label='Preregistered threshold')
ax.axhline(1.0,linestyle=':',linewidth=1.0,label='No spread change')
ax.set_xlabel(r'Noise level $\sigma$')
ax.set_ylabel('Across-seed spread ratio\nCFG 4.5 cond / CFG 1 cond')
ax.set_title('Guidance contracts conditional-branch seed spread late in denoising')
ax.invert_xaxis()
ax.legend(frameon=False,fontsize=8.5,ncol=2,loc='best')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['svg','pdf','png']:
    fig.savefig(BASE/f'figures/F01_guidance_seed_contraction.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
