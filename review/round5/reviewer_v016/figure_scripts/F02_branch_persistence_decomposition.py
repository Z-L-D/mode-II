#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE=Path('/mnt/data/r4_r5_audit_v016')
# Combine CFG1 cond with guided branch decomposition
s=pd.read_csv(BASE/'results/POSTHOC_R1_seedcentred_summary.csv')
b=pd.read_csv(BASE/'results/POSTHOC_R1_branch_persistence_decomposition.csv')
vr1=float(s[(s.arm=='CFG1_cond')&(s.estimator=='seed-centred')].VR4.iloc[0])
vals={
    'CFG 1 cond':vr1,
    'CFG 4.5 cond':float(b[b.representation=='cond'].VR4_seedcentred.iloc[0]),
    'CFG 4.5 uncond':float(b[b.representation=='uncond'].VR4_seedcentred.iloc[0]),
    'CFG 4.5 branch mean':float(b[b.representation=='branchmean'].VR4_seedcentred.iloc[0]),
}
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.bar(list(vals.keys()),list(vals.values()))
ax.axhline(1.0,linestyle='--',linewidth=1.1,label='VR(4) = 1')
ax.set_ylabel('Variance ratio VR(4)')
ax.set_title('Persistence changes under guidance and branch mixing')
ax.tick_params(axis='x',rotation=18)
ax.legend(frameon=False,fontsize=8.5)
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0,max(vals.values())*1.18)
fig.tight_layout()
for ext in ['svg','pdf','png']:
    fig.savefig(BASE/f'figures/F02_branch_persistence_decomposition.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
