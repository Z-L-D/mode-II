#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE=Path('/mnt/data/r4_r5_audit_v016')
d=pd.read_csv(BASE/'results/CURRENT_RESOLUTION_TOPOLOGY_BOOTSTRAP.csv')
# Order sequences by crossing count for readability.
seq_order=['+-','+-+-','+-+-+-']
markers=['o','s','^']
fig,ax=plt.subplots(figsize=(6.6,4.2))
for seq,m in zip(seq_order,markers):
    q=d[d.sequence==seq].copy()
    if q.empty: continue
    q['resolution_num']=q.resolution.astype(int)
    q=q.sort_values('resolution_num')
    ax.plot(q.resolution_num,q.bootstrap_fraction,marker=m,markersize=6,linewidth=1.5,label=f'{seq} ({len(seq)-1} crossings)')
ax.set_xlabel('Resolution')
ax.set_ylabel('Prompt-bootstrap fraction')
ax.set_title('Passive beta topology depends on resolution')
ax.set_xticks([512,768,1024])
ax.set_ylim(-.02,1.02)
ax.legend(frameon=False,fontsize=8.5)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['svg','pdf','png']:
    fig.savefig(BASE/f'figures/F04_resolution_topology_bootstrap.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
