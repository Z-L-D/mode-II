#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE=Path('/mnt/data/r4_r5_audit_v016')
a=pd.read_csv(BASE/'results/B7_MATCHED100_LEGACY_BETA.csv')
b=pd.read_csv(BASE/'results/B7_MATCHED100_PASSIVE_BETA.csv')
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.plot(a.g,a.beta,marker='o',markersize=3.5,linewidth=1.5,label='Legacy 100-step res-suite')
ax.plot(b.g,b.beta,marker='s',markersize=3.3,linewidth=1.5,label='Current passive 100-step')
ax.axhline(0,linewidth=1.0,linestyle='--')
ax.set_xlabel(r'$g$')
ax.set_ylabel(r'$\beta_{layer}(g)=g_{L+1}-g_L$')
ax.set_title('Same prompt and 100 evaluations do not restore the legacy beta topology')
ax.legend(frameon=False,fontsize=8.5)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['svg','pdf','png']:
    fig.savefig(BASE/f'figures/F03_legacy_vs_passive_beta100.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
