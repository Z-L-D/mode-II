# E4 / P4a supplement: prior ZIT 8/16/32-step CSVs (older capture, seq 1152, 4 layers, 10 gens each, one prompt).
import pandas as pd, numpy as np
D={s:pd.read_csv(f'z-image-turbo-{f}.csv') for s,f in [(8,5),(16,7),(32,8)]}
V=['g_micro','spectral_gap','sv_entropy']
M={s:d.groupby(['sigma','layer'])[V].mean().unstack('layer') for s,d in D.items()}
s8=M[8].index.values
for s in [16,32]:
    devs=[]
    for v in V:
        A=M[s][v]; B=M[8][v]
        Ai=np.stack([np.interp(s8,A.index.values,A[c].values) for c in A.columns],1)
        rel=np.abs(Ai/B.values-1); devs.append((v,np.median(rel),rel.max()))
    print(f'{s} vs 8 steps:',[(v,round(a,4),round(b,4)) for v,a,b in devs])
# seed floor: half-split of 8-step gens
d=D[8]; gi=d.gen_idx.unique(); a=d[d.gen_idx.isin(gi[::2])].groupby(['sigma','layer'])[V].mean(); b=d[d.gen_idx.isin(gi[1::2])].groupby(['sigma','layer'])[V].mean()
print('8-step half-split seed floor median rel:',np.round(np.median(np.abs(a/b-1),0),4))
