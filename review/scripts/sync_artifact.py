# Reproduces ledger sync peaks and shows the same interior peak from monotone trend surrogates with no coordination.
import pickle; from common import *
from scipy.stats import spearmanr
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]:
    gids,meta,X,sig=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); M=X.mean(0); T,L,_=M.shape; t=np.arange(T)
    R=syncR(M); lin=np.empty_like(M); srt=np.empty_like(M)
    for l in range(L):
        for k in range(3):
            lin[:,l,k]=np.polyval(np.polyfit(t,M[:,l,k],1),t)
            v=np.sort(M[:,l,k]); srt[:,l,k]=v if M[-1,l,k]>=M[0,l,k] else v[::-1]
    mono=np.median([abs(spearmanr(t,M[:,l,k])[0]) for l in range(L) for k in range(3)])
    print(f'{c} {sub}: observed {R.max():.4f}@{R.argmax()}/{T-1}  linear {syncR(lin).max():.3f}@{syncR(lin).argmax()}  sorted {syncR(srt).max():.3f}@{syncR(srt).argmax()}  median|rho| {mono:.2f}')
