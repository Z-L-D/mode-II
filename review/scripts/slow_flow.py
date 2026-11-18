# |dA/dsigma| by sigma band, per-step vs per-sigma late/early ratios, Mochi 30 vs 100 step invariance.
import pickle; from common import *
bins=[(0.9,1.01),(0.75,0.9),(0.6,0.75),(0.45,0.6),(0.3,0.45),(0.0,0.3)]
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('MOCHI1','attn'),('SD35M','attn'),('SD35L','attn')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    st=np.linalg.norm(np.diff(A,axis=0),axis=1); ds=np.abs(np.diff(s)); cm=(s[1:]+s[:-1])/2; k=max(1,len(st)//3)
    band=[(st[(cm>lo)&(cm<=hi)].sum()/ds[(cm>lo)&(cm<=hi)].sum()) if ((cm>lo)&(cm<=hi)).any() else np.nan for lo,hi in bins]
    st1=st[1:]; k1=max(1,len(st1)//3)
    print(c,sub,'bands',np.round(band,2),'per-sigma late/early %.3f per-step %.3f per-step excl first %.2f dsigma ratio %.2f'%(
        (st[-k:]/ds[-k:]).mean()/(st[:k]/ds[:k]).mean(), st[-k:].mean()/st[:k].mean(), st1[-k1:].mean()/st1[:k1].mean(), ds[-k:].mean()/ds[:k].mean()))
