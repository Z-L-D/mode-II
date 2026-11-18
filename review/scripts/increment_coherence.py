# Centering-free cross-layer co-movement K(t) with per-layer circular-shift null.
import pickle; from common import *
rng=np.random.default_rng(0)
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; dY=np.diff(Y,axis=1)
    U=dY/np.sqrt((dY**2).mean(1,keepdims=True)+1e-15)
    K=(np.abs(U.mean(2)).mean(-1)/np.sqrt((U**2).mean(2)).mean(-1)).mean(0)
    Kn=[]
    for _ in range(50):
        sh=rng.integers(0,U.shape[1],size=(U.shape[0],U.shape[2])); idx=(np.arange(U.shape[1])[None,:,None]-sh[:,None,:])%U.shape[1]
        Us=np.take_along_axis(U,idx[...,None].repeat(3,-1),axis=1); Kn.append((np.abs(Us.mean(2)).mean(-1)/np.sqrt((Us**2).mean(2)).mean(-1)).mean())
    print(c,sub,'K(t)',np.round(K,2),'argmax',K.argmax(),'null',round(np.mean(Kn),2))
