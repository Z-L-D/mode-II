# seq_len (CFG-branch pooling) audit by step and domain; SD35M first-step artifact; SD35L batch-state decodability.
import pickle; from common import *
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline; from sklearn.model_selection import GroupKFold, cross_val_score
for c in ['SD35M','SD35L','ZIB','ZIT','MOCHI','MOCHI1']:
    df,meta,ns,L=load(c,'attn'); s=df.groupby(['gid','token_idx']).seq_len.first().reset_index()
    print(c, s.groupby('token_idx').seq_len.value_counts(normalize=True).unstack().fillna(0).iloc[[0,1,2,-1]].round(2).to_dict())
    big=s.seq_len.max(); fr=s.assign(b=s.seq_len==big).groupby('gid').b.mean().to_frame().join(meta.dom); print(fr.groupby('dom').b.mean().round(2).to_dict())
for c in ['SD35M','SD35L']:
    df,meta,ns,L=load(c,'attn'); S=df.groupby(['gid','token_idx']).seq_len.first().unstack()
    g,m,X,s=pickle.load(open(f'F_{c}_attn.pkl','rb')); S=S.loc[g]; Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]
    j=np.linalg.norm((Y[:,1]-Y[:,0]).reshape(len(Y),-1),axis=1); b=(S[1]==8192).values
    print(c,'step0->1 change batched %.2f unbatched %.2f'%(j[b].mean(),j[~b].mean()))
    t=ns//2; F=Y[:,t].reshape(len(Y),-1); grp=pd.factorize(np.asarray(m.pt.astype(str)))[0]; flag=(S[t]==8192).values
    if 0<flag.mean()<1: print('  batch flag decodable:',cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000)),F,flag,groups=grp,cv=GroupKFold(5)).mean())
