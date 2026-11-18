import pickle,sys,numpy as np,pandas as pd
D=sys.argv[1]; rows=[]
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn'),('MOCHI1','attn')]:
    g,m,X,s=pickle.load(open(f'{D}/F_{c}_{sub}.pkl','rb'))
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    sp=np.linalg.norm(np.diff(A,axis=0),axis=1)/np.abs(np.diff(s)); cm=(s[1:]+s[:-1])/2
    r=dict(ch=f'{c}-{sub}',G=X.shape[0])
    for w in (0.5,0.6,0.7):
        for k in (1.5,2,3):
            plat=np.median(sp[cm<w]); i=int(np.argmax(sp<k*plat)); r[f'w{w}_k{k}']=round(float(cm[i]),3)
    rows.append(r)
O=pd.DataFrame(rows).set_index('ch'); print(O.T.to_string())
main=O.drop('MOCHI1-attn',errors='ignore').iloc[:,1:]
print('\nrange over 7 archive channels (excl MOCHI1) per setting:'); print(pd.DataFrame({'min':main.min(),'max':main.max()}).T.to_string())
print('\nall 8 channels:'); a=O.iloc[:,1:]; print(pd.DataFrame({'min':a.min(),'max':a.max()}).T.to_string())
