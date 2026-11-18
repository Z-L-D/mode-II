import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Load the Round 3 matched-study CSVs (data/r3_captures) into per-arm field cubes.
# Output (scratch): F3_<arm>_<sub>.pkl = (gids, meta, X[G,T,L,3] of g/gap/H, sigma[T]); intake table to out/.
import glob, os, re, pickle, sys, numpy as np, pandas as pd
D=f'{_COSPAFS_ROOT}/data/r3_captures'; S=os.environ.get('SCR','.')
OUT=f'{_COSPAFS_ROOT}/review/round3/audit/out'
DIRS={'2c7cf154':'S45_1024_s1002','4244ed53':'S45_1024_s1001','0c6de754':'S45_768_s1001','8706ae48':'S45_512_s1001',
      '4429ec00':'S10_1024_s1002','90c87dde':'S10_1024_s1001','5ccfeade':'Z40_1024_s1001','c9ad092f':'Z10_1024_s1001'}
# c63512cd is a byte-identical copy of 4244ed53 (checked by md5) and is skipped
frames=[]; intake=[]
for d,arm in DIRS.items():
    for f in sorted(glob.glob(f'{D}/{d}/*.csv')):
        x=pd.read_csv(f); theme=re.search(r'G60-(.+?)[-.]0\.csv$',os.path.basename(f)).group(1)
        x['arm']=arm; x['theme']=theme; x['file']=os.path.basename(f); frames.append(x)
df=pd.concat(frames,ignore_index=True)
df['prompt_text']=df.groupby(['file','gen_idx']).prompt_text.transform(lambda s: s.dropna().iloc[0] if s.notna().any() else '')
df['gid']=df.arm+':'+df.file+':'+df.gen_idx.astype(str)
summ=[]
for arm,a in df.groupby('arm'):
    ns=a.token_idx.nunique(); L=a.layer.nunique(); nsub=a.submodule.nunique()
    c=a.groupby('gid').agg(n=('token_idx','nunique'),r=('layer','size'),fin=('g_micro',lambda v: np.isfinite(v).all()))
    ok=c[(c.n==ns)&(c.r==ns*L*nsub)&c.fin].index
    bad=c[~c.index.isin(ok)]
    summ.append(dict(arm=arm,gens=len(c),complete=len(ok),dropped=';'.join(bad.index),steps=ns,layers=L,subs=nsub,
                     prompts=a[a.gid.isin(ok)].prompt_hash.nunique(),seq=','.join(map(str,sorted(a.seq_len.unique()))),
                     hook_mode=','.join(a.hook_mode.unique()),branch=','.join(a.branch.unique()),seed=','.join(map(str,a.seed.unique()))))
    a=a[a.gid.isin(ok)]
    for sub in sorted(a.submodule.unique()):
        b=a[a.submodule==sub]
        piv=b.pivot_table(index=['gid','token_idx'],columns='layer',values=['g_micro','spectral_gap','sv_entropy'])
        gids=list(piv.index.get_level_values(0).unique())
        X=np.stack([piv[v].values.reshape(len(gids),ns,L) for v in ['g_micro','spectral_gap','sv_entropy']],-1)
        meta=b.groupby('gid').agg(pt=('prompt_text','first'),ph=('prompt_hash','first'),theme=('theme','first'),seed=('seed','first'),
                                  seq=('seq_len','first'),seqs=('seq_len',lambda v: ','.join(map(str,sorted(set(v)))))).loc[gids]
        sig=b[b.gid==gids[0]].groupby('token_idx').sigma.first().values
        pickle.dump((gids,meta,X,sig),open(f'{S}/F3_{arm}_{sub}.pkl','wb'))
S_=pd.DataFrame(summ); print(S_.to_string()); S_.to_csv(f'{OUT}/R3_intake.csv',index=False)
df.to_parquet(f'{S}/r3_all.parquet')
