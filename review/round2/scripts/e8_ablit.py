import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E8 / P8: Mistral Instruct vs ablit weight-only effect on identical text (prefill = same prompt tokens in both models).
import glob, numpy as np, pandas as pd
a=pd.read_parquet(glob.glob('pq/LLM_mistral-03-7B-instruct_*.parquet')[0]); b=pd.read_parquet(glob.glob('pq/LLM_mistral-03-7B-instruct-ablit_*.parquet')[0])
pa=a[a.token_type=='prefill']; pb=b[b.token_type=='prefill']
ta=pa.groupby('gen_idx').prompt_text.first(); tb=pb.groupby('gen_idx').prompt_text.first()
same=[g for g in ta.index if g in tb.index and ta[g]==tb[g]]
print('prefill gens with identical prompt text',len(same),'of',len(ta))
mm=pa[pa.gen_idx.isin(same)].merge(pb[pb.gen_idx.isin(same)],on=['gen_idx','layer'],suffixes=('_a','_b'))
assert (mm.seq_len_a==mm.seq_len_b).all()
for v in ['g_micro','spectral_gap']: mm['d_'+v]=np.log(mm[v+'_b']/mm[v+'_a'])
mm['d_H']=mm.sv_entropy_b-mm.sv_entropy_a
L=mm.layer.nunique()
prof=mm.groupby('layer')[['d_g_micro','d_spectral_gap','d_H']].agg(lambda x: np.median(np.abs(x)))
sgn=mm.groupby('layer').d_g_micro.median()
print('median |dlog g| overall %.5f  p90 %.5f  max %.5f'%(mm.d_g_micro.abs().median(),mm.d_g_micro.abs().quantile(.9),mm.d_g_micro.abs().max()))
print('median |dlog gap| %.5f  median |dH| %.5f'%(mm.d_spectral_gap.abs().median(),mm.d_H.abs().median()))
am=int(prof.d_g_micro.values.argmax()); print('argmax layer',am,'depth frac %.3f'%(am/(L-1)))
print('per-layer median |dlog g| x1e3:',np.round(prof.d_g_micro.values*1e3,2))
print('per-layer median signed dlog g x1e3:',np.round(sgn.values*1e3,2))
# prompt dependence: per-gen mean |dlog g| vs prompt (harmful-sounding prompts?)
pg=mm.groupby('gen_idx').d_g_micro.apply(lambda x: x.abs().mean()).sort_values()
print('per-prompt mean |dlog g|: min %.5f median %.5f max %.5f'%(pg.min(),pg.median(),pg.max()))
for g in pg.index[-5:]: print('  top',round(pg[g],5),ta[g][:90])
for g in pg.index[:3]: print('  bottom',round(pg[g],5),ta[g][:90])
out=prof.assign(signed_dlogg=sgn).reset_index(); out.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E8_ablit_prefill_by_layer.csv',index=False)
pd.DataFrame({'gen_idx':pg.index,'mean_abs_dlogg':pg.values,'prompt':[ta[g] for g in pg.index]}).to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E8_ablit_prefill_by_prompt.csv',index=False)
