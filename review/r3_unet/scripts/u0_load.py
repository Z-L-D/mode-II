"""U0: load the SD1.5 UNet CSVs (R3C_SCORING_PLAN.md) -> cache npz with arrays [G, T, L] per (channel, branch, field).
Complete gens only (50 steps x 16 blocks x 2 subs x 2 branches); gens keyed by (file, gen_idx)."""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
fs = sorted(glob.glob(f'{BASE}/data/*.csv'))
dfs = []
for f in fs:
    d = pd.read_csv(f, keep_default_na=True)
    d['file'] = os.path.basename(f).split('G480-')[1].rsplit('-0.csv', 1)[0]
    d['prompt_text'] = d.groupby('gen_idx').prompt_text.transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    dfs.append(d)
D = pd.concat(dfs, ignore_index=True)
D['gid'] = D.file + ':' + D.gen_idx.astype(str)
print('rows', len(D), 'files', len(fs), 'gens raw', D.gid.nunique(), 'hook_mode', D.hook_mode.unique(), 'branches', D.branch.unique())
cnt = D.groupby('gid').size(); T = D.token_idx.nunique(); L = D.layer.nunique()
ok = cnt[cnt == T * L * 2 * 2].index; D = D[D.gid.isin(ok)]
print('complete gens', len(ok), 'T', T, 'L', L, 'prompts', D.prompt_text.nunique(), 'seeds', D.seed.nunique(), 'gen_len', D.gen_len.unique())
gids = sorted(ok); out = {}
meta = D.groupby('gid').agg(pt=('prompt_text', 'first'), dom=('file', 'first'), seed=('seed', 'first'))
meta = meta.loc[gids]
for sub in ('attn', 'mlp'):
    for br in ('cond', 'uncond'):
        d = D[(D.submodule == sub) & (D.branch == br)]
        for v in ('g_micro', 'spectral_gap', 'sv_entropy', 'phi_micro', 'prompt_influence'):
            piv = d.pivot_table(index=['gid', 'token_idx'], columns='layer', values=v, dropna=False).reindex(pd.MultiIndex.from_product([gids, range(T)]))
            out[f'{sub}_{br}_{v}'] = piv.values.reshape(len(gids), T, L)
sig = D[D.gid == gids[0]].groupby('token_idx').sigma.first().values
NL = D[D.gid == gids[0]].groupby('layer')[['seq_len', 'hidden_dim']].first().values
np.savez_compressed(f'{BASE}/out/u0_cache.npz', gids=np.array(gids), pt=meta.pt.values.astype(str), dom=meta.dom.values.astype(str), seed=meta.seed.values,
                    sigma=sig, NL=NL, **out)
print('sigma', sig[:3], '...', sig[-3:]); print('N,D per block', NL.tolist())
print('NaN in cond g:', {s: int(np.isnan(out[f'{s}_cond_g_micro']).sum()) for s in ('attn', 'mlp')})
print('seeds per prompt', pd.Series(meta.pt.values).value_counts().describe().round(2).to_dict())
