"""A0: build the ACE cache with the analyzer DataManager unchanged, as standing_wave/sw_load.py (R3B_SCORING_PLAN.md).
Writes out/cache_{attn,mlp}.npz with g_gen, gap_gen, ent_gen [n, t, L], sigma, layers, prompts, dom, seed, file."""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, glob, types, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
import numpy as np, pandas as pd, csv
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location('cpla', f'{_COSPAFS_ROOT}/review/round6/analyzer/cos_pafs_live_analyzer.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
fs = sorted(glob.glob(f'{BASE}/data/*.csv'))
dm = A.DataManager(); dm.load_paths(','.join(fs)); df = dm.get_df(); df = df[df.g_micro.notna()]
raw = sum(len(pd.read_csv(f)) for f in fs)
print('DataManager rows', len(df), 'raw rows', raw, 'gen_idx unique', df.gen_idx.nunique(), 'cols', [c for c in df.columns if c not in pd.read_csv(fs[0], nrows=1).columns])
tsv = {r['tags'].strip(): r['domain'] for r in csv.DictReader(open(f'{_COSPAFS_ROOT}/capture_code_r3/ace_step/ace_prompts_30.tsv'), delimiter='\t')}
ptext = df.dropna(subset=['prompt_text']).groupby('gen_idx').prompt_text.first()
seed = df.groupby('gen_idx').seed.first()
for sub in ('attn', 'mlp'):
    d = df[df.submodule == sub]; nl = d.layer.nunique()
    cnt = d.groupby('gen_idx').size(); steps_per = d.groupby('gen_idx').token_idx.nunique(); T = int(steps_per.mode()[0])
    keep = cnt.index[(steps_per == T) & (cnt == T * nl)]; d = d[d.gen_idx.isin(keep)]
    gens = np.array(sorted(keep)); steps = np.sort(d.token_idx.unique()); layers = np.sort(d.layer.unique())
    piv = lambda col: d.pivot_table(index=['gen_idx', 'token_idx'], columns='layer', values=col, aggfunc='mean') \
                        .reindex(pd.MultiIndex.from_product([gens, steps])).values.reshape(len(gens), len(steps), len(layers))
    g, gap, ent = piv('g_micro'), piv('spectral_gap'), piv('sv_entropy')
    sig = d.groupby('token_idx').sigma.mean().reindex(steps).values
    prompts = ptext.reindex(gens).astype(str).str.strip().values
    dom = np.array([tsv[p] for p in prompts])
    np.savez(f'{BASE}/out/cache_{sub}.npz', g_gen=g, gap_gen=gap, ent_gen=ent, sigma=sig, layers=layers, prompts=prompts, dom=dom,
             seed=seed.reindex(gens).values, gens=gens, G=np.nanmean(g, 0), GAP=np.nanmean(gap, 0), ENT=np.nanmean(ent, 0))
    print(f'{sub}: {len(gens)} gens, {len(steps)} steps, {len(layers)} layers, {len(set(prompts))} prompts, {len(set(dom))} domains, '
          f'nan={np.isnan(g).sum()}, sigma {np.round(sig, 4).tolist()}', flush=True)
