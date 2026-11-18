# RC14 (post hoc, round-2 review Kimi): the aggregate Surrogates A and B of Table beta with 1000 draws each
# instead of 30. Same analyzer call, channels and permutation rules as review/round6/scripts/r6c_analyzer_beta_nulls.py.
# Surrogate A: each layer's g permuted across all (generation, step) rows. Surrogate B: g permuted across generations
# within each (layer, step) cell. Reports the share of draws that reproduce the observed aggregate pattern and +,-,+,-.
import os, sys, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
spec = importlib.util.spec_from_file_location('cpla', f'{ROOT}/review/round6/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from multiprocessing import Pool
TEL = sys.argv[1]; ND = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; OUT = os.path.join(HERE, 'out', 'rc14_surrogates_1000.csv')
SETS = [('SD35M archive attn', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', None), ('SD35L archive attn', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None),
        ('ZIT archive mlp', 'ZIT-CFG-1.0-attn-1024-120', 'mlp'), ('ZIT-25 archive mlp', 'ZIB-CFG-4.0-attn-1024-120', 'mlp'),
        ('ZIT-25 archive attn', 'ZIB-CFG-4.0-attn-1024-120', 'attn'), ('ZIT archive attn', 'ZIT-CFG-1.0-attn-1024-120', 'attn'),
        ('MOCHI archive CFG4.5 attn', 'MOCHI-CFG-4.5-attn-480-120', None)]
P3 = '+- -+ +-'; D = {}
def pat(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return ' '.join('+-' if d == 'pos→neg' else '-+' for d in re.findall(r'\(([a-z]+→[a-z]+)\)', t))
def surrogate(args):
    lab, kind, seed = args; df, codes = D[lab]; rng = np.random.default_rng(seed); d = df.copy(); g = d['g_micro'].to_numpy().copy()
    for idx in codes[kind]: g[idx] = g[rng.permutation(idx)]
    d['g_micro'] = g; return pat(d)
if __name__ == '__main__':
    rows = []
    for lab, folder, sub in SETS:
        dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv')))); df = dm.get_df(); df = df[df.g_micro.notna()]
        if sub: df = df[df.submodule == sub]
        df = df.reset_index(drop=True)
        codes = {'A': list(df.groupby('layer').indices.values()), 'B': list(df.groupby(['layer', 'token_idx']).indices.values())}
        D[lab] = (df, codes); obs = pat(df)
        with Pool(os.cpu_count()) as p:
            ra = p.map(surrogate, [(lab, 'A', 10 + k) for k in range(ND)], chunksize=8); rb = p.map(surrogate, [(lab, 'B', 50000 + k) for k in range(ND)], chunksize=8)
        rows.append(dict(channel=lab, observed=obs, n_draws=ND, surrA_n_obs=sum(x == obs for x in ra), surrA_n_3cross=sum(x == P3 for x in ra), surrA_modal=pd.Series(ra).mode()[0],
                         surrB_n_obs=sum(x == obs for x in rb), surrB_n_3cross=sum(x == P3 for x in rb), surrB_modal=pd.Series(rb).mode()[0]))
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(OUT, index=False)
