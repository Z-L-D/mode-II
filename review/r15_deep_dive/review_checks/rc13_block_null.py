# RC13 (post hoc, round-2 review GLM M7): a step-joint null for the per-step +,-,+,- counts of Sec. 7.2.
# The per-step surrogate permutes each layer's g across generations independently at every step, and the
# paper's binomial treated steps as independent. Here one permutation of generations per layer is drawn
# and applied at every step at once, so the dependence between steps of one generation is kept and only the
# within-generation coupling across layers is broken. Statistic: number of steps whose analyzer pattern is
# +,-,+,- (unchanged plot_beta_layer). p = (1 + #draws with count >= observed) / (1 + draws).
import os, sys, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
spec = importlib.util.spec_from_file_location('cpla', f'{ROOT}/review/round6/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from multiprocessing import Pool
TEL = sys.argv[1]; ND = int(sys.argv[2]) if len(sys.argv) > 2 else 60; OUT = os.path.join(HERE, 'out', 'rc13_block_null.csv')
ONLY = os.environ.get('RC13_ONLY')
SETS = [('SD35M archive attn', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', None), ('SD35L archive attn', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None),
        ('MOCHI archive CFG4.5 attn', 'MOCHI-CFG-4.5-attn-480-120', None), ('ZIT-25 archive attn', 'ZIB-CFG-4.0-attn-1024-120', 'attn'),
        ('ZIT-25 archive mlp', 'ZIB-CFG-4.0-attn-1024-120', 'mlp'), ('ZIT archive mlp', 'ZIT-CFG-1.0-attn-1024-120', 'mlp')]
P3 = '+- -+ +-'
def pat(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return ' '.join('+-' if d == 'pos→neg' else '-+' for d in re.findall(r'\(([a-z]+→[a-z]+)\)', t))
D = {}
def count(args):
    key, seed = args; df, steps = D[key]
    d = df
    if seed >= 0:
        rng = np.random.default_rng(seed); d = df.copy()
        G = d['gk'].unique()
        for l, idx in d.groupby('layer').groups.items():
            perm = dict(zip(G, rng.permutation(G)))  # one permutation per layer, shared by all steps
            sub = d.loc[idx]; lut = sub.set_index(['gk', 'token_idx'])['g_micro']
            d.loc[idx, 'g_micro'] = lut.reindex(list(zip(sub['gk'].map(perm), sub['token_idx']))).values
    d = d[d.g_micro.notna()]
    return sum(pat(d[d.token_idx == t]) == P3 for t in steps)
if __name__ == '__main__':
    rows = []
    for lab, folder, sub in SETS:
        if ONLY and lab not in ONLY.split(';'): continue
        dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv')))); df = dm.get_df(); df = df[df.g_micro.notna()]
        if sub: df = df[df.submodule == sub]
        df = df.copy(); df['gk'] = df['prompt_hash'].astype(str) + '|' + df['gen_idx'].astype(str)
        cnt = df.groupby('token_idx').gk.nunique(); steps = cnt[cnt >= cnt.max() / 2].index.tolist()
        D[lab] = (df, steps)
        with Pool(os.cpu_count()) as p: res = p.map(count, [(lab, -1)] + [(lab, 7000 + k) for k in range(ND)])
        obs, sur = res[0], np.array(res[1:])
        rows.append(dict(channel=lab, n_steps=len(steps), observed_3cross_steps=obs, null_mean=sur.mean(), null_q95=float(np.quantile(sur, 0.95)), null_max=int(sur.max()),
                         n_draws=ND, p=(1 + int((sur >= obs).sum())) / (1 + ND)))
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(OUT, index=False)
