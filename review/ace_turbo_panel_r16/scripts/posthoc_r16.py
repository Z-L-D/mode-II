# Post hoc (not preregistered): why T8 <-> T6 misses 0.94 while the same-prompt panel arms agree. Tests prompt set vs step count.
import os, sys, pickle
sys.path.insert(0, '/mnt/project-files/review/round6/scripts')
from r6c_common import *
SCR = os.environ['SCR']; OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
def cube(n, sub):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{n}_{sub}.pkl', 'rb')); return meta, X, np.asarray(s, float)
rows = []
m6, _, _ = cube('ACET', 'attn'); m8, _, _ = cube('T8', 'attn')
shared = sorted(set(m6.pt.str.strip()) & set(m8.pt.str.strip())); print('prompts shared T6/T8:', len(shared), 'of', m6.pt.nunique(), m8.pt.nunique())
for sub in ('attn', 'mlp'):
    for k in [1, 2, 3, 4, 5, 7, 8, 9, 10]:
        _, XA, sa = cube(f'T{k}', sub); _, XB, sb = cube('ACET', sub); rows.append(dict(test='T_k <-> T6', a=f'T{k}', sub=sub, tangent=tangent(XA, sa, XB, sb)[0]))
    meta, X, s = cube('T8', sub); seeds = meta.seed.astype(int).values; a = seeds < np.median(seeds)
    sa_ = sorted(set(seeds))[:30]; ia = np.isin(seeds, sa_)
    rows.append(dict(test='T8 seed_a <-> seed_b (same prompts, same steps)', a='T8a', sub=sub, tangent=tangent(X[ia], s, X[~ia], s)[0]))
    # random half split of T6 by prompt (different prompts, same model, same steps)
    meta6, X6, s6 = cube('ACET', sub); up = np.array(sorted(meta6.pt.unique())); vals = []
    for r in range(50):
        rng = np.random.default_rng(r); h = rng.permutation(up)[:len(up) // 2]; i = meta6.pt.isin(h).values
        vals.append(tangent(X6[i], s6, X6[~i], s6)[0])
    rows.append(dict(test='T6 prompt-half <-> other half (different prompts, same steps), mean of 50 splits', a='T6', sub=sub, tangent=float(np.mean(vals)), lo=float(np.min(vals))))
    meta8, X8, s8 = cube('T8', sub); up = np.array(sorted(meta8.pt.unique())); vals = []
    for r in range(50):
        rng = np.random.default_rng(r); h = rng.permutation(up)[:len(up) // 2]; i = meta8.pt.isin(h).values
        vals.append(tangent(X8[i], s8, X8[~i], s8)[0])
    rows.append(dict(test='T8 prompt-half <-> other half, mean of 50 splits', a='T8', sub=sub, tangent=float(np.mean(vals)), lo=float(np.min(vals))))
d = pd.DataFrame(rows); d.to_csv(f'{OUTD}/posthoc_r16_prompt_vs_steps.csv', index=False); print(d.round(3).to_string(index=False))
