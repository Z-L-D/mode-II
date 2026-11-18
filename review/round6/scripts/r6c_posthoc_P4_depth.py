# P4 (post hoc, asked by Zac 2026-10-08 07:38Z): the three-regime reading along the LAYER axis instead of sigma.
# Same continuous 3-segment fit as S4 (r6c_S4_topology.fit3), with x = layer index. Output: out/P4_depth_topology.csv, out/P4_depth_per_sigma.csv
import importlib.util
from r6c_common import *
src = open(os.path.join(os.path.dirname(__file__), 'r6c_S4_topology.py')).read().split('rows = []')[0]
exec(src)  # defines fit3, extrema, per_prompt with the S4 code unchanged
rng = np.random.default_rng(65); NB = 500
def depth_pp(X, meta):
    k = meta.pt.astype(str).str.strip().values; up = np.unique(k)
    return np.stack([X[k == p][..., 0].mean(0) for p in up])          # [P, T, L]
def fitd(y):
    L = len(y); top, i, j, sl = fit3(y, -np.arange(L, dtype=float)); return top, i / (L - 1), j / (L - 1), sl
rows = []; rps = []
chans = [('ZIT', 'mlp'), ('ZIB', 'mlp'), ('ZIT', 'attn'), ('ZIB', 'attn'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
sets = [(f'archive {m}-{sub}',) + tuple(arch(m, sub)[1:]) for m, sub in chans]
for sub in ['mlp', 'attn']:
    z4 = R5('Z40_1024_s1001', sub); sets.append((f'R4 ZIB CFG4 cond-{sub}', z4['meta'], z4['X']['cond'], np.asarray(z4['sigma'], float)))
for lab, meta, X, s in sets:
    PP = depth_pp(X, meta); L = PP.shape[2]
    y = PP.mean((0, 1)); top, b1, b2, sl = fitd(y)
    boot = np.array([fitd(PP[rng.integers(0, len(PP), len(PP))].mean((0, 1)))[0] for _ in range(NB)])
    vals, cnt = np.unique(boot, return_counts=True)
    per = [fitd(PP[:, t].mean(0))[0] for t in range(len(s))]
    for t, tp in enumerate(per): rps.append(dict(set=lab, step=t, sigma=s[t], topology=tp))
    pv, pc = np.unique(per, return_counts=True)
    rows.append(dict(set=lab, layers=L, prompts=len(PP), topology_time_avg=top, boot_modal=vals[cnt.argmax()], boot_modal_frac=cnt.max() / NB,
                     boot_frac_pmp=float((boot == '+-+').mean()), break1_depth=round(b1, 3), break2_depth=round(b2, 3),
                     per_sigma_modal=pv[pc.argmax()], per_sigma_modal_frac=round(pc.max() / len(per), 3), per_sigma_frac_pmp=round(float(np.mean(np.array(per) == '+-+')), 3),
                     argmin_depth=round(int(np.argmin(y)) / (L - 1), 3), argmax_depth=round(int(np.argmax(y)) / (L - 1), 3), g_first=y[0], g_min=y.min(), g_last=y[-1]))
D = pd.DataFrame(rows); D.to_csv(f'{OUT6}/P4_depth_topology.csv', index=False); pd.DataFrame(rps).to_csv(f'{OUT6}/P4_depth_per_sigma.csv', index=False)
pd.set_option('display.width', 300); print(D.round(3).to_string())
