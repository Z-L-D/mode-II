# P4b (post hoc): is the layer-axis +-+ carried by the last layers or by the linear scale? Refit on log g and with the final 1 / 2 layers dropped.
from r6c_common import *
exec(open(os.path.join(os.path.dirname(__file__), 'r6c_S4_topology.py')).read().split('rows = []')[0])
rng = np.random.default_rng(66); NB = 300
def depth_pp(X, meta):
    k = meta.pt.astype(str).str.strip().values; up = np.unique(k); return np.stack([X[k == p][..., 0].mean(0) for p in up])
def fitd(y):
    L = len(y); top, i, j, sl = fit3(y, -np.arange(L, dtype=float)); return top, i / (L - 1), j / (L - 1)
rows = []
chans = [('ZIT', 'mlp'), ('ZIB', 'mlp'), ('ZIT', 'attn'), ('ZIB', 'attn'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
for m, sub in chans:
    _, meta, X, s = arch(m, sub); PP = depth_pp(X, meta).mean(1)     # [P, L] time-averaged depth profile
    for scale in ['g', 'log g']:
        Q = np.log(PP) if scale == 'log g' else PP
        for drop in [0, 1, 2]:
            Z = Q[:, :Q.shape[1] - drop] if drop else Q
            top, b1, b2 = fitd(Z.mean(0)); boot = np.array([fitd(Z[rng.integers(0, len(Z), len(Z))].mean(0))[0] for _ in range(NB)])
            v, c = np.unique(boot, return_counts=True)
            rows.append(dict(channel=f'{m}-{sub}', scale=scale, dropped_last_layers=drop, topology=top, break1_depth=round(b1, 3), break2_depth=round(b2, 3),
                             boot_frac_pmp=round(float((boot == '+-+').mean()), 3), boot_modal=v[c.argmax()]))
D = pd.DataFrame(rows); D.to_csv(f'{OUT6}/P4b_depth_robustness.csv', index=False)
pd.set_option('display.width', 250); print(D.to_string())
# the raw time-averaged depth profiles, normalised to each channel's max, for the reply
for m, sub in chans:
    _, meta, X, s = arch(m, sub); y = X[..., 0].mean((0, 1)); print(f'{m}-{sub}', np.round(y / y.max(), 2).tolist())
