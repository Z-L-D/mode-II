"""POST HOC (suggested by LIT_LENSES.md T1; not preregistered): finite-sample isotropic baseline for the amplitude g.
White Gaussian X (N x D) at each channel's own N and D -> g_iso; compared with the archive's median g (all rows) and its range.
Output: out/PH_isotropy_baseline.csv"""
import numpy as np, pandas as pd
from lens_common import CH, OUT, load
rng = np.random.default_rng(1); NN = {'Z-Image': {'ZIT': 4160, 'ZIB': 8320}, 'Mochi': 11130, 'SD3.5': 4096}
rows = []
for lab, key, D, fam in CH:
    N = NN['Z-Image']['ZIT' if lab.startswith('ZIT') else 'ZIB'] if fam == 'Z-Image' else NN[fam]
    s = np.linalg.svd(rng.standard_normal((N, D)) / np.sqrt(N), compute_uv=False); giso = s.sum() ** 2 / (D * (s ** 2).sum())
    g = load(key)['g_gen']; rows.append(dict(channel=lab, N=N, D=D, q=D / N, g_iso=giso, g_median=np.nanmedian(g), g_p01=np.nanquantile(g, 0.01), g_p99=np.nanquantile(g, 0.99),
                                             ratio_median_to_iso=np.nanmedian(g) / giso))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/PH_isotropy_baseline.csv', index=False); print(df.round(4).to_string(index=False))
