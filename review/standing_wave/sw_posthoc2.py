"""POST HOC: does the time-varying part of z(L,t) (time mean removed) oscillate? A drift gives one sign change of the
leading temporal coefficient (forced by centring); an oscillation needs >= 2. Also: share of energy in the time-mean."""
import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'sw_tests.py')).read().split("CHANNELS = [")[0]
ns = {'__file__': os.path.join(HERE, 'sw_tests.py')}; exec(compile(src, 'sw_tests_lib', 'exec'), ns)
rows = []
for ch in ['ZIT_mlp', 'ZIB_mlp', 'ZIT_attn', 'ZIB_attn', 'Mochi_attn', 'SD35M_attn', 'SD35L_attn']:
    d = ns['load'](ch); Z = ns['field'](d['G'], d['GAP'], d['layers'], 'z')
    mean_share = np.sum(np.abs(Z.mean(0)) ** 2) * Z.shape[0] / np.sum(np.abs(Z) ** 2)
    C = Z - Z.mean(0); U, s, Vh = np.linalg.svd(C.T, full_matrices=False)
    a = s[0] * Vh[0].conj(); W = np.array([a.real, a.imag]); ev, evec = np.linalg.eigh(W @ W.T); p = evec[:, -1] @ W
    sc = int(np.sum(np.sign(p[1:]) != np.sign(p[:-1])))
    # monotone-drift check: Spearman of p with step index
    rho = pd.Series(p).corr(pd.Series(np.arange(len(p))), method='spearman')
    rows.append(dict(channel=ch, steps=len(p), time_mean_energy_share=mean_share, fluct_mode1_share=s[0] ** 2 / np.sum(s ** 2),
                     fluct_coef_sign_changes=sc, fluct_coef_spearman_vs_step=rho))
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, 'out', 'SW_posthoc_fluct.csv'), index=False); print(df.round(3).to_string(index=False))
