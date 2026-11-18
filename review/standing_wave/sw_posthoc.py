"""POST HOC (after the frozen tests): (a) chance level of the P4 'central nodes within +/-1' count under the P3 nulls;
(b) period-3 content of the detrended time-mean depth profiles behind the ZIB MLP lattice."""
import numpy as np, pandas as pd, importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ['x']
spec = importlib.util.spec_from_file_location('swt_lib', os.path.join(HERE, 'sw_tests.py'))
src = open(os.path.join(HERE, 'sw_tests.py')).read().split("CHANNELS = [")[0]
ns = {'__file__': os.path.join(HERE, 'sw_tests.py')}; exec(compile(src, 'sw_tests_lib', 'exec'), ns)
rows = []
for ch in ['ZIT_mlp', 'ZIB_mlp', 'ZIT_attn', 'ZIB_attn', 'SD35M_attn']:
    d = ns['load'](ch); layers, sigma = d['layers'], d['sigma']
    r = ns['summarise'](d['G'], d['GAP'], layers, sigma, 'z')
    rec = lambda mins: sum(any(abs(m - c) <= 1 for m in layers[mins]) for c in ns['CENTRAL'])
    n1 = [rec(ns['interior_minima'](np.abs(ns['phase_rand_complex'](r['psi'])))) for _ in range(5000)]
    n2 = []
    for _ in range(500):
        z = ns['summarise'](ns['phase_rand_profiles'](d['G']), ns['phase_rand_profiles'](d['GAP']), layers, sigma, 'z'); n2.append(rec(z['mins']))
    # period-3 share of detrended time-mean profile power
    x = np.arange(len(layers)); p3 = {}
    for nm, X in (('g', d['G'].mean(0)), ('gap', d['GAP'].mean(0))):
        R = X - np.polyval(np.polyfit(x, X, 1), x); P = np.abs(np.fft.rfft(R)) ** 2; f = np.fft.rfftfreq(len(x))
        band = (f >= 1 / 3.5) & (f <= 1 / 2.5); p3[nm] = P[band].sum() / P[1:].sum()
    rows.append(dict(channel=ch, recovered_obs=rec(r['mins']), null1_mean=np.mean(n1), null1_p_ge=np.mean(np.array(n1) >= rec(r['mins'])),
                     null2_mean=np.mean(n2), null2_p_ge=np.mean(np.array(n2) >= rec(r['mins'])),
                     g_power_share_period_2p5_3p5=p3['g'], gap_power_share_period_2p5_3p5=p3['gap'],
                     A_abs_cv=np.std(np.abs(r['A'])) / np.mean(np.abs(r['A']))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, 'out', 'SW_posthoc.csv'), index=False); print(df.round(3).to_string(index=False))
