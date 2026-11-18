# Infer the sampling shift from the logged per-step sigma schedule.
# Assumes a linear-timestep scheduler (ComfyUI 'simple'/'normal'): t_k = 1 - k/N,
# sigma = s*t / (1 + (s-1)*t)  =>  s = sigma*(1-t) / (t*(1-sigma)).
import csv, glob, sys, os
import numpy as np
for f in sys.argv[1:]:
    sig = {}
    with open(f, newline='') as fh:
        r = csv.reader(fh)
        hdr = next(r)
        if hdr[0] != 'gen_idx':
            continue
        gi, ti, si = hdr.index('gen_idx'), hdr.index('token_idx'), hdr.index('sigma')
        first = None
        for row in r:
            if first is None: first = row[gi]
            if row[gi] != first: break
            sig.setdefault(int(row[ti]), float(row[si]))
    s = np.array([sig[k] for k in sorted(sig)])
    N = len(s)
    k = np.arange(N); t = 1 - k / N
    m = (k > 0) & (s < 1)
    est = s[m] * (1 - t[m]) / (t[m] * (1 - s[m]))
    print(f"{os.path.basename(os.path.dirname(f))[:45]:45s} N={N} sigma[0:4]={np.round(s[:4],4)} last={s[-1]:.4f} shift median={np.median(est):.3f} range=({est.min():.3f},{est.max():.3f})")
