# RC32 (post hoc, round-7 review GLM M10a): the Round 6 S2 prompt-RSA Mantel tests re-run with 10,000 permutations
# (the S2 design unchanged: 117 shared prompts, attention, first/middle/last state, Spearman Mantel). Reuses
# review/round6/scripts/r6c_S2_content.py up to its pair loop; rng 32.
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); S6 = os.path.join(HERE, '..', '..', 'round6', 'scripts'); sys.path.insert(0, S6); os.chdir(S6)
src = open(os.path.join(S6, 'r6c_S2_content.py')).read().split('\nudom =')[0].replace('NPERM = 2000', 'NPERM = 10000').replace('default_rng(62)', 'default_rng(32)')
exec(compile(src, 'r6c_S2', 'exec'))
pairs = [('ZIT', 'ZIB')] + E10_UNREL + [('SD35M', 'SD35L')]; rows = []
for a, b in pairs:
    for tau in TAUS:
        r, p, n95 = mantel(RD[(a, tau)], RD[(b, tau)]); rows.append(dict(a=a, b=b, kind=kind(a, b), tau=tau, rsa=r, p10k=p, null_p95=n95)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'out', 'rc32_mantel_10k.csv'), index=False)
