import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Write EXPLORE (_E) or HOLDOUT (_H) copies of cube pickles, by prompt, per SPLIT.tsv. Usage: SCR=... python split_cubes.py E|H name...
import os, sys, csv, pickle, numpy as np
SCR = os.environ.get('SCR', _COSPAFS_CACHE); half = {'E': 'EXPLORE', 'H': 'HOLDOUT'}[sys.argv[1]]
S = {r['prompt'].strip(): r['half'] for r in csv.DictReader(open(f'{_COSPAFS_ROOT}/review/zib_turbo_mixup_2026-10-18/r13/SPLIT.tsv'), delimiter='\t')}
for name in sys.argv[2:]:
    for kind in ('F', 'V'):
        p = f'{SCR}/{kind}_{name}.pkl'
        if not os.path.exists(p): continue
        o = pickle.load(open(p, 'rb'))
        meta = o[1] if kind == 'F' else o['meta']
        k = np.array([S.get(str(t).strip()) == half for t in meta.pt])
        if kind == 'F': o = [o[0], meta[k], o[2][k], o[3]]
        else: o = dict(o, meta=meta[k], V=o['V'][k])
        base, sub = name.rsplit('_', 1)
        pickle.dump(o, open(f'{SCR}/{kind}_{base}{sys.argv[1]}_{sub}.pkl', 'wb')); print(kind, name, half, int(k.sum()))
