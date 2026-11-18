# Round 12 / Round 11: build archive-format cubes (g, meta, X[G,T,L,3], sigma) from the new captures.
# X[...,0]=g_micro, 1=spectral_gap, 2=sv_entropy (+ variant columns in a separate array when present).
# Rules (fixed before scoring, RERUN_SPEC / PREREG_R12): keep only complete generations (every step x layer present,
# all three statistics finite); one branch at a time; prompt text is carried per (file, gen_idx, seed);
# balanced design: at most `cap` complete generations per prompt, the first ones in file order (Z1: 4).
# Usage: python build_cubes.py <name> <branch> <cap|0> <out_dir> <csv...>
import sys, os, csv, math, pickle, collections
import numpy as np, pandas as pd
VAR = ['g_img', 'gap_img', 'H_img', 'g_txt', 'gap_txt', 'H_txt', 'g_ctr', 'gap_ctr', 'H_ctr', 'g_topk', 'gap_topk', 'H_topk',
       'n_img_rows', 'n_txt_rows', 'txt_energy_share', 'mean_energy_share', 'top1_chan_share', 'n_massive_100x']
def f(x):
    try: return float(x)
    except Exception: return float('nan')
def main():
    name, branch, cap, outd = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; files = sys.argv[5:]
    gens = collections.OrderedDict(); info = {}
    for p in files:
        dom = os.path.basename(p).rsplit('-', 2)[-2] if 'G' in os.path.basename(p) else ''
        with open(p, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                k = (os.path.basename(p), r['gen_idx'], r['seed'])
                if r['prompt_text']: info.setdefault(k, dict(pt=r['prompt_text'].strip(), dom=dom, seq=int(r['seq_len']), pl=int(r['prompt_len'] or 0), seed=r['seed']))
                if r['branch'] not in (branch, 'all'): continue
                gens.setdefault(k, {})
                gens[k][(r['submodule'], int(r['token_idx']), int(r['layer']))] = r
    subs = sorted({s for g in gens.values() for (s, _, _) in g})
    for sub in subs:
        cells = [{(t, l): r for (s, t, l), r in g.items() if s == sub} for g in gens.values()]
        T = max(max(t for t, _ in c) + 1 for c in cells if c); L = max(max(l for _, l in c) + 1 for c in cells if c)
        keep = []; per = collections.Counter(); dropped = collections.Counter()
        for k, c in zip(gens.keys(), cells):
            if k not in info: dropped['no prompt'] += 1; continue
            if len(c) != T * L: dropped['incomplete'] += 1; continue
            X = np.array([[[f(c[(t, l)][m]) for m in ('g_micro', 'spectral_gap', 'sv_entropy')] for l in range(L)] for t in range(T)])
            if not np.isfinite(X).all() or (X[..., 0] <= 0).any(): dropped['nonfinite'] += 1; continue
            pt = info[k]['pt']
            if cap and per[pt] >= cap: dropped['over cap'] += 1; continue
            per[pt] += 1
            V = None
            if 'g_img' in c[(0, 0)]:
                V = np.array([[[f(c[(t, l)][m]) for m in VAR] for l in range(L)] for t in range(T)])
            sig = [f(c[(t, 0)]['sigma']) for t in range(T)]
            keep.append((k, X, V, sig))
        Xa = np.stack([x for _, x, _, _ in keep]); s = np.array(keep[0][3])
        meta = pd.DataFrame([dict(gid=f'{k[0]}:{k[1]}:{k[2]}', **info[k]) for k, _, _, _ in keep]).set_index('gid')
        meta['ph'] = pd.factorize(meta.pt)[0]
        out = [list(meta.index), meta, Xa, s]
        pickle.dump(out, open(f'{outd}/F_{name}_{sub}.pkl', 'wb'))
        if keep[0][2] is not None:
            pickle.dump(dict(meta=meta, V=np.stack([v for _, _, v, _ in keep]), cols=VAR, s=s), open(f'{outd}/V_{name}_{sub}.pkl', 'wb'))
        print(f'{name} {sub} branch={branch}: kept {len(keep)} gens, {meta.pt.nunique()} prompts, T={T} L={L}, '
              f'min/max per prompt {min(per.values())}/{max(per.values())}, seq {sorted(meta.seq.unique())}, dropped {dict(dropped)}')
main()
