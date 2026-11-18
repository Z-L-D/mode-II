import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
"""Structural audit of the Round 12 / Round 11 captures delivered 2026-10-27 (no statistics of interest computed).
Per arm: generations (keyed by file, gen_idx, seed), completeness, steps and sigma schedule, layers, submodules,
branches, D, seq_len, prompts and gens per prompt against the spec prompt lists, seeds, variant-column coverage.
Usage: python3 r12_audit.py <data_root>  ->  ../out/R12_audit.json, ../out/R12_audit_gens.csv"""
import os, sys, glob, json, csv
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'out'); ROOT = sys.argv[1]
RR = f'{_COSPAFS_ROOT}/review/zib_turbo_mixup_2026-10-18/rerun'
def plist(p): return [l.rstrip('\n') for l in open(p, encoding='utf-8') if l.strip()]
P120 = set(plist(f'{RR}/prompts_primary120.txt')); P60 = set(plist(f'{RR}/prompts_r3_cond60.txt'))
SPEC = {'Z1_ZIB_CFG4_primary': dict(prompts=P120, per=4, steps=25, branches={'cond', 'uncond'}),
        'Z2_ZIB_CFG1_variants': dict(prompts=P60, per=1, steps=25, branches={'cond'}),
        'Z3_ZIT_CFG4_matched': dict(prompts=P120, per=1, steps=25, branches={'cond', 'uncond'}),
        'R11_ZIT_CFG1_variants': dict(prompts=P60, per=1, steps=None, branches={'cond'}),
        'R11_SD35M_CFG1_variants': dict(prompts=P60, per=1, steps=None, branches={'cond'}),
        'ACE_turbo_matched': dict(prompts=P120, per=1, steps=6, branches={'cond'})}
VAR = ['g_img', 'g_txt', 'g_ctr', 'g_topk']
res, grows = {}, []
for arm, sp in SPEC.items():
    frames = []
    for f in sorted(glob.glob(f'{ROOT}/{arm}/*.csv')):
        d = pd.read_csv(f, dtype={'prompt_text': str}, low_memory=False); d['file'] = os.path.basename(f); frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['prompt_text'] = d.groupby(['file', 'gen_idx', 'seed'])['prompt_text'].transform(lambda s: s.ffill().bfill())
    key = ['file', 'gen_idx', 'seed']
    L = sorted(d.layer.unique()); subs = sorted(d.submodule.unique()); brs = sorted(d.branch.dropna().unique())
    g = d.groupby(key).agg(rows=('g_micro', 'size'), steps=('token_idx', 'nunique'), gen_len=('gen_len', 'first'), prompt=('prompt_text', 'first'),
                           s0=('sigma', 'max'), sN=('sigma', 'min'), seq=('seq_len', 'first')).reset_index()
    full = len(L) * len(subs) * len(brs)
    g['complete'] = (g.steps == g.gen_len) & (g.rows == g.gen_len * full)
    sched = d[d.file == g.file.iloc[0]].query('layer == @L[0] and submodule == @subs[0]').drop_duplicates('token_idx').sort_values('token_idx').sigma.round(4).tolist()
    gc = g[g.complete]; pp = gc.groupby('prompt').size()
    in_spec = set(gc.prompt) & sp['prompts']
    r = dict(files=len(frames), rows=int(len(d)), gens_total=int(len(g)), gens_complete=int(g.complete.sum()),
             incomplete=g[~g.complete][['file', 'gen_idx', 'seed', 'rows', 'steps', 'gen_len']].astype(str).values.tolist()[:12],
             steps=sorted(map(int, gc.gen_len.unique())), sigma_schedule_first_gen=sched[:4] + ['…'] + sched[-3:],
             layers=f'{min(L)}–{max(L)} ({len(L)})', submodules=subs, branches=brs, hidden_dim=sorted(map(int, d.hidden_dim.unique())),
             seq_len=sorted(map(int, d.seq_len.unique()))[:6], hook_mode=sorted(d.hook_mode.unique()),
             prompts_unique=int(gc.prompt.nunique()), prompts_in_spec=len(in_spec), spec_prompts=len(sp['prompts']),
             prompts_not_in_spec=sorted(set(gc.prompt) - sp['prompts'])[:5], spec_prompts_missing=len(sp['prompts'] - set(gc.prompt)),
             gens_per_prompt=pp.value_counts().sort_index().to_dict(), seeds_unique=int(gc.seed.nunique()),
             dup_seed_prompt=int(gc.duplicated(['prompt', 'seed']).sum()))
    if all(v in d.columns for v in VAR):
        r['variant_nonnull_frac'] = {v: float(d[v].notna().mean()) for v in VAR}
    res[arm] = r; gc2 = g.assign(arm=arm); grows.append(gc2)
    print(arm, json.dumps(r, default=str)[:1500], '\n')
json.dump(res, open(f'{OUT}/R12_audit.json', 'w'), indent=1, default=str)
pd.concat(grows).to_csv(f'{OUT}/R12_audit_gens.csv', index=False)
