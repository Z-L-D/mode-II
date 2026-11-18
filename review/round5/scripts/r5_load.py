import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 5 loader: the corrected per-branch captures (data/r4_captures, uploaded 2026-10-08 as telemetry-10-7-26.zip).
# Output (scratch): R5_<arm>_<sub>.pkl = dict(gids, meta, X={branch: [G,T,L,3] of g/gap/H}, PI=[G,T,L], sigma=[T]);
#                   intake table to review/round5/out/R5_intake.csv.
# Generation key = arm:file:gen_idx:seed. The CFG 4.5 40-step pyramid file reuses gen_idx values for two different
# seeds (two sessions appended), so gen_idx alone is not a generation id. One 512 CSV has no header row; the frozen
# 21-column schema is applied to it.
import glob, os, re, pickle, sys, hashlib, numpy as np, pandas as pd
D = f'{_COSPAFS_ROOT}/data/r4_captures'
S = os.environ.get('SCR', '.')
OUT = f'{_COSPAFS_ROOT}/review/round5/out'; os.makedirs(OUT, exist_ok=True)
COLS = ["gen_idx","token_idx","token_type","layer","submodule","g_micro","phi_micro","phi_micro_skip","spectral_gap",
        "sv_entropy","sigma","seq_len","hidden_dim","hook_mode","prompt_hash","prompt_len","gen_len","prompt_text",
        "branch","prompt_influence","seed"]
ARMS = {
 'SD3.5M-attn-TE3-CFG1.0-R1024-S1001-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'S10_1024_s1001',
 'SD3.5M-attn-TE3-CFG1.0-R1024-S1002-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'S10_1024_s1002',
 'SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'S45_1024_s1001',
 'SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'S45_1024_s1002',
 'SD3.5M-attn-TE3-CFG4.5-R768-S1001-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'S45_768_s1001',
 'SD3.5M-attn-TE3-CFG4.5-R512-S1001-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'S45_512_s1001',
 'ZIB-attn-TE1-CFG1.0-R1024-S1001-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'Z10_1024_s1001',
 'ZIB-attn-TE1-CFG4.0-R1024-S1001-STUDY_CFG-RES-SEED-CORRECTION_-G60': 'Z40_1024_s1001',
 'SD3.5M-attn-TE3-CFG1.0-R1024-S_RAND_-STUDY_CFG-PYRAMID-DEEPSTEP100_-G60-pyramid-deep-step-0': 'P10_100',
 'SD3.5M-attn-TE3-CFG1.0-R1024-S_RAND_-STUDY_CFG-PYRAMID-DEEPSTEP40_-G60-pyramid-deep-step-0': 'P10_40',
 'SD3.5M-attn-TE3-CFG4.5-R1024-S_RAND_-STUDY_CFG-PYRAMID-DEEPSTEP100_-G60-pyramid-deep-step-0': 'P45_100',
 'SD3.5M-attn-TE3-CFG4.5-R1024-S_RAND_-STUDY_CFG-PYRAMID-DEEPSTEP40_-G60-pyramid-deep-step-0': 'P45_40',
}
def read(f):
    with open(f, encoding='utf-8') as h: first = h.readline()
    hdr = first.startswith('gen_idx,')
    x = pd.read_csv(f, low_memory=False) if hdr else pd.read_csv(f, header=None, names=COLS, low_memory=False)
    return x, hdr
import csv
# matched_prompts_30.tsv stores the sea-otter prompt as a quoted field that starts with a newline ("\nA sea otter ...");
# csv parsing plus strip() recovers the intended text.
_rows = list(csv.reader(open(f'{_COSPAFS_ROOT}/capture_code_r3/matched_prompts_30.tsv', encoding='utf-8'), delimiter='\t'))
frozen = {r[-1].strip() for r in _rows[1:]}
assert len(frozen) == 30
rows = []; frames = []
for d, arm in ARMS.items():
    for f in sorted(glob.glob(f'{D}/{d}/*.csv')):
        x, hdr = read(f)
        m = re.search(r'G60-(.+?)-0\.csv$', os.path.basename(f)); theme = m.group(1) if m else ''
        x['arm'] = arm; x['theme'] = theme; x['file'] = os.path.basename(f); x['_hdr'] = hdr
        frames.append(x)
df = pd.concat(frames, ignore_index=True)
df['gid'] = df.arm + ':' + df.file + ':' + df.gen_idx.astype(str) + ':' + df.seed.astype('int64').astype(str)
pt = df.dropna(subset=['prompt_text']).groupby('gid').prompt_text.first()
df['pt'] = df.gid.map(pt).fillna('')
summ = []
for arm, a in df.groupby('arm', sort=False):
    ns = int(a.gen_len.max()); L = a.layer.nunique(); subs = sorted(a.submodule.unique()); brs = sorted(a.branch.unique())
    need = ns * L * len(subs) * len(brs)
    c = a.groupby('gid').agg(n=('token_idx', 'nunique'), r=('layer', 'size'), fin=('g_micro', lambda v: np.isfinite(v).all()),
                             pif=('prompt_influence', lambda v: np.isfinite(pd.to_numeric(v, errors='coerce')).all()))
    ok = c[(c.n == ns) & (c.r == need) & c.fin].index
    bad = c[~c.index.isin(ok)]
    ap = a[a.gid.isin(ok)].groupby('gid').pt.first()
    summ.append(dict(arm=arm, folder=[k for k, v in ARMS.items() if v == arm][0], csv_files=a.file.nunique(),
                     headerless_files=int((~a.groupby('file')._hdr.first()).sum()), gens=len(c), complete=len(ok),
                     dropped=';'.join(bad.index), gens_nonfinite_prompt_influence=int((~c.pif).sum()),
                     steps=ns, layers=L, subs=','.join(subs), branches=','.join(brs), prompts=int(ap.nunique()),
                     prompts_in_frozen30=int(sum(p in frozen for p in ap.unique())),
                     seeds=a.seed.nunique(), seed_values=','.join(map(str, sorted(a.seed.unique())[:4])) + ('…' if a.seed.nunique() > 4 else ''),
                     seq_len=','.join(map(str, sorted(a.seq_len.unique()))), hook_mode=','.join(a.hook_mode.unique()),
                     dup_gen_idx=int(a.groupby('gen_idx').seed.nunique().gt(1).sum()),
                     sigma_first=float(a.sigma.max()), sigma_last=float(a.sigma.min())))
    a = a[a.gid.isin(ok)]
    for sub in subs:
        b = a[a.submodule == sub]
        gids = sorted(b.gid.unique())
        X = {}
        for br in brs:
            q = b[b.branch == br]
            piv = q.pivot_table(index=['gid', 'token_idx'], columns='layer', values=['g_micro', 'spectral_gap', 'sv_entropy']).loc[gids]
            X[br] = np.stack([piv[v].values.reshape(len(gids), ns, L) for v in ['g_micro', 'spectral_gap', 'sv_entropy']], -1)
        q = b[b.branch == brs[0]]
        PI = pd.to_numeric(q.prompt_influence, errors='coerce')
        # prompt_influence = cos(mean_cond, mean_uncond) of the hidden state at each (step, layer); identical on the two branch rows
        PI = q.assign(pi=PI).pivot_table(index=['gid', 'token_idx'], columns='layer', values='pi', dropna=False).loc[gids].values.reshape(len(gids), ns, L)
        meta = b.groupby('gid').agg(pt=('pt', 'first'), ph=('prompt_hash', 'first'), theme=('theme', 'first'), seed=('seed', 'first'),
                                    seq=('seq_len', 'first')).loc[gids]
        sig = b[b.gid == gids[0]].groupby('token_idx').sigma.first().values
        pickle.dump(dict(gids=gids, meta=meta, X=X, PI=PI, sigma=sig), open(f'{S}/R5_{arm}_{sub}.pkl', 'wb'))
S_ = pd.DataFrame(summ)
pd.set_option('display.width', 250)
print(S_.drop(columns=['folder']).to_string())
S_.to_csv(f'{OUT}/R5_intake.csv', index=False)
# prompt membership of the core arms against the frozen 30
core = df[df.arm.isin(['S10_1024_s1001', 'S45_1024_s1001'])].groupby('pt').size().index
missing = sorted(frozen - set(core)); extra = sorted(set(core) - frozen)
open(f'{OUT}/R5_prompt_membership.txt', 'w').write(
    f'core prompts {len(core)}; in frozen 30: {len(set(core) & frozen)}\nmissing from frozen:\n' + '\n'.join(missing) + '\n\nextra:\n' + '\n'.join(extra) + '\n')
print(open(f'{OUT}/R5_prompt_membership.txt').read())
