#!/usr/bin/env python3
"""cka_r18b.py: activation-based baseline for the Round 18 pairs (preregistered in review/r18/PREREG_R18_ADDENDUM_B.md).

Reads the activation sketches the capture node writes with act_sketch on (dm_telemetry/<session>_sketch/g<gen>_s<step>.npz:
per attn / mlp hook of the cond branch, the token mean over image rows and 6 x 6 image tokens at fixed image positions, at
step 0 and the steps nearest flow time 0.75, 0.5, 0.25) and compares every pair of sessions on their common prompts.

Statistics (per pair, submodule and sampling state; states matched by label):
  token  linear CKA on token rows (prompts x 36 tokens), Gram form <Kc, Lc> / (|Kc| |Lc|)          [primary]
  prompt linear CKA on prompt rows (token means), and its debiased form (unbiased HSIC, Song et al. 2012)
  procr  orthogonal Procrustes similarity on prompt rows: nuclear norm of X^T Y, X and Y centred, unit Frobenius norm
Layer summaries:
  best     symmetric best match, (mean_i max_j S_ij + mean_j max_i S_ij) / 2: no layer correspondence assumed  [primary]
  matched  mean over the layers of the first model of S at the layer of the second model nearest in normalised depth
The primary value of a pair is token / best, averaged over the four states. Null: the second model's prompts permuted
(token rows move with their prompt), the same permutation for every layer, state and statistic; --perm_token draws for
the token statistics (default 200), --perm_prompt for the prompt ones (default 1000); p = (k + 1) / (N + 1).
Procrustes is reported without a null.

    python cka_r18b.py "D:\\ComfyUI\\custom_nodes\\comfyui-cos-pafs\\dm_telemetry"            # all preregistered pairs
    python cka_r18b.py <telemetry dir> --pairs R18S_SD3M:R18S_SD35M --perm_token 50          # a quick look

Writes cka_r18b_values.csv (one row per pair x submodule x state x statistic x summary), cka_r18b_pairs.csv (primary values,
nulls and p per pair and submodule) and cka_r18b_matrices.npz (layer x layer matrices) next to this file, or to --out.
Needs numpy; uses torch on the GPU when it is available (the token-level null is about 200 x 8 Gram products per pair).
"""
import argparse, csv, glob, json, os, re, sys, time
import numpy as np

STATES = ('first', 't0.75', 't0.5', 't0.25')
SUBS = ('attn', 'mlp')
# kind of each preregistered pair (Addendum B); unrelated = different developers
FAM = {'R18_SD35L': 'sd3', 'R18_SD35LT': 'sd3', 'R18S_SD3M': 'sd3', 'R18S_SD35M': 'sd3', 'R18S_SD35M1': 'sd3',
       'R18_ZIT2': 'z', 'R18_ZB6': 'z', 'R18S_PIXART': 'pixart', 'R18S_COSMOS2B': 'cosmos'}
RELATED = {('R18S_SD35M', 'R18S_SD35M1'): 'same weights (CFG 4.5 vs 1)',
           ('R18_SD35L', 'R18_SD35LT'): 'distillation, documented teacher-student',
           ('R18S_SD3M', 'R18S_SD35M'): 'successor lineage',
           ('R18_ZB6', 'R18_ZIT2'): 'sibling lineage'}


def prereg_pairs():
    P = list(RELATED)
    main = [s for s in FAM if s != 'R18S_SD35M1']
    for i, a in enumerate(main):
        for b in main[i + 1:]:
            if (a, b) in RELATED or (b, a) in RELATED: continue
            P.append((a, b))
    return P


def kind_of(a, b):
    if (a, b) in RELATED: return RELATED[(a, b)]
    if (b, a) in RELATED: return RELATED[(b, a)]
    return 'unrelated' if FAM[a] != FAM[b] else 'same vendor, relation undocumented'


# ------------------------------------------------------------------ loading
def sketch_files(tel, session):
    dirs = sorted(d for d in glob.glob(os.path.join(tel, f'{session}*_sketch')) if re.fullmatch(re.escape(session) + r'(_sv|_rd|_sv_rd)?_sketch', os.path.basename(d)))
    if not dirs: raise SystemExit(f'no sketch directory for {session} under {tel}')
    idx = {}   # (state, prompt_hash) -> (gen_idx, path); the last generation of a prompt wins (resumed runs)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, 'g*_s*.npz'))):
            with np.load(f) as z: m = json.loads(str(z['meta']))
            for lab in m['state'].split('+'):
                k = (lab, m['prompt_hash'])
                if k not in idx or m['gen_idx'] >= idx[k][0]: idx[k] = (m['gen_idx'], f, m)
    return idx


def layers_of(path, sub):
    with np.load(path) as z: return sorted(int(k[1:4]) for k in z.files if k.endswith(f'_{sub}_mean'))


def load_state(idx, state, prompts, sub):
    """Prompt-row and token-row matrices per layer for one state: lists of [n, D] and [n * G^2, D] (float32)."""
    paths = [idx[(state, p)][1] for p in prompts]
    L = layers_of(paths[0], sub); M = {l: [] for l in L}; T = {l: [] for l in L}
    for f in paths:
        with np.load(f) as z:
            for l in L:
                M[l].append(z[f'L{l:03d}_{sub}_mean'].astype(np.float32)); T[l].append(z[f'L{l:03d}_{sub}_tok'].astype(np.float32))
    return L, [np.stack(M[l]) for l in L], [np.concatenate(T[l]) for l in L]


# ------------------------------------------------------------------ statistics
class Back:
    def __init__(self, use_torch):
        self.t = None
        if use_torch:
            try:
                import torch
                self.t = torch; self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.t = None
    def gram_stack(self, mats):
        """Centred Gram matrices with unit Frobenius norm, flattened: [L, n * n]."""
        out = []
        for X in mats:
            X = X - X.mean(0, keepdims=True)
            if self.t is not None:
                Xt = self.t.from_numpy(X).to(self.dev); K = Xt @ Xt.T; K = K / K.norm(); out.append(K.reshape(-1))
            else:
                K = X @ X.T; K /= np.linalg.norm(K); out.append(K.reshape(-1))
        return self.t.stack(out) if self.t is not None else np.stack(out)
    def cross(self, A, B, perm_idx=None):
        """S[i, j] = <A_i, B_j permuted>; perm_idx a flat index map of the row/column permutation."""
        if self.t is not None:
            Bp = B if perm_idx is None else B[:, perm_idx]
            return (A @ Bp.T).float().cpu().numpy()
        Bp = B if perm_idx is None else B[:, perm_idx]
        return A @ Bp.T
    def flat_perm(self, rows):
        n = len(rows); f = (rows[:, None] * n + rows[None, :]).reshape(-1)
        return self.t.from_numpy(f).to(self.dev) if self.t is not None else f


def summaries(S, La, Lb):
    best = 0.5 * (S.max(1).mean() + S.max(0).mean())
    j = [int(round(i / max(La - 1, 1) * (Lb - 1))) for i in range(La)]
    matched = float(np.mean([S[i, j[i]] for i in range(La)]))
    return float(best), matched


def hsic_u_parts(K):
    """Pieces of the unbiased HSIC estimator for a Gram matrix with zeroed diagonal."""
    Kt = K.copy(); np.fill_diagonal(Kt, 0.0); return Kt, Kt.sum(), Kt.sum(1)


def hsic_u(Kt, sK, rK, Lt, sL, rL):
    n = Kt.shape[0]
    return ((Kt * Lt).sum() + sK * sL / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * (rK @ rL)) / (n * (n - 3))


def debiased_matrix(Pa, Pb, perm=None):
    A = []; B = []
    for X in Pa:
        X = X - X.mean(0, keepdims=True); A.append(hsic_u_parts(X @ X.T))
    for Y in Pb:
        Y = Y - Y.mean(0, keepdims=True); K = Y @ Y.T
        if perm is not None: K = K[np.ix_(perm, perm)]
        B.append(hsic_u_parts(K))
    sa = [hsic_u(*a, *a) for a in A]; sb = [hsic_u(*b, *b) for b in B]
    S = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            S[i, j] = hsic_u(*a, *b) / np.sqrt(max(sa[i] * sb[j], 1e-30))
    return S


def procrustes_matrix(Pa, Pb):
    def prep(X):
        X = X.astype(np.float64); X = X - X.mean(0, keepdims=True); X = X / np.linalg.norm(X); K = X @ X.T
        w, V = np.linalg.eigh(K); return K, (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    A = [prep(X) for X in Pa]; B = [prep(Y) for Y in Pb]
    S = np.zeros((len(A), len(B)))
    for i, (_, Ra) in enumerate(A):
        for j, (Kb, _) in enumerate(B):
            S[i, j] = np.sqrt(np.clip(np.linalg.eigvalsh(Ra @ Kb @ Ra), 0, None)).sum()
    return S


# ------------------------------------------------------------------ one pair
def score_pair(tel_idx, a, b, args, be, rng_seed=18):
    ia, ib = tel_idx[a], tel_idx[b]
    rows_v, rows_p, mats = [], [], {}
    pa = {p for (s, p) in ia if s == 'first'}; pb = {p for (s, p) in ib if s == 'first'}
    prompts = sorted(pa & pb)
    if len(prompts) < args.min_prompts:
        print(f'  {a} vs {b}: {len(prompts)} common prompts (< {args.min_prompts}); skipped'); return [], []
    n = len(prompts); rng = np.random.default_rng(rng_seed)
    perms_t = [rng.permutation(n) for _ in range(args.perm_token)]
    perms_p = perms_t + [rng.permutation(n) for _ in range(max(args.perm_prompt - args.perm_token, 0))]
    for sub in SUBS:
        obs = {}; nul = {}
        for st in STATES:
            if not all((st, p) in ia and (st, p) in ib for p in prompts): continue
            La_idx, Pa, Ta = load_state(ia, st, prompts, sub); Lb_idx, Pb, Tb = load_state(ib, st, prompts, sub)
            if not La_idx or not Lb_idx: continue
            La, Lb = len(La_idx), len(Lb_idx); G2 = Ta[0].shape[0] // n
            # token level (primary)
            KA = be.gram_stack(Ta); KB = be.gram_stack(Tb)
            S = be.cross(KA, KB); mats[f'{a}|{b}|{sub}|{st}|token'] = S
            obs.setdefault(('token', 'best'), []).append(summaries(S, La, Lb)[0]); obs.setdefault(('token', 'matched'), []).append(summaries(S, La, Lb)[1])
            nb, nm = [], []
            for pi in perms_t:
                rows = (pi[:, None] * G2 + np.arange(G2)[None, :]).reshape(-1)
                Sp = be.cross(KA, KB, be.flat_perm(rows)); x = summaries(Sp, La, Lb); nb.append(x[0]); nm.append(x[1])
            nul.setdefault(('token', 'best'), []).append(nb); nul.setdefault(('token', 'matched'), []).append(nm)
            del KA, KB
            # prompt level: linear, debiased, Procrustes
            PA = be.gram_stack(Pa); PB = be.gram_stack(Pb); S = be.cross(PA, PB); mats[f'{a}|{b}|{sub}|{st}|prompt'] = S
            for k, v in zip(('best', 'matched'), summaries(S, La, Lb)): obs.setdefault(('prompt', k), []).append(v)
            nb, nm = [], []
            for pi in perms_p:
                x = summaries(be.cross(PA, PB, be.flat_perm(pi)), La, Lb); nb.append(x[0]); nm.append(x[1])
            nul.setdefault(('prompt', 'best'), []).append(nb); nul.setdefault(('prompt', 'matched'), []).append(nm)
            S = debiased_matrix(Pa, Pb); mats[f'{a}|{b}|{sub}|{st}|prompt_debiased'] = S
            for k, v in zip(('best', 'matched'), summaries(S, La, Lb)): obs.setdefault(('prompt_debiased', k), []).append(v)
            nb, nm = [], []
            for pi in perms_p[:args.perm_debiased]:
                x = summaries(debiased_matrix(Pa, Pb, pi), La, Lb); nb.append(x[0]); nm.append(x[1])
            nul.setdefault(('prompt_debiased', 'best'), []).append(nb); nul.setdefault(('prompt_debiased', 'matched'), []).append(nm)
            S = procrustes_matrix(Pa, Pb); mats[f'{a}|{b}|{sub}|{st}|procrustes'] = S
            for k, v in zip(('best', 'matched'), summaries(S, La, Lb)): obs.setdefault(('procrustes', k), []).append(v)
            for (stat, summ), vals in obs.items():
                rows_v.append(dict(a=a, b=b, kind=kind_of(a, b), sub=sub, state=st, stat=stat, summary=summ, value=vals[-1], La=La, Lb=Lb, n_prompts=n))
        for (stat, summ), vals in obs.items():
            r = dict(a=a, b=b, kind=kind_of(a, b), sub=sub, stat=stat, summary=summ, value=float(np.mean(vals)), n_states=len(vals), n_prompts=n,
                     La=La, Lb=Lb, depth_matched=La == Lb, primary=(stat, summ) == ('token', 'best'))
            if (stat, summ) in nul:
                draws = np.mean(np.array(nul[(stat, summ)]), axis=0)   # mean over states, per permutation
                r.update(null_p95=float(np.percentile(draws, 95)), null_mean=float(draws.mean()), n_perm=len(draws),
                         p=(int((draws >= r['value']).sum()) + 1) / (len(draws) + 1))
            rows_p.append(r)
    np.savez_compressed(os.path.join(args.out, f'cka_r18b_matrices_{a}__{b}.npz'), **{k.replace('|', '__'): v for k, v in mats.items()})
    return rows_v, rows_p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('telemetry'); ap.add_argument('--pairs', default='')
    ap.add_argument('--perm_token', type=int, default=200); ap.add_argument('--perm_prompt', type=int, default=1000)
    ap.add_argument('--perm_debiased', type=int, default=200); ap.add_argument('--min_prompts', type=int, default=30)
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__))); ap.add_argument('--no_torch', action='store_true')
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    pairs = [tuple(p.split(':')) for p in args.pairs.split(',') if p] or prereg_pairs()
    for a, b in pairs:
        for s in (a, b): FAM.setdefault(s, s)
    be = Back(not args.no_torch)
    print(f'backend: {"torch " + be.dev if be.t is not None else "numpy"}; {len(pairs)} pairs')
    sessions = sorted({s for p in pairs for s in p}); idx = {}
    for s in sessions:
        idx[s] = sketch_files(args.telemetry, s); print(f'  {s}: {len({p for (_, p) in idx[s]})} prompts, states {sorted({st for (st, _) in idx[s]})}')
    V, P = [], []
    for a, b in pairs:
        t0 = time.time(); v, p = score_pair(idx, a, b, args, be); V += v; P += p
        prim = [r for r in p if r['primary']]
        print(f'{a} vs {b} ({kind_of(a, b)}): ' + '; '.join(f'{r["sub"]} {r["value"]:.3f} (p95 {r.get("null_p95", float("nan")):.3f})' for r in prim) + f'  [{time.time() - t0:.0f} s]', flush=True)
    for name, rows in (('cka_r18b_values.csv', V), ('cka_r18b_pairs.csv', P)):
        if not rows: continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(args.out, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f'wrote cka_r18b_values.csv, cka_r18b_pairs.csv and cka_r18b_matrices_*.npz to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
