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

Amendment B.2 (2026-11-11, after the first run stopped at R18_ZB6 vs R18_ZIT2 and before any Z-Image value was seen): the
sketches are float16, so an activation above 65504 is stored as inf. A layer whose matrix (token rows or prompt rows) holds a
non-finite value for any prompt at a state is left out of every statistic of that state, in the observed value and in every
null draw alike (its row or column of S is NaN and the summaries skip NaN). The rows record how many layers were left out
(nonfinite_a / nonfinite_b). Results are also written after every pair, and --resume skips pairs already in cka_r18b_pairs.csv.
B.2a (06:55Z, after the second run stopped in eigvalsh on finite data at R18_SD35L vs R18_ZB6): Procrustes is computed from thin
SVDs (same quantity, no eigen-solver), and a layer whose Procrustes step cannot be computed is left out of that statistic only.
"""
import argparse, csv, glob, json, os, re, sys, time
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
            if not np.isfinite(X).all():   # amendment B.2: float16 overflow; the layer is left out (NaN row)
                nn = X.shape[0] * X.shape[0]
                out.append(self.t.full((nn,), float('nan'), device=self.dev) if self.t is not None else np.full(nn, np.nan)); continue
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
    """NaN rows / columns (layers left out under amendment B.2) are skipped; with no NaN this equals the original."""
    S = np.asarray(S, float); r = np.isfinite(S).any(1); c = np.isfinite(S).any(0)
    if not r.any() or not c.any(): return float('nan'), float('nan')
    Sb = S[np.ix_(r, c)]; Sb = np.where(np.isfinite(Sb), Sb, -np.inf)
    best = 0.5 * (Sb.max(1).mean() + Sb.max(0).mean())
    j = [int(round(i / max(La - 1, 1) * (Lb - 1))) for i in range(La)]
    v = [S[i, j[i]] for i in range(La) if np.isfinite(S[i, j[i]])]
    matched = float(np.mean(v)) if v else float('nan')
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
        if not np.isfinite(X).all(): A.append(None); continue
        X = X - X.mean(0, keepdims=True); A.append(hsic_u_parts(X @ X.T))
    for Y in Pb:
        if not np.isfinite(Y).all(): B.append(None); continue
        Y = Y - Y.mean(0, keepdims=True); K = Y @ Y.T
        if perm is not None: K = K[np.ix_(perm, perm)]
        B.append(hsic_u_parts(K))
    sa = [hsic_u(*a, *a) if a is not None else np.nan for a in A]; sb = [hsic_u(*b, *b) if b is not None else np.nan for b in B]
    S = np.full((len(A), len(B)), np.nan)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if a is None or b is None: continue
            S[i, j] = hsic_u(*a, *b) / np.sqrt(max(sa[i] * sb[j], 1e-30))
    return S


def procrustes_matrix(Pa, Pb):
    """Nuclear norm of Xa^T Xb (X centred, unit Frobenius norm). Amendment B.2: computed from thin SVDs (Xa = U s V^T, so
    |Xa^T Xb|_* = |diag(s) U^T Xb|_*), which equals the frozen trace((Ka^1/2 Kb Ka^1/2)^1/2) form but does not need an
    eigen-solver that can fail to converge; a layer that is non-finite, constant, or whose SVD fails is left out (NaN)."""
    def prep(X):
        if not np.isfinite(X).all(): return None
        X = X.astype(np.float64); X = X - X.mean(0, keepdims=True); nrm = np.linalg.norm(X)
        if not np.isfinite(nrm) or nrm == 0: return None
        return X / nrm
    def left(X):
        try: U, s_, _ = np.linalg.svd(X, full_matrices=False); return (U * s_).T
        except np.linalg.LinAlgError: return None
    A = [prep(X) for X in Pa]; B = [prep(Y) for Y in Pb]; CA = [left(X) if X is not None else None for X in A]
    S = np.full((len(A), len(B)), np.nan)
    for i, Ca in enumerate(CA):
        for j, Xb in enumerate(B):
            if Ca is None or Xb is None: continue
            try: S[i, j] = np.linalg.svd(Ca @ Xb, compute_uv=False).sum()
            except np.linalg.LinAlgError: pass
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
    nonfin = {}
    for sub in SUBS:
        obs = {}; nul = {}
        for st in STATES:
            if not all((st, p) in ia and (st, p) in ib for p in prompts): continue
            La_idx, Pa, Ta = load_state(ia, st, prompts, sub); Lb_idx, Pb, Tb = load_state(ib, st, prompts, sub)
            if not La_idx or not Lb_idx: continue
            La, Lb = len(La_idx), len(Lb_idx); G2 = Ta[0].shape[0] // n
            nfa = [l for l, x, y in zip(La_idx, Pa, Ta) if not (np.isfinite(x).all() and np.isfinite(y).all())]
            nfb = [l for l, x, y in zip(Lb_idx, Pb, Tb) if not (np.isfinite(x).all() and np.isfinite(y).all())]
            for l, x, y in zip(La_idx, Pa, Ta):   # a layer is left out of both views if either view is non-finite
                if l in nfa: x[...] = np.nan; y[...] = np.nan
            for l, x, y in zip(Lb_idx, Pb, Tb):
                if l in nfb: x[...] = np.nan; y[...] = np.nan
            if nfa or nfb: print(f'    {sub} {st}: non-finite layers left out: {a} {nfa}, {b} {nfb}', flush=True)
            nonfin[(sub, st)] = (len(nfa), len(nfb))
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
                rows_v.append(dict(a=a, b=b, kind=kind_of(a, b), sub=sub, state=st, stat=stat, summary=summ, value=vals[-1], La=La, Lb=Lb, n_prompts=n,
                                   nonfinite_a=nonfin[(sub, st)][0], nonfinite_b=nonfin[(sub, st)][1]))
        for (stat, summ), vals in obs.items():
            nf = [v for (s_, _), v in nonfin.items() if s_ == sub]
            r = dict(a=a, b=b, kind=kind_of(a, b), sub=sub, stat=stat, summary=summ, value=float(np.nanmean(vals)), n_states=len(vals), n_prompts=n,
                     La=La, Lb=Lb, depth_matched=La == Lb, primary=(stat, summ) == ('token', 'best'),
                     nonfinite_a=max((x[0] for x in nf), default=0), nonfinite_b=max((x[1] for x in nf), default=0))
            if (stat, summ) in nul:
                draws = np.nanmean(np.array(nul[(stat, summ)], float), axis=0)   # mean over states, per permutation
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
    ap.add_argument('--resume', action='store_true', help='skip pairs already in cka_r18b_pairs.csv (keeps their rows)')
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
    def write():
        for name, rows in (('cka_r18b_values.csv', V), ('cka_r18b_pairs.csv', P)):
            if not rows: continue
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(args.out, name), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    done = set()
    if args.resume:
        for name, rows in (('cka_r18b_values.csv', V), ('cka_r18b_pairs.csv', P)):
            fp = os.path.join(args.out, name)
            if os.path.exists(fp): rows += list(csv.DictReader(open(fp, encoding='utf-8')))
        done = {(r['a'], r['b']) for r in P}; print(f'resume: {len(done)} pairs already done')
    for a, b in pairs:
        if (a, b) in done: continue
        t0 = time.time(); v, p = score_pair(idx, a, b, args, be); V += v; P += p
        prim = [r for r in p if r['primary']]
        print(f'{a} vs {b} ({kind_of(a, b)}): ' + '; '.join(f'{r["sub"]} {r["value"]:.3f} (p95 {r.get("null_p95", float("nan")):.3f})' for r in prim) + f'  [{time.time() - t0:.0f} s]', flush=True)
        write()
    write()
    print(f'wrote cka_r18b_values.csv, cka_r18b_pairs.csv and cka_r18b_matrices_*.npz to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
