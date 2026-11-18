import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 12 scoring, exactly as frozen in rerun/PREREG_R12.md (sha ad06a1bb, 2026-10-18 06:21:27Z).
# Tangent/shape: Round 6 S1 tangent() unchanged. Null: B's layer order permuted (same permutation at every sigma), 1000 draws, p95.
# Cubes: Z1 = Base primary cond branch (120 prompts x <=4, complete gens), Z3 = Turbo matched cond, Z2 = Base CFG 1 cond,
# ZIT = Turbo-8 archive (480 gens), unrelated = MOCHI, SD35M, SD35L archive attn cubes.
# Usage: SCR=<cube cache> python score_r12.py
import os, sys, json
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
OUTD = f'{_COSPAFS_ROOT}/review/zib_turbo_mixup_2026-10-18/r12/out'
rng = np.random.default_rng(12)
C = {}
def cube(m, sub):
    if (m, sub) not in C: C[(m, sub)] = arch(m, sub)
    return C[(m, sub)]
def tan(a, b, sub, null=True):
    _, _, XA, sa = cube(a, sub); _, _, XB, sb = cube(b, sub)
    tc, sr = tangent(XA, sa, XB, sb)
    p95 = float('nan')
    if null:
        ns = [tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)]
        p95 = float(np.percentile(ns, 95))
    return dict(a=a, b=b, sub=sub, tangent=tc, shape=sr, null_p95=p95)
rows = []
def T(a, b, sub, null=True):
    r = tan(a, b, sub, null); rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
UNREL = ['MOCHI', 'SD35M', 'SD35L']
res = {}
for base in ('Z1', 'Z2'):
    for sub in ('attn', 'mlp'):
        res[(base, 'ZIT', sub)] = T(base, 'ZIT', sub)
for sub in ('attn', 'mlp'):
    res[('Z1', 'Z3', sub)] = T('Z1', 'Z3', sub)
    res[('Z3', 'ZIT', sub)] = T('Z3', 'ZIT', sub, null=False)
    res[('Z1', 'Z1U', sub)] = T('Z1', 'Z1U', sub, null=False)      # descriptive: cond vs uncond of the same run
    if os.path.exists(f"{os.environ.get('SCR', _COSPAFS_CACHE)}/F_BMAY_{sub}.pkl"):  # release: the May workflow-test cubes are not distributed (exploratory only)
        res[('Z1', 'BMAY', sub)] = T('Z1', 'BMAY', sub, null=False)    # descriptive: Base primary vs the May workflow test
    res[('Z1', 'ZIB', sub)] = T('Z1', 'ZIB', sub, null=False)      # descriptive: Base vs the archive "ZIB" (Turbo-25 pooled)
    res[('Z3', 'ZIB', sub)] = T('Z3', 'ZIB', sub, null=False)      # descriptive: Turbo matched vs archive "ZIB" (same weights, pooled)
unrel = {}
for z in ('Z1', 'ZIT', 'Z2', 'Z3'):
    for u in UNREL:
        unrel[(z, u)] = T(z, u, 'attn', null=False)['tangent']
pd.DataFrame(rows).to_csv(f'{OUTD}/r12_tangents.csv', index=False)
def mx(zs): return max(unrel[(z, u)] for z in zs for u in UNREL)
V = []
def verdict(i, test, ok, detail): V.append(dict(id=i, test=test, verdict='SUPPORTED' if ok else 'REFUTED', detail=detail))
a, m = res[('Z1', 'ZIT', 'attn')], res[('Z1', 'ZIT', 'mlp')]
verdict('R12-1', 'Z1 cond <-> Turbo-8 above null p95 (attn and mlp)', a['tangent'] > a['null_p95'] and m['tangent'] > m['null_p95'],
        f"attn {a['tangent']:.3f} vs p95 {a['null_p95']:.3f}; mlp {m['tangent']:.3f} vs p95 {m['null_p95']:.3f}")
verdict('R12-2', 'lineage below same weights: attn <= 0.85, mlp <= 0.87', a['tangent'] <= 0.85 and m['tangent'] <= 0.87,
        f"attn {a['tangent']:.3f}, mlp {m['tangent']:.3f}")
ua = mx(['Z1', 'ZIT'])
verdict('R12-3', 'lineage above unrelated: attn >= max unrelated + 0.30; mlp >= null p95 + 0.30',
        a['tangent'] >= ua + 0.30 and m['tangent'] >= m['null_p95'] + 0.30,
        f"attn {a['tangent']:.3f} vs {ua:.3f}+0.30={ua+0.30:.3f}; mlp {m['tangent']:.3f} vs {m['null_p95']:.3f}+0.30={m['null_p95']+0.30:.3f}")
a3, m3 = res[('Z1', 'Z3', 'attn')], res[('Z1', 'Z3', 'mlp')]
ra, rm = res[('Z3', 'ZIT', 'attn')]['tangent'], res[('Z3', 'ZIT', 'mlp')]['tangent']
u3 = mx(['Z1', 'Z3'])
ok2 = a3['tangent'] <= ra - 0.10 and m3['tangent'] <= rm - 0.10
ok3 = a3['tangent'] >= u3 + 0.30 and m3['tangent'] >= m3['null_p95'] + 0.30
verdict('R12-4', 'Z1 <-> Z3 (identical settings): R12-2 and R12-3 bounds, same-weights ref = Z3<->Turbo-8', ok2 and ok3,
        f"attn {a3['tangent']:.3f} (<= {ra:.3f}-0.10={ra-0.10:.3f}; >= {u3:.3f}+0.30={u3+0.30:.3f}); "
        f"mlp {m3['tangent']:.3f} (<= {rm:.3f}-0.10={rm-0.10:.3f}; >= p95 {m3['null_p95']:.3f}+0.30={m3['null_p95']+0.30:.3f})")
a2, m2 = res[('Z2', 'ZIT', 'attn')], res[('Z2', 'ZIT', 'mlp')]
u2 = mx(['Z2', 'ZIT'])
ok = (a2['tangent'] > a2['null_p95'] and m2['tangent'] > m2['null_p95'] and a2['tangent'] <= 0.85 and m2['tangent'] <= 0.87
      and a2['tangent'] >= u2 + 0.30 and m2['tangent'] >= m2['null_p95'] + 0.30)
verdict('R12-5', 'Z2 (Base CFG 1) <-> Turbo-8: R12-1 to R12-3', ok,
        f"attn {a2['tangent']:.3f} (p95 {a2['null_p95']:.3f}, unrel max {u2:.3f}); mlp {m2['tangent']:.3f} (p95 {m2['null_p95']:.3f})")
V.append(dict(id='R12-6', test='ACE Base <-> ACE Turbo', verdict='NOT SCORED', detail='ACE-Step 1.5 Base not run yet'))
V.append(dict(id='R12-7', test='ACE Base layer-axis crossings', verdict='NOT SCORED', detail='ACE-Step 1.5 Base not run yet'))
pd.DataFrame(V).to_csv(f'{OUTD}/r12_verdicts.csv', index=False)
json.dump({f'{k[0]}|{k[1]}': v for k, v in unrel.items()}, open(f'{OUTD}/r12_unrelated_attn.json', 'w'), indent=1)
print(pd.DataFrame(V).to_string())
