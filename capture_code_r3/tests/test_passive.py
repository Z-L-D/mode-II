import sys, importlib, csv, shutil, torch, torch.nn as nn
sys.path.insert(0, '.'); sys.path.insert(0, 'pkg')
import comfy.samplers as cs
nodes = importlib.import_module('cospafs.nodes')
D, N = 32, 48
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.proj = nn.Linear(D, D)
    def forward(s, x): return s.proj(torch.tanh(x))
class Block(nn.Module):
    def __init__(s): super().__init__(); s.attn = Attn(); s.mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
    def forward(s, x, ctx): x = x + s.attn(x + ctx); return x + s.mlp(x)
class DM(nn.Module):
    def __init__(s): super().__init__(); torch.manual_seed(0); s.transformer_blocks = nn.ModuleList([Block() for _ in range(4)])
    def forward(s, x, ctx):
        for b in s.transformer_blocks: x = b(x, ctx)
        return x
class Wrap: pass
dm = DM(); mp = Wrap(); mp.model = Wrap(); mp.model.diffusion_model = dm
torch.manual_seed(1); pos = torch.randn(1, N, D); neg = torch.randn(1, N, D) * 0.3; x0 = torch.randn(1, N, D)
def run(tag, cfg, pool, order=(0, 1), split=True, steps=3):
    cs.POOL, cs.ORDER = pool, order
    g = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, pos, neg, cfg=cfg, csv_session=tag, prompt_text='a test prompt', split_cfg_branches=split)[0]
    x = x0.clone()
    for i, t in enumerate(torch.linspace(1.0, 0.2, steps)):
        x = x - 0.1 * g.predict_noise(x, t.reshape(1), {}, seed=1234)
    return list(csv.DictReader(open(nodes.TELEMETRY_DIR / f'{tag}.csv')))
shutil.rmtree(nodes.TELEMETRY_DIR, ignore_errors=True)
A = run('pooled', 4.5, True); B = run('separate', 4.5, False); C = run('pooled_rev', 4.5, True, (1, 0)); E = run('cfg1', 1.0, True); L = run('legacy', 4.5, True, split=False)
print('header:', list(A[0].keys())[-4:], '| rows/step pooled', sum(r['token_idx'] == '0' for r in A), 'cfg1', sum(r['token_idx'] == '0' for r in E))
key = lambda r: (r['token_idx'], r['layer'], r['submodule'], r['branch'])
def maxdiff(P, Q, col):
    q = {key(r): r for r in Q}
    return max(abs(float(r[col]) - float(q[key(r)][col])) for r in P if r[col] != '')
for col in ['g_micro', 'spectral_gap', 'sv_entropy', 'phi_micro', 'prompt_influence']:
    print(f'{col:16s} pooled vs separate {maxdiff(A, B, col):.2e} | pooled vs reversed order {maxdiff(A, C, col):.2e}')
print('branches', sorted({r['branch'] for r in A}), 'cfg1 branches', sorted({r['branch'] for r in E}), 'legacy branches', sorted({r['branch'] for r in L}))
s0 = lambda R, b: {(r['layer'], r['submodule']): float(r['g_micro']) for r in R if r['token_idx'] == '0' and r['branch'] == b}
a, e = s0(A, 'cond'), s0(E, 'cond'); print('step-0 cond rows identical across CFG 4.5 and 1.0:', max(abs(a[k] - e[k]) for k in a))
print('seed col', A[0]['seed'], '| hook_mode', A[0]['hook_mode'], L[0]['hook_mode'], '| prompt_hash', A[0]['prompt_hash'], '| seq_len cond row', A[0]['seq_len'])
# R3: split_cfg_branches=False is now ignored, so the 'legacy' run is per-branch too
l = {(r['token_idx'], r['layer'], r['submodule']): float(r['g_micro']) for r in L}
print('flag-off run vs cond-branch g (R3: rows no longer pooled):', round(max(abs(l[(r['token_idx'], r['layer'], r['submodule'])] - float(r['g_micro'])) for r in A if r['branch'] == 'cond'), 4))
# reference check: recompute cond-branch g at step 0 layer 0 attn directly
acts = {}
h = dm.transformer_blocks[0].attn.register_forward_hook(lambda m, i, o: acts.setdefault('a', o))
dm(x0, pos); h.remove()
X = acts['a'].reshape(-1, D).double(); ev = torch.linalg.eigvalsh(X.T @ X / X.shape[0]).clamp(min=1e-12); s = ev.sqrt()
ref = (s.sum() ** 2 / ev.sum() / D).item()
print('direct recompute g (step0, L0 attn, cond):', round(ref, 6), 'csv:', [r['g_micro'] for r in A if r['token_idx'] == '0' and r['layer'] == '0' and r['submodule'] == 'attn' and r['branch'] == 'cond'])
