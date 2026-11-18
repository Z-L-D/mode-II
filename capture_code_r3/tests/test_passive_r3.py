# R3 fix tests: per-branch rows can no longer be switched off, a second branch channel,
# shifted-widget detection, step-0 branch check, gen_len, and hook cleanup after sample().
import sys, importlib, csv, shutil, io, contextlib, torch, torch.nn as nn
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
    def forward(s, x, ctx, transformer_options=None):
        for b in s.transformer_blocks: x = b(x, ctx)
        return x
class Wrap: pass
dm = DM(); mp = Wrap(); mp.model = Wrap(); mp.model.diffusion_model = dm
torch.manual_seed(1); pos = torch.randn(1, N, D); neg = torch.randn(1, N, D) * 0.3; x0 = torch.randn(1, N, D)
sig = torch.linspace(1.0, 0.0, 5)  # 4 steps
fails = []
def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond: fails.append(name)
def run(tag, cfg=4.5, pool=True, wrapper=True, kw=False, **gk):
    cs.POOL, cs.WRAPPER, cs.TO_KWARG = pool, wrapper, kw
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, pos, neg, cfg=cfg, csv_session=tag, prompt_text='p', **gk)[0]
        g.sample(x0.clone(), x0, None, sig, seed=77)
    rows = list(csv.DictReader(open(nodes.TELEMETRY_DIR / f'{tag}.csv')))
    return rows, out.getvalue()
def n_hooks():
    mods = [dm] + list(dm.modules())
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in mods)
shutil.rmtree(nodes.TELEMETRY_DIR, ignore_errors=True)
base_hooks = n_hooks()

R, log = run('ref')
check('default: hook_mode passive-branch', {r['hook_mode'] for r in R} == {'passive-branch'})
check('default: branches cond+uncond', {r['branch'] for r in R} == {'cond', 'uncond'})
check('default: prompt_influence finite on every row', all(r['prompt_influence'] not in ('', 'nan') for r in R))
check('default: gen_len = 4 steps', {r['gen_len'] for r in R} == {'4'})
check('default: seed written', {r['seed'] for r in R} == {'77'})
check('default: branch check OK printed', 'branch check gen 1' in log and 'OK' in log and 'FAILED' not in log)
check('hooks removed after sample()', n_hooks() == base_hooks)

R2, log2 = run('flag_off', split_cfg_branches=False)
check('flag False is ignored (still passive-branch)', {r['hook_mode'] for r in R2} == {'passive-branch'} and {r['branch'] for r in R2} == {'cond', 'uncond'})
check('flag False prints the shift warning', 'split_cfg_branches arrived as False' in log2)
key = lambda r: (r['token_idx'], r['layer'], r['submodule'], r['branch'])
ref = {key(r): r for r in R}
check('flag False rows identical to reference', all(r['g_micro'] == ref[key(r)]['g_micro'] for r in R2))

R3, log3 = run('shifted', layer_range=True, log_telemetry='')
check('shifted widgets detected', 'widget values look shifted' in log3)
check('shifted widgets: all 4 layers captured', len({r['layer'] for r in R3}) == 4)

R4, log4 = run('root_channel', wrapper=False, kw=True)
check('wrapper dropped, transformer_options channel: cond+uncond', {r['branch'] for r in R4} == {'cond', 'uncond'})
check('root channel rows identical to wrapper rows', all(r['g_micro'] == ref[key(r)]['g_micro'] and r['prompt_influence'] == ref[key(r)]['prompt_influence'] for r in R4))

R5, log5 = run('no_channel', wrapper=False, kw=False)
check('no channel at all: rows marked pooled', {r['branch'] for r in R5} == {'pooled'})
check('no channel at all: BRANCH CHECK FAILED printed', 'BRANCH CHECK FAILED' in log5)

R6, log6 = run('separate', pool=False)
check('separate forwards: identical rows', all(r['g_micro'] == ref[key(r)]['g_micro'] for r in R6))
R7, log7 = run('cfg1', cfg=1.0)
check('cfg 1: cond rows only, check OK', {r['branch'] for r in R7} == {'cond'} and 'OK' in log7)

# two guiders sharing one model, sequential (multi-arm workflow)
cs.POOL, cs.WRAPPER, cs.TO_KWARG = True, True, False
with contextlib.redirect_stdout(io.StringIO()):
    gA = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, pos, neg, cfg=4.5, csv_session='armA', prompt_text='p')[0]
    gB = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, pos, neg, cfg=1.0, csv_session='armB', prompt_text='p')[0]
    gA.sample(x0.clone(), x0, None, sig, seed=1); gB.sample(x0.clone(), x0, None, sig, seed=1); gA.sample(x0.clone(), x0, None, sig, seed=2)
A = list(csv.DictReader(open(nodes.TELEMETRY_DIR / 'armA.csv')))
check('multi-arm: no hooks left on the shared model', n_hooks() == base_hooks)
check('multi-arm: arm A has 2 gens, both cond+uncond', {r['gen_idx'] for r in A} == {'1', '2'} and {r['branch'] for r in A} == {'cond', 'uncond'})
print('\n%d failures' % len(fails)); sys.exit(1 if fails else 0)
