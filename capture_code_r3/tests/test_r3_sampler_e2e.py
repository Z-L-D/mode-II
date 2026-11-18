# End-to-end sampler test on REAL ComfyUI: model_base.BaseModel + ModelPatcher + comfy.samplers sampling
# through the Passive Telemetry Guider with CFG > 1 and a CSV session, exactly the path a workflow takes.
# Covers what the hook-only tests missed: predict_noise, phi from mean vectors, and the CSV writer.
# Usage: COMFY=/path/to/ComfyUI [NODE_DIR=<node folder>] python tests/test_r3_sampler_e2e.py
import os, sys, csv, math, tempfile, importlib, importlib.util, pathlib
sys.dont_write_bytecode = True
import torch
COMFY = os.environ['COMFY']
HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.environ.get('NODE_DIR', os.path.abspath(os.path.join(HERE, '..')))
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, COMFY)
import comfy.options; comfy.options.enable_args_parsing()
import comfy.supported_models as sm, comfy.model_patcher, comfy.samplers, comfy.sample

def load_nodes(src_dir, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(src_dir, '__init__.py'), submodule_search_locations=[src_dir])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    return importlib.import_module(f'{name}.nodes')
nodes = load_nodes(NODE_DIR, 'cospafs_e2e')
TMP = pathlib.Path(tempfile.mkdtemp()); nodes.TELEMETRY_DIR = TMP

fails = []
def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''), flush=True)
    if not cond: fails.append(name)

def tiny_unet(kind):
    if kind == 'sd15':
        cfg = dict(use_checkpoint=False, image_size=32, use_spatial_transformer=True, legacy=False, adm_in_channels=None,
                   in_channels=4, out_channels=4, model_channels=32, num_res_blocks=[1, 1, 1, 1], channel_mult=[1, 2, 4, 4],
                   transformer_depth=[1, 1, 1, 0], transformer_depth_middle=1, transformer_depth_output=[1, 1, 1, 1, 0, 0, 0, 0],
                   context_dim=48, num_heads=4, use_linear_in_transformer=False)
        mc = sm.SD15(cfg)
    else:
        cfg = dict(use_checkpoint=False, image_size=32, use_spatial_transformer=True, legacy=False, adm_in_channels=2816, num_classes='sequential',
                   in_channels=4, out_channels=4, model_channels=64, num_res_blocks=[1, 1, 1], channel_mult=[1, 2, 4],
                   transformer_depth=[0, 1, 2], transformer_depth_middle=2, transformer_depth_output=[0, 1, 1, 2, 2, 2],
                   context_dim=64, use_linear_in_transformer=True)  # ComfyUI adds num_head_channels 64
        mc = sm.SDXL(cfg)
    model = mc.get_model({}, device=torch.device('cpu'))
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    model.eval()
    return model, cfg['context_dim']

def conds(kind, ctx_dim, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(1, 7, ctx_dim, generator=g)
    extra = {}
    if kind == 'sdxl':
        extra = dict(pooled_output=torch.randn(1, 1280, generator=g), width=256, height=256, crop_w=0, crop_h=0,
                     target_width=256, target_height=256)
    return [[c, extra]]

for kind in ('sd15', 'sdxl'):
    model, ctx = tiny_unet(kind)
    mp = comfy.model_patcher.ModelPatcher(model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
    session = f'e2e_{kind}'
    g = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, conds(kind, ctx, 1), conds(kind, ctx, 2), cfg=4.5,
                                                           csv_session=session, prompt_text='a test prompt')[0]
    latent = torch.zeros(1, 4, 16, 16)
    noise = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(0))
    sigmas = comfy.samplers.calculate_sigmas(mp.get_model_object('model_sampling'), 'normal', 3)
    sampler = comfy.samplers.sampler_object('euler')
    err = None
    try:
        out = g.sample(noise, latent, sampler, sigmas, seed=0, disable_pbar=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    check(f'{kind}: sampling with CFG 4.5 and a CSV session runs without error', err is None, err or '')
    if err:
        continue
    check(f'{kind}: output finite', bool(torch.isfinite(out).all()))
    rows = list(csv.DictReader(open(TMP / f'{session}.csv', encoding='utf-8')))
    steps = sorted({int(r['token_idx']) for r in rows})
    layers = sorted({int(r['layer']) for r in rows})
    check(f'{kind}: CSV has 3 steps', steps == [0, 1, 2], str(steps))
    check(f'{kind}: CSV has both branches', {r['branch'] for r in rows} == {'cond', 'uncond'})
    check(f'{kind}: g_micro finite on every row', all(r['g_micro'] and math.isfinite(float(r['g_micro'])) for r in rows))
    # phi: blank exactly where the 3 consecutive captures do not share one width, finite elsewhere
    bad = []
    for st in steps:
        for br in ('cond', 'uncond'):
            seq = sorted([r for r in rows if int(r['token_idx']) == st and r['branch'] == br],
                         key=lambda r: (int(r['layer']), r['submodule'] != 'attn'))
            for i, r in enumerate(seq):
                if i < 2:
                    continue
                same = len({seq[j]['hidden_dim'] for j in (i - 2, i - 1, i)}) == 1
                if same != (r['phi_micro'] != ''):
                    bad.append((st, br, r['layer'], r['submodule'], r['hidden_dim'], r['phi_micro']))
    n_blank = sum(1 for r in rows if r['phi_micro'] == '')
    check(f'{kind}: phi blank only across a width change, a number everywhere else', not bad, f'{bad[:3]} blanks={n_blank}')
    dims = sorted({int(r['hidden_dim']) for r in rows})
    check(f'{kind}: several block widths captured (the case that crashed)', len(dims) > 1, str(dims))
    print(f'  {kind}: {len(rows)} rows, {len(layers)} blocks, widths {dims}, blank phi rows {n_blank}')

print(f'\n{len(fails)} failed: {fails}' if fails else '\nALL PASS')
sys.exit(1 if fails else 0)
