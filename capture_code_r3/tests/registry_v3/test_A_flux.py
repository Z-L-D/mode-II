# Group A_flux registry test against REAL ComfyUI classes:
#   Flux (comfy.ldm.flux.model.Flux), Chroma, ChromaRadiance, HunyuanVideo.
# Usage: COMFY=/path/to/ComfyUI python test_A_flux.py (from capture_code_r3/tests/registry_v3)
# Loads the telemetry node read-only from the node folder two levels up (no bytecode written there),
# tests the entries in the shipped model_registry.json, then per class:
#   1. real-size config(s) on the meta device: block list, paths resolve on every block, D
#   2. tiny config(s), real CPU forward through the node's guider hooks: every double block's
#      attn and MLP hook fire exactly once, captured D == hidden_size, captured rows == image tokens
#   3. single blocks: which paths resolve (none of the registry/fallback paths; linear2 alternative)
import os, sys, json, importlib, importlib.util
sys.dont_write_bytecode = True
import torch
COMFY = os.environ['COMFY']
NODE_DIR = os.environ.get('NODE_DIR', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, COMFY)
import comfy.options; comfy.options.enable_args_parsing()
import comfy.ops
from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from comfy.ldm.chroma.model import Chroma
from comfy.ldm.chroma_radiance.model import ChromaRadiance
from comfy.ldm.hunyuan_video.model import HunyuanVideo
import comfy.model_base as mb

def load_nodes(src_dir, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(src_dir, '__init__.py'),
                                                  submodule_search_locations=[src_dir])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    return importlib.import_module(f'{name}.nodes')
nodes = load_nodes(NODE_DIR, 'cospafs_r3')
ops = comfy.ops.disable_weight_init
# v3.0: the entries under test are the shipped ones, read from the node folder's model_registry.json
_REG = json.load(open(os.path.join(NODE_DIR, 'model_registry.json')))
ENTRIES = {k: _REG[k] for k in ['Flux', 'Chroma', 'ChromaRadiance', 'HunyuanVideo']}
OLD_REG = json.load(open(os.path.join(HERE, '..', 'fixtures', 'model_registry_v2.1.json')))
fails = []
def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''), flush=True)
    if not cond: fails.append(name)

# ---- 0. class mapping in model_base and the current registry ----
check('model_base.Flux2 subclasses model_base.Flux (no own unet_model -> comfy.ldm.flux.model.Flux)',
      issubclass(mb.Flux2, mb.Flux) and '__init__' not in mb.Flux2.__dict__)
check('model_base.LongCatImage subclasses model_base.Flux (same unet_model)',
      issubclass(mb.LongCatImage, mb.Flux) and '__init__' not in mb.LongCatImage.__dict__)
check('ldm Chroma is NOT a subclass of ldm Flux', not issubclass(Chroma, Flux))
check('ldm ChromaRadiance subclasses ldm Chroma (distinct class name)', issubclass(ChromaRadiance, Chroma))
import inspect
src = inspect.getsource(mb)
for cls in ('HunyuanImage21', 'HunyuanVideo'):
    seg = src.split(f'class {cls}(BaseModel)')[1].split('\nclass ')[0]
    check(f'model_base.{cls} uses unet_model=comfy.ldm.hunyuan_video.model.HunyuanVideo',
          'unet_model=comfy.ldm.hunyuan_video.model.HunyuanVideo' in seg)
for cls in ('HunyuanVideoI2V', 'HunyuanVideoSkyreelsI2V', 'HunyuanImage21Refiner', 'HunyuanVideo15', 'HunyuanVideo15_SR_Distilled'):
    c = getattr(mb, cls)
    check(f'model_base.{cls} inherits from model_base.HunyuanVideo or HunyuanImage21',
          issubclass(c, mb.HunyuanVideo) or issubclass(c, mb.HunyuanImage21))
cur = OLD_REG
check('v2.1 model_registry.json had none of the keys Flux/Chroma/ChromaRadiance/HunyuanVideo (why R2 wrote nothing)',
      not any(k in cur for k in ENTRIES), str([k for k in ENTRIES if k in cur]))
check('shipped v3.0 registry has all four keys', all(k in nodes._MODEL_REGISTRY for k in ENTRIES))

class Wrap: pass
def init_(m):
    torch.manual_seed(0)
    for q in m.parameters(): torch.nn.init.normal_(q, std=0.05)
    return m.eval()
def guider_for(dm):
    mp = Wrap(); mp.model = Wrap(); mp.model.diffusion_model = dm; mp.model_options = {}
    mp.is_dynamic = lambda: False
    g = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, [], [], cfg=1.0, csv_session='', prompt_text='t')[0]
    g._log_steps = False
    return g, mp

# v2.1 registry (fixture) vs the shipped v3.0 registry
with torch.device('meta'):
    _f = Flux(image_model='flux', in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
              mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000,
              patch_size=2, qkv_bias=True, guidance_embed=True, txt_ids_dims=[], device='meta', operations=ops)
g, mp = guider_for(_f)
_saved = dict(nodes._MODEL_REGISTRY); nodes._MODEL_REGISTRY.clear()
nodes._MODEL_REGISTRY.update({k: v for k, v in OLD_REG.items() if not k.startswith('_')})
check('v2.1 registry: Flux finds 0 blocks (old diffusers-named key never matches)', len(g._find_blocks(mp.model)) == 0)
nodes._MODEL_REGISTRY.clear(); nodes._MODEL_REGISTRY.update(_saved)
check('v3.0 registry: Flux finds its 19 double blocks', len(g._find_blocks(mp.model)) == 19)
del _f

# inject the new entries
nodes._MODEL_REGISTRY.update(json.loads(json.dumps(ENTRIES)))

def run_hooks(dm, forward):
    g, mp = guider_for(dm)
    blocks = g._find_blocks(mp.model); g._register_hooks(blocks)
    for d in (g._micro_g_per_layer, g._micro_hdim_per_layer, g._micro_seqlen_per_layer): d.clear()
    with torch.no_grad(): out = forward()
    g._remove_hooks()
    per = {}
    for k, v in g._micro_g_per_layer.items():
        per.setdefault((k[0], k[1]), 0); per[(k[0], k[1])] += len(v)
    return g, blocks, per, out

def mlp_out(b):
    for p in ENTRIES_KEY_MLP:
        m = nodes._resolve_dotted_attr(b, p)
        if m is not None: return p, m
    return None, None

def real_check(key, label, dm, depth, D):
    global ENTRIES_KEY_MLP
    ENTRIES_KEY_MLP = ENTRIES[key]['mlp_attrs']['mlp']
    g, mp = guider_for(dm)
    blocks = g._find_blocks(mp.model)
    check(f'{label} real config: registry entry {key} used, {depth} blocks via {ENTRIES[key]["block_path"]!r}',
          g._registry_config is nodes._MODEL_REGISTRY.get(key) and len(blocks) == depth
          and all(b is dm.double_blocks[i] for i, b in enumerate(blocks)), f'found {len(blocks)}')
    a_ok = all(nodes._resolve_dotted_attr(b, ENTRIES[key]['attn_attrs']['self_attn'][0]) is not None for b in blocks)
    m_ok = all(mlp_out(b)[1] is not None for b in blocks)
    dims = {nodes._resolve_dotted_attr(b, 'img_attn.proj').out_features for b in blocks} | {mlp_out(b)[1].out_features for b in blocks}
    check(f'{label} real config: attn + MLP paths resolve on all {depth} blocks, D = {D}', a_ok and m_ok and dims == {D},
          f'mlp path {mlp_out(blocks[0])[0]}; dims {dims}')
    # registration count, as the node does it
    g._register_hooks(blocks)
    n = len(g._hook_handles); g._remove_hooks()
    check(f'{label} real config: node registers {2 * depth} hooks (attn+MLP per double block)', n == 2 * depth, f'{n}')

def tiny_check(key, label, dm, fwd, D, n_tok, batch=1):
    global ENTRIES_KEY_MLP
    ENTRIES_KEY_MLP = ENTRIES[key]['mlp_attrs']['mlp']
    g, blocks, per, out = run_hooks(dm, fwd)
    L = len(blocks)
    check(f'{label} tiny forward: output finite, shape {tuple(out.shape)}', torch.isfinite(out).all().item())
    check(f'{label} tiny forward: attn hook (img_attn.proj) fired once on all {L} double blocks',
          L == len(dm.double_blocks) and all(per.get((i, 0)) == 1 for i in range(L)), str(per))
    check(f'{label} tiny forward: MLP hook ({mlp_out(blocks[0])[0]}) fired once on all {L} double blocks',
          all(per.get((i, 1)) == 1 for i in range(L)))
    hd = set(g._micro_hdim_per_layer.values())
    check(f'{label} tiny forward: captured D == hidden_size {D}', hd == {D}, str(hd))
    sl = {}
    for k, v in g._micro_seqlen_per_layer.items(): sl.setdefault(k[0:2], 0); sl[k[0:2]] += v
    check(f'{label} tiny forward: rows per hook = {batch} x {n_tok} image tokens (no text tokens)',
          set(sl.values()) == {batch * n_tok}, str(set(sl.values())))
    keys = {k[2] for k in g._micro_g_per_layer if len(k) == 3}
    return g, keys

def single_block_report(label, dm, key):
    sb = dm.single_blocks
    if len(sb) == 0:
        check(f'{label}: no single blocks (nothing to report)', True); return
    b = sb[0]
    reg = [p for p in ENTRIES[key]['attn_attrs']['self_attn'] + ENTRIES[key]['mlp_attrs']['mlp']
           if nodes._resolve_dotted_attr(b, p) is not None]
    fb = [a for a in ('attn1', 'self_attn', 'attn', 'attention', 'mlp', 'feed_forward', 'ffn', 'ff') if hasattr(b, a)]
    check(f'{label}: on SingleStreamBlock no registry path and no node fallback attr resolves',
          not reg and not fb, f'children={[n for n, _ in b.named_children()]}')

# alternative: block_path 'double_blocks + single_blocks' with linear2 for single blocks
def alt_single_check(label, dm, fwd, D, key):
    alt = json.loads(json.dumps(ENTRIES[key]))
    alt['block_path'] = 'double_blocks + single_blocks'
    alt['attn_attrs']['self_attn'] = ['img_attn.proj', 'linear2']
    saved = nodes._MODEL_REGISTRY[key]; nodes._MODEL_REGISTRY[key] = alt
    try:
        g, blocks, per, _ = run_hooks(dm, fwd)
    finally:
        nodes._MODEL_REGISTRY[key] = saved
    nd, ns = len(dm.double_blocks), len(dm.single_blocks)
    ok_d = all(per.get((i, 0)) == 1 and per.get((i, 1)) == 1 for i in range(nd))
    ok_s = all(per.get((i, 0)) == 1 and (i, 1) not in per for i in range(nd, nd + ns))
    hd = {k[0]: v for k, v in g._micro_hdim_per_layer.items()}
    check(f'{label} ALTERNATIVE (not in entries) double+single with linear2: single blocks get one fused "attn" row, no MLP row, D={D}',
          len(blocks) == nd + ns and ok_d and ok_s and all(hd[i] == D for i in range(nd + ns)), str(per))

torch.manual_seed(0)
# =========================== Flux ===========================
def flux_kw(**over):
    kw = dict(image_model='flux', in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
              mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000,
              patch_size=2, qkv_bias=True, guidance_embed=True, txt_ids_dims=[])
    kw.update(over); return kw
FLUX2_REAL = dict(image_model='flux2', in_channels=128, out_channels=128, vec_in_dim=None, context_in_dim=15360, hidden_size=6144,
                  mlp_ratio=3.0, num_heads=48, depth=8, depth_single_blocks=48, axes_dim=[32, 32, 32, 32], theta=2000,
                  patch_size=1, qkv_bias=False, guidance_embed=True, txt_ids_dims=[3], global_modulation=True,
                  mlp_silu_act=True, ops_bias=False, default_ref_method='index', ref_index_scale=10.0)
OVIS_REAL = flux_kw(vec_in_dim=None, context_in_dim=2048, depth=6, depth_single_blocks=27, guidance_embed=False,
                    txt_ids_dims=[1, 2], yak_mlp=True, txt_norm=True)
for label, kw, depth, D in (('Flux.1-dev', flux_kw(), 19, 3072), ('Flux.2-dev', FLUX2_REAL, 8, 6144), ('Ovis-Image', OVIS_REAL, 6, 3072)):
    with torch.device('meta'):
        big = Flux(**kw, device='meta', operations=ops)
    real_check('Flux', label, big, depth, D)
    if label == 'Flux.1-dev': single_block_report('Flux real', big, 'Flux')
    del big

# tiny Flux.1-style
D = 64
tiny = init_(Flux(**flux_kw(in_channels=4, out_channels=4, vec_in_dim=8, context_in_dim=16, hidden_size=D, num_heads=4,
                            depth=2, depth_single_blocks=3, axes_dim=[4, 6, 6]), operations=ops))
x = torch.randn(1, 4, 16, 16); t = torch.tensor([0.7]); ctx = torch.randn(1, 7, 16); y = torch.randn(1, 8); gd = torch.tensor([3.5])
fwd = lambda: tiny(x, t, ctx, y=y, guidance=gd)
tiny_check('Flux', 'Flux.1-style', tiny, fwd, D, n_tok=(16 // 2) * (16 // 2))
single_block_report('Flux.1-style tiny', tiny, 'Flux')
alt_single_check('Flux.1-style', tiny, fwd, D, 'Flux')

# tiny Flux.2-style (global modulation, SiLU-gated MLP, no bias), with two reference latents
tiny2 = init_(Flux(image_model='flux2', in_channels=8, out_channels=8, vec_in_dim=None, context_in_dim=16, hidden_size=D,
                   mlp_ratio=3.0, num_heads=4, depth=3, depth_single_blocks=2, axes_dim=[4, 4, 4, 4], theta=2000, patch_size=1,
                   qkv_bias=False, guidance_embed=False, txt_ids_dims=[3], global_modulation=True, mlp_silu_act=True,
                   ops_bias=False, default_ref_method='index', ref_index_scale=10.0, operations=ops))
check('Flux.2-style: double blocks have no per-block img_mod (global modulation)', not hasattr(tiny2.double_blocks[0], 'img_mod'))
check('Flux.2-style: img_mlp is Sequential(Linear, SiLUActivation, Linear)', type(tiny2.double_blocks[0].img_mlp[1]).__name__ == 'SiLUActivation')
x2 = torch.randn(1, 8, 6, 8); ctx2 = torch.randn(1, 5, 16); refs = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 6, 6)]
tiny_check('Flux', 'Flux.2-style + 2 ref_latents', tiny2, lambda: tiny2(x2, torch.tensor([0.5]), ctx2, ref_latents=refs), D,
           n_tok=6 * 8 + 4 * 4 + 6 * 6)
tiny_check('Flux', "Flux.2-style ref 'index_timestep_zero'", tiny2,
           lambda: tiny2(x2, torch.tensor([0.5]), ctx2, ref_latents=refs[:1], ref_latents_method='index_timestep_zero'), D,
           n_tok=6 * 8 + 4 * 4)
single_block_report('Flux.2-style tiny', tiny2, 'Flux')

# tiny Ovis-style (YakMLP -> img_mlp.down_proj)
tiny3 = init_(Flux(**flux_kw(in_channels=4, out_channels=4, vec_in_dim=None, context_in_dim=16, hidden_size=D, num_heads=4,
                             depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], guidance_embed=False, txt_ids_dims=[1, 2],
                             yak_mlp=True, txt_norm=True), operations=ops))
check('Ovis-style: img_mlp.2 absent, img_mlp.down_proj present',
      nodes._resolve_dotted_attr(tiny3.double_blocks[0], 'img_mlp.2') is None and hasattr(tiny3.double_blocks[0].img_mlp, 'down_proj'))
tiny_check('Flux', 'Ovis-style yak_mlp', tiny3, lambda: tiny3(x, t, ctx), D, n_tok=64)

# =========================== Chroma ===========================
CHROMA_REAL = dict(image_model='chroma', in_channels=64, out_channels=64, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0,
                   num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, patch_size=2, qkv_bias=True,
                   in_dim=64, out_dim=3072, hidden_dim=5120, n_layers=5, txt_ids_dims=[], vec_in_dim=None)
with torch.device('meta'):
    big = Chroma(**CHROMA_REAL, device='meta', operations=ops)
check('Chroma real config: modulation layout 3*S + 12*D + 2 == 344 (hard-coded mod_index_length)', 3 * 38 + 12 * 19 + 2 == 344)
real_check('Chroma', 'Chroma', big, 19, 3072)
single_block_report('Chroma real', big, 'Chroma')
del big
ch_kw = dict(CHROMA_REAL, in_channels=16, out_channels=16, context_in_dim=16, hidden_size=D, num_heads=4, depth=2,
             depth_single_blocks=3, axes_dim=[4, 6, 6], out_dim=D, hidden_dim=32, n_layers=2)
tinyc = init_(Chroma(**ch_kw, operations=ops))
xc = torch.randn(2, 4, 16, 16); ctxc = torch.randn(2, 7, 16)
def fwd_c():
    return tinyc(xc, torch.tensor([0.7, 0.7]), ctxc, torch.tensor([0.0, 0.0]))
# batch 2 (cond+uncond as ComfyUI batches them); activations are 3-D [B, L, D]
gC, keys = tiny_check('Chroma', 'Chroma (batch 2)', tinyc, fwd_c, D, n_tok=64, batch=2)
alt_single_check('Chroma', tinyc, fwd_c, D, 'Chroma')
# branch split as the node does it when the model wrapper reports cond_or_uncond = [0, 1]
g, mp = guider_for(tinyc); blocks = g._find_blocks(mp.model); g._register_hooks(blocks)
g._micro_g_per_layer.clear(); g._micro_seqlen_per_layer.clear(); g._cur_branches = [0, 1]
with torch.no_grad(): fwd_c()
g._remove_hooks(); g._cur_branches = None
ks = set(g._micro_g_per_layer)
want = {(i, s, b) for i in range(len(blocks)) for s in (0, 1) for b in ('cond', 'uncond')}
check('Chroma batch 2 with cond_or_uncond [0,1]: every hook row split into cond and uncond, 64 rows each',
      ks == want and set(g._micro_seqlen_per_layer.values()) == {64}, str(sorted(ks)))

# =========================== ChromaRadiance ===========================
RAD_REAL = dict(CHROMA_REAL, image_model='chroma_radiance', in_channels=3, out_channels=3, patch_size=16, nerf_hidden_size=64,
                nerf_mlp_ratio=4, nerf_depth=4, nerf_max_freqs=8, nerf_tile_size=512, nerf_final_head_type='conv',
                nerf_embedder_dtype=torch.float32, use_x0=False, use_sequential_txt_ids=True)
with torch.device('meta'):
    big = ChromaRadiance(**RAD_REAL, device='meta', operations=ops)
real_check('ChromaRadiance', 'ChromaRadiance', big, 19, 3072)
del big
rad_kw = dict(RAD_REAL, context_in_dim=16, hidden_size=D, num_heads=4, depth=2, depth_single_blocks=3, axes_dim=[4, 6, 6],
              out_dim=D, hidden_dim=32, n_layers=2, patch_size=4, nerf_hidden_size=16, nerf_mlp_ratio=2, nerf_depth=2,
              nerf_max_freqs=4, nerf_tile_size=0, nerf_final_head_type='linear', nerf_embedder_dtype=None)
tinyr = init_(ChromaRadiance(**rad_kw, operations=ops))
xr = torch.randn(1, 3, 32, 32)
tiny_check('ChromaRadiance', 'ChromaRadiance', tinyr, lambda: tinyr(xr, torch.tensor([0.7]), ctx, torch.tensor([0.0])), D,
           n_tok=(32 // 4) * (32 // 4))
single_block_report('ChromaRadiance tiny', tinyr, 'ChromaRadiance')

# =========================== HunyuanVideo ===========================
def hv_kw(**over):
    kw = dict(image_model='hunyuan_video', in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
              mlp_ratio=4.0, num_heads=24, depth=20, depth_single_blocks=40, axes_dim=[16, 56, 56], theta=256,
              patch_size=[1, 2, 2], qkv_bias=True, guidance_embed=True, byt5=False, meanflow=False,
              use_cond_type_embedding=False, vision_in_dim=None, meanflow_sum=False)
    kw.update(over); return kw
for label, kw, depth, D_ in (('HunyuanVideo 1.0', hv_kw(), 20, 3072),
                             ('HunyuanImage 2.1', hv_kw(in_channels=64, out_channels=64, vec_in_dim=None, context_in_dim=3584,
                                                         hidden_size=3584, num_heads=28, patch_size=[1, 1], axes_dim=[64, 64],
                                                         byt5=True), 20, 3584),
                             ('HunyuanVideo 1.5', hv_kw(in_channels=65, out_channels=32, vec_in_dim=None, context_in_dim=3584,
                                                         hidden_size=2048, num_heads=16, depth=54, depth_single_blocks=0,
                                                         patch_size=[1, 1, 1], byt5=True, use_cond_type_embedding=True,
                                                         vision_in_dim=1152, meanflow_sum=True), 54, 2048)):
    with torch.device('meta'):
        big = HunyuanVideo(**kw, device='meta', operations=ops)
    real_check('HunyuanVideo', label, big, depth, D_)
    if label == 'HunyuanVideo 1.0': single_block_report('HunyuanVideo real', big, 'HunyuanVideo')
    del big

tinyh = init_(HunyuanVideo(**hv_kw(in_channels=4, out_channels=4, vec_in_dim=8, context_in_dim=16, hidden_size=D, num_heads=4,
                                   depth=2, depth_single_blocks=3, axes_dim=[4, 6, 6]), operations=ops))
xh = torch.randn(1, 4, 3, 8, 8); ctxh = torch.randn(1, 6, 16); mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
fwd_h = lambda: tinyh(xh, torch.tensor([700.0]), ctxh, y=torch.randn(1, 8), guidance=torch.tensor([6.0]), attention_mask=mask)
tiny_check('HunyuanVideo', 'HunyuanVideo 1.0-style (video, txt mask)', tinyh, fwd_h, D, n_tok=3 * 4 * 4)
check('HunyuanVideo: txt_in TokenRefiner blocks are not under double_blocks (not hooked)',
      not any(m is tinyh.txt_in.individual_token_refiner.blocks[0] for m in tinyh.double_blocks))
tiny_check('HunyuanVideo', 'HunyuanVideo I2V guiding_frame_index', tinyh,
           lambda: tinyh(xh, torch.tensor([700.0]), ctxh, y=torch.randn(1, 8), guidance=torch.tensor([6.0]),
                         guiding_frame_index=torch.tensor([0.0])), D, n_tok=3 * 4 * 4)
single_block_report('HunyuanVideo tiny', tinyh, 'HunyuanVideo')
alt_single_check('HunyuanVideo', tinyh, fwd_h, D, 'HunyuanVideo')

tinyi = init_(HunyuanVideo(**hv_kw(in_channels=4, out_channels=4, vec_in_dim=None, context_in_dim=16, hidden_size=D, num_heads=4,
                                   depth=2, depth_single_blocks=2, patch_size=[1, 1], axes_dim=[8, 8], byt5=True), operations=ops))
xi = torch.randn(2, 4, 8, 8)
tiny_check('HunyuanVideo', 'HunyuanImage 2.1-style (2-D, ByT5, batch 2)', tinyi,
           lambda: tinyi(xi, torch.tensor([700.0, 700.0]), torch.randn(2, 6, 16), txt_byt5=torch.randn(2, 5, 1472),
                         guidance=torch.tensor([6.0, 6.0])), D, n_tok=64, batch=2)

tiny15 = init_(HunyuanVideo(**hv_kw(in_channels=9, out_channels=4, vec_in_dim=None, context_in_dim=16, hidden_size=D, num_heads=4,
                                    depth=3, depth_single_blocks=0, patch_size=[1, 1, 1], axes_dim=[4, 6, 6], byt5=True,
                                    use_cond_type_embedding=True, vision_in_dim=32, meanflow_sum=True), operations=ops))
x15 = torch.randn(1, 9, 2, 4, 4); ref = torch.randn(1, 9, 1, 4, 4)
tiny_check('HunyuanVideo', 'HunyuanVideo 1.5-style (0 single, ByT5+vision tokens, ref_latent)', tiny15,
           lambda: tiny15(x15, torch.tensor([700.0]), torch.randn(1, 6, 16), txt_byt5=torch.randn(1, 5, 1472),
                          clip_fea=torch.randn(1, 4, 32), guidance=torch.tensor([6.0]), ref_latent=ref), D,
           n_tok=2 * 4 * 4 + 1 * 4 * 4)
single_block_report('HunyuanVideo 1.5 tiny', tiny15, 'HunyuanVideo')

print(f'\n{len(fails)} failed: {fails}' if fails else '\nALL PASS')
sys.exit(1 if fails else 0)
