# Group C_image: registry entries checked against REAL ComfyUI classes (58b176f).
# Usage: COMFY=/path/to/ComfyUI python test_C_image.py
# Loads the telemetry node read-only from the node folder two levels up and tests the entries in
# its shipped model_registry.json, then per class:
#   1. real-size config on the meta device: block count, registry entry used, every listed path
#      resolves on every block, hidden dim;
#   2. tiny config on CPU, one (or more) real forward: attn and MLP hooks fire exactly once per
#      block and the captured D equals the block width.
import os, sys
sys.dont_write_bytecode = True   # do not write __pycache__ into the read-only node folder
import json, importlib, importlib.util, torch
COMFY = os.environ['COMFY']
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, COMFY)
import comfy.options; comfy.options.enable_args_parsing()
import comfy.ops
import comfy.model_detection
ops = comfy.ops.disable_weight_init

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.environ.get('NODE_DIR', os.path.abspath(os.path.join(HERE, '..', '..')))

def load_nodes(src_dir, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(src_dir, '__init__.py'),
                                                  submodule_search_locations=[src_dir])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    return importlib.import_module(f'{name}.nodes')
nodes = load_nodes(NODE_DIR, 'cospafs_r3')
# v3.0: the entries under test are the shipped ones, read from the node folder's model_registry.json
_REG = json.load(open(os.path.join(NODE_DIR, 'model_registry.json')))
ENTRIES = {k: _REG[k] for k in ['HiDreamImageTransformer2DModel', 'HiDreamO1Transformer', 'OmniGen2Transformer2DModel', 'Kandinsky5', 'Anima', 'QwenImageTransformer2DModel', 'ErnieImageModel']}
OLD_REG = json.load(open(os.path.join(HERE, '..', 'fixtures', 'model_registry_v2.1.json')))
ORIG_QWEN = dict(OLD_REG.get('QwenImageTransformer2DModel', {}))
nodes._MODEL_REGISTRY.update(ENTRIES)   # in-memory only; nothing on disk is changed

fails = []
def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''), flush=True)
    if not cond: fails.append(name)

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

def resolve(block, path):
    return nodes._resolve_dotted_attr(block, path)

def static_checks(label, key, dm, n_exp, D_exp):
    e = ENTRIES[key]
    g, mp = guider_for(dm)
    blocks = g._find_blocks(mp.model)
    check(f'{label} real config: {n_exp} blocks via {e["block_path"]!r}', len(blocks) == n_exp, f'found {len(blocks)}')
    check(f'{label} real config: registry entry {key} used', g._registry_config is nodes._MODEL_REGISTRY.get(key))
    a = [resolve(b, e['attn_attrs']['self_attn'][0]) for b in blocks]
    m = [resolve(b, e['mlp_attrs']['mlp'][0]) for b in blocks]
    check(f'{label} real config: attn path resolves on every block', all(x is not None for x in a))
    check(f'{label} real config: mlp path resolves on every block', all(x is not None for x in m))
    outs = {x.out_features for x in a if hasattr(x, 'out_features')}
    check(f'{label} real config: attn out_features = {D_exp}', outs == {D_exp}, str(outs))
    return blocks

def run_hooks(dm, forwards):
    """Register the node's hooks, run each forward in turn; return per-forward counts."""
    g, mp = guider_for(dm)
    blocks = g._find_blocks(mp.model); g._register_hooks(blocks)
    results = []
    for fwd in forwards:
        for d in (g._micro_g_per_layer, g._micro_hdim_per_layer, g._micro_seqlen_per_layer): d.clear()
        with torch.no_grad(): out = fwd()
        per = {}
        for k, v in g._micro_g_per_layer.items():
            per[(k[0], k[1])] = per.get((k[0], k[1]), 0) + len(v)
        hd = {(k[0], k[1]): v for k, v in g._micro_hdim_per_layer.items()}
        sl = {(k[0], k[1]): v for k, v in g._micro_seqlen_per_layer.items()}
        results.append((per, hd, sl, out))
    n_hooks = len(g._hook_handles)
    g._remove_hooks()
    return g, blocks, results, n_hooks

def forward_checks(label, dm, forwards, D, n_blocks, names=None):
    g, blocks, results, n_hooks = run_hooks(dm, forwards)
    check(f'{label} tiny: {n_blocks} blocks, {2 * n_blocks} hooks registered (dual hook active)',
          len(blocks) == n_blocks and n_hooks == 2 * n_blocks and g._dual_hook_active, f'blocks {len(blocks)}, hooks {n_hooks}')
    for j, (per, hd, sl, out) in enumerate(results):
        tag = names[j] if names else f'forward {j}'
        check(f'{label} tiny {tag}: attn hook fired once on all {n_blocks} blocks', all(per.get((i, 0)) == 1 for i in range(n_blocks)), str(per))
        check(f'{label} tiny {tag}: MLP hook fired once on all {n_blocks} blocks', all(per.get((i, 1)) == 1 for i in range(n_blocks)))
        check(f'{label} tiny {tag}: captured D = {D} for attn and MLP', all(hd.get((i, s)) == D for i in range(n_blocks) for s in (0, 1)), str(set(hd.values())))
        check(f'{label} tiny {tag}: output finite', torch.isfinite(out).all().item())
    return blocks, results

def call_order(blocks, mods, fwd):
    order = []
    hs = [m.register_forward_hook(lambda mm, i, o, k=k: order.append(k)) for k, m in enumerate(mods)]
    with torch.no_grad(): fwd()
    for h in hs: h.remove()
    return order

# =====================================================================================
# HiDream-I1
# =====================================================================================
from comfy.ldm.hidream.model import HiDreamImageTransformer2DModel
hid_cfg = dict(attention_head_dim=128, axes_dims_rope=[64, 32, 32], caption_channels=[4096, 4096], max_resolution=[128, 128],
               in_channels=16, llama_layers=list(range(32)) + [31] * 16, num_attention_heads=20, num_routed_experts=4,
               num_activated_experts=2, num_layers=16, num_single_layers=32, out_channels=16, patch_size=2, text_emb_dim=2048)
with torch.device('meta'):
    big = HiDreamImageTransformer2DModel(**hid_cfg, device='meta', operations=ops)
blocks = static_checks('HiDream-I1', 'HiDreamImageTransformer2DModel', big, 48, 2560)
check('HiDream-I1 real config: blocks 0-15 double-stream, 16-47 single-stream',
      all(type(b.block).__name__ == 'HiDreamImageTransformerBlock' for b in blocks[:16]) and
      all(type(b.block).__name__ == 'HiDreamImageSingleTransformerBlock' for b in blocks[16:]))
check('HiDream-I1 real config: ff_i is the MoE module on every block', all(type(b.block.ff_i).__name__ == 'MOEFeedForwardSwiGLU' for b in blocks))
del big

D = 32
tiny = init_(HiDreamImageTransformer2DModel(patch_size=2, in_channels=4, out_channels=4, num_layers=2, num_single_layers=3,
                                             attention_head_dim=16, num_attention_heads=2, caption_channels=[24, 24], text_emb_dim=8,
                                             num_routed_experts=4, num_activated_experts=2, axes_dims_rope=(8, 4, 4),
                                             max_resolution=(32, 32), llama_layers=[0, 1, 2, 3, 3], operations=ops))
x = torch.randn(1, 4, 16, 16); t = torch.tensor([0.5]); y = torch.randn(1, 8)
t5 = torch.randn(1, 5, 24); llama = torch.randn(1, 4, 6, 24)
fwd = lambda: tiny(x, t, y=y, context=t5, encoder_hidden_states_llama3=llama)
x2 = torch.randn(2, 4, 16, 16)
fwd_b2 = lambda: tiny(x2, torch.tensor([0.5, 0.5]), y=y.repeat(2, 1), context=t5.repeat(2, 1, 1), encoder_hidden_states_llama3=llama.repeat(2, 1, 1, 1))
blocks, res = forward_checks('HiDream-I1', tiny, [fwd, fwd_b2], D, 5, ['B=1', 'B=2'])
sl = res[0][2]
n_img = 64; n_txt0 = 5 + 6  # T5 + last llama layer
check('HiDream-I1 tiny: double-stream rows = image tokens only (64)', all(sl[(i, s)] == n_img for i in range(2) for s in (0, 1)), str(sl))
check('HiDream-I1 tiny: single-stream rows = image + T5 + llama + per-block llama tokens (64+5+6+6)',
      all(sl[(i, s)] == n_img + n_txt0 + 6 for i in range(2, 5) for s in (0, 1)))
check('HiDream-I1 tiny: B=2 rows = 2x B=1 rows', all(res[1][2][k] == 2 * v for k, v in sl.items()))
order = call_order(blocks, [b.block.attn1.to_out for b in blocks], fwd)
check('HiDream-I1 tiny: block list (double + single) is in forward order', order == list(range(5)), str(order))
# MoE: the expert w2 Linears fire a variable number of times, which is why the whole ff_i is hooked
cnt = {'w2': 0}
hs = [e.w2.register_forward_hook(lambda *a: cnt.__setitem__('w2', cnt['w2'] + 1)) for e in blocks[0].block.ff_i.experts]
with torch.no_grad(): fwd()
for h in hs: h.remove()
check('HiDream-I1 tiny: MoE expert w2 calls per block != 1 (so ff_i, not a w2, is hooked)', cnt['w2'] != 1, f'expert w2 calls in block 0: {cnt["w2"]}')
del tiny

# =====================================================================================
# HiDream-O1
# =====================================================================================
from comfy.ldm.hidream_o1.model import HiDreamO1Transformer, TMS_TOKEN_ID
from comfy.ldm.hidream_o1.conditioning import build_extra_conds
with torch.device('meta'):
    big = HiDreamO1Transformer(device='meta', operations=ops)
blocks = static_checks('HiDream-O1', 'HiDreamO1Transformer', big, 36, 4096)
check('HiDream-O1 real config: GQA 32 q / 8 kv heads, mlp not merged',
      blocks[0].self_attn.num_heads == 32 and blocks[0].self_attn.num_kv_heads == 8 and not blocks[0].mlp.merged_mlp)
del big

D = 64
tiny = init_(HiDreamO1Transformer(operations=ops,
             text_config_overrides=dict(hidden_size=D, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=16, rope_dims=[4, 2, 2]),
             vision_config_overrides=dict(hidden_size=32, num_heads=2, intermediate_size=64, depth=1, num_position_embeddings=16)))
x = torch.randn(1, 3, 128, 96)  # 4 x 3 = 12 pixel-patch tokens
ids = torch.randint(0, 1000, (1, 8)); ids[0, -1] = TMS_TOKEN_ID
conds = build_extra_conds(ids, x)
T_total = ids.shape[1] + 12; ar = conds['ar_len']
fwd = lambda: tiny(x, torch.tensor([700.0]), **conds)
tiny.clear_kv_cache()
blocks, res = forward_checks('HiDream-O1', tiny, [fwd, fwd], D, 3, ['cold (cache miss)', 'hot (KV cache hit)'])
check('HiDream-O1 tiny: cold path rows = full sequence (text + image)', all(v == T_total for v in res[0][2].values()), f'{set(res[0][2].values())} vs {T_total}')
check('HiDream-O1 tiny: hot path rows = sequence minus cached text prefix (T - ar_len)', all(v == T_total - ar for v in res[1][2].values()), f'{set(res[1][2].values())} vs {T_total - ar}')
check('HiDream-O1 tiny: hot and cold outputs agree', torch.allclose(res[0][3], res[1][3], atol=1e-3), f'max diff {(res[0][3] - res[1][3]).abs().max().item():.2e}')
del tiny

# =====================================================================================
# OmniGen2
# =====================================================================================
from comfy.ldm.omnigen.omnigen2 import OmniGen2Transformer2DModel
og_cfg = dict(axes_dim_rope=[40, 40, 40], axes_lens=[1024, 1664, 1664], ffn_dim_multiplier=None, hidden_size=2520, in_channels=16,
              multiple_of=256, norm_eps=1e-05, num_attention_heads=21, num_kv_heads=7, num_layers=32, num_refiner_layers=2,
              out_channels=None, patch_size=2, text_feat_dim=2048, timestep_scale=1000.0)
with torch.device('meta'):
    big = OmniGen2Transformer2DModel(**og_cfg, device='meta', operations=ops)
blocks = static_checks('OmniGen2', 'OmniGen2Transformer2DModel', big, 32, 2520)
check('OmniGen2 real config: GQA 21 / 7 heads, FFN inner 10240', blocks[0].attn.heads == 21 and blocks[0].attn.kv_heads == 7
      and blocks[0].feed_forward.linear_2.in_features == 10240)
del big

D = 64
tiny = init_(OmniGen2Transformer2DModel(patch_size=2, in_channels=4, hidden_size=D, num_layers=3, num_refiner_layers=2,
                                         num_attention_heads=4, num_kv_heads=2, multiple_of=16, axes_dim_rope=(4, 6, 6),
                                         axes_lens=(64, 64, 64), text_feat_dim=24, timestep_scale=1000.0, operations=ops))
x = torch.randn(1, 4, 16, 16); ctx = torch.randn(1, 7, 24); ref = [torch.randn(1, 4, 8, 16)]
fwd_t2i = lambda: tiny(x, torch.tensor([0.5]), ctx, num_tokens=7)
fwd_edit = lambda: tiny(x, torch.tensor([0.5]), ctx, num_tokens=7, ref_latents=ref)
blocks, res = forward_checks('OmniGen2', tiny, [fwd_t2i, fwd_edit], D, 3, ['t2i', 'with ref image'])
check('OmniGen2 tiny t2i: rows = text 7 + image 64', all(v == 7 + 64 for v in res[0][2].values()), str(set(res[0][2].values())))
check('OmniGen2 tiny edit: rows = text 7 + ref 32 + image 64', all(v == 7 + 32 + 64 for v in res[1][2].values()), str(set(res[1][2].values())))
del tiny

# =====================================================================================
# Kandinsky 5
# =====================================================================================
from comfy.ldm.kandinsky5.model import Kandinsky5
for lbl, kcfg, D_exp in (('Kandinsky5 lite-video (class default)', {}, 1792),
                          ('Kandinsky5 image-lite', dict(model_dim=2560, visual_embed_dim=64, axes_dims=(32, 48, 48),
                                                         rope_scale_factor=(1.0, 1.0, 1.0)), 2560)):
    with torch.device('meta'):
        big = Kandinsky5(**kcfg, device='meta', operations=ops)
    static_checks(lbl, 'Kandinsky5', big, 32, D_exp)
    del big

D = 64
tiny = init_(Kandinsky5(in_visual_dim=4, out_visual_dim=4, in_text_dim=32, in_text_dim2=16, time_dim=32, model_dim=D, ff_dim=128,
                        visual_embed_dim=16, patch_size=(1, 2, 2), num_text_blocks=2, num_visual_blocks=3, axes_dims=(4, 6, 6),
                        operations=ops))
ctx = torch.randn(1, 7, 32); yp = torch.randn(1, 16)
x_img = torch.randn(1, 4, 16, 16)                 # image: 64 tokens
x_vid = torch.randn(1, 4, 3, 16, 16)              # video: 3 x 64 = 192 tokens
x_big = torch.randn(1, 4, 184, 184)               # 92 x 92 = 8464 tokens > 8192 -> chunked FFN path
fwds = [lambda: tiny(x_img, torch.tensor([500.0]), ctx, yp),
        lambda: tiny(x_vid, torch.tensor([500.0]), ctx, yp),
        lambda: tiny(x_big, torch.tensor([500.0]), ctx, yp)]
blocks, res = forward_checks('Kandinsky5', tiny, fwds, D, 3, ['image 64 tok', 'video 192 tok', 'image 8464 tok (chunked)'])
check('Kandinsky5 tiny: rows = visual tokens only (64 / 192 / 8464)',
      [set(r[2].values()) for r in res] == [{64}, {192}, {8464}], str([set(r[2].values()) for r in res]))
cnt = {'n': 0}
h = blocks[0].feed_forward.out_layer.register_forward_hook(lambda *a: cnt.__setitem__('n', cnt['n'] + 1))
with torch.no_grad(): fwds[2]()
h.remove()
check('Kandinsky5 tiny: >8192 tokens -> feed_forward.out_layer fires 4x per block (why the module is hooked)', cnt['n'] == 4, f'{cnt["n"]} calls')
del tiny

# =====================================================================================
# Anima
# =====================================================================================
from comfy.ldm.anima.model import Anima
an_cfg = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
              concat_padding_mask=True, model_channels=2048, num_blocks=28, num_heads=16, crossattn_emb_channels=1024,
              pos_emb_cls='rope3d', pos_emb_learnable=True, pos_emb_interpolation='crop', min_fps=1, max_fps=30,
              use_adaln_lora=True, adaln_lora_dim=256, extra_per_block_abs_pos_emb=False, rope_h_extrapolation_ratio=4.0,
              rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0, extra_h_extrapolation_ratio=1.0,
              extra_w_extrapolation_ratio=1.0, extra_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=False)
with torch.device('meta'):
    big = Anima(**an_cfg, device='meta', operations=ops)
static_checks('Anima (2048 / 28 assumed)', 'Anima', big, 28, 2048)
check('Anima real config: auto-detect would NOT pick llm_adapter.blocks (registry path is the DiT)',
      len(big.llm_adapter.blocks) == 6 and big.blocks[0] is not big.llm_adapter.blocks[0])
del big

D = 64
cfg_t = dict(an_cfg, max_img_h=64, max_img_w=64, max_frames=8, in_channels=4, out_channels=4, model_channels=D, num_blocks=3,
             num_heads=4, adaln_lora_dim=16)
tiny = init_(Anima(**cfg_t, operations=ops))
x = torch.randn(1, 4, 1, 16, 16)                  # 8 x 8 = 64 tokens
qwen_states = torch.randn(1, 9, 1024); t5_ids = torch.randint(0, 32128, (1, 12))
fwd = lambda: tiny(x, torch.tensor([0.5]), qwen_states, t5xxl_ids=t5_ids)
blocks, res = forward_checks('Anima', tiny, [fwd], D, 3, ['image'])
check('Anima tiny: rows = image tokens only (64)', set(res[0][2].values()) == {64}, str(set(res[0][2].values())))
del tiny

# =====================================================================================
# Qwen-Image
# =====================================================================================
from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
with torch.device('meta'):
    big = QwenImageTransformer2DModel(device='meta', operations=ops)
static_checks('Qwen-Image', 'QwenImageTransformer2DModel', big, 60, 3072)
del big

D = 32
tiny = init_(QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4, num_layers=3, attention_head_dim=16,
                                          num_attention_heads=2, joint_attention_dim=24, axes_dims_rope=(4, 6, 6), operations=ops))
x = torch.randn(1, 4, 1, 16, 16); ctx = torch.randn(1, 7, 24); ref = [torch.randn(1, 4, 1, 16, 16)]
fwds = [lambda: tiny(x, torch.tensor([0.5]), ctx),
        lambda: tiny(x, torch.tensor([0.5]), ctx, ref_latents=ref),
        lambda: tiny(x, torch.tensor([0.5]), ctx, ref_latents=ref, ref_latents_method='index_timestep_zero')]
blocks, res = forward_checks('Qwen-Image', tiny, fwds, D, 3, ['t2i', 'edit (1 ref)', 'edit 2511 timestep-zero'])
check('Qwen-Image tiny: rows = image-stream tokens only (64; 128 with one ref)',
      [set(r[2].values()) for r in res] == [{64}, {128}, {128}], str([set(r[2].values()) for r in res]))
# which MLP the node hooks under the OLD entry vs the new one
g, mp = guider_for(tiny)
g._find_blocks(mp.model); g._register_hooks(g._find_blocks(mp.model))
new_mlp_hooked = [bool(b.img_mlp.net[2]._forward_hooks) for b in tiny.transformer_blocks]
txt_hooked = [bool(b.txt_mlp.net[2]._forward_hooks) for b in tiny.transformer_blocks]
g._remove_hooks()
check('Qwen-Image: new entry hooks img_mlp.net.2 on every block, never txt_mlp', all(new_mlp_hooked) and not any(txt_hooked))
old = dict(ORIG_QWEN)
nodes._MODEL_REGISTRY['QwenImageTransformer2DModel'] = old
g, mp = guider_for(tiny); g._register_hooks(g._find_blocks(mp.model))
n_old = len(g._hook_handles); g._remove_hooks()
check('Qwen-Image: OLD entry (both mlp paths, dual_hook_supported false) registers attn hooks only', n_old == 3, f'{n_old} hooks')
old2 = dict(old, dual_hook_supported=True)
nodes._MODEL_REGISTRY['QwenImageTransformer2DModel'] = old2
g, mp = guider_for(tiny); g._register_hooks(g._find_blocks(mp.model))
img_h = [bool(b.img_mlp.net[2]._forward_hooks) for b in tiny.transformer_blocks]
txt_h = [bool(b.txt_mlp.net[2]._forward_hooks) for b in tiny.transformer_blocks]
g._remove_hooks()
check('Qwen-Image: OLD mlp list with dual on -> only the first path (img_mlp.net.2) is hooked, txt_mlp never',
      all(img_h) and not any(txt_h))
nodes._MODEL_REGISTRY['QwenImageTransformer2DModel'] = ENTRIES['QwenImageTransformer2DModel']
del tiny

# =====================================================================================
# ERNIE Image
# =====================================================================================
from comfy.ldm.ernie.model import ErnieImageModel
with torch.device('meta'):
    big = ErnieImageModel(device='meta', operations=ops)
static_checks('ERNIE-Image', 'ErnieImageModel', big, 36, 4096)
del big

D = 64
tiny = init_(ErnieImageModel(hidden_size=D, num_attention_heads=4, num_layers=3, ffn_hidden_size=128, in_channels=8, out_channels=8,
                             patch_size=1, text_in_dim=24, rope_axes_dim=(4, 6, 6), operations=ops))
x = torch.randn(1, 8, 8, 8); ctx = torch.randn(1, 5, 24)
fwd = lambda: tiny(x, torch.tensor([500.0]), ctx)
fwd_b2 = lambda: tiny(x.repeat(2, 1, 1, 1), torch.tensor([500.0, 500.0]), ctx.repeat(2, 1, 1))
blocks, res = forward_checks('ERNIE-Image', tiny, [fwd, fwd_b2], D, 3, ['B=1', 'B=2'])
check('ERNIE-Image tiny: rows = image 64 + text 5 (B=1), doubled for B=2',
      set(res[0][2].values()) == {69} and set(res[1][2].values()) == {138}, str([set(r[2].values()) for r in res]))
del tiny

print(f'\n{len(fails)} failed' if fails else '\nALL PASS')
sys.exit(1 if fails else 0)
