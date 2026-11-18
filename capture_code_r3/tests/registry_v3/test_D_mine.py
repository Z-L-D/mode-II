# Real-class hook tests: Ideogram 4, Krea 2, MiniMax H3, Qwen-Image 2.1, and archive identity
# (Z-Image NextDiT, SD3.5 MMDiT, Mochi). Usage: COMFY=/path/to/ComfyUI [NODES_R2=<capture_code_r2 dir>] python test_D_mine.py
import os, sys, json, importlib, importlib.util, torch
COMFY = os.environ['COMFY']; sys.argv = [sys.argv[0], '--cpu']; sys.path.insert(0, COMFY)
import comfy.options; comfy.options.enable_args_parsing(); import comfy.ops
ops = comfy.ops.disable_weight_init
NODES = os.environ.get('NODES', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))); R2 = os.environ.get('NODES_R2')  # NODES_R2 optional: R2 archive-identity checks
def load_nodes(src_dir, name, reg=None):
    spec = importlib.util.spec_from_file_location(name, os.path.join(src_dir, '__init__.py'), submodule_search_locations=[src_dir])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    n = importlib.import_module(f'{name}.nodes')
    if reg:  # test a candidate registry without touching the node folder
        R = json.load(open(reg)); n._MODEL_REGISTRY.clear(); n._MODEL_REGISTRY.update({k: v for k, v in R.items() if not k.startswith('_')})
    return n
nodes = load_nodes(NODES, 'cospafs_r3', os.environ.get('REG'))
fails = []
def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else '')); (fails.append(name) if not cond else None)
def init_(m):
    torch.manual_seed(0)
    for q in m.parameters(): torch.nn.init.normal_(q, std=0.05)
    return m.eval().requires_grad_(False)  # ComfyUI runs inference-only (kitchen in-place RoPE refuses grad)
class Wrap: pass
def guider_for(dm, n=nodes):
    mp = Wrap(); mp.model = Wrap(); mp.model.diffusion_model = dm; mp.model_options = {}; mp.is_dynamic = lambda: False
    g = n.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, [], [], cfg=1.0, csv_session='', prompt_text='t')[0]
    g._log_steps = False
    return g, mp
def run(dm, fwd, n=nodes, branches=None):
    g, mp = guider_for(dm, n); blocks = g._find_blocks(mp.model); g._register_hooks(blocks)
    g._micro_g_per_layer.clear(); g._micro_hdim_per_layer.clear(); g._micro_seqlen_per_layer.clear()
    g._cur_branches = branches
    with torch.no_grad(): fwd()
    g._remove_hooks()
    per = {}
    for k, v in g._micro_g_per_layer.items(): per[(k[0], k[1])] = per.get((k[0], k[1]), 0) + len(v)
    return g, blocks, per
def once_each(per, L, subs=(0, 1)):
    return all(per.get((i, s)) == 1 for i in range(L) for s in subs)
def std_checks(label, key, dm, fwd, D, L, subs=(0, 1)):
    g, blocks, per = run(dm, fwd)
    check(f'{label}: registry entry {key} used', g._registry_config is nodes._MODEL_REGISTRY.get(key) and g._registry_config is not None)
    check(f'{label}: {L} blocks found', len(blocks) == L, f'{len(blocks)}')
    check(f'{label}: attn{"+MLP" if 1 in subs else ""} hooks fire once per block', once_each(per, L, subs), str(per))
    hd = {k[0]: v for k, v in g._micro_hdim_per_layer.items() if k[1] == 0}
    check(f'{label}: captured D = {D}', set(hd.values()) == {D}, str(set(hd.values())))
    return g, per

# ---------- Ideogram 4 ----------
from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
with torch.device('meta'):
    big = Ideogram4Transformer2DModel(device='meta', operations=ops)
g, mp = guider_for(big); bl = g._find_blocks(mp.model)
check('Ideogram 4 real config: 34 layers, D 4608', len(bl) == 34 and bl[0].attention.o.out_features == 4608, f'{len(bl)}')
m = init_(Ideogram4Transformer2DModel(in_channels=128, num_layers=2, num_attention_heads=2, attention_head_dim=16, intermediate_size=64,
                                      adaln_dim=16, llm_features_dim=24, mrope_section=(4, 2, 2), operations=ops))
x = torch.randn(1, 128, 4, 4); ctx = torch.randn(1, 5, 24)
std_checks('Ideogram 4 tiny forward (cond, text+image tokens)', 'Ideogram4Transformer2DModel', m, lambda: m(x, torch.tensor([0.5]), context=ctx), 32, 2)
g, _, _ = run(m, lambda: m(x, torch.tensor([0.5]), context=ctx))
sl_c = {k[0]: v for k, v in g._micro_seqlen_per_layer.items()}
g, _, _ = run(m, lambda: m(x, torch.tensor([0.5]), context=None))
sl_u = {k[0]: v for k, v in g._micro_seqlen_per_layer.items()}
check('Ideogram 4: uncond pass (context=None) has image tokens only (16 vs 21)', set(sl_c.values()) == {21} and set(sl_u.values()) == {16}, f'{sl_c} {sl_u}')

# ---------- Krea 2 ----------
from comfy.ldm.krea2.model import SingleStreamDiT
with torch.device('meta'):
    big = SingleStreamDiT(device='meta', operations=ops)
g, mp = guider_for(big); bl = g._find_blocks(mp.model)
check('Krea 2 real config (class defaults): 28 blocks, D 6144', len(bl) == 28 and bl[0].attn.wo.out_features == 6144, f'{len(bl)}')
m = init_(SingleStreamDiT(features=64, tdim=32, txtdim=16, heads=4, kvheads=2, multiplier=2, layers=2, patch=2, channels=16,
                          txtlayers=2, txtheads=2, txtkvheads=2, operations=ops))
x = torch.randn(1, 16, 8, 8); ctx = torch.randn(1, 5, 2 * 16)
std_checks('Krea 2 tiny forward', 'SingleStreamDiT', m, lambda: m(x, torch.tensor([0.5]), ctx), 64, 2)
check('Krea 2: text-fusion blocks not hooked (only .blocks)', all(b in list(m.blocks) for b in nodes._resolve_block_list(m, 'blocks')))

# ---------- Qwen-Image 2.1 ----------
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
with torch.device('meta'):
    big = QwenImage21Transformer2DModel(device='meta', operations=ops)
g, mp = guider_for(big); bl = g._find_blocks(mp.model)
check('Qwen-Image 2.1 real config: 32 blocks, D 4096', len(bl) == 32 and bl[0].attn.to_out[0].out_features == 4096, f'{len(bl)}')
m = init_(QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=2,
                                        context_in_dim=24, axes_dims_rope=(4, 6, 6), operations=ops))
class _P:  # stand-in patcher for the cache's memory query
    hook_patches = {}
    def get_free_memory(self, dev): return 1 << 40
m.current_patcher = _P()
x = torch.randn(1, 64, 4, 4); ctx = torch.randn(1, 5, 24)
tos_off = nodes._MODEL_REGISTRY['QwenImage21Transformer2DModel'].get('transformer_options_set', {})
check('Qwen-Image 2.1: registry turns the prefix cache off', tos_off.get('qwen_image21_cache', {}).get('device') == 'off', str(tos_off))
def two_steps(to):
    m.reset_prefix_cache(True)
    m(x, torch.tensor([0.9]), ctx, transformer_options=dict(to)); m(x, torch.tensor([0.5]), ctx, transformer_options=dict(to))
g, blocks, per = run(m, lambda: two_steps({}))
sl = sorted({v for k, v in g._micro_seqlen_per_layer.items()})
check('Qwen-Image 2.1 WITHOUT the fix: cache on gives extra prefix-only calls (documents the trap)', any(v != 2 for v in per.values()) or len(sl) > 1, f'{per} seq {sl}')
g, blocks, per = run(m, lambda: two_steps(tos_off))
check('Qwen-Image 2.1 with cache off: attn+MLP fire once per block per forward (2 steps -> 2)', all(v == 2 for v in per.values()) and len(per) == 4, str(per))
check('Qwen-Image 2.1 with cache off: full text+image sequence (5+16=21) every step', set(g._micro_seqlen_per_layer.values()) == {21}, str(g._micro_seqlen_per_layer))
hd = {k[0]: v for k, v in g._micro_hdim_per_layer.items() if k[1] == 0}
check('Qwen-Image 2.1: captured D = 32', set(hd.values()) == {32})

# ---------- MiniMax H3 ----------
from comfy.ldm.minimax.model import MiniMaxH3Model
with torch.device('meta'):
    big = MiniMaxH3Model(device='meta', operations=ops)
g, mp = guider_for(big); bl = g._find_blocks(mp.model)
check('MiniMax H3 real config: 50 blocks, D 5376', len(bl) == 50 and bl[0].attn.out_proj.out_features == 5376, f'{len(bl)}')
m = init_(MiniMaxH3Model(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
                         ffn_hidden_size=96, latents_dim=24, audio_latents_dim=32, text_dim=40, timestep_input_dim=32,
                         time_embed_hidden_size=64, time_embed_dim=48, rope_inv_freq_len=4, operations=ops))
m.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(4).float() / 4)))
blk = m.blocks[0]
def block_fwd():
    S = 10; xh = torch.randn(S, 64); t_emb = torch.randn(1, 48)
    for b in m.blocks: xh = b(xh, t_emb, [(0, S, 0)], None)
g, blocks, per = run(m, block_fwd, branches=[0])
check('MiniMax H3 blocks (2-D packed [tokens, D] input): attn (attn.out_proj) and MLP (mlp) fire once per block', once_each(per, 2), str(per))
check('MiniMax H3: 2-D activation attributed to the cond branch, not cut along tokens', all(k[2] == 'cond' for k in g._micro_g_per_layer) and set(g._micro_seqlen_per_layer.values()) == {10},
      f'{set(k[2] for k in g._micro_g_per_layer)} {set(g._micro_seqlen_per_layer.values())}')
check('MiniMax H3: token refiner not hooked (only .blocks)', all(b in list(m.blocks) for b in blocks))
vid = torch.randn(1, 24, 2, 4, 4); aud = torch.randn(1, 32, 2, 6); ctx = torch.randn(1, 5, 40)
g, blocks, per = run(m, lambda: m([vid, aud], torch.tensor([500.0]), ctx), branches=[0])
check('MiniMax H3 full model forward (text 5 + audio 12 + video 8 rows): hooks fire once per block', once_each(per, 2), str(per))
check('MiniMax H3 full model forward: one packed sequence of 25 rows per hook', set(g._micro_seqlen_per_layer.values()) == {25}, str(set(g._micro_seqlen_per_layer.values())))

# ---------- Archive identity: R3 hooks the same modules as R2 ----------
if R2:
    r2 = load_nodes(R2, 'cospafs_r2')
    def hooked(n, dm):
        g, mp = guider_for(dm, n); bl = g._find_blocks(mp.model); g._register_hooks(bl)
        mods = [(i, type(m_).__name__) for i, b in enumerate(bl) for m_ in b.modules() if m_._forward_hooks]
        ids = [id(m_) for b in bl for m_ in b.modules() if m_._forward_hooks]
        g._remove_hooks(); return ids, mods
    from comfy.ldm.lumina.model import NextDiT
    zi = init_(NextDiT(patch_size=2, in_channels=16, dim=64, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=32,
                       axes_dims=(8, 12, 12), axes_lens=(64, 64, 64), qk_norm=True, z_image_modulation=True, operations=ops))
    from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
    sd = init_(OpenAISignatureMMDITWrapper(patch_size=2, in_channels=16, depth=3, num_patches=64, pos_embed_max_size=8, adm_in_channels=32,
                                           context_embedder_config={'target': 'torch.nn.Linear', 'params': {'in_features': 32, 'out_features': 3 * 64}},
                                           qk_norm='rms', x_block_self_attn_layers=[], operations=ops))
    for lab, dm in (('Z-Image (NextDiT)', zi), ('SD3.5 (MMDiT)', sd)):
        a, ma = hooked(r2, dm); b, mb = hooked(nodes, dm)
        check(f'archive identity {lab}: R3 hooks exactly the R2 modules', a == b, f'R2 {ma[:4]} R3 {mb[:4]}')
    from comfy.ldm.genmo.joint_model.asymm_models_joint import AsymmDiTJoint
    mo = init_(AsymmDiTJoint(depth=2, patch_size=2, num_heads=2, hidden_size_x=64, hidden_size_y=32, mlp_ratio_x=192.0, mlp_ratio_y=4.0, learn_sigma=False,
                             in_channels=12, qk_norm=True, qkv_bias=False, out_bias=True, attn_drop=0.0, patch_embed_bias=True, posenc_preserve_area=True,
                             timestep_mlp_bias=True, attend_to_padding=False, timestep_scale=1000.0, use_t5=True, t5_feat_dim=16, t5_token_length=8,
                             rope_theta=10000.0, operations=ops))
    xm = torch.randn(1, 12, 2, 8, 8); cm = torch.randn(1, 8, 16); mk = torch.ones(1, 8, dtype=torch.bool)
    fm = lambda: mo(xm, torch.tensor([0.5]), cm, mk, num_tokens=8)
    g2, b2, p2 = run(mo, fm, r2); g3, b3, p3 = run(mo, fm, nodes)
    check('Mochi: R3 uses the new AsymmDiTJoint entry', g3._registry_config is nodes._MODEL_REGISTRY.get('AsymmDiTJoint') and g3._registry_config is not None)
    v2 = {k: v for k, v in g2._micro_g_per_layer.items() if k[1] == 0}; v3 = {k: v for k, v in g3._micro_g_per_layer.items() if k[1] == 0}
    check('archive identity Mochi: attn g values identical R2 (fallback attn[0]) vs R3 (attn.proj_x)', v2.keys() == v3.keys() and all(v2[k] == v3[k] for k in v2), f'{list(v2.items())[:2]} {list(v3.items())[:2]}')
    check('archive identity Mochi: MLP still not hooked (archive is attn-only)', not any(k[1] == 1 for k in g3._micro_g_per_layer) and not any(k[1] == 1 for k in g2._micro_g_per_layer))

print(f'\n{len(fails)} failed' if fails else '\nALL PASS'); sys.exit(1 if fails else 0)
