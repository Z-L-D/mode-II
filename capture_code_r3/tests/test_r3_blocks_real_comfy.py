# R3 block-resolution test against REAL ComfyUI model classes (not the mock).
# Usage: COMFY=/path/to/ComfyUI python tests/test_r3_blocks_real_comfy.py   (run from capture_code_r3/)
# Needs a ComfyUI checkout importable on CPU. Checked on ComfyUI master 58b176f (2026-10-08).
#
# What it checks, per architecture:
#   1. Real-size config (built on the meta device): the Passive guider finds the expected
#      number of blocks and the registry entry is the one used.
#   2. Tiny config with the same structure, real forward on CPU: every block's attn hook and
#      MLP hook fire exactly once per forward, and the captured hidden dim matches the block.
# R2 nodes.py finds 0 blocks for UNetModel and AceStepConditionGenerationModel (checked at the end).
import os, sys, importlib, json, torch
COMFY = os.environ['COMFY']
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, COMFY)
import comfy.options; comfy.options.enable_args_parsing()
import comfy.ops
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.ace.ace_step15 import AceStepConditionGenerationModel
from comfy.ldm.ace.model import ACEStepTransformer2DModel

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
def load_nodes(src_dir, name):
    # import <src_dir> as package <name> (no symlinks: the project share does not support them)
    spec = importlib.util.spec_from_file_location(name, os.path.join(src_dir, '__init__.py'),
                                                  submodule_search_locations=[src_dir])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    return importlib.import_module(f'{name}.nodes')
nodes = load_nodes(os.path.dirname(HERE), 'cospafs_r3')
ops = comfy.ops.disable_weight_init
fails = []
def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''))
    if not cond: fails.append(name)

class Wrap: pass
def init_(m):
    # comfy ops skip weight init (torch.empty), so fill tiny models with small random weights
    torch.manual_seed(0)
    for q in m.parameters(): torch.nn.init.normal_(q, std=0.05)
    return m.eval()
def guider_for(dm, nodes_mod=nodes):
    mp = Wrap(); mp.model = Wrap(); mp.model.diffusion_model = dm; mp.model_options = {}
    mp.is_dynamic = lambda: False
    g = nodes_mod.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, [], [], cfg=1.0, csv_session='', prompt_text='t')[0]
    g._log_steps = False
    return g, mp

def run_hooks(dm, forward):
    g, mp = guider_for(dm)
    blocks = g._find_blocks(mp.model); g._register_hooks(blocks)
    for d in (g._micro_g_per_layer, g._micro_hdim_per_layer): d.clear()
    with torch.no_grad(): forward()
    g._remove_hooks()
    per = {}
    for k, v in g._micro_g_per_layer.items():
        per.setdefault((k[0], k[1]), 0); per[(k[0], k[1])] += len(v)
    return g, blocks, per

def unet_cfg(kind, mc, ctx, heads):
    if kind == 'sd15':
        return dict(image_size=32, in_channels=4, out_channels=4, model_channels=mc, num_res_blocks=[2, 2, 2, 2],
                    channel_mult=[1, 2, 4, 4], transformer_depth=[1, 1, 1, 1, 1, 1, 0, 0], transformer_depth_middle=1,
                    transformer_depth_output=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], context_dim=ctx, num_heads=heads,
                    use_spatial_transformer=True, use_linear_in_transformer=False, legacy=False)
    return dict(image_size=32, in_channels=4, out_channels=4, model_channels=mc, num_res_blocks=[2, 2, 2],
                channel_mult=[1, 2, 4], transformer_depth=[0, 0, 2, 2, 10, 10], transformer_depth_middle=10,
                transformer_depth_output=[0, 0, 0, 2, 2, 2, 10, 10, 10], context_dim=ctx, num_head_channels=64 if mc >= 320 else 8,
                adm_in_channels=None, use_spatial_transformer=True, use_linear_in_transformer=True, legacy=False)

# ---------------- UNet: SD1.5 and SDXL ----------------
for kind, n_exp, dims_exp in (('sd15', 16, {320, 640, 1280}), ('sdxl', 70, {640, 1280})):
    with torch.device('meta'):
        big = UNetModel(**unet_cfg(kind, 320, 768 if kind == 'sd15' else 2048, 8), device='meta', operations=ops)
    g, mp = guider_for(big)
    blocks = g._find_blocks(mp.model)
    dims = [b.attn1.to_out[0].out_features for b in blocks]
    check(f'{kind} real config: {n_exp} blocks found', len(blocks) == n_exp, f'found {len(blocks)}')
    check(f'{kind} real config: registry entry UNetModel used', g._registry_config is not None and g._registry_config is nodes._MODEL_REGISTRY.get('UNetModel'))
    check(f'{kind} real config: hidden dims {sorted(dims_exp)}', set(dims) == dims_exp, f'{sorted(set(dims))}; per block {dims}')

    torch.manual_seed(0)
    tiny = init_(UNetModel(**unet_cfg(kind, 32, 32, 2), operations=ops))
    x = torch.randn(1, 4, 32, 32); t = torch.tensor([500.0]); c = torch.randn(1, 7, 32)
    g, blocks, per = run_hooks(tiny, lambda: tiny(x, t, context=c))
    L = len(blocks)
    check(f'{kind} tiny forward: attn hook fired once on all {L} blocks', all(per.get((i, 0)) == 1 for i in range(L)), str(per))
    check(f'{kind} tiny forward: MLP hook fired once on all {L} blocks', all(per.get((i, 1)) == 1 for i in range(L)))
    hd = {k[0]: v for k, v in g._micro_hdim_per_layer.items() if k[1] == 0}
    check(f'{kind} tiny forward: captured D equals block width', all(hd[i] == blocks[i].attn1.to_out[0].out_features for i in range(L)))
    # forward order: record call order of attn1 and compare with list order
    order = []
    hs = [b.attn1.register_forward_hook(lambda m, i, o, k=k: order.append(k)) for k, b in enumerate(blocks)]
    with torch.no_grad(): tiny(x, t, context=c)
    for h in hs: h.remove()
    check(f'{kind} tiny forward: block list is in forward (depth) order', order == list(range(L)))

# ---------------- ACE-Step 1.5 ----------------
# FSQ calls .item() at init, so the meta device fails; build on CPU in bf16 without weight init (~4 GB)
big = AceStepConditionGenerationModel(dtype=torch.bfloat16, operations=ops)
g, mp = guider_for(big)
blocks = g._find_blocks(mp.model)
check('ACE-Step 1.5 real config: 24 DiT layers via decoder.layers', len(blocks) == 24 and blocks[0] is big.decoder.layers[0], f'found {len(blocks)}')
check('ACE-Step 1.5 real config: registry entry used', g._registry_config is nodes._MODEL_REGISTRY.get('AceStepConditionGenerationModel'))
check('ACE-Step 1.5 real config: D = 2048', blocks[0].self_attn.o_proj.out_features == 2048)
check('ACE-Step 1.5 real config: even layers sliding-window, odd full',
      all((b.self_attn.sliding_window is not None) == (i % 2 == 0) for i, b in enumerate(blocks)))

torch.manual_seed(0)
D = 64
tiny = init_(AceStepConditionGenerationModel(in_channels=24, hidden_size=D, text_hidden_dim=32, timbre_hidden_dim=8, audio_acoustic_hidden_dim=8,
                                       num_dit_layers=4, num_lyric_layers=1, num_timbre_layers=1, num_tokenizer_layers=1, num_heads=4,
                                       num_kv_heads=2, head_dim=16, intermediate_size=128, fsq_dim=D, encoder_hidden_size=D,
                                       encoder_intermediate_size=128, encoder_num_heads=4, sliding_window=4, operations=ops))
Tn = 40
hs_ = torch.randn(1, Tn, 8); ctx_lat = torch.randn(1, Tn, 16); enc = torch.randn(1, 5, D)
fwd = lambda: tiny.decoder(hidden_states=hs_, timestep=torch.tensor([0.5]), timestep_r=torch.tensor([0.5]), attention_mask=None,
                           encoder_hidden_states=enc, encoder_attention_mask=None, context_latents=ctx_lat)
g, blocks, per = run_hooks(tiny, fwd)
check('ACE-Step 1.5 tiny forward: attn (self_attn.o_proj) once per layer', all(per.get((i, 0)) == 1 for i in range(4)), str(per))
check('ACE-Step 1.5 tiny forward: MLP (mlp.down_proj) once per layer', all(per.get((i, 1)) == 1 for i in range(4)))
sl = {k[0]: v for k, v in g._micro_seqlen_per_layer.items() if k[1] == 0}
check('ACE-Step 1.5 tiny forward: seq len = frames / patch 2', all(v == Tn // 2 for v in sl.values()), str(sl))

# ---------------- ACE-Step v1 ----------------
with torch.device('meta'):
    big = ACEStepTransformer2DModel(device='meta', operations=ops)
g, mp = guider_for(big)
blocks = g._find_blocks(mp.model)
check('ACE-Step v1 real config: 28 blocks via transformer_blocks', len(blocks) == 28, f'found {len(blocks)}')
check('ACE-Step v1 real config: registry entry used', g._registry_config is nodes._MODEL_REGISTRY.get('ACEStepTransformer2DModel'))
check('ACE-Step v1 real config: D = 1536', blocks[0].attn.to_out[0].out_features == 1536)
# v1 forward needs the lyric/speaker encoders; hook firing is checked on a block directly.
torch.manual_seed(0)
tiny = init_(ACEStepTransformer2DModel(num_layers=2, inner_dim=32, attention_head_dim=16, num_attention_heads=2, text_embedding_dim=16,
                                 speaker_embedding_dim=8, lyric_hidden_size=32, ssl_names=[], operations=ops)) \
    if 'ssl_names' in ACEStepTransformer2DModel.__init__.__code__.co_varnames else None
if tiny is not None:
    g, mp = guider_for(tiny); blocks = g._find_blocks(mp.model); g._register_hooks(blocks)
    b = blocks[0]; Tn = 12
    xin = torch.randn(1, Tn, 32); enc = torch.randn(1, 5, 32)
    temb = torch.randn(1, 6 * 32)
    from comfy.ldm.ace.model import Qwen2RotaryEmbedding
    rot = Qwen2RotaryEmbedding(dim=16, max_position_embeddings=64, base=1e6)
    freqs = rot(xin, seq_len=Tn); freqs_c = rot(enc, seq_len=5)
    with torch.no_grad():
        b(xin, encoder_hidden_states=enc, rotary_freqs_cis=freqs, rotary_freqs_cis_cross=freqs_c, temb=temb)
    g._remove_hooks()
    per = {(k[0], k[1]): len(v) for k, v in g._micro_g_per_layer.items()}
    check('ACE-Step v1 tiny block: attn (attn.to_out.0) and MLP (ff) fire once', per.get((0, 0)) == 1 and per.get((0, 1)) == 1, str(per))

# ---------------- Regression: R2 finds nothing on these ----------------
r2 = load_nodes(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'capture_code_r2'), 'cospafs_r2')
for label, dm in (('SD1.5', UNetModel(**unet_cfg('sd15', 32, 32, 2), operations=ops)),
                  ('ACE-Step 1.5', AceStepConditionGenerationModel(in_channels=24, hidden_size=64, text_hidden_dim=32, timbre_hidden_dim=8,
                        audio_acoustic_hidden_dim=8, num_dit_layers=2, num_lyric_layers=1, num_timbre_layers=1, num_tokenizer_layers=1,
                        num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128, fsq_dim=64, encoder_hidden_size=64,
                        encoder_intermediate_size=128, encoder_num_heads=4, operations=ops))):
    g, mp = guider_for(dm, r2)
    check(f'R2 nodes.py on {label}: 0 blocks (the reported failure, reproduced)', len(g._find_blocks(mp.model)) == 0)

print(f'\n{len(fails)} failed' if fails else '\nALL PASS')
sys.exit(1 if fails else 0)
