# Group B (video DiTs) registry test against REAL ComfyUI classes.
# Usage: COMFY=/path/to/ComfyUI python test_B_video.py      (checked on ComfyUI master 58b176f, 2026-10-08)
#
# Loads the R3 telemetry node read-only from the node folder two levels up and tests the entries
# in its shipped model_registry.json, then per class:
#   1. real-size config on the meta device: guider finds the expected blocks, registry entry is used, D matches
#   2. tiny config, real forward on CPU: every attn hook and MLP hook fires exactly once per block, captured D
#      equals block width, block list is in forward order
#   3. the linear_input_act trap (Wan ffn.2 / LTX ff.net.2) reproduced by forcing the fused path
import os, sys, json, contextlib, importlib, importlib.util, traceback
import torch
import torch.nn.functional as F

COMFY = os.environ['COMFY']
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, COMFY)
import comfy.options; comfy.options.enable_args_parsing()
import comfy.ops
import comfy.quant_ops

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.environ.get('NODE_DIR', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))


def load_nodes(src_dir, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(src_dir, '__init__.py'),
                                                  submodule_search_locations=[src_dir])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    return importlib.import_module(f'{name}.nodes')


nodes = load_nodes(NODE_DIR, 'cospafs_r3')
# v3.0: the entries under test are the shipped ones, read from the node folder's model_registry.json
_REG = json.load(open(os.path.join(NODE_DIR, 'model_registry.json')))
ENTRIES = {k: _REG[k] for k in ['WanModel', 'VaceWanModel', 'CameraWanModel', 'HumoWanModel', 'WanModel_S2V', 'AnimateWanModel', 'SCAILWanModel', 'SCAIL2WanModel', 'LTXVModel', 'LTXAVModel', 'CogVideoXTransformer3DModel', 'GeneralDIT', 'MiniTrainDIT']}
OLD_REG = json.load(open(os.path.join(HERE, '..', 'fixtures', 'model_registry_v2.1.json')))
for k, v in ENTRIES.items():           # in-memory only; the node's model_registry.json is not touched
    nodes._MODEL_REGISTRY[k] = v

ops = comfy.ops.disable_weight_init
fails = []


def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{extra}]' if extra else ''), flush=True)
    if not cond:
        fails.append(name)


class Wrap: pass


def init_(m):
    torch.manual_seed(0)
    for q in m.parameters():
        torch.nn.init.normal_(q, std=0.05)
    return m.eval()


def guider_for(dm):
    mp = Wrap(); mp.model = Wrap(); mp.model.diffusion_model = dm; mp.model_options = {}
    mp.is_dynamic = lambda: False
    g = nodes.CoSPAFS_PassiveTelemetryGuider().get_guider(mp, [], [], cfg=1.0, csv_session='', prompt_text='t')[0]
    g._log_steps = False
    return g, mp


def run_hooks(dm, forward, blocks=None, entry=None, branches=None):
    g, mp = guider_for(dm)
    if blocks is None:
        blocks = g._find_blocks(mp.model)
    else:                                   # test-only: caller supplies the block list (GeneralDIT)
        g._registry_config = entry
    g._register_hooks(blocks)
    n_handles = len(g._hook_handles)
    for d in (g._micro_g_per_layer, g._micro_hdim_per_layer, g._micro_seqlen_per_layer):
        d.clear()
    g._cur_branches = branches
    with torch.no_grad():
        forward()
    g._remove_hooks()
    per = {}
    for k, v in g._micro_g_per_layer.items():
        per.setdefault((k[0], k[1]), 0); per[(k[0], k[1])] += len(v)
    return g, blocks, per, n_handles


def resolve(obj, path):
    return nodes._resolve_dotted_attr(obj, path)


def tiny_checks(label, key, dm, forward, width, attn_path, mlp_path, L, order_mod=None):
    """Forward the tiny model with the node's hooks and check firing / dims / order."""
    entry = ENTRIES[key]
    g, blocks, per, nh = run_hooks(dm, forward)
    check(f'{label} tiny: registry entry {key} used', g._registry_config is entry)
    check(f'{label} tiny: {L} blocks found', len(blocks) == L, f'found {len(blocks)}')
    check(f'{label} tiny: {attn_path} and {mlp_path} resolve on every block',
          all(resolve(b, attn_path) is not None and resolve(b, mlp_path) is not None for b in blocks))
    check(f'{label} tiny: 2 hooks per block registered', nh == 2 * L, f'{nh} handles')
    check(f'{label} tiny forward: attn hook ({attn_path}) fired once on all {L} blocks',
          all(per.get((i, 0)) == 1 for i in range(L)), str(per))
    check(f'{label} tiny forward: MLP hook ({mlp_path}) fired once on all {L} blocks',
          all(per.get((i, 1)) == 1 for i in range(L)), str(per))
    hd = {(k[0], k[1]): v for k, v in g._micro_hdim_per_layer.items()}
    check(f'{label} tiny forward: captured D == {width} for attn and MLP',
          all(hd.get((i, s)) == width for i in range(L) for s in (0, 1)), str(sorted(set(hd.values()))))
    order = []
    mods = [resolve(b, order_mod or attn_path) for b in blocks]
    hs = [m.register_forward_hook(lambda m, i, o, k=k: order.append(k)) for k, m in enumerate(mods)]
    with torch.no_grad():
        forward()
    for h in hs:
        h.remove()
    check(f'{label} tiny forward: block list is in forward (depth) order', order == list(range(L)), str(order))
    return g


def meta_checks(label, key, build, L_exp, D_exp, attn_path, mlp_path):
    try:
        with torch.device('meta'):
            big = build()
    except Exception as e:
        check(f'{label} real config (meta): builds', False, repr(e)[:200]); return
    g, mp = guider_for(big)
    blocks = g._find_blocks(mp.model)
    check(f'{label} real config (meta): {L_exp} blocks, entry {key} used',
          len(blocks) == L_exp and g._registry_config is ENTRIES[key], f'found {len(blocks)}, key={type(big).__name__}')
    a = [resolve(b, attn_path) for b in blocks]; m = [resolve(b, mlp_path) for b in blocks]
    ok = all(x is not None for x in a + m)
    dims = sorted(set(x.out_features for x in a)) if ok and hasattr(a[0], 'out_features') else None
    check(f'{label} real config (meta): paths resolve on all blocks, attn out_features == {D_exp}',
          ok and dims == [D_exp], f'dims {dims}')


@contextlib.contextmanager
def force_fused_linear():
    """Make comfy.ops.linear_input_act take its fused branch (as with --fast fp16_accumulation on CUDA fp16, or
    TensorWise INT8 weights). The fused kernel is replaced by plain F.linear so it runs on CPU."""
    old_w = comfy.ops._fp16_linear_wanted
    ck = comfy.quant_ops.ck
    had = hasattr(ck, 'fp16_linear'); old_k = getattr(ck, 'fp16_linear', None)
    comfy.ops._fp16_linear_wanted = lambda x: True
    def _fp16_linear(x, w, b, residual=None, residual_scale=None):
        out = F.linear(x, w, b)
        return out if residual is None else torch.addcmul(residual, out, residual_scale)
    setattr(ck, 'fp16_linear', _fp16_linear)
    try:
        yield
    finally:
        comfy.ops._fp16_linear_wanted = old_w
        if had: setattr(ck, 'fp16_linear', old_k)
        else: delattr(ck, 'fp16_linear')


def trap_check(label, dm, forward, inner_path, outer_path):
    blocks = [b for b in guider_for(dm)[0]._find_blocks(guider_for(dm)[1].model)]
    def count(fused):
        n_in, n_out = [0], [0]
        hs = []
        for b in blocks:
            hs.append(resolve(b, inner_path).register_forward_hook(lambda *a: n_in.__setitem__(0, n_in[0] + 1)))
            hs.append(resolve(b, outer_path).register_forward_hook(lambda *a: n_out.__setitem__(0, n_out[0] + 1)))
        with torch.no_grad():
            if fused:
                with force_fused_linear(): forward()
            else:
                forward()
        for h in hs: h.remove()
        return n_in[0], n_out[0]
    L = len(blocks)
    i0, o0 = count(False); i1, o1 = count(True)
    check(f'{label} trap: default CPU path: {inner_path} fires {L}x, {outer_path} fires {L}x', (i0, o0) == (L, L), f'{i0},{o0}')
    check(f'{label} trap: fused linear_input_act path: {inner_path} fires 0x (bypassed), {outer_path} still {L}x',
          (i1, o1) == (0, L), f'{i1},{o1}')


def section(name, fn):
    print(f'\n---------------- {name} ----------------', flush=True)
    try:
        fn()
    except Exception as e:
        traceback.print_exc()
        check(f'{name}: section ran without exception', False, repr(e)[:300])


# =========================================================================================
# Wan family
# =========================================================================================
import comfy.ldm.wan.model as wanm
import comfy.ldm.wan.model_animate as wan_anim
import comfy.ldm.wan.model_animate2 as wan_anim2
import comfy.ldm.wan.model_wandancer as wan_dancer
import comfy.ldm.wan.ar_model as wan_ar

D = 64; FFN = 128; HEADS = 4; TXT = 32; L = 3
WAN_TINY = dict(patch_size=(1, 2, 2), in_dim=16, dim=D, ffn_dim=FFN, freq_dim=32, text_dim=TXT, out_dim=16,
                num_heads=HEADS, num_layers=L, operations=ops)
WAN_14B = dict(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40, in_dim=16, device='meta', operations=ops)

x_lat = torch.randn(1, 16, 2, 8, 8); t1 = torch.tensor([0.5]); ctx = torch.randn(1, 7, TXT)
clip = torch.randn(1, 257, 1280)


def wan_basic():
    for mt in ('t2v', 'i2v'):
        torch.manual_seed(0)
        m = init_(wanm.WanModel(model_type=mt, **WAN_TINY))
        kw = {'clip_fea': clip} if mt == 'i2v' else {}
        tiny_checks(f'WanModel {mt}', 'WanModel', m, lambda: m(x_lat, t1, ctx, **kw), D, 'self_attn.o', 'ffn', L)
    m = init_(wanm.WanModel(model_type='t2v', **WAN_TINY))
    trap_check('WanModel', m, lambda: m(x_lat, t1, ctx), 'ffn.2', 'ffn')
    # cond/uncond batched: B=2 with branches [0, 1] -> separate cond/uncond keys, rows = tokens per sample
    m2 = init_(wanm.WanModel(model_type='t2v', **WAN_TINY))
    xb = torch.randn(2, 16, 2, 8, 8); tb = torch.tensor([0.5, 0.5]); cb = torch.randn(2, 7, TXT)
    g, blocks, per, _ = run_hooks(m2, lambda: m2(xb, tb, cb), branches=[0, 1])
    keys = set(k[2] for k in g._micro_g_per_layer)
    rows = set(g._micro_seqlen_per_layer.values())
    check('WanModel tiny forward B=2 cond/uncond: split into cond+uncond, rows = 2*4*4 tokens each',
          keys == {'cond', 'uncond'} and rows == {32}, f'keys {keys} rows {rows}')
    meta_checks('WanModel 14B', 'WanModel', lambda: wanm.WanModel(model_type='t2v', **WAN_14B), 40, 5120, 'self_attn.o', 'ffn')
    meta_checks('WanModel 1.3B', 'WanModel', lambda: wanm.WanModel(model_type='t2v', dim=1536, ffn_dim=8960, num_heads=12,
                num_layers=30, device='meta', operations=ops), 30, 1536, 'self_attn.o', 'ffn')
    meta_checks('WanModel 2.2-5B', 'WanModel', lambda: wanm.WanModel(model_type='t2v', dim=3072, ffn_dim=14336, num_heads=24,
                num_layers=30, in_dim=48, out_dim=48, device='meta', operations=ops), 30, 3072, 'self_attn.o', 'ffn')


def wan_vace():
    m = init_(wanm.VaceWanModel(vace_layers=2, vace_in_dim=32, **{**WAN_TINY, 'num_layers': 4}))
    vc = torch.randn(1, 1, 32, 2, 8, 8)
    fwd = lambda: m(x_lat, t1, ctx, vace_context=vc, vace_strength=[1.0])
    g = tiny_checks('VaceWanModel', 'VaceWanModel', m, fwd, D, 'self_attn.o', 'ffn', 4)
    # the VACE side blocks are not in .blocks, so they are not hooked even though they run
    ran = [0]
    hs = [vb.self_attn.o.register_forward_hook(lambda *a: ran.__setitem__(0, ran[0] + 1)) for vb in m.vace_blocks]
    with torch.no_grad(): fwd()
    for h in hs: h.remove()
    check('VaceWanModel tiny forward: vace_blocks run (2x) but are outside block_path', ran[0] == 2, f'{ran[0]}')
    meta_checks('VaceWanModel 14B', 'VaceWanModel', lambda: wanm.VaceWanModel(vace_layers=8, vace_in_dim=96, **WAN_14B),
                40, 5120, 'self_attn.o', 'ffn')


def wan_camera():
    m = init_(wanm.CameraWanModel(model_type='camera', **WAN_TINY))
    cam = torch.randn(1, 24, 2, 64, 64)
    tiny_checks('CameraWanModel', 'CameraWanModel', m,
                lambda: m(x_lat, t1, ctx, clip_fea=clip, camera_conditions=cam), D, 'self_attn.o', 'ffn', L)
    meta_checks('CameraWanModel 14B', 'CameraWanModel', lambda: wanm.CameraWanModel(model_type='camera', **WAN_14B),
                40, 5120, 'self_attn.o', 'ffn')


def wan_humo():
    m = init_(wanm.HumoWanModel(**WAN_TINY))
    aud = torch.randn(1, 2, 8, 5, 1280)
    fwd = lambda: m(x_lat, t1, ctx, audio_embed=aud)
    tiny_checks('HumoWanModel', 'HumoWanModel', m, fwd, D, 'self_attn.o', 'ffn', L)
    n = [0]
    hs = [b.audio_cross_attn_wrapper.audio_cross_attn.o.register_forward_hook(lambda *a: n.__setitem__(0, n[0] + 1)) for b in m.blocks]
    with torch.no_grad(): fwd()
    for h in hs: h.remove()
    check('HumoWanModel tiny forward: in-block audio cross-attn ran on every block (not hooked by the node)', n[0] == L, f'{n[0]}')
    meta_checks('HumoWanModel 17B-class (dim 5120)', 'HumoWanModel', lambda: wanm.HumoWanModel(**WAN_14B), 40, 5120, 'self_attn.o', 'ffn')


def wan_s2v():
    m = init_(wanm.WanModel_S2V(audio_dim=32, **WAN_TINY))
    aud = torch.randn(1, 25, 32, 8)
    fwd = lambda: m(x_lat, t1, ctx, audio_embed=aud)
    tiny_checks('WanModel_S2V', 'WanModel_S2V', m, fwd, D, 'self_attn.o', 'ffn', L)
    meta_checks('WanModel_S2V 14B', 'WanModel_S2V', lambda: wanm.WanModel_S2V(**WAN_14B), 40, 5120, 'self_attn.o', 'ffn')


def wan_animate():
    m = init_(wan_anim.AnimateWanModel(**WAN_TINY))
    pose = torch.randn(1, 16, 1, 8, 8)
    tiny_checks('AnimateWanModel', 'AnimateWanModel', m, lambda: m(x_lat, t1, ctx, clip_fea=clip, pose_latents=pose),
                D, 'self_attn.o', 'ffn', L)
    meta_checks('AnimateWanModel 14B', 'AnimateWanModel', lambda: wan_anim.AnimateWanModel(**WAN_14B), 40, 5120, 'self_attn.o', 'ffn')


def wan_scail():
    for cls, key in ((wanm.SCAILWanModel, 'SCAILWanModel'), (wanm.SCAIL2WanModel, 'SCAIL2WanModel')):
        kw = dict(WAN_TINY); kw.pop('patch_size')
        extra = {'mask_in_dim': 28} if key == 'SCAIL2WanModel' else {}
        m = init_(cls(**kw, **extra))
        pose = torch.randn(1, 16, 2, 8, 8)
        tiny_checks(key, key, m, lambda: m(x_lat, t1, ctx, clip_fea=clip, pose_latents=pose), D, 'self_attn.o', 'ffn', L)
        kb = dict(WAN_14B); kb['in_dim'] = 20
        meta_checks(f'{key} 14B', key, lambda: cls(**kb), 40, 5120, 'self_attn.o', 'ffn')


def wan_others():
    # Classes not in the assignment list but in the same family; checked so the report can say what happens.
    # WanAnimate2Model: pose branch runs every block a second time via block.forward_pose (uncached first pass)
    a2 = init_(wan_anim2.WanAnimate2Model(**{**WAN_TINY, 'in_dim': 36}))
    x36 = torch.randn(1, 36, 3, 8, 8); pose = torch.randn(1, 16, 2, 8, 8)
    g, mp = guider_for(a2)
    blocks = list(a2.blocks)
    n_o = [0]; n_f = [0]
    hs = [b.self_attn.o.register_forward_hook(lambda *a: n_o.__setitem__(0, n_o[0] + 1)) for b in blocks]
    hs += [b.ffn.register_forward_hook(lambda *a: n_f.__setitem__(0, n_f[0] + 1)) for b in blocks]
    with torch.no_grad(): a2(x36, t1, ctx, clip_fea=clip, pose_latents=pose)
    for h in hs: h.remove()
    check('WanAnimate2Model (not given an entry): with pose latents and no cache, self_attn.o and ffn fire 2x per block',
          (n_o[0], n_f[0]) == (2 * L, 2 * L), f'o {n_o[0]}, ffn {n_f[0]} for {L} blocks')
    # CausalWanModel: same block structure; static path check only
    c = wan_ar.CausalWanModel(**WAN_TINY)
    check('CausalWanModel (not given an entry): self_attn.o and ffn resolve on every block',
          all(resolve(b, 'self_attn.o') is not None and resolve(b, 'ffn') is not None for b in c.blocks))


section('WanModel', wan_basic)
section('VaceWanModel', wan_vace)
section('CameraWanModel', wan_camera)
section('HumoWanModel', wan_humo)
section('WanModel_S2V', wan_s2v)
section('AnimateWanModel', wan_animate)
section('SCAILWanModel / SCAIL2WanModel', wan_scail)
section('Wan classes without an entry', wan_others)

# =========================================================================================
# LTX
# =========================================================================================
import comfy.ldm.lightricks.model as ltxm
import comfy.ldm.lightricks.av_model as ltxav


def ltxv():
    kw = dict(in_channels=16, cross_attention_dim=32, attention_head_dim=16, num_attention_heads=2, caption_channels=24,
              num_layers=3, operations=ops)
    m = init_(ltxm.LTXVModel(**kw))
    xv = torch.randn(1, 16, 2, 4, 4); c = torch.randn(1, 7, 24)
    fwd = lambda: m(xv, torch.tensor([0.5]), c, None, frame_rate=25, transformer_options={})
    tiny_checks('LTXVModel', 'LTXVModel', m, fwd, 32, 'attn1.to_out.0', 'ff', 3)
    trap_check('LTXVModel', m, fwd, 'ff.net.2', 'ff')
    # cross_attention_adaln variant (LTX 2.3-style block, also used inside LTXAVModel)
    m2 = init_(ltxm.LTXVModel(cross_attention_adaln=True, **kw))
    tiny_checks('LTXVModel cross_attention_adaln', 'LTXVModel', m2,
                lambda: m2(xv, torch.tensor([0.5]), c, None, frame_rate=25, transformer_options={}), 32, 'attn1.to_out.0', 'ff', 3)
    meta_checks('LTXVModel 0.9 2B (class defaults)', 'LTXVModel', lambda: ltxm.LTXVModel(device='meta', operations=ops),
                28, 2048, 'attn1.to_out.0', 'ff')
    meta_checks('LTXVModel 0.9 13B', 'LTXVModel', lambda: ltxm.LTXVModel(attention_head_dim=128, cross_attention_dim=4096,
                num_layers=48, device='meta', operations=ops), 48, 4096, 'attn1.to_out.0', 'ff')


def ltxav_():
    kw = dict(in_channels=16, audio_in_channels=128, cross_attention_dim=32, audio_cross_attention_dim=16,
              attention_head_dim=16, audio_attention_head_dim=8, num_attention_heads=2, audio_num_attention_heads=2,
              caption_channels=24, num_layers=3, connector_attention_head_dim=12, connector_num_attention_heads=2,
              operations=ops)
    m = init_(ltxav.LTXAVModel(**kw))
    vx = torch.randn(1, 16, 2, 4, 4); ax = torch.randn(1, 8, 5, 16); c = torch.randn(1, 7, 48)
    fwd = lambda: m([vx, ax], torch.tensor([0.5]), c, None, frame_rate=25, transformer_options={})
    tiny_checks('LTXAVModel', 'LTXAVModel', m, fwd, 32, 'attn1.to_out.0', 'ff', 3)
    trap_check('LTXAVModel', m, fwd, 'ff.net.2', 'ff')
    # audio stream is not hooked by the node, but runs: audio_attn1.to_out.0 / audio_ff fire once per block, D=16
    n = {}
    hs = []
    for i, b in enumerate(m.transformer_blocks):
        for p in ('audio_attn1.to_out.0', 'audio_ff', 'audio_to_video_attn.to_out.0', 'video_to_audio_attn.to_out.0'):
            hs.append(resolve(b, p).register_forward_hook(lambda mod, inp, out, p=p: n.__setitem__(p, n.get(p, 0) + 1)))
    with torch.no_grad(): fwd()
    for h in hs: h.remove()
    check('LTXAVModel tiny forward: audio + a2v/v2a modules each fire once per block (unhooked by the node)',
          all(v == 3 for v in n.values()) and len(n) == 4, str(n))
    # video-only use of the AV model (no audio latent): ax is empty, video hooks still fire once per block
    g, blocks, per, _ = run_hooks(m, lambda: m([vx], torch.tensor([0.5]), c, None, frame_rate=25, transformer_options={}))
    check('LTXAVModel tiny forward, video-only input: attn+MLP fire once per block',
          all(per.get((i, s)) == 1 for i in range(3) for s in (0, 1)), str(per))
    meta_checks('LTXAVModel LTX-2 19B (class defaults)', 'LTXAVModel', lambda: ltxav.LTXAVModel(device='meta', operations=ops),
                48, 4096, 'attn1.to_out.0', 'ff')


section('LTXVModel', ltxv)
section('LTXAVModel', ltxav_)

# =========================================================================================
# CogVideoX
# =========================================================================================
import comfy.ldm.cogvideo.model as cogm


def cog():
    for label, extra, xs in (('CogVideoX 1.0-style', {}, (1, 16, 2, 8, 8)),
                             ('CogVideoX 1.5-style (patch_size_t=2)', {'patch_size_t': 2}, (1, 16, 4, 8, 8))):
        m = init_(cogm.CogVideoXTransformer3DModel(num_attention_heads=2, attention_head_dim=16, time_embed_dim=16,
                                                   text_embed_dim=24, num_layers=3, sample_width=16, sample_height=16,
                                                   sample_frames=9, use_rotary_positional_embeddings=True,
                                                   operations=ops, **extra))
        x = torch.randn(*xs); c = torch.randn(1, 7, 24)
        g = tiny_checks(label, 'CogVideoXTransformer3DModel', m, lambda: m(x, torch.tensor([500.0]), c), 32, 'attn_out', 'ff_out', 3)
        n_img = (xs[2] // extra.get('patch_size_t', 1)) * (xs[3] // 2) * (xs[4] // 2)
        rows = set(g._micro_seqlen_per_layer.values())
        check(f'{label} tiny forward: captured rows = text (7) + video tokens ({n_img})', rows == {7 + n_img}, str(rows))
    meta_checks('CogVideoX 2B', 'CogVideoXTransformer3DModel', lambda: cogm.CogVideoXTransformer3DModel(
        num_attention_heads=30, num_layers=30, device='meta', operations=ops), 30, 1920, 'attn_out', 'ff_out')
    meta_checks('CogVideoX 5B', 'CogVideoXTransformer3DModel', lambda: cogm.CogVideoXTransformer3DModel(
        num_attention_heads=48, num_layers=42, device='meta', operations=ops), 42, 3072, 'attn_out', 'ff_out')


section('CogVideoXTransformer3DModel', cog)

# =========================================================================================
# Cosmos 1 (GeneralDIT) and Cosmos Predict2 (MiniTrainDIT)
# =========================================================================================
import comfy.ldm.cosmos.model as cosm
import comfy.ldm.cosmos.predict2 as cosp2

COS1_TINY = dict(max_img_h=16, max_img_w=16, max_frames=8, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
                 concat_padding_mask=True, block_config='FA-CA-MLP', model_channels=64, num_blocks=3, num_heads=4,
                 crossattn_emb_channels=32, pos_emb_cls='rope3d', pos_emb_learnable=False, pos_emb_interpolation='crop',
                 block_x_format='THWBD', affline_emb_norm=True, use_adaln_lora=True, adaln_lora_dim=16,
                 extra_per_block_abs_pos_emb=True, extra_per_block_abs_pos_emb_type='learnable', operations=ops)


def cosmos1():
    entry = ENTRIES['GeneralDIT']
    m = init_(cosm.GeneralDIT(**COS1_TINY))
    x = torch.randn(1, 16, 2, 8, 8); c = torch.randn(1, 7, 32)
    fwd = lambda: m(x, torch.tensor([1.0]), c, fps=torch.tensor([24.0]))
    ap, mpth = entry['attn_attrs']['self_attn'][0], entry['mlp_attrs']['mlp'][0]
    # 1. R2 behaviour: list() of the nn.ModuleDict gives key strings, so no hooks and no telemetry
    check('GeneralDIT: list(blocks) is ModuleDict KEYS (strings), which is what R2 hooked',
          all(isinstance(k, str) for k in list(m.blocks)), str(list(m.blocks)))
    # 2. R3 nodes.py (_as_block_seq) iterates the ModuleDict by value: registry path, hooks fire once per block
    tiny_checks('GeneralDIT', 'GeneralDIT', m, fwd, 64, ap, mpth, 3)
    g, blocks, per, nh = run_hooks(m, fwd)
    # 2b. R3 batch_dim: with batched CFG, cond/uncond are separated along the batch axis, not along tokens.
    #     Each branch's g must equal the g of a B=1 forward of that sample alone.
    xb2 = torch.randn(2, 16, 2, 8, 8); cb2 = torch.randn(2, 7, 32); t2 = torch.tensor([1.0, 1.0])
    gB, _, _, _ = run_hooks(m, lambda: m(xb2, t2, cb2, fps=torch.tensor([24.0, 24.0])), branches=[0, 1])
    keys = set(k[2] for k in gB._micro_g_per_layer); rows = set(gB._micro_seqlen_per_layer.values())
    check('GeneralDIT B=2 cond/uncond: split into cond+uncond, rows = 2*4*4 each',
          keys == {'cond', 'uncond'} and rows == {32}, f'keys {keys} rows {rows}')
    worst = 0.0
    for bi, name in ((0, 'cond'), (1, 'uncond')):
        g1, _, _, _ = run_hooks(m, lambda: m(xb2[bi:bi + 1], t2[:1], cb2[bi:bi + 1], fps=torch.tensor([24.0])), branches=[0])
        for i in range(3):
            for sp in (0, 1):
                v2 = gB._micro_g_per_layer[(i, sp, name)][0]; v1 = g1._micro_g_per_layer[(i, sp, 'cond')][0]
                worst = max(worst, abs(v2 - v1) / max(abs(v1), 1e-12))
    check('GeneralDIT B=2: each branch g equals its own B=1 forward (split is along batch, not tokens)',
          worst < 1e-3, f'max rel diff {worst:.2e}')
    blocks = list(m.blocks.values())
    # 3. layout: the hooked tensors are sequence-first, batch is NOT dim 0
    shp = {}
    hs = [resolve(blocks[0], ap).register_forward_hook(lambda mod, i, o: shp.__setitem__('attn', tuple(o.shape))),
          resolve(blocks[0], mpth).register_forward_hook(lambda mod, i, o: shp.__setitem__('mlp', tuple(o.shape)))]
    xb = torch.randn(2, 16, 2, 8, 8); cb = torch.randn(2, 7, 32)
    with torch.no_grad(): m(xb, torch.tensor([1.0, 1.0]), cb, fps=torch.tensor([24.0]))
    for h in hs: h.remove()
    check('GeneralDIT B=2: attn output is (THW, B, D) and MLP output (T, H, W, B, D): batch is not dim 0',
          shp.get('attn') == (32, 2, 64) and shp.get('mlp') == (2, 4, 4, 2, 64), str(shp))
    # 4. real configs
    for label, mc, nb, nh_ in (('Cosmos-1 7B', 4096, 28, 32), ('Cosmos-1 14B', 5120, 36, 40)):
        kw = dict(COS1_TINY); kw.update(model_channels=mc, num_blocks=nb, num_heads=nh_, crossattn_emb_channels=1024,
                                        max_img_h=240, max_img_w=240, max_frames=128, device='meta')
        with torch.device('meta'):
            big = cosm.GeneralDIT(**kw)
        bl = list(big.blocks.values())
        ok = all(resolve(b, ap) is not None and resolve(b, mpth) is not None for b in bl)
        check(f'GeneralDIT {label} (meta): {nb} blocks, paths resolve, D={mc}',
              len(bl) == nb and ok and resolve(bl[0], ap).out_features == mc)


def cosmos2():
    m = init_(cosp2.MiniTrainDIT(max_img_h=16, max_img_w=16, max_frames=8, in_channels=16, out_channels=16, patch_spatial=2,
                                 patch_temporal=1, concat_padding_mask=True, model_channels=64, num_blocks=3, num_heads=4,
                                 crossattn_emb_channels=32, pos_emb_cls='rope3d', pos_emb_learnable=True,
                                 use_adaln_lora=True, adaln_lora_dim=16, operations=ops))
    x = torch.randn(1, 16, 2, 8, 8); c = torch.randn(1, 7, 32)
    fwd = lambda: m(x, torch.tensor([0.5]), c, fps=torch.tensor([16.0]))
    tiny_checks('MiniTrainDIT', 'MiniTrainDIT', m, fwd, 64, 'self_attn.output_proj', 'mlp.layer2', 3)
    xb = torch.randn(2, 16, 2, 8, 8); cb = torch.randn(2, 7, 32)
    g, blocks, per, _ = run_hooks(m, lambda: m(xb, torch.tensor([0.5, 0.5]), cb, fps=torch.tensor([16.0])), branches=[0, 1])
    keys = set(k[2] for k in g._micro_g_per_layer); rows = set(g._micro_seqlen_per_layer.values())
    check('MiniTrainDIT tiny forward B=2 cond/uncond: split into cond+uncond, rows = 2*4*4 each',
          keys == {'cond', 'uncond'} and rows == {32}, f'keys {keys} rows {rows}')
    for label, mc, nb, nh_ in (('Predict2 2B', 2048, 28, 16), ('Predict2 14B', 5120, 36, 40)):
        meta_checks(f'MiniTrainDIT {label}', 'MiniTrainDIT', lambda: cosp2.MiniTrainDIT(
            max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
            model_channels=mc, num_blocks=nb, num_heads=nh_, crossattn_emb_channels=1024, pos_emb_cls='rope3d',
            pos_emb_learnable=True, use_adaln_lora=True, adaln_lora_dim=256, device='meta', operations=ops),
            nb, mc, 'self_attn.output_proj', 'mlp.layer2')


section('GeneralDIT (Cosmos 1)', cosmos1)
section('MiniTrainDIT (Cosmos Predict2)', cosmos2)

print(f'\n{len(fails)} failed: {fails}' if fails else '\nALL PASS')
sys.exit(1 if fails else 0)
