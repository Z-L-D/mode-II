"""Latent data spectrum of a model's own generated images (gate "latent Λ", ledger v0.24).

Encodes a folder of images with the model's VAE, maps each latent into the space the
diffusion model actually sees (ComfyUI latent_format.process_in), and writes the
covariance spectrum of the flattened latents. No generation, no sampling.

Run inside the Python environment of your ComfyUI install, on the GPU:

    python latent_spectrum.py --comfy C:/path/to/ComfyUI \
        --vae models/vae/ae.safetensors --format Flux \
        --images D:/outputs/zit_campaign --name ZIT --max 1000

    --format: Flux  for Z-Image Turbo / Base (Flux VAE)
              SD3   for SD3.5 Medium / Large
    (Mochi needs its video VAE and is not handled here.)

Output, next to --out (default: current folder):
    <name>_latent_spectrum.npz   eigenvalues, per-image prompt/seed hashes, split-half check
    <name>_latent_spectrum.json  summary: n, latent shape, top eigenvalues, implied speciation sigma

Upload the .json and .npz (a few hundred KB). Images are only read, never modified.

Speciation sigma: Biroli et al. 2024 put speciation where the signal variance along the top
principal direction equals the noise variance. For a rectified-flow path x = (1-s) x0 + s eps
that is (1-s)^2 * Lambda = s^2, i.e. s_spec = sqrt(Lambda) / (1 + sqrt(Lambda)).
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--comfy', required=True, help='ComfyUI root folder')
ap.add_argument('--vae', required=True, help='VAE .safetensors (absolute, or relative to --comfy)')
ap.add_argument('--format', required=True, choices=['Flux', 'SD3'])
ap.add_argument('--images', required=True, help='folder of generated images (searched recursively)')
ap.add_argument('--name', required=True, help='label for the output files, e.g. ZIT')
ap.add_argument('--max', type=int, default=1000, help='max images (a fixed random subset if more)')
ap.add_argument('--out', default='.')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--cpu', action='store_true', help='run ComfyUI on CPU (testing only; slow)')
a = ap.parse_args()

sys.argv = [sys.argv[0]] + (['--cpu'] if a.cpu else [])
sys.path.insert(0, a.comfy)
import torch
import comfy.options
comfy.options.enable_args_parsing()
import comfy.utils, comfy.sd, comfy.latent_formats
from PIL import Image

def prompt_and_seed(path):
    """Prompt text and seed from ComfyUI's embedded API graph (PNG 'prompt' chunk), if present."""
    try:
        g = json.loads(Image.open(path).info.get('prompt', '') or '{}')
    except Exception:
        return '', ''
    texts, seeds = [], []
    for node in g.values():
        ins = node.get('inputs', {}) if isinstance(node, dict) else {}
        for k in ('prompt_text', 'text', 'text_g', 'text_l'):
            if isinstance(ins.get(k), str) and ins[k].strip():
                texts.append((0 if k == 'prompt_text' else 1, ins[k].strip()))
        for k in ('seed', 'noise_seed'):
            if isinstance(ins.get(k), (int, float)):
                seeds.append(str(int(ins[k])))
    texts.sort()
    return (texts[0][1] if texts else ''), (seeds[0] if seeds else '')

files = sorted(f for ext in ('png', 'jpg', 'jpeg', 'webp')
               for f in glob.glob(os.path.join(a.images, '**', f'*.{ext}'), recursive=True))
if not files:
    sys.exit(f'no images under {a.images}')
sizes = {}
for f in files:
    with Image.open(f) as im:
        sizes.setdefault(im.size, []).append(f)
size, files = max(sizes.items(), key=lambda kv: len(kv[1]))
print(f'{sum(len(v) for v in sizes.values())} images; using the most common size {size}: {len(files)} '
      f'(other sizes: { {k: len(v) for k, v in sizes.items() if k != size} })')
rng = np.random.default_rng(a.seed)
if len(files) > a.max:
    files = sorted(rng.choice(files, a.max, replace=False).tolist())

vae_path = a.vae if os.path.isabs(a.vae) else os.path.join(a.comfy, a.vae)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
fmt = getattr(comfy.latent_formats, a.format)()

lat, meta = [], []
t0 = time.time()
for i, f in enumerate(files):
    px = torch.from_numpy(np.asarray(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0)[None]
    with torch.no_grad():
        z = fmt.process_in(vae.encode(px).float().cpu())
    lat.append(z.numpy().reshape(-1).astype(np.float32))
    p, s = prompt_and_seed(f)
    meta.append((os.path.relpath(f, a.images), hashlib.md5(p.encode()).hexdigest()[:12] if p else '', s))
    if i % 100 == 0:
        print(f'  encoded {i + 1}/{len(files)}  ({time.time() - t0:.0f} s)  latent {tuple(z.shape)}')
shape = tuple(z.shape)
X = np.stack(lat).astype(np.float64)
n, d = X.shape
del lat

def top_eigs(M, k=50):
    """Eigenvalues of the covariance of rows of M, via the n x n Gram matrix."""
    Mc = M - M.mean(0, keepdims=True)
    G = Mc @ Mc.T / Mc.shape[0]
    ev = np.linalg.eigvalsh(G)[::-1]
    return ev[:k]

ev = top_eigs(X)
idx = rng.permutation(n)
h1, h2 = top_eigs(X[idx[: n // 2]], 10), top_eigs(X[idx[n // 2:]], 10)
per_dim_var = float(X.var(0).mean())

# Within-prompt spectrum (prompt-conditional distribution), if prompts repeat enough
pid = np.array([m[1] for m in meta])
groups = [np.where(pid == u)[0] for u in np.unique(pid) if u and (pid == u).sum() >= 8]
within = None
if len(groups) >= 3:
    W = np.concatenate([X[g] - X[g].mean(0, keepdims=True) for g in groups])
    G = W @ W.T / W.shape[0]
    within = np.linalg.eigvalsh(G)[::-1][:10]

s_spec = lambda L: float(np.sqrt(L) / (1 + np.sqrt(L)))
summary = {
    'name': a.name, 'format': a.format, 'vae': os.path.basename(vae_path), 'image_size': list(size),
    'n_images': n, 'latent_shape': list(shape), 'latent_dim': d,
    'mean_per_dim_variance': per_dim_var,
    'top_eigenvalues': [float(x) for x in ev[:10]],
    'Lambda': float(ev[0]), 'sigma_speciation': s_spec(ev[0]),
    'Lambda_over_trace': float(ev[0] / (per_dim_var * d)),
    'split_half_Lambda': [float(h1[0]), float(h2[0])],
    'split_half_sigma_speciation': [s_spec(h1[0]), s_spec(h2[0])],
    'n_prompts_found': int(len(np.unique(pid[pid != '']))),
    'within_prompt': None if within is None else {
        'n_prompts_used': len(groups), 'n_images_used': int(sum(len(g) for g in groups)),
        'Lambda': float(within[0]), 'sigma_speciation': s_spec(within[0]),
        'top_eigenvalues': [float(x) for x in within]},
    'images_sha256': hashlib.sha256('\n'.join(m[0] for m in meta).encode()).hexdigest(),
    'comfy_latent_format_check': f'{a.format}.process_in applied (scale/shift as the diffusion model sees it)',
}
os.makedirs(a.out, exist_ok=True)
stem = os.path.join(a.out, f'{a.name}_latent_spectrum')
np.savez_compressed(stem + '.npz', eigenvalues=ev, split_half_1=h1, split_half_2=h2,
                    within_prompt=np.array([]) if within is None else within,
                    files=np.array([m[0] for m in meta]), prompt_md5=pid, seed=np.array([m[2] for m in meta]))
json.dump(summary, open(stem + '.json', 'w'), indent=2)
print(json.dumps({k: summary[k] for k in ('n_images', 'latent_shape', 'Lambda', 'sigma_speciation',
                                          'split_half_sigma_speciation', 'n_prompts_found')}, indent=1))
print(f'wrote {stem}.json and .npz')
