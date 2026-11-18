"""Shared loaders for the post hoc follow-ups of 2026-11-10 (review/posthoc_2026-11-10/PLAN.md, sha 27d056e4).
Paths: COSPAFS_ROOT (repository root), SCR (archive and Round 11-14 cubes; default <root>/cache), R16CUBES, R17CUBES."""
import os, sys, pickle, csv, types, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = os.path.join(BASE, 'out'); os.makedirs(OUT, exist_ok=True)
ROOT = os.environ.get('COSPAFS_ROOT', os.path.abspath(os.path.join(BASE, '..', '..')))
CA = os.environ.get('SCR', os.path.join(ROOT, 'cache')); R16 = os.environ.get('R16CUBES', CA); R17 = os.environ.get('R17CUBES', CA)
PROMPTS = os.environ.get('PROMPTS60', os.path.join(HERE, 'prompts_r3_cond60.tsv'))
R17N = {'PIXART', 'PIXARTU', 'COSMOS2B', 'COSMOS2BU', 'SD3M', 'SD3MU', 'SD35M17', 'SD35M17U'}
FILE = {'SD35M17': 'SD35M', 'SD35M17U': 'SD35MU'}          # Round 17 SD3.5 Medium recapture (the archive cube is also called SD35M)
TCLK = {'PIXART', 'PIXARTU', 'COSMOS2B', 'COSMOS2BU'}      # sigma is VP/EDM: clock t = sigma / (1 + sigma), as score_r17.py
DIM = {'PIXART': 1152, 'COSMOS2B': 2048, 'SD3M': 1536, 'SD35M17': 1536, 'Z1': 3840, 'Z3': 3840}   # hidden width; N = 4096 image tokens (R17)
LABEL = {'PIXART': 'PixArt-Sigma', 'COSMOS2B': 'Cosmos-Predict2 2B', 'SD3M': 'SD3 Medium', 'SD35M17': 'SD3.5 Medium (R17)',
         'SD3SV': 'SD3.5 Medium (R11)', 'Z2': 'Z-Image Base (Z2)', 'ZITSV': 'Z-Image Turbo (R11)', 'Z1': 'Z-Image Base (Z1)', 'Z3': 'Z-Image Turbo (Z3)'}
def _dir(n): return R17 if n in R17N else CA
def strip(v): return np.asarray([str(x).strip() for x in v])
def cube(n, sub, raw_clock=False):
    """-> prompts [G], X [G, T, L, 3] (g, gap, H), clock [T]"""
    g, meta, X, s = pickle.load(open(f'{_dir(n)}/F_{FILE.get(n, n)}_{sub}.pkl', 'rb')); s = np.asarray(s, float)
    if n in TCLK and not raw_clock: s = s / (1 + s)
    return strip(meta.pt), np.asarray(X, float), s
VSETS = {'img': ['g_img', 'gap_img', 'H_img'], 'txt': ['g_txt', 'gap_txt', 'H_txt'], 'ctr': ['g_ctr', 'gap_ctr', 'H_ctr'], 'topk': ['g_topk', 'gap_topk', 'H_topk']}
def vcube(n, sub, rs):
    """variant cube: rs in img / txt / ctr / topk, or a single column name"""
    o = pickle.load(open(f'{_dir(n)}/V_{FILE.get(n, n)}_{sub}.pkl', 'rb')); s = np.asarray(o['s'], float)
    if n in TCLK: s = s / (1 + s)
    cols = VSETS.get(rs, [rs]); idx = [o['cols'].index(c) for c in cols]
    return strip(o['meta'].pt), np.asarray(o['V'][..., idx], float), s
def have(n, sub, v=False): return os.path.exists(f"{_dir(n)}/{'V' if v else 'F'}_{FILE.get(n, n)}_{sub}.pkl")
def domains():
    r = list(csv.reader(open(PROMPTS), delimiter='\t'))[1:]; return {x[1].strip(): x[0].strip() for x in r}
def analyzer():
    sys.modules.setdefault('gradio', types.ModuleType('gradio')); sys.path.insert(0, f'{ROOT}/review/lenses')
    import lens_L2_rg as L2; return L2
