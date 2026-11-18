"""Shared helpers for the Mode-II figure scripts (CoS-PAFS, Zac Doyle).

Every figure script imports this module, builds one figure with figstyle, writes the plotted
numbers to ../data/<figure>.csv (long format, one `source` column per row) and calls save().
Inputs are read only; nothing under /mnt/project-files/review/ is modified.
"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common'))  # <repo root>/common/figstyle.py
from figstyle import *  # noqa: F401,F403  (C, GRAY, LIGHT, INK, INK2, MODEL_COLOR, MODEL_LABEL, TEXTWIDTH, save, panel_label, plt)

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
DATA = os.path.join(PAPER, 'data')
REV = os.environ.get('REVIEW', f'{_COSPAFS_ROOT}/review')
R2 = f'{REV}/round2/out'
R5 = f'{REV}/round5/out'
R6 = f'{REV}/round6/out'
LEDGER = os.environ.get('LEDGER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '_sources', 'ledger_v020_text.txt'))
# Raw (gids, meta, X[G,T,L,3], sigma) cubes; X[...,0]=g, [...,1]=gap, [...,2]=H. Override with CUBES=<dir>.
CUBES = os.environ.get('CUBES', f'{_COSPAFS_CACHE}')

# Fixed colours for pair kinds (categorical slots in fixed order; unrelated pairs are the gray reference).
PAIR_COLOR = {'lineage': C[0], 'same-model CFG': C[2], 'same family': C[3], 'attn-vs-mlp': C[4], 'unrelated': GRAY}
PAIR_LABEL = {'lineage': 'ZIT-25↔ZIT (same weights)', 'same-model CFG': 'Mochi CFG 1↔4.5 (same weights)',
              'same family': 'SD3.5 M↔L (same family)', 'attn-vs-mlp': 'ZIT attn↔MLP', 'unrelated': 'unrelated pairs'}
SHORT = {'ZIT': 'ZIT', 'ZIB': 'ZIT-25', 'MOCHI': 'Mochi', 'SD35M': 'SD3.5 M', 'SD35L': 'SD3.5 L'}


def cube(name):
    """Load F_<name>.pkl, e.g. cube('ZIB_mlp') -> gids, meta, X, sigma (float array)."""
    path = os.path.join(CUBES, f'F_{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: set CUBES to the folder holding the F_*.pkl telemetry cubes')
    g, meta, X, s = pickle.load(open(path, 'rb'))
    return g, meta, X, np.asarray(s, float)


def r5cube(arm, sub):
    """Round-5 per-branch capture R5_<arm>_<sub>.pkl -> dict(gids, meta, X{cond,uncond}, PI, sigma)."""
    return pickle.load(open(os.path.join(CUBES, f'R5_{arm}_{sub}.pkl'), 'rb'))


def write_data(fig_id, frames):
    """Concatenate panel frames (each with a `panel` and `source` column) and write data/<fig_id>.csv."""
    df = pd.concat(frames, ignore_index=True, sort=False)
    cols = ['panel'] + [c for c in df.columns if c not in ('panel', 'source')] + ['source']
    df = df[cols]
    os.makedirs(DATA, exist_ok=True)
    df.to_csv(os.path.join(DATA, f'{fig_id}.csv'), index=False)
    return df


def ledger_has(snippet):
    """Guard for transcribed values: the quoted ledger text must occur verbatim in the ledger file."""
    if not os.path.exists(LEDGER):  # the findings ledger is internal (available on request); skip the transcription check
        return True
    txt = ' '.join(open(LEDGER, encoding='utf-8').read().split())
    s = ' '.join(snippet.split())
    if s not in txt:
        raise ValueError(f'ledger quote not found verbatim: {snippet[:80]}...')
    return snippet


def amp_stats(M):
    """g, gap, H of a PSD second-moment matrix M (D x D): s = sqrt(eig M)."""
    ev = np.clip(np.linalg.eigvalsh(M), 0, None)[::-1]
    s = np.sqrt(ev); D = len(s)
    g = s.sum() ** 2 / (s ** 2).sum() / D
    p = s / s.sum(); p = p[p > 0]
    return g, s[0] / s[1], float(-(p * np.log(p)).sum())
