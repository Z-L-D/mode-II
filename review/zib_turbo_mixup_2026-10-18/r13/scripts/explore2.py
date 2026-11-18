import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 13 EXPLORATION pass 2 (EXPLORE prompts only; descriptive).
import os, pickle, numpy as np, pandas as pd
SCR = os.environ.get('SCR', _COSPAFS_CACHE); np.set_printoptions(precision=3, suppress=True, linewidth=220)
L = lambda n: pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); LV = lambda n: pickle.load(open(f'{SCR}/V_{n}.pkl', 'rb'))
# cols: 0 g_img 1 gap_img 2 H_img 3 g_txt 4 gap_txt 5 H_txt 6 g_ctr 7 gap_ctr 8 H_ctr 9 g_topk 10 gap_topk 11 H_topk 12 n_img 13 n_txt 14 txt_share 15 mean_share 16 top1 17 n_massive
for sub in ('attn', 'mlp'):
    VB = LV(f'Z2E_{sub}'); VT = LV(f'ZITSVE_{sub}'); sB, sT = VB['s'], VT['s']
    print(f'== {sub}: caption energy share, median over gens, by layer, first/last step')
    for nm, V in (('Base', VB['V']), ('Turbo', VT['V'])):
        print(f' {nm} step0', np.median(V[:, 0, :, 14], 0).round(2)); print(f' {nm} last ', np.median(V[:, -1, :, 14], 0).round(2))
    print(f' first layer with median share >= 0.5, by step  Base:', [int(np.argmax(np.median(VB['V'][:, t, :, 14], 0) >= 0.5)) for t in range(len(sB))])
    print(f'                                              Turbo:', [int(np.argmax(np.median(VT['V'][:, t, :, 14], 0) >= 0.5)) for t in range(len(sT))])
    print(f' Base minus Turbo mean log g by variant, by layer (Base step closest to each Turbo sigma)')
    j = [int(np.argmin(np.abs(sB - x))) for x in sT]
    for nm, c in (('V1 image', 0), ('V2 caption', 3), ('V3 centred', 6), ('V4 top8', 9)):
        d = np.log(VB['V'][:, j, :, c]).mean((0, 1)) - np.log(VT['V'][..., c]).mean((0, 1)); print(f'  {nm:10s}', d)
    print(' mean_energy_share median by layer  Base', np.median(VB['V'][..., 15], (0, 1)).round(2))
    print('                                   Turbo', np.median(VT['V'][..., 15], (0, 1)).round(2))
    print(' top1_chan_share median by layer     Base', np.median(VB['V'][..., 16], (0, 1)).round(3))
    print('                                   Turbo', np.median(VT['V'][..., 16], (0, 1)).round(3))
VS = LV('SD3SVE_attn')
print('== SD3.5 M attn mean_energy_share by layer', np.median(VS['V'][..., 15], (0, 1)).round(2))
print('   top1 chan share', np.median(VS['V'][..., 16], (0, 1)).round(3))
