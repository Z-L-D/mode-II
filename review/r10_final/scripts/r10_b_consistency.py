"""R10-4: cross-paper numeric consistency (preregistered, R10_PREREGISTRATION.md).
Each row is a quantity quoted in both papers, with the numeric strings it should carry. The script searches both
main.tex files (TeX stripped of the \\ and $ characters) and reports the strings present in only one. Output: out/R10_4_consistency.csv."""
import os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'out')
PAP = os.path.join(HERE, '..', '..', '..', 'papers')
tex = {p: re.sub(r'[\\$,{}~]', '', open(f'{PAP}/{p}/main.tex').read()) for p in ['mode-I', 'mode-II']}
CHECKS = [
    ('lineage tangent attn/MLP, pooled archive', ['0.94', '0.97']),
    ('capture ladder: unconditional', ['0.981', '0.989']),
    ('capture ladder: branch mean', ['0.988', '0.991']),
    ('capture ladder: conditional CFG 4', ['0.992']),
    ('capture ladder: Base CFG 1', ['0.999']),
    ('unrelated-pair tangent bound', ['0.24']),
    ('SD3.5 M-L tangent', ['-0.25']),
    ('hierarchy margins', ['-0.049', '+0.047', '+0.131', '+0.152', '+0.703']),
    ('Qwen derivative r', ['0.63', '-0.12', '-0.06']),
    ('Qwen 1.5-3B interval and null', ['0.61', '0.65', '0.38']),
    ('abliteration prefill tangent', ['0.9998']),
    ('D10 derivative correlations', ['0.937', '0.007']),
    ('LLM RSA ranges', ['0.44', '0.48', '0.72', '0.89']),
    ('pooling excess held-out R2', ['0.62', '0.64']),
    ('ZIT VR4/walk', ['1.13']),
    ('n-stability of the lineage tangent', ['0.937', '0.946', '0.965', '0.968']),
    ('divergence token median', ['token 7']),
    ('prereg SHA prefixes R8, R9', ['acc578ac', '09c1f170']),
    ('Round 6 confirmed/refuted', ['17', '5']),
    ('nested i.i.d. retirement (R9-B1)', ['R9-B1']),
]
rows = []
for q, nums in CHECKS:
    for n in nums:
        a, b = n in tex['mode-I'], n in tex['mode-II']
        rows.append(dict(quantity=q, string=n, in_mode_I=a, in_mode_II=b, status='both' if a and b else ('mode-I only' if a else ('mode-II only' if b else 'neither'))))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/R10_4_consistency.csv', index=False)
print(D.to_string(index=False))
print(f"{(D.status == 'both').sum()} of {len(D)} strings in both papers")
