#!/usr/bin/env python3
"""Model-identity gate for the Base reruns: is this capture really NOT the Turbo model?

    python fingerprint_check.py --model zimage <csv> [<csv> ...]
    python fingerprint_check.py --model ace15  <csv> [<csv> ...]

At the first step (sigma = 1) the depth profile of mean log g over layers is mostly a fingerprint of the weights.
It is compared (shape only: Pearson r of the mean-removed profiles) with the Turbo reference in turbo_fingerprints.json:
the Z-Image Turbo archive (480 gens) or the ACE-Step 1.5 Turbo run (120 gens). Only cond / all rows at step 0 are used.

Reference values (2026-10-18): Turbo against itself under other settings 0.99-1.00 (25 vs 8 steps, CFG 1 vs 4, pooled or not);
Zac's May Z-Image Base run 0.79 (attn) / 0.59 (mlp); Mochi across node versions and CFG 0.994.
Verdict per submodule, Z-Image: r >= 0.98 TURBO (stop: wrong model file); r < 0.95 NOT TURBO (pass); otherwise UNCLEAR.
ACE-Step is noisier: single-genre files of the Turbo run itself give 0.92-0.997 (attn) and 0.96-0.997 (mlp), and there is no
ACE Base reference yet, so its thresholds are provisional: r >= 0.92 TURBO; r < 0.85 NOT TURBO; otherwise UNCLEAR.
UNCLEAR means: check the output-file metadata (capture_code_r4/scan_png_models.py for images). Exit code 0 only if every
submodule passes.
Works on a 1-prompt preflight: one generation is enough for a clear verdict, more is better.
"""
import argparse, csv, json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=['zimage', 'ace15'], required=True); ap.add_argument('csv', nargs='+')
    a = ap.parse_args()
    ref = json.load(open(os.path.join(HERE, 'turbo_fingerprints.json')))
    acc = defaultdict(lambda: defaultdict(list))
    for p in a.csv:
        with open(p, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('token_idx') != '0' or r.get('branch', 'all') not in ('cond', 'all'):
                    continue
                try:
                    v = float(r['g_micro'])
                except (ValueError, KeyError):
                    continue
                if v > 0 and math.isfinite(v):
                    acc[r['submodule']][int(r['layer'])].append(math.log(v))
    ok = True
    for sub in ('attn', 'mlp'):
        R = ref.get(f'{a.model}_turbo_{sub}')
        if sub not in acc:
            print(f'{sub}: no step-0 rows'); ok = False; continue
        L = sorted(acc[sub])
        if len(L) != len(R):
            print(f'{sub}: {len(L)} layers, reference has {len(R)}'); ok = False; continue
        x = [sum(acc[sub][l]) / len(acc[sub][l]) for l in L]
        mx, mr = sum(x) / len(x), sum(R) / len(R)
        num = sum((u - mx) * (w - mr) for u, w in zip(x, R))
        r = num / math.sqrt(sum((u - mx) ** 2 for u in x) * sum((w - mr) ** 2 for w in R))
        n = len(acc[sub][L[0]])
        hi, lo = (0.98, 0.95) if a.model == 'zimage' else (0.92, 0.85)
        verdict = 'TURBO (wrong model file: stop)' if r >= hi else 'NOT TURBO (pass)' if r < lo else 'UNCLEAR (check the output metadata)'
        print(f'{sub}: depth-profile r with {a.model} Turbo = {r:.3f} over {n} generation(s): {verdict}')
        ok &= r < lo
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
