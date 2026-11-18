#!/usr/bin/env python3
"""Preflight / acceptance check for Passive Telemetry Guider CSVs (R3 capture gates).

Run it on the CSV from a 1-prompt x 1-seed x 2-step test before any long capture,
and again on every finished CSV before analysis:

    python preflight_check.py run.csv [more.csv ...] [--cfg 4.5] [--steps 40]
        [--layers 24] [--subs attn] [--seed 1001] [--prompts matched_prompts_30.tsv]

Exit code 0 = PASS, 1 = FAIL. CFG is read from the file name (".CFG4.5-") when
--cfg is not given. Checks:
  hook_mode is passive-branch; branches are {cond, uncond} at CFG>1 and {cond} at CFG 1;
  no 'pooled' rows; prompt_influence finite at CFG>1; every step of every gen has the
  same (layer, submodule, branch) set; step count; seed present (and equal to --seed);
  prompt_hash = md5(prompt_text); seed reuse across prompts (seed-design gate);
  prompt membership against a frozen list (prompt-protocol gate).
R5 additions (2026-10-08): a generation is keyed by (gen_idx, seed), because two sessions appended to one CSV
reuse gen_idx; a CSV without a header row is read with the frozen 21-column schema (warning); nonfinite
g/gap/H cells are reported (those generations are dropped in analysis); a missing output-image column is a
D14 warning; the frozen prompt list is parsed as TSV with quotes (matched_prompts_30.tsv stores one prompt as a
quoted field starting with a newline; matched_prompts_30_v2.tsv is the clean copy).
"""
import argparse, csv, hashlib, math, re, sys
from collections import defaultdict

SCHEMA = ['gen_idx', 'token_idx', 'token_type', 'layer', 'submodule', 'g_micro', 'phi_micro', 'phi_micro_skip',
          'spectral_gap', 'sv_entropy', 'sigma', 'seq_len', 'hidden_dim', 'hook_mode', 'prompt_hash', 'prompt_len',
          'gen_len', 'prompt_text', 'branch', 'prompt_influence', 'seed']
OUTPUT_ID_COLS = ('output_image', 'image_name', 'output_filename')

def load(path, notes=None):
    gens = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        first = f.readline(); f.seek(0)
        if first.startswith('gen_idx,'):
            rd = csv.DictReader(f)
        else:
            rd = csv.DictReader(f, fieldnames=SCHEMA)
            if notes is not None:
                notes.append('no header row; read with the frozen 21-column schema (writer bug: check the CSV session setting)')
        cols = rd.fieldnames or []
        for r in rd:
            gens[(r['gen_idx'], r.get('seed', ''))].append(r)
    if notes is not None:
        idx = defaultdict(set)
        for gi, sd in gens:
            idx[gi].add(sd)
        dup = sorted(k for k, v in idx.items() if len(v) > 1)
        if dup:
            notes.append(f'gen_idx reused for different seeds ({len(dup)} values, e.g. {dup[0]}): two sessions appended to one CSV; '
                         'generations are keyed by (gen_idx, seed)')
        if not any(c in cols for c in OUTPUT_ID_COLS):
            notes.append('no output-image column (D14: save the output file name keyed to seed and prompt_hash)')
    return gens

def frozen_prompts(path):
    with open(path, encoding='utf-8', newline='') as f:
        if path.endswith('.tsv'):
            rows = [r for r in csv.reader(f, delimiter='\t') if r and any(x.strip() for x in r)]
            if rows and rows[0][0].strip().lower() == 'domain':
                rows = rows[1:]
            return {r[-1].strip() for r in rows}
        return {l.strip() for l in f if l.strip()}

def check(path, a):
    fails, warns = [], []
    m = re.search(r'CFG(\d+(?:\.\d+)?)', path)
    cfg = a.cfg if a.cfg is not None else (float(m.group(1)) if m else None)
    if cfg is None:
        warns.append('CFG unknown (pass --cfg); branch rule not applied')
    notes = []
    gens = load(path, notes)
    warns.extend(notes)
    if not gens:
        return ['no data rows (header only?)'], warns, {}
    need = None if cfg is None else ({'cond'} if abs(cfg - 1) < 1e-9 else {'cond', 'uncond'})
    prompts, seed_prompts = {}, defaultdict(set)
    for key, rows in gens.items():
        g = f'{key[0]} (seed {key[1]})'
        hm = {r.get('hook_mode', '') for r in rows}
        if hm != {'passive-branch'}:
            fails.append(f'gen {g}: hook_mode {sorted(hm)} (need passive-branch)')
        br = {r.get('branch', '') for r in rows}
        if 'pooled' in br:
            fails.append(f'gen {g}: pooled rows (branch could not be attributed)')
        if need is not None and br != need:
            fails.append(f'gen {g}: branches {sorted(br)} (need {sorted(need)})')
        if need == {'cond', 'uncond'}:
            bad = [r for r in rows if r.get('prompt_influence', '') in ('', 'nan')
                   or not math.isfinite(float(r['prompt_influence']))]
            if bad:
                fails.append(f'gen {g}: {len(bad)}/{len(rows)} rows without finite prompt_influence')
        bad_num = 0
        for r in rows:
            try:
                if not all(math.isfinite(float(r[k])) for k in ('g_micro', 'spectral_gap', 'sv_entropy')):
                    bad_num += 1
            except (ValueError, TypeError, KeyError):
                bad_num += 1
        if bad_num:
            warns.append(f'gen {g}: {bad_num} row(s) with nonfinite g/gap/H (generation is dropped in analysis)')
        steps = defaultdict(set)
        for r in rows:
            steps[int(r['token_idx'])].add((r['layer'], r['submodule'], r.get('branch', '')))
        sets = list(steps.values())
        if any(s != sets[0] for s in sets):
            fails.append(f'gen {g}: steps carry different layer/submodule/branch sets (incomplete capture)')
        layers = {k[0] for k in sets[0]}; subs = {k[1] for k in sets[0]}
        if a.layers and len(layers) != a.layers:
            fails.append(f'gen {g}: {len(layers)} layers (expected {a.layers})')
        if a.subs and subs != set(a.subs.split(',')):
            fails.append(f'gen {g}: submodules {sorted(subs)} (expected {a.subs})')
        n_steps = len(steps)
        gl = rows[0].get('gen_len', '-1')
        if a.steps and n_steps != a.steps:
            fails.append(f'gen {g}: {n_steps} steps (expected {a.steps})')
        elif gl not in ('', '-1') and int(gl) != n_steps:
            fails.append(f'gen {g}: {n_steps} steps written but gen_len={gl} (partial run?)')
        seeds = {r.get('seed', '') for r in rows}
        if seeds == {''}:
            fails.append(f'gen {g}: seed missing')
        elif a.seed is not None and seeds != {str(a.seed)}:
            fails.append(f'gen {g}: seed {sorted(seeds)} (expected {a.seed})')
        text = next((r['prompt_text'] for r in rows if r.get('prompt_text')), '')
        ph = rows[0].get('prompt_hash', '')
        if text and ph and int(ph) != int(hashlib.md5(text.encode()).hexdigest()[:8], 16):
            fails.append(f'gen {g}: prompt_hash does not match md5(prompt_text)')
        prompts[g] = text
        for s in seeds:
            seed_prompts[s].add(text)
    reused = {s: len(p) for s, p in seed_prompts.items() if len(p) > 1}
    if reused:
        warns.append('seed reused across prompts: ' + ', '.join(f'{s} x{n}' for s, n in reused.items())
                     + ' (seed-design gate: use a unique seed pair per prompt for fate/commitment work)')
    if a.prompts:
        fz = frozen_prompts(a.prompts); got = set(prompts.values())
        extra = got - fz
        if extra:
            fails.append(f'{len(extra)} prompt(s) not in the frozen list, e.g. {sorted(extra)[0][:70]!r}')
    return fails, warns, prompts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+'); ap.add_argument('--cfg', type=float)
    ap.add_argument('--steps', type=int); ap.add_argument('--layers', type=int)
    ap.add_argument('--subs'); ap.add_argument('--seed', type=int); ap.add_argument('--prompts')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    allp, bad = set(), 0
    for p in a.csv:
        fails, warns, prompts = check(p, a)
        allp |= set(prompts.values())
        print(f"{'FAIL' if fails else 'PASS'}  {p}  ({len(prompts)} gens)")
        for f in (fails[:8] if a.quiet else fails):
            print('   x', f)
        if a.quiet and len(fails) > 8:
            print(f'   ... {len(fails) - 8} more')
        for w in warns:
            print('   !', w)
        bad += bool(fails)
    if a.prompts:
        missing = frozen_prompts(a.prompts) - allp
        print(f'frozen prompts missing across all files: {len(missing)}')
        for m in sorted(missing):
            print('   -', m[:90])
        bad += bool(missing)
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
