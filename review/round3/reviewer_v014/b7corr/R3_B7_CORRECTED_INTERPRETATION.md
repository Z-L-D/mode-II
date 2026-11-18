# R3 B7 correction — four-crossing interpretation

## Correction

The v0.13 statement that the historical four-crossing SD3.5-Medium beta topology “is not robust to passive capture” is too strong and is withdrawn.

The comparison confounded at least:

1. **instrument/capture regime** — historical resolution suite: legacy EMAG-hook schema; new matched study: passive schema;
2. **actual denoising traversal** — historical resolution suite: **100 steps**; new matched study: **40 steps**;
3. **prompt composition** — historical resolution suite: one long pyramid-construction prompt; new matched study: multi-theme campaign.

Therefore the existing data do not identify which factor changed the crossing topology.

## What the existing-data re-audit establishes

The historical 100-step trajectory retains its four-crossing `-+-+-` topology at 512, 768 and 1024 when:

- all observations below the minimum sigma reached by the new 40-step schedule are removed; and
- the 100-step observations are thinned to the nearest 40 sigma locations of the new schedule.

So **simple plotting density and omission of the very-late sigma tail are not sufficient to erase the fourth crossing in the historical trajectory**.

This does **not** emulate a genuine 40-step denoising run. A 40-step sampler follows a different numerical trajectory because its state at a given sigma depends on the preceding integration schedule. The 100-vs-40 traversal confound therefore remains real.

## Correct status

> The legacy 100-step EMAG-schema SD3.5-Medium runs show a robust four-crossing `-+-+-` beta topology. The newer 40-step passive runs show a three-crossing `+-+-` topology. Existing data do not determine whether the difference is caused by capture functional, denoising-step density/trajectory, prompt composition, or their interaction.

The four-crossing card is **UNRESOLVED ACROSS INSTRUMENT / TRAVERSAL**, not refuted.

## Required matched experiment

Minimum decisive capture:

- SD3.5-Medium;
- 1024 first priority, then 512/768 if needed;
- CFG 1.0 to avoid branch mixing;
- **100 denoising steps**;
- current Passive instrument;
- the historical pyramid prompt;
- ideally the original seeds if recoverable, otherwise >=30 seeds.

Best experiment: same prompt/seeds/100-step schedule under both the historical Diagnostic-EMAG build and the current Passive build.

This becomes a **pre-paper gate if the four-crossing/rebound result remains in Mode-I or Mode-II**. If the result is omitted, it can be deferred.

## Consequence for B7

The formal B7 preregistration was already NOT SCORABLE because the required branch-separated CFG>1 rows were absent. That formal status does not change.

What changes is the post-hoc interpretation:

- v0.13: “strong evidence against the legacy four-crossing card” — **withdrawn**;
- v0.14: “new 40-step passive study differs, but instrument and traversal are confounded” — **current**.
