# Independent audit preregistration — new Round-4 captures

Frozen before numerical inspection of `uncond-cond+deepstep-telemetry.zip`.

## Disclosure
The uploaded v0.15 ledger already reveals the high-level results of Round-4 P1–P5.
Therefore:
- P1–P5 are **replication/audit targets**, not blind independent predictions.
- R1/R2 are **genuinely preregistered new-capture tests** because the raw numerical
  contents of `uncond-cond+deepstep-telemetry.zip` have not yet been opened.

The controlling thresholds are copied from `R4_PREREGISTRATION.md` and are not
modified here.

## R1 — repaired per-branch core

Expected only if the actual capture satisfies the preregistered design.

- R1a: preflight PASS:
  - CFG4.5 hook_mode = passive-branch;
  - branches = {cond, uncond};
  - CFG1 branch = {cond};
  - finite prompt_influence at CFG4.5;
  - 24 layers at every step.
- R1b: cond branch at CFG4.5 equals CFG1 at step 0:
  median |log(g_cond,4.5/g_CFG1)| <= 1e-3.
- R1c: with unique seed pairs per prompt, step-0 seed-identity held-out-prompt
  accuracy <= 0.65.
- R1d: E5 sigma50 on cond branch should lie in [0.85,0.96] for CFG1 and CFG4.5.
- R1e: with unique seeds, CFG1 cond VR(4) <= 1.0.
- R1f: guidance contracts across-seed spread:
  median SD(cond,4.5)/SD(CFG1) over last three quarters < 0.7.
- R1g: cond/uncond become distinguishable somewhere at sigma <= 0.9:
  prompt_influence < 0.999.
- R1h: report branch-mean/cond difference without a preregistered sign threshold.

### Audit guard
If the uploaded capture reuses global seeds 1001/1002 across prompts rather than
the preregistered unique-pair design, R1c/R1d/R1e are NOT SCORABLE under the
formal R1 rules. They may be analyzed post hoc with in-fold seed centering but
must not be called R1 hits/misses.

## R2 — decisive 100-step passive pyramid test

- R2a: current-passive 100-step CFG1 pyramid run should show the passive topology:
  first sign '+', sequence != '-+-+-', <= 3 crossings.
  If '-+-+-' with four crossings appears, actual 100-step traversal/prompt is
  sufficient to restore the legacy topology and the capture-functional
  explanation is falsified.
- R2b: matched-sigma level should agree with the 40-step passive arm:
  median |log(g_100/g_40)| <= 0.05, with prompt-composition spread reported as
  a control.

## Additional audits fixed before calculation

A1. Reproduce the supplied Round-4 P1–P5 result JSON directly from the source
scripts where the required inputs are packaged or recoverable.

A2. Re-run all topology calculations from raw CSVs using a separately written,
minimal implementation of the frozen 24-bin beta estimator.

A3. Bootstrap R2a over seeds. The preregistered verdict uses the pooled curve;
bootstrap stability is post hoc and cannot change that verdict.

A4. For any CFG4.5 per-branch run, quantify:
  - cond vs uncond g, gap, H by sigma;
  - branch-mean vs cond;
  - prompt_influence by sigma;
  - whether the difference is concentrated in operating point or normalized
    depth geometry.
These are post-hoc mechanism analyses unless already named in R1.

A5. Revisit every v0.15 gate only after the formal R1/R2 scorecard is fixed.
