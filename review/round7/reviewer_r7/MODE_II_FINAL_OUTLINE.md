# Mode-II final outline — mathematical paper

## Working thesis
The telemetry is best modeled as a family of explicitly nonautonomous depth-time flows observed through low-dimensional spectral coordinates. Coarse scalar projections can be similar across unrelated architectures, while registered fields, content relations and especially tangent dynamics define successively stricter equivalence notions. Distillation preserves a much stronger dynamical relation than ordinary cross-architecture resemblance, but the data reject a single universal microscopic vector field.

## 1. Mathematical problem statement
- Define model m, prompt p, seed s, layer/depth u and sampling/context clock τ.
- Define the observed spectral state z=(g,gap,H).
- Distinguish trajectory similarity, field similarity, shape similarity, content relational similarity and tangent/dynamical similarity.
- State what is not claimed: full-state conjugacy, orbit equivalence, RG universality class, thermodynamic criticality.

## 2. Spectral observables
### 2.1 Uncentred second moment
- X^T X / N and singular/eigen spectrum.
### 2.2 g, gap and H
- Exact definitions and normalization.
- Distinguish the project’s g from Roy–Vetterli effective rank terminology where needed.
### 2.3 Capture-functional dependence
- pooled versus branch statistics.

**Necessary graph:** II-F1.

## 3. Pooling theorem
- Prove the concavity bound for tr sqrt(M) / the induced g change under pooling.
- Show numerical validation on identical trajectories.
- Explain why pooling can raise g without changing the underlying conditional trajectory.

**Necessary graph:** II-F2.

## 4. Projection degeneracy and equivalence hierarchy
- Formalize P(z) as a lower-dimensional projection.
- Show that high corr(PΓ_a,PΓ_b) does not identify a shared vector field.
- Data anchor: linear scalar lineage margin <0 while field/shape/tangent margins rise monotonically.
- Define increasingly stringent empirical relations:
  envelope resemblance < field correspondence < registered shape < tangent correspondence.

**Necessary graph:** II-F3.

## 5. Nonautonomous flow family
- Start from dz/du = F_m(z,u,τ).
- Show why beta_g=F(g) and even F(g,gap,H) are insufficient.
- D08/D09: flexible u,τ account for most predictive variance; state adds smaller residual information.
- D03/D15: cross-architecture transfer fails; Base↔Turbo remains exceptional.
- Preferred decomposition:
  dz/du = F_m^(0)(u,τ) + R_m(z,u,τ).

**Necessary graph:** II-F4.

## 6. Trajectory family and structured neighborhoods
- Descriptive decomposition:
  M_{m,p,s}(τ) ≈ Γ_m(τ)+δ_{m,p}(τ)+ε_{m,p,s}(τ).
- Clarify non-orthogonality.
- System×time dominates global variance, while prompt identity is large within fixed model-state cells.
- Define deterministic content geometry and stochastic neighborhoods separately.

**Necessary graph:** II-F5.

## 7. Dynamical inheritance under distillation
- Equal-sigma registration rather than a learned compressed clock.
- Tangent cosine / depth derivative as the central lineage statistic.
- D10: state-space alignment can improve unrelated representation similarity without aligning derivatives.
- Mathematical interpretation: partial orientation-preserving trajectory correspondence in the measured coordinates, weaker than formal orbit equivalence.

**Necessary graph:** II-F6.

## 8. Content geometry is a different quotient
- Define prompt-distance/RSA object.
- Show broad cross-model relational geometry beyond theme and text.
- Show weak rank coupling to tangent similarity (rho≈0.297 in final pass).
- Interpretation: shared content organization and shared microscopic dynamics are distinct equivalence relations.

**Necessary graph:** II-F7.

## 9. Two distinct phase/topology constructions
### 9.1 Evolution along sampling clock
- Piecewise sign of d g_bar/dτ in Z-Image MLP: +−+.
- Robustness under bootstrap and clean branches.
### 9.2 Evolution along depth
- β_layer(g)=g(u+du)-g(u), +−+− sign topology.
- Per-sigma recurrence across architectures.
- Surrogate classes demonstrate which channels contain non-null layer coupling.
### 9.3 Why crossing ratios are not invariants
- Capture, scale, sample and marginal null sensitivity.

**Necessary graph:** II-F8.

## 10. Kinematic nulls and surrogate theory
- Derive E[phi]→−1/2 for adjacent differences under independent vectors with common covariance.
- Derive stride-2 expectation 0.
- Present cloud-beta surrogate A/B logic.
- Explain why a null-generic crossing is not a phase boundary.

**Necessary graph:** II-F9.

## 11. Local linearization
- X_{u+du}=A X_u+b as a local descriptive model.
- Complex pairs appear only in selected regimes.
- Domain census: contraction modulus more stable than rotation.
- Geometry-derived Base→Turbo map closes MLP local eigenmodes but not attention.
- Conclude: recurrent relaxational component; rotational component architecture/state dependent.

**Necessary graph:** II-F10.

## 12. Estimator theory for seed predictability
- Show why threshold-crossing sigma50 depends on ceiling/sample size.
- Define relative onset.
- Define VR(q) and calibrate against finite-length random walk/AR/iid references.
- Explicitly separate estimator calibration from claims about commitment.

**Necessary graph:** II-F11.

## 13. What kind of universality remains possible?
- Reject universal microscopic F in current coordinates.
- Candidate universality can only be coarser: ordering/topology, quotient geometry, or equivalence class under projections.
- State the empirical hierarchy:
  coarse envelope recurrence;
  cross-model content geometry;
  lineage-specific tangent dynamics.
- List conditions a stronger mathematical universality claim would have to satisfy.

**Necessary graph:** II-F12.

## 14. Open causal mathematics
- Committor q(x) and clone experiment for a true separatrix.
- Response/Jacobian and susceptibility under finite perturbations.
- Context-invariant AR observables.
- Massive-activation robustness of uncentred spectral statistics.
- Additional submodule measurements needed for cross-architecture sigma-axis claims.

## Appendices
- Full proofs.
- Exact estimator definitions.
- Null/surrogate derivations.
- Intrinsic arc-length negative result.
- Legacy rebound/complex-mode study labeled by instrument.
- Complete preregistration/falsification table.
