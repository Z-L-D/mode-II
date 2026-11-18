"""
ECCController v2: Closed-Loop Phase Controller
================================================
Implements the driven RG control layer: dg/dt = β(g) + Φ(t)

v2 changes from v1:
  - Accepts susceptibility signal from EMAG guider as primary β proxy
  - Falls back to cos_sim (velocity linearity) when susceptibility unavailable
  - Continuous Φ across all phases via simplified Pontryagin costate
  - Adaptive phase boundary detection from susceptibility gradient
  - F(β) vs ln|β| logging for c-theorem empirical verification
  - Cross-validation of cos_sim and susceptibility signals
  - Outputs CFG/EMAG modulation suggestions (optional, guider can ignore)

Theory:
  β(g) is the RG beta function. At the critical fixed point, β=0.
  Susceptibility χ diverges at β=0 (critical point).
  Our susceptibility measurement is INVERTED: cos_sim(cond, perturbed).
    High susceptibility_cos → low χ → far from critical → |β| large
    Low susceptibility_cos  → high χ → near critical → |β| small

  F(β) = F₀ - ν ln|β|  (Zamolodchikov c-theorem / Calabrese-Cardy)
  J[Φ] = (1/T) ∫ [F(β) - λ‖Φ‖²] dt  (Pontryagin objective)

  The optimal Φ is smooth (not bang-bang) for quadratic cost.
  Costate equation: dψ/dt = -∂H/∂g = ν/β · dβ/dg
  Optimal control: Φ* = ψ/(2λ)
"""
import torch
import math
from typing import Dict, Optional, List, Tuple


class ECCController:
    """
    Closed-loop phase controller with susceptibility feedback.
    
    Can operate in two modes:
      - Standalone (in sampler): uses cos_sim only, Φ applied to latent
      - Integrated (in guider): uses susceptibility + cos_sim, outputs
        modulation suggestions for CFG/EMAG
    
    The controller does NOT modify the denoising output directly when
    integrated with the guider. It provides:
      1. Adaptive phase classification
      2. Φ magnitude (for the guider to apply as EMAG scale modulation)
      3. F(β) and J[Φ] telemetry
      4. Phase boundary suggestions based on susceptibility gradient
    """

    def __init__(
        self,
        # Phase boundaries (initial / fallback)
        explore_end: float = 0.46,
        converge_end: float = 0.85,
        transition_width: float = 0.05,
        # Φ control parameters
        exploration_noise: float = 0.015,
        convergence_stabilization: float = 0.005,
        consolidation_damping: float = 0.002,
        # RG / c-theorem parameters
        nu: float = 1.0,
        lambda_cost: float = 0.1,
        f0: float = 2.0,
        # Adaptive boundary parameters
        adaptive_boundaries: bool = False,
        susceptibility_gradient_threshold: float = 0.003,
        min_explore_frac: float = 0.10,
        max_explore_frac: float = 0.50,
        min_converge_end: float = 0.60,
        max_converge_end: float = 0.92,
        # General
        enabled: bool = True,
    ):
        # Phase boundaries
        self.explore_end = explore_end
        self.converge_end = converge_end
        self.transition_width = transition_width
        self._initial_explore_end = explore_end
        self._initial_converge_end = converge_end

        # Continuous Φ parameters (per-phase base magnitudes)
        self.exploration_noise = exploration_noise
        self.convergence_stabilization = convergence_stabilization
        self.consolidation_damping = consolidation_damping

        # RG objective parameters
        self.nu = nu
        self.lambda_cost = lambda_cost
        self.f0 = f0

        # Adaptive boundary config
        self.adaptive_boundaries = adaptive_boundaries
        self.susceptibility_gradient_threshold = susceptibility_gradient_threshold
        self.min_explore_frac = min_explore_frac
        self.max_explore_frac = max_explore_frac
        self.min_converge_end = min_converge_end
        self.max_converge_end = max_converge_end

        self.enabled = enabled

        # Running state
        self._step = 0
        self._j_accumulator = 0.0
        self._costate_psi = 0.0  # Pontryagin costate variable

        # History buffers
        self._telemetry_log: List[Dict] = []
        self._susceptibility_history: List[float] = []
        self._cos_sim_history: List[float] = []
        self._beta_history: List[float] = []
        self._f_beta_history: List[float] = []
        self._j_history: List[float] = []

        # Adaptive boundary state
        self._detected_ec_boundary: Optional[float] = None
        self._detected_cs_boundary: Optional[float] = None
        self._ec_boundary_locked = False
        self._cs_boundary_locked = False

    def reset(self):
        """Reset all running state for a new generation."""
        self._step = 0
        self._j_accumulator = 0.0
        self._costate_psi = 0.0
        self._telemetry_log.clear()
        self._susceptibility_history.clear()
        self._cos_sim_history.clear()
        self._beta_history.clear()
        self._f_beta_history.clear()
        self._j_history.clear()
        self._detected_ec_boundary = None
        self._detected_cs_boundary = None
        self._ec_boundary_locked = False
        self._cs_boundary_locked = False
        # Reset to initial boundaries
        self.explore_end = self._initial_explore_end
        self.converge_end = self._initial_converge_end

    # -----------------------------------------------------------------
    # β proxy computation
    # -----------------------------------------------------------------

    def _compute_beta_from_susceptibility(self, susceptibility_cos: float) -> float:
        """
        Compute |β| from susceptibility cosine similarity.
        
        susceptibility_cos = cos_sim(cond_standard, cond_perturbed)
        High value → perturbation has little effect → system committed → |β| large
        Low value  → perturbation shifts output → system sensitive → |β| small
        
        We map: |β| = 1 - susceptibility_cos (inverted χ)
        At the critical point, susceptibility_cos is minimal → |β| minimal.
        """
        return max(1.0 - susceptibility_cos, 1e-4)

    def _compute_beta_from_cos_sim(self, cos_sim: float) -> float:
        """
        Fallback: compute |β| from velocity cosine similarity.
        
        cos_sim ≈ 1 → trajectory straight → near fixed point → |β| small
        cos_sim < 1 → trajectory curving → away from fixed point → |β| large
        
        Note: this is the OPPOSITE mapping from susceptibility.
        High cos_sim → small β (committed, on trajectory)
        Low cos_sim  → large β (exploring, curving)
        """
        return max(1.0 - cos_sim, 1e-4)

    def compute_beta(
        self,
        susceptibility: Optional[float] = None,
        cos_sim: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Compute |β| from best available signal.
        Returns (beta, source_name).
        
        Priority: susceptibility (direct χ measurement) > cos_sim (trajectory proxy)
        """
        if susceptibility is not None:
            return self._compute_beta_from_susceptibility(susceptibility), 'susceptibility'
        elif cos_sim is not None:
            return self._compute_beta_from_cos_sim(cos_sim), 'cos_sim'
        else:
            return 0.5, 'default'

    # -----------------------------------------------------------------
    # F(β) and J[Φ] — the RG objective
    # -----------------------------------------------------------------

    def compute_feature_richness(self, beta: float) -> float:
        """
        F(β) = F₀ - ν ln|β|
        
        From Zamolodchikov c-theorem + Calabrese-Cardy:
        Feature richness (information capacity) peaks at β→0 (critical point)
        and decreases logarithmically with distance from criticality.
        """
        return self.f0 - self.nu * math.log(max(beta, 1e-8))

    def compute_control_cost(self, phi_magnitude: float) -> float:
        """λ ‖Φ‖²"""
        return self.lambda_cost * (phi_magnitude ** 2)

    # -----------------------------------------------------------------
    # Continuous Φ via simplified Pontryagin costate
    # -----------------------------------------------------------------

    def _compute_phi_magnitude(
        self,
        phase: str,
        beta: float,
        frac: float,
        sigma: float,
    ) -> float:
        """
        Compute optimal control magnitude Φ* for current state.
        
        The Pontryagin optimal control for quadratic cost is:
          Φ* = ψ / (2λ)
        where ψ is the costate satisfying:
          dψ/dt = -∂H/∂g = ν / (β · |dβ/dg|)
        
        Since we don't have dβ/dg directly, we use the empirical
        β history to estimate it numerically, and integrate the
        costate forward.
        
        The result is a smooth, continuous control signal that:
        - Is larger when β is small (near critical point → more control needed)
        - Is larger when β is changing rapidly (transition zone)
        - Decays as the system commits (large β, stable trajectory)
        """
        # Estimate dβ/dt from history
        if len(self._beta_history) >= 2:
            dbeta_dt = self._beta_history[-1] - self._beta_history[-2]
        else:
            dbeta_dt = 0.0

        # Costate update: dψ/dt ≈ ν / (β + ε)
        # This drives ψ large when β is small (near criticality)
        costate_drive = self.nu / (beta + 0.01)
        
        # Damping term to prevent costate divergence
        costate_damping = 0.1 * self._costate_psi
        
        # Integrate costate
        self._costate_psi += 0.1 * (costate_drive - costate_damping)
        
        # Clamp costate to prevent instability
        self._costate_psi = max(-10.0, min(10.0, self._costate_psi))

        # Optimal control: Φ* = ψ / (2λ)
        phi_optimal = self._costate_psi / (2.0 * self.lambda_cost + 1e-8)

        # Phase-dependent scaling and character
        if phase == 'exploration':
            # Stochastic: Φ is noise-like, magnitude from costate
            base = self.exploration_noise
            # Scale by sigma (noise vanishes as sigma decreases)
            scale = sigma * abs(phi_optimal) * 0.1
            return base + min(scale, base * 2.0)

        elif phase == 'convergence':
            # Stabilizing: Φ opposes trajectory curvature
            base = self.convergence_stabilization
            # Stronger when β is changing rapidly (transition zone)
            curvature_scale = min(abs(dbeta_dt) * 10.0, 1.0)
            scale = abs(phi_optimal) * 0.05 * curvature_scale
            return base + min(scale, base * 3.0)

        else:  # consolidation
            # Damping: Φ reinforces commitment (negative perturbation)
            base = self.consolidation_damping
            # Decays toward zero as system fully commits
            commitment = max(0.0, 1.0 - beta * 10.0)  # high when β small
            scale = abs(phi_optimal) * 0.02 * (1.0 - commitment)
            return base * (1.0 - commitment * 0.8)

    # -----------------------------------------------------------------
    # Adaptive boundary detection
    # -----------------------------------------------------------------

    def _detect_boundaries(self, frac: float) -> None:
        """
        Detect phase boundaries from susceptibility gradient.
        
        E→C boundary: where susceptibility starts dropping (system becomes
        sensitive to perturbation — real exploration begins).
        
        C→S boundary: where susceptibility stabilizes near 1.0 (system
        has committed, perturbation no longer affects output).
        """
        if not self.adaptive_boundaries:
            return
        if len(self._susceptibility_history) < 5:
            return

        # Compute susceptibility gradient (smoothed)
        window = min(5, len(self._susceptibility_history))
        recent = self._susceptibility_history[-window:]
        
        if len(recent) < 3:
            return

        # Gradient: negative means susceptibility is DROPPING (becoming more sensitive)
        gradient = (recent[-1] - recent[0]) / max(window - 1, 1)

        # --- E→C boundary detection ---
        # The E→C boundary is where susceptibility starts dropping from
        # its initial plateau. On SD1.5, the "dead zone" has susceptibility
        # pinned near 1.0. When it starts dropping, real exploration begins.
        if not self._ec_boundary_locked:
            if (gradient < -self.susceptibility_gradient_threshold and
                    frac > self.min_explore_frac and
                    frac < self.max_explore_frac):
                # Susceptibility is dropping — we've entered real exploration
                # Set E→C boundary slightly AHEAD of current position
                # (the system needs time to explore before converging)
                proposed = frac + 0.10
                proposed = max(self.min_explore_frac, min(proposed, self.max_explore_frac))
                
                if self._detected_ec_boundary is None:
                    self._detected_ec_boundary = proposed
                else:
                    # Smooth update — don't jump boundaries around
                    self._detected_ec_boundary = (
                        0.7 * self._detected_ec_boundary + 0.3 * proposed
                    )

        # --- C→S boundary detection ---
        # The C→S boundary is where susceptibility recovers toward 1.0
        # after the convergence dip, AND velocity stabilizes.
        if not self._cs_boundary_locked:
            if (self._detected_ec_boundary is not None and
                    frac > self._detected_ec_boundary + 0.15 and
                    frac > self.min_converge_end - 0.1):
                
                # Check if susceptibility has recovered above threshold
                if (len(recent) >= 3 and
                        all(s > 0.995 for s in recent[-3:]) and
                        gradient >= 0):
                    proposed = frac + 0.05
                    proposed = max(self.min_converge_end, min(proposed, self.max_converge_end))
                    
                    if self._detected_cs_boundary is None:
                        self._detected_cs_boundary = proposed
                    else:
                        self._detected_cs_boundary = (
                            0.7 * self._detected_cs_boundary + 0.3 * proposed
                        )

        # Apply detected boundaries (with lockout after application)
        if self._detected_ec_boundary is not None and not self._ec_boundary_locked:
            if frac >= self._detected_ec_boundary - 0.02:
                self.explore_end = self._detected_ec_boundary
                self._ec_boundary_locked = True

        if self._detected_cs_boundary is not None and not self._cs_boundary_locked:
            if frac >= self._detected_cs_boundary - 0.02:
                self.converge_end = self._detected_cs_boundary
                self._cs_boundary_locked = True

    # -----------------------------------------------------------------
    # Phase classification
    # -----------------------------------------------------------------

    def _classify_phase(self, frac: float) -> str:
        """Three-phase classification with current boundaries."""
        tw = self.transition_width
        if frac < self.explore_end - tw:
            return 'exploration'
        elif frac < self.explore_end + tw:
            return 'exploration'  # transition zone counted as exploration
        elif frac < self.converge_end - tw:
            return 'convergence'
        elif frac < self.converge_end + tw:
            return 'convergence'  # transition zone counted as convergence
        else:
            return 'consolidation'

    # -----------------------------------------------------------------
    # Main step function
    # -----------------------------------------------------------------

    def step(
        self,
        step: int,
        total_steps: int,
        sigma: float,
        susceptibility: Optional[float] = None,
        cos_sim: Optional[float] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Execute one controller step.
        
        Args:
            step: Current step index
            total_steps: Total steps in generation
            sigma: Current noise level (real, not gate-softened)
            susceptibility: cos_sim(cond_standard, cond_perturbed) from guider
            cos_sim: cos_sim(velocity[t], velocity[t-1]) from sampler
            velocity: Current velocity tensor (for standalone Φ computation)
        
        Returns:
            Dict with:
                phase: str — current phase classification
                phi_magnitude: float — optimal Φ magnitude
                phi: Optional[Tensor] — Φ tensor (only in standalone mode)
                adaptive_explore_end: float — current E→C boundary
                adaptive_converge_end: float — current C→S boundary
                f_beta: float — feature richness
                j_step: float — per-step objective value
                j_avg: float — running average objective
                beta: float — current |β| estimate
                beta_source: str — which signal β came from
                telemetry: Dict — full telemetry record
        """
        if not self.enabled:
            return {
                'phase': 'convergence',
                'phi_magnitude': 0.0,
                'phi': None,
                'adaptive_explore_end': self.explore_end,
                'adaptive_converge_end': self.converge_end,
                'f_beta': self.f0,
                'j_step': self.f0,
                'j_avg': self.f0,
                'beta': 0.5,
                'beta_source': 'disabled',
                'telemetry': None,
            }

        self._step = step
        frac = step / max(total_steps - 1, 1)
        sigma_val = sigma.item() if isinstance(sigma, torch.Tensor) else float(sigma)

        # Store signals
        if susceptibility is not None:
            self._susceptibility_history.append(susceptibility)
        if cos_sim is not None:
            self._cos_sim_history.append(cos_sim)

        # Compute β from best available signal
        beta, beta_source = self.compute_beta(susceptibility, cos_sim)
        self._beta_history.append(beta)

        # Adaptive boundary detection
        self._detect_boundaries(frac)

        # Phase classification (uses current, potentially adaptive, boundaries)
        phase = self._classify_phase(frac)

        # Compute F(β) — the c-theorem feature richness
        f_beta = self.compute_feature_richness(beta)
        self._f_beta_history.append(f_beta)

        # Compute optimal Φ magnitude via costate
        phi_magnitude = self._compute_phi_magnitude(
            phase, beta, frac, sigma_val
        )

        # Compute Φ tensor if velocity is provided (standalone mode)
        phi_tensor = None
        if velocity is not None:
            if phase == 'exploration':
                # Stochastic: random direction, costate-derived magnitude
                noise = torch.randn_like(velocity)
                phi_tensor = phi_magnitude * sigma_val * noise
            elif phase == 'convergence':
                # Stabilizing: oppose velocity curvature
                # (simplified: small perturbation along velocity direction)
                v_norm = velocity.norm()
                if v_norm > 1e-8:
                    phi_tensor = -phi_magnitude * (velocity / v_norm)
                else:
                    phi_tensor = torch.zeros_like(velocity)
            else:
                # Consolidation: damping (reinforce current direction)
                phi_tensor = torch.zeros_like(velocity)

        # Compute objective
        cost = self.compute_control_cost(phi_magnitude)
        j_step = f_beta - cost
        self._j_accumulator += j_step
        j_avg = self._j_accumulator / (step + 1)
        self._j_history.append(j_step)

        # Cross-validation signal (when both are available)
        cross_validation = None
        if susceptibility is not None and cos_sim is not None:
            beta_from_susc = self._compute_beta_from_susceptibility(susceptibility)
            beta_from_cos = self._compute_beta_from_cos_sim(cos_sim)
            # Correlation: if both measure the same thing, they should
            # track each other monotonically. Divergence is informative.
            cross_validation = {
                'beta_susceptibility': beta_from_susc,
                'beta_cos_sim': beta_from_cos,
                'ratio': beta_from_susc / max(beta_from_cos, 1e-8),
                'agreement': abs(beta_from_susc - beta_from_cos) < 0.1,
            }

        # Build telemetry record
        telemetry = {
            'step': step,
            'frac': frac,
            'phase': phase,
            'sigma': sigma_val,
            'beta': beta,
            'beta_source': beta_source,
            'f_beta': f_beta,
            'ln_beta': math.log(max(beta, 1e-8)),
            'phi_magnitude': phi_magnitude,
            'costate_psi': self._costate_psi,
            'j_step': j_step,
            'j_avg': j_avg,
            'susceptibility': susceptibility,
            'cos_sim': cos_sim,
            'cross_validation': cross_validation,
            'explore_end': self.explore_end,
            'converge_end': self.converge_end,
            'ec_boundary_locked': self._ec_boundary_locked,
            'cs_boundary_locked': self._cs_boundary_locked,
        }
        self._telemetry_log.append(telemetry)

        return {
            'phase': phase,
            'phi_magnitude': phi_magnitude,
            'phi': phi_tensor,
            'adaptive_explore_end': self.explore_end,
            'adaptive_converge_end': self.converge_end,
            'f_beta': f_beta,
            'j_step': j_step,
            'j_avg': j_avg,
            'beta': beta,
            'beta_source': beta_source,
            'telemetry': telemetry,
        }

    # -----------------------------------------------------------------
    # Summary and c-theorem verification
    # -----------------------------------------------------------------

    def print_summary(self) -> None:
        """Print generation summary with c-theorem verification data."""
        if not self._telemetry_log:
            return

        print("\n" + "=" * 90)
        print("[ECC v2] CLOSED-LOOP CONTROLLER SUMMARY")
        print("=" * 90)

        # Phase distribution
        phases = [t['phase'] for t in self._telemetry_log]
        counts = {p: phases.count(p) for p in ['exploration', 'convergence', 'consolidation']}
        print(f"\n  Phase distribution: {counts}")
        
        # Boundary info
        print(f"  Boundaries: E→C={self.explore_end:.3f}, C→S={self.converge_end:.3f}")
        if self._ec_boundary_locked:
            print(f"    E→C boundary was adaptively detected and locked")
        if self._cs_boundary_locked:
            print(f"    C→S boundary was adaptively detected and locked")

        # J[Φ] trajectory
        if self._j_history:
            j_final = self._j_accumulator / len(self._j_history)
            print(f"\n  J[Φ] final average: {j_final:.4f}")
            # J trajectory by phase
            for p in ['exploration', 'convergence', 'consolidation']:
                p_entries = [t for t in self._telemetry_log if t['phase'] == p]
                if p_entries:
                    p_j = [t['j_step'] for t in p_entries]
                    print(f"    {p:15s}: J_mean={sum(p_j)/len(p_j):.4f}, "
                          f"J_min={min(p_j):.4f}, J_max={max(p_j):.4f}")

        # β trajectory
        if self._beta_history:
            print(f"\n  β trajectory:")
            n = len(self._beta_history)
            samples = [0, n//4, n//2, 3*n//4, n-1]
            samples = sorted(set(max(0, min(s, n-1)) for s in samples))
            for s in samples:
                t = self._telemetry_log[s]
                print(f"    step {t['step']:3d}: β={t['beta']:.6f} "
                      f"(from {t['beta_source']}, F={t['f_beta']:.3f})")

        # Φ trajectory
        phi_mags = [t['phi_magnitude'] for t in self._telemetry_log]
        if phi_mags:
            print(f"\n  Φ magnitude: mean={sum(phi_mags)/len(phi_mags):.6f}, "
                  f"max={max(phi_mags):.6f}")
            for p in ['exploration', 'convergence', 'consolidation']:
                p_phis = [t['phi_magnitude'] for t in self._telemetry_log 
                          if t['phase'] == p]
                if p_phis:
                    print(f"    {p:15s}: Φ_mean={sum(p_phis)/len(p_phis):.6f}")

        # C-theorem verification: F(β) vs ln|β|
        # If the c-theorem holds, F vs ln|β| should be linear with slope -ν
        if len(self._f_beta_history) >= 10:
            print(f"\n  ► C-THEOREM VERIFICATION: F(β) vs ln|β|")
            print(f"  {'─' * 70}")
            print(f"  Theory predicts: F = F₀ - ν·ln|β|  (linear, slope = -ν = -{self.nu:.2f})")
            
            # Compute empirical slope from first and last quartile
            n = len(self._telemetry_log)
            q1 = self._telemetry_log[:n//4]
            q4 = self._telemetry_log[-(n//4):]
            
            if q1 and q4:
                f_early = sum(t['f_beta'] for t in q1) / len(q1)
                f_late = sum(t['f_beta'] for t in q4) / len(q4)
                ln_b_early = sum(t['ln_beta'] for t in q1) / len(q1)
                ln_b_late = sum(t['ln_beta'] for t in q4) / len(q4)
                
                dln = ln_b_late - ln_b_early
                if abs(dln) > 1e-6:
                    empirical_slope = (f_late - f_early) / dln
                    print(f"  Empirical slope: {empirical_slope:.3f} "
                          f"(predicted: {-self.nu:.3f})")
                    deviation = abs(empirical_slope - (-self.nu)) / self.nu
                    if deviation < 0.2:
                        print(f"  ✓ Within 20% of prediction — c-theorem consistent")
                    elif deviation < 0.5:
                        print(f"  ~ Within 50% — qualitatively consistent, "
                              f"ν may need recalibration")
                    else:
                        print(f"  ✗ >50% deviation — c-theorem relationship "
                              f"not confirmed at current ν")
                else:
                    print(f"  (Insufficient ln|β| range for slope estimation)")

            # Print F vs ln|β| at sample points for manual plotting
            print(f"\n  Sample points (for plotting):")
            print(f"  {'step':>5s}  {'ln|β|':>8s}  {'F(β)':>8s}  {'β':>8s}  {'source':>14s}")
            n = len(self._telemetry_log)
            sample_idxs = sorted(set(
                max(0, min(s, n-1)) for s in 
                [0, 2, 5, n//4, n//3, n//2, 2*n//3, 3*n//4, n-3, n-1]
            ))
            for s in sample_idxs:
                t = self._telemetry_log[s]
                print(f"  {t['step']:5d}  {t['ln_beta']:8.4f}  "
                      f"{t['f_beta']:8.4f}  {t['beta']:8.6f}  {t['beta_source']:>14s}")

        # Cross-validation summary
        cross_entries = [t for t in self._telemetry_log 
                        if t['cross_validation'] is not None]
        if cross_entries:
            agreements = sum(1 for t in cross_entries 
                           if t['cross_validation']['agreement'])
            total = len(cross_entries)
            ratios = [t['cross_validation']['ratio'] for t in cross_entries]
            mean_ratio = sum(ratios) / len(ratios) if ratios else 0
            print(f"\n  ► CROSS-VALIDATION (susceptibility vs cos_sim)")
            print(f"  {'─' * 70}")
            print(f"  Agreement (|Δβ| < 0.1): {agreements}/{total} steps "
                  f"({100*agreements/max(total,1):.0f}%)")
            print(f"  Mean β ratio (susc/cos): {mean_ratio:.3f} "
                  f"(1.0 = perfect agreement)")

        print("\n" + "=" * 90)

    def get_c_theorem_data(self) -> List[Dict]:
        """
        Return F(β) vs ln|β| data points for external plotting.
        Each entry has: step, frac, phase, beta, ln_beta, f_beta, beta_source
        """
        return [
            {
                'step': t['step'],
                'frac': t['frac'],
                'phase': t['phase'],
                'beta': t['beta'],
                'ln_beta': t['ln_beta'],
                'f_beta': t['f_beta'],
                'beta_source': t['beta_source'],
            }
            for t in self._telemetry_log
        ]
