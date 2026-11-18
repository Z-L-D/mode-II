"""
CoS-PAFS: Chain-of-Steps Phase-Adaptive Flow Solver
====================================================
A reasoning-aware sampler for flow-matching video diffusion models.

Based on findings from "Demystifying Video Reasoning" (Wang et al., 2026)
which demonstrates that diffusion steps serve three functionally distinct
phases: exploration, convergence, and consolidation.

Designed for LTX-2.3 (22B) but works with any flow-matching DiT in ComfyUI.
"""

import torch
import torch.nn.functional as F
import math
from tqdm.auto import trange
from .ecc_controller import ECCController


# =============================================================================
# Phase Detection
# =============================================================================

class PhaseDetector:
    """
    Detects the current reasoning phase from velocity field statistics.

    Phases:
        exploration:   Model is generating multiple hypotheses in superposition.
                       High spatial variance in velocity, unstable x0 estimates.
        convergence:   Model is committing to specific paths, pruning alternatives.
                       Rapid velocity changes, peak noise sensitivity.
        consolidation: Reasoning is done. Texture refinement and detail.
                       Stable velocities, smooth x0 estimates.
    """

    def __init__(self, mode="auto",
                 exploration_end_frac=0.30,
                 convergence_end_frac=0.75,
                 spatial_var_threshold=0.15,
                 velocity_delta_threshold=0.08,
                 x0_stability_threshold=0.92,
                 is_eps_model=False):  # <-- NEW
        self.mode = mode
        self.exploration_end_frac = exploration_end_frac
        self.convergence_end_frac = convergence_end_frac

        # Auto-scale thresholds for EPS models (SD1.5/SDXL)
        scale = 1.6 if is_eps_model else 1.0
        self.spatial_var_threshold = spatial_var_threshold * scale
        self.velocity_delta_threshold = velocity_delta_threshold * scale
        self.x0_stability_threshold = min(0.99, x0_stability_threshold * scale)

        # State
        self._prev_velocity = None
        self._prev_x0 = None
        self._prev_phase = None
        self._step_phases = []

    def reset(self):
        self._prev_velocity = None
        self._prev_x0 = None
        self._prev_phase = None
        self._step_phases = []

    def detect(self, velocity, x0_estimate, step_index, total_steps, sigma):
        """
        Detect the current phase.

        Args:
            velocity: Current velocity field v = (x - denoised) / sigma
            x0_estimate: Current clean latent estimate (denoised output)
            step_index: Current step index (0 = first/noisiest)
            total_steps: Total number of steps
            sigma: Current noise level

        Returns:
            str: 'exploration', 'convergence', or 'consolidation'
        """
        if self.mode == "fixed":
            return self._fixed_detection(step_index, total_steps)
        elif self.mode == "auto":
            return self._auto_detection(velocity, x0_estimate,
                                        step_index, total_steps, sigma)
        elif self.mode == "hybrid":
            return self._hybrid_detection(velocity, x0_estimate,
                                          step_index, total_steps, sigma)
        else:
            return self._fixed_detection(step_index, total_steps)

    def _fixed_detection(self, step_index, total_steps):
        """Fixed phase boundaries based on step fraction."""
        frac = step_index / max(total_steps - 1, 1)
        if frac < self.exploration_end_frac:
            return 'exploration'
        elif frac < self.convergence_end_frac:
            return 'convergence'
        else:
            return 'consolidation'

    def _auto_detection(self, velocity, x0_estimate, step_index, total_steps, sigma):
        """
        Automatic phase detection from velocity statistics.
        Uses hysteresis to prevent rapid phase switching.
        """
        # Compute velocity spatial variance (normalized)
        v_flat = velocity.reshape(velocity.shape[0], -1)
        v_spatial_var = v_flat.var(dim=-1).mean().item()
        v_norm = v_flat.norm(dim=-1).mean().item()
        v_spatial_var_normalized = v_spatial_var / max(v_norm ** 2, 1e-8)

        # Compute velocity temporal derivative
        v_delta = float('inf')
        if self._prev_velocity is not None:
            diff = (velocity - self._prev_velocity).reshape(velocity.shape[0], -1)
            v_delta = diff.norm(dim=-1).mean().item() / max(v_norm, 1e-8)

        # Compute x0 stability (cosine similarity with previous estimate)
        x0_stability = 0.0
        if self._prev_x0 is not None:
            x0_flat = x0_estimate.reshape(1, -1)
            prev_flat = self._prev_x0.reshape(1, -1)
            x0_stability = F.cosine_similarity(x0_flat, prev_flat, dim=-1).item()

        # Update state
        self._prev_velocity = velocity.detach().clone()
        self._prev_x0 = x0_estimate.detach().clone()

        # Phase classification with hysteresis
        # Once we leave exploration, we don't go back
        if self._prev_phase == 'consolidation':
            phase = 'consolidation'
        elif self._prev_phase == 'convergence':
            # Can transition to consolidation but not back to exploration
            if x0_stability > self.x0_stability_threshold and v_delta < self.velocity_delta_threshold:
                phase = 'consolidation'
            else:
                phase = 'convergence'
        else:
            # In exploration or first step
            if v_spatial_var_normalized > self.spatial_var_threshold and x0_stability < 0.7:
                phase = 'exploration'
            elif x0_stability > self.x0_stability_threshold and v_delta < self.velocity_delta_threshold:
                phase = 'consolidation'
            else:
                phase = 'convergence'

        self._prev_phase = phase
        self._step_phases.append(phase)
        return phase

    def _hybrid_detection(self, velocity, x0_estimate, step_index, total_steps, sigma):
        """
        Hybrid: use fixed boundaries as guardrails, auto-detection for fine-tuning.
        This is the recommended mode for production use.
        """
        frac = step_index / max(total_steps - 1, 1)

        # Hard boundaries: can't be in exploration past 45% or consolidation before 50%
        if frac > 0.45:
            # Can't be exploration
            if frac < 0.50:
                return 'convergence'
            auto = self._auto_detection(velocity, x0_estimate,
                                        step_index, total_steps, sigma)
            if auto == 'exploration':
                return 'convergence'
            return auto
        elif frac < 0.15:
            # Always exploration for very early steps
            self._prev_velocity = velocity.detach().clone()
            self._prev_x0 = x0_estimate.detach().clone()
            self._prev_phase = 'exploration'
            self._step_phases.append('exploration')
            return 'exploration'
        else:
            return self._auto_detection(velocity, x0_estimate,
                                        step_index, total_steps, sigma)

    def get_phase_log(self):
        """Return the phase assignments for all steps so far."""
        return list(self._step_phases)


# =============================================================================
# Core Solver Strategies
# =============================================================================

class ExplorationSolver:
    """
    Phase 1: Stochastic Euler with diversity preservation.

    The paper shows that early denoising steps maintain multiple hypotheses
    in superposition. Standard ODE solvers collapse this by precisely tracking
    a single trajectory. We deliberately use low-order solving plus controlled
    stochastic perturbation to preserve hypothesis diversity.

    This is NOT the same as SDE sampling. The noise is:
    - Tiny (default 1.5% of sigma)
    - Only applied during the exploration phase
    - Scaled by remaining noise level (vanishes as sigma decreases)
    """

    def __init__(self, noise_scale=0.015):
        self.noise_scale = noise_scale

    def step(self, x, denoised, sigma, sigma_next, extra_noise=None):
        dt = sigma_next - sigma
        d = (x - denoised) / sigma  # ODE direction

        # Standard Euler update
        x_next = x + d * dt

        # Diversity-preserving perturbation
        if self.noise_scale > 0 and sigma_next > 0:
            # Scale noise by: base_scale * remaining_sigma * step_fraction
            # This ensures noise vanishes as we approach clean latent
            noise_magnitude = self.noise_scale * sigma_next.item()
            if noise_magnitude > 1e-7:
                if extra_noise is not None:
                    noise = extra_noise
                else:
                    noise = torch.randn_like(x_next)
                x_next = x_next + noise_magnitude * noise

        return x_next


class ConvergenceSolver:
    """
    Phase 2: RF-Solver-2 with Adams-Bashforth correction.

    The paper shows peak noise sensitivity at steps 20-30 (of 50).
    This is where discretization error matters most. We use:
    - 2nd order Taylor expansion of the velocity integral (RF-Solver-2)
    - Adams-Bashforth-3 multistep correction using velocity history
    - Adaptive blending based on velocity smoothness

    The Taylor expansion handles the nonlinear velocity integral more
    accurately than Euler. The AB3 correction leverages the velocity
    history buffer for additional accuracy at zero extra NFE cost.
    """

    def __init__(self, taylor_order=2, ab_blend_range=(0.3, 0.7)):
        self.taylor_order = taylor_order
        self.ab_blend_min = ab_blend_range[0]
        self.ab_blend_max = ab_blend_range[1]

    def step(self, x, denoised, sigma, sigma_next, velocity_history, sigma_history):
        dt = sigma_next - sigma
        v_current = (x - denoised) / sigma

        # --- RF-Solver-2: Taylor expansion ---
        if self.taylor_order >= 2 and len(velocity_history) >= 1:
            v_prev = velocity_history[-1]
            ds_prev = sigma - sigma_history[-1] if len(sigma_history) >= 1 else dt

            # Finite difference estimate of dv/ds
            if abs(ds_prev.item()) > 1e-8:
                dv_ds = (v_current - v_prev) / ds_prev
            else:
                dv_ds = torch.zeros_like(v_current)

            # 2nd order Taylor: x_next = x + dt*v + 0.5*dt^2*dv/ds
            x_taylor = x + dt * v_current + 0.5 * dt * dt * dv_ds

            if self.taylor_order >= 3 and len(velocity_history) >= 2:
                # 3rd order: add (1/6)*dt^3*d2v/ds2
                v_prev2 = velocity_history[-2]
                ds_prev2 = sigma_history[-1] - sigma_history[-2] if len(sigma_history) >= 2 else ds_prev
                if abs(ds_prev2.item()) > 1e-8 and abs(ds_prev.item()) > 1e-8:
                    dv_ds_prev = (v_prev - v_prev2) / ds_prev2
                    d2v_ds2 = (dv_ds - dv_ds_prev) / ((ds_prev + ds_prev2) * 0.5)
                    x_taylor = x_taylor + (1.0 / 6.0) * dt * dt * dt * d2v_ds2
        else:
            # Fall back to standard Euler
            x_taylor = x + dt * v_current

        # --- Adams-Bashforth correction ---
        if len(velocity_history) >= 2:
            v_n = v_current
            v_n1 = velocity_history[-1]
            v_n2 = velocity_history[-2]

            # AB3 coefficients
            x_ab = x + dt * (23.0/12.0 * v_n - 16.0/12.0 * v_n1 + 5.0/12.0 * v_n2)

            # Adaptive blend: higher Taylor weight when velocity is smooth
            v_diff = (v_current - v_n1).reshape(v_current.shape[0], -1)
            v_norm = v_current.reshape(v_current.shape[0], -1).norm(dim=-1, keepdim=True)
            smoothness = 1.0 - (v_diff.norm(dim=-1, keepdim=True) / (v_norm + 1e-8))
            # Clamp to blend range
            blend = smoothness.mean().clamp(self.ab_blend_min, self.ab_blend_max)

            # Clamp AB: if it wants to go further than Taylor by >30%, pull it back
            ab_displacement = (x_ab - x).norm()
            taylor_displacement = (x_taylor - x).norm()
            if taylor_displacement > 1e-8:
                overshoot_ratio = ab_displacement / taylor_displacement
                if overshoot_ratio > 1.3:
                    # AB is overshooting — reduce its influence
                    clamp_blend = min(blend + 0.3, 0.95)
                    x_next = clamp_blend * x_taylor + (1.0 - clamp_blend) * x_ab
                else:
                    x_next = blend * x_taylor + (1.0 - blend) * x_ab
            else:
                x_next = x_taylor
        elif len(velocity_history) >= 1:
            # AB2 fallback
            v_n = v_current
            v_n1 = velocity_history[-1]
            x_ab2 = x + dt * (1.5 * v_n - 0.5 * v_n1)

            x_next = 0.5 * x_taylor + 0.5 * x_ab2
        else:
            x_next = x_taylor

        return x_next


class ConsolidationSolver:
    """
    Phase 3: Momentum-Heun for stable texture refinement.

    Reasoning is complete. The priority is smooth, stable detail rendering.
    EMA momentum on the velocity prevents micro-oscillations that cause
    texture flickering in video output.

    Optional Heun corrector (predictor-corrector) for extra accuracy,
    but costs one additional NFE per step. Default is OFF for VRAM reasons.
    """

    def __init__(self, momentum=0.85, use_heun=False):
        self.momentum = momentum
        self.use_heun = use_heun
        self._ema_velocity = None

    def reset(self):
        self._ema_velocity = None

    def step(self, x, denoised, sigma, sigma_next, model=None, extra_args=None):
        dt = sigma_next - sigma
        v_current = (x - denoised) / sigma

        # EMA-smooth the velocity
        if self._ema_velocity is not None:
            v_smooth = self.momentum * self._ema_velocity + (1.0 - self.momentum) * v_current
        else:
            v_smooth = v_current
        self._ema_velocity = v_smooth.detach().clone()

        if self.use_heun and model is not None and sigma_next > 0:
            # Heun predictor-corrector (costs 1 extra NFE)
            x_predict = x + dt * v_smooth

            # Corrector: evaluate at predicted point
            denoised_next = model(x_predict, sigma_next, **extra_args)
            v_corrector = (x_predict - denoised_next) / sigma_next

            # Average velocities for the corrector update
            v_avg = 0.5 * (v_smooth + v_corrector)
            x_next = x + dt * v_avg
        else:
            # Momentum-Euler (no extra NFE)
            x_next = x + dt * v_smooth

        return x_next


# =============================================================================
# Main CoS-PAFS Sampling Function
# =============================================================================

@torch.no_grad()
def sample_cos_pafs(model, x, sigmas, extra_args=None, callback=None, disable=None,
                    # Exploration params
                    exploration_noise=0.015,
                    # Convergence params
                    taylor_order=2,
                    ab_blend_min=0.3,
                    ab_blend_max=0.7,
                    # Consolidation params
                    consolidation_momentum=0.85,
                    use_heun_corrector=False,
                    # Phase detection params
                    phase_mode="hybrid",
                    exploration_end=0.30,
                    convergence_end=0.75,
                    # Sigma lying test
                    sigma_lie_mode="none",
                    sigma_lie_value=0.5,
                    # Gate softening
                    gate_softening_mode="none",
                    gate_blend_factor=0.10,
                    gate_anchor_sigma=0.50,
                    gate_explore_mult=1.5,
                    gate_converge_mult=0.5,
                    gate_consolidate_mult=0.0,
                    # Diagnostics
                    print_phases=False,
                    **kwargs):
    """
    CoS-PAFS: Chain-of-Steps Phase-Adaptive Flow Solver.

    A reasoning-aware sampler that adapts its numerical strategy based
    on the current phase of the denoising process.

    This function follows the ComfyUI/k-diffusion sampler signature
    and can be used with SamplerCustomAdvanced via KSAMPLER wrapper.

    Args:
        model: Model wrapper (handles CFG). Call as model(x, sigma, **extra_args).
        x: Initial noisy latent [B, C, T, H, W] or [B, C, H, W].
        sigmas: Noise schedule tensor, descending from high to low (to 0).
        extra_args: Dict with conditioning info passed to model.
        callback: Progress callback function.
        disable: If True, disable progress bar.
        exploration_noise: Stochastic perturbation scale in exploration phase.
        taylor_order: Taylor expansion order for convergence solver (1, 2, or 3).
        ab_blend_min/max: Adams-Bashforth blend range in convergence phase.
        consolidation_momentum: EMA momentum for consolidation velocity smoothing.
        use_heun_corrector: Whether to use Heun predictor-corrector in consolidation.
        phase_mode: Phase detection mode ('fixed', 'auto', 'hybrid').
        exploration_end: Fraction of steps for exploration phase end (fixed/hybrid).
        convergence_end: Fraction of steps for convergence phase end (fixed/hybrid).
        sigma_lie_mode: EXPERIMENTAL sigma lying mode ('none', 'fixed', etc).
        sigma_lie_value: Value for sigma lying.
        gate_softening_mode: Timestep gate softening ('none', 'uniform', 'phase_adaptive').
            Compresses effective sigma range toward anchor, reducing timestep
            embedding dominance to allow freer latent comprehension.
        gate_blend_factor: Base blend factor for gate softening (0.0 = no change,
            0.5 = halfway to anchor). Start at 0.05-0.10 and sweep up.
        gate_anchor_sigma: Neutral sigma midpoint to blend toward (0.5 default).
        gate_explore_mult: Phase-adaptive multiplier on blend during exploration.
        gate_converge_mult: Phase-adaptive multiplier on blend during convergence.
        gate_consolidate_mult: Phase-adaptive multiplier on blend during consolidation.
        print_phases: If True, print phase assignments and gate values to console.

    Returns:
        Denoised latent tensor.
    """
    extra_args = {} if extra_args is None else extra_args

    # DEBUG: dump model parameterization (only when diagnostics enabled)
    if print_phases:
        if hasattr(model, 'inner_model'):
            m = model.inner_model
            if hasattr(m, 'inner_model'):
                m = m.inner_model  # unwrap CFG wrapper
            if hasattr(m, 'model_sampling'):
                ms = m.model_sampling
                print(f"[CoS-PAFS DEBUG] model_sampling class: {ms.__class__.__name__}")
                print(f"[CoS-PAFS DEBUG] sigma_min: {ms.sigma_min}, sigma_max: {ms.sigma_max}")
                if hasattr(ms, 'timestep'):
                    test_sigmas = [0.0, 0.1, 0.5, 0.9, 1.0]
                    for s in test_sigmas:
                        try:
                            t = ms.timestep(torch.tensor([s]))
                            print(f"[CoS-PAFS DEBUG] sigma={s} -> timestep={t.item():.4f}")
                        except:
                            pass
            if hasattr(m, 'model_type'):
                print(f"[CoS-PAFS DEBUG] model_type: {m.model_type}")
            if hasattr(m, 'predict_eps_from_noise'):
                print("[CoS-PAFS DEBUG] has predict_eps_from_noise")
            if hasattr(m, 'model_config'):
                mc = m.model_config
                print(f"[CoS-PAFS DEBUG] model_config: {mc.__class__.__name__}")
                if hasattr(mc, 'unet_config'):
                    print(f"[CoS-PAFS DEBUG] unet_config keys: {list(mc.unet_config.keys())[:10]}")
        print(f"[CoS-PAFS DEBUG] sigmas range: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}, count: {len(sigmas)}")
        print(f"[CoS-PAFS DEBUG] x shape: {x.shape}")
        if gate_softening_mode != "none":
            print(f"[CoS-PAFS GATE] mode={gate_softening_mode}, blend={gate_blend_factor}, "
                  f"anchor={gate_anchor_sigma}")
            if gate_softening_mode == "phase_adaptive":
                print(f"[CoS-PAFS GATE] phase multipliers: explore={gate_explore_mult}, "
                      f"converge={gate_converge_mult}, consolidate={gate_consolidate_mult}")

    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1

    # Initialize components
    # Detect EPS vs Flow model for threshold calibration
    is_eps = hasattr(model, 'predict_eps_from_noise') or 'unet_config' in str(model)
    phase_detector = PhaseDetector(
        mode=phase_mode,
        exploration_end_frac=exploration_end,
        convergence_end_frac=convergence_end,
        is_eps_model=is_eps,
    )
    
    # ECC Controller for explicit Φ and J[Φ] tracking (optional, backward compatible)
    from .ecc_controller import ECCController
    ecc_controller = ECCController(
        explore_end=exploration_end,
        converge_end=convergence_end,
        transition_width=0.05,
        exploration_noise=exploration_noise,
        convergence_stabilization=0.005,
        consolidation_damping=0.002,
        nu=0.8,
        lambda_cost=0.1,
        f0=2.0,
        adaptive_boundaries=False,
        enabled=True,
    )
    
    exploration_solver = ExplorationSolver(noise_scale=exploration_noise)
    convergence_solver = ConvergenceSolver(
        taylor_order=taylor_order,
        ab_blend_range=(ab_blend_min, ab_blend_max),
    )
    consolidation_solver = ConsolidationSolver(
        momentum=consolidation_momentum,
        use_heun=use_heun_corrector,
    )

    # Velocity history buffer for Adams-Bashforth multistep
    velocity_history = []
    sigma_history = []
    max_history = 4  # Keep last N velocities

    # Velocity cosine similarity tracking (ProbeFlow-inspired)
    # Measures directional alignment between consecutive velocity vectors.
    # High cos_sim = trajectory is linear (model committed to a direction)
    # Low cos_sim = trajectory is curving (model evaluating alternatives)
    cos_sim_history = []
    cos_sim_window = 3  # Smoothing window for phase transition detection
    actual_phases = []  # Track phases actually used (ECC overrides PhaseDetector)

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i] * s_in
        sigma_next = sigmas[i + 1] * s_in

        # =====================================================================
        # Compute the sigma the model actually sees (gate softening / sigma lie)
        # ODE math ALWAYS uses the real sigma — only the model's internal
        # timestep embedding is affected.
        # =====================================================================
        model_sigma = sigma  # default: honest

        # --- Gate softening (Approach 1 / Approach 3 from research notes) ---
        # Compresses the effective sigma range toward an anchor point.
        # Uniform: same blend at every step.
        # Phase-adaptive: soften MORE during exploration (free up latent
        #   comprehension for hypothesis diversity), LESS during convergence
        #   (need precise mode control), near-ZERO during consolidation.
        if gate_softening_mode != "none":
            if gate_softening_mode == "uniform":
                blend = gate_blend_factor
            elif gate_softening_mode == "phase_adaptive":
                # Use step-fraction phase (pre-model-call, so can't use
                # auto-detected phase which needs the denoised output)
                frac = i / max(total_steps - 1, 1)
                if frac < exploration_end:
                    blend = gate_blend_factor * gate_explore_mult
                elif frac < convergence_end:
                    blend = gate_blend_factor * gate_converge_mult
                else:
                    blend = gate_blend_factor * gate_consolidate_mult
            else:
                blend = 0.0

            if blend > 0.0:
                anchor = sigmas[i].new_tensor(gate_anchor_sigma)
                softened = (1.0 - blend) * sigmas[i] + blend * anchor
                model_sigma = softened * s_in

                if print_phases and (i == 0 or i == total_steps // 2 or i == total_steps - 1):
                    print(f"[CoS-PAFS GATE] step {i}: real_sigma={sigmas[i]:.4f}, "
                          f"model_sigma={softened:.4f}, blend={blend:.3f}")

        # --- Sigma lying (experimental, overrides gate softening if active) ---
        if sigma_lie_mode != "none":
            if sigma_lie_mode == "fixed":
                fake_sigma = sigmas[i].new_tensor(sigma_lie_value) * s_in
            elif sigma_lie_mode == "reversed":
                reversed_idx = total_steps - 1 - i
                fake_sigma = sigmas[reversed_idx] * s_in
            elif sigma_lie_mode == "shifted":
                shifted_idx = min(int(i * 1.5), total_steps - 1)
                fake_sigma = sigmas[shifted_idx] * s_in
            elif sigma_lie_mode == "proportional":
                fake_sigma = torch.clamp(sigmas[i] - sigma_lie_value, min=0.01) * s_in
            else:
                fake_sigma = sigma

            model_sigma = fake_sigma  # sigma lie overrides gate softening

            if print_phases and (i == 0 or i == total_steps // 2 or i == total_steps - 1):
                print(f"[CoS-PAFS SIGMA LIE] step {i}: real_sigma={sigmas[i]:.4f}, "
                      f"fake_sigma={model_sigma[0]:.4f}")

        # Forward pass — model sees model_sigma, ODE uses real sigma
        denoised = model(x, model_sigma, **extra_args)

        # Compute velocity for phase detection and history
        if sigmas[i] > 1e-8:
            velocity = (x - denoised) / sigmas[i]
        else:
            velocity = torch.zeros_like(x)

        # =====================================================================
        # Velocity cosine similarity — trajectory linearity signal
        # From ProbeFlow (Fang et al., 2026): cosine similarity between
        # consecutive velocity vectors is a geometric proxy for trajectory
        # curvature. Uses existing velocity history — zero extra NFE.
        #
        # Interpretation:
        #   cos_sim ~0.3-0.7: model exploring, velocity direction unstable
        #   cos_sim jumps to ~0.9+: model committing (explore→converge)
        #   cos_sim stable ~0.95+: trajectory locked (consolidation)
        #   cos_sim drops in late steps: model fighting spatial commitments
        #                                (restart candidate)
        # =====================================================================
        cos_sim = 0.0
        if len(velocity_history) >= 1:
            v_prev = velocity_history[-1]
            cos_sim = F.cosine_similarity(
                velocity.reshape(1, -1),
                v_prev.reshape(1, -1),
                dim=-1
            ).item()

        cos_sim_history.append(cos_sim)
            
        # Smoothed cosine similarity (windowed average for robustness)
        if len(cos_sim_history)  >= cos_sim_window:
            cos_sim_smoothed = sum(cos_sim_history[-cos_sim_window:]) / cos_sim_window
        else:
            cos_sim_smoothed = cos_sim

        # Clamp to prevent extreme direction flips (EPS model stability)
        cos_sim_smoothed = max(0.3, cos_sim_smoothed)

        if print_phases and (i  <= 2 or i == total_steps // 4 or i == total_steps // 2
                             or i == 3 * total_steps // 4 or i  >= total_steps - 2):
            print(f"[CoS-PAFS COS_SIM] step {i}: cos_sim={cos_sim:.4f},  "
                  f"smoothed={cos_sim_smoothed:.4f} ")

        # Detect current phase (kept for backward compatibility & logging)
        phase = phase_detector.detect(
            velocity, denoised,
            step_index=i,
            total_steps=total_steps,
            sigma=sigmas[i],
        )
    
        # ECC Controller step (computes explicit Φ and telemetry)
        if ecc_controller.enabled:
            ecc_output = ecc_controller.step(
                velocity=velocity,
                step=i,
                total_steps=total_steps,
                sigma=sigmas[i],
                cos_sim=cos_sim_smoothed if len(cos_sim_history) > 0 else 0.5,
            )
            phase = ecc_output['phase']
            phi = ecc_output['phi']
        else:
            phi = torch.zeros_like(velocity)
    
        actual_phases.append(phase)

        # Execute phase-appropriate solver strategy
        if phase == 'exploration':
            x_next = exploration_solver.step(x, denoised, sigmas[i], sigmas[i + 1])

        elif phase == 'convergence':
            x_next = convergence_solver.step(
                x, denoised, sigmas[i], sigmas[i + 1],
                velocity_history=velocity_history,
                sigma_history=sigma_history,
            )

        elif phase == 'consolidation':
            x_next = consolidation_solver.step(
                x, denoised, sigmas[i], sigmas[i + 1],
                model=model if use_heun_corrector else None,
                extra_args=extra_args,
            )

        else:
            # Fallback: standard Euler
            d = (x - denoised) / sigmas[i]
            dt = sigmas[i + 1] - sigmas[i]
            x_next = x + d * dt


        # Update velocity history
        velocity_history.append(velocity.detach().clone())
        sigma_history.append(sigmas[i].clone())
        if len(velocity_history) > max_history:
            velocity_history.pop(0)
            sigma_history.pop(0)

        # Callback for ComfyUI progress
        if callback is not None:
            callback({
                'x': x_next,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigmas[i],
                'denoised': denoised,
                'cos_sim': cos_sim,
                'cos_sim_smoothed': cos_sim_smoothed,
                'phase': phase,
            })

        x = x_next

    if print_phases:
        # Use actual phases (ECC-overridden when active) for summary
        if actual_phases:
            phases = actual_phases
        elif 'phase_detector' in locals() and phase_detector is not None:
            phases = phase_detector.get_phase_log()
        else:
            phases = []
        
        if phases:
            phase_counts = {p: phases.count(p) for p in ['exploration', 'convergence', 'consolidation']}
            print(f"[CoS-PAFS] Phase distribution: {phase_counts}")
    
        # ECC Telemetry print
        if ecc_controller.enabled and ecc_controller._telemetry_log:
            log = ecc_controller._telemetry_log
            avg_j = log[-1]['j_avg']
            print(f"[ECC] Final J[Φ] average: {avg_j:.4f}")
            print(f"[ECC] Telemetry saved: {len(log)} steps.")
        # Print phase transitions
        transitions = []
        for j in range(1, len(phases)):
            if phases[j] != phases[j-1]:
                transitions.append(f"step {j}: {phases[j-1]} → {phases[j]}")
        if transitions:
            print(f"[CoS-PAFS] Transitions: {', '.join(transitions)}")

        # Cosine similarity trajectory profile
        if len(cos_sim_history) > 5:
            n = len(cos_sim_history)
            # Sample at key points: start, 25%, 50%, 75%, end
            sample_idxs = [0, 1, 2, n // 4, n // 2, 3 * n // 4, n - 3, n - 2, n - 1]
            sample_idxs = sorted(set(i for i in sample_idxs if 0 <= i < n))
            profile = ', '.join(f's{idx}={cos_sim_history[idx]:.3f}' for idx in sample_idxs)
            print(f"[CoS-PAFS COS_SIM] Profile: {profile}")

            # Compute per-phase averages
            phase_sims = {'exploration': [], 'convergence': [], 'consolidation': []}
            for j, (cs, ph) in enumerate(zip(cos_sim_history, phases)):
                if j > 0:  # skip step 0 (cos_sim=0 by definition)
                    phase_sims[ph].append(cs)
            avg_by_phase = {ph: (sum(v) / len(v) if v else 0.0) for ph, v in phase_sims.items()}
            print(f"[CoS-PAFS COS_SIM] Phase averages: "
                  f"explore={avg_by_phase['exploration']:.3f}, "
                  f"converge={avg_by_phase['convergence']:.3f}, "
                  f"consolidate={avg_by_phase['consolidation']:.3f}")

            # Restart signal: average cos_sim in last 5 steps
            last_n = min(5, n)
            tail_avg = sum(cos_sim_history[-last_n:]) / last_n
            print(f"[CoS-PAFS COS_SIM] Tail avg (last {last_n}): {tail_avg:.3f}"
                  f" {'← RESTART CANDIDATE' if tail_avg < 0.85 else '← clean convergence'}")
            
        if print_phases and ecc_controller.enabled:
            log = ecc_controller._telemetry_log
            if log:
                avg_j = log[-1]['j_avg']
                print(f"[ECC] Final J[Φ] average: {avg_j:.4f}")
                print(f"[ECC] Telemetry saved: {len(log)} steps. Check ecc_controller._telemetry_log for full trace.")

    return x


# =============================================================================
# Sigma Schedule Generator
# =============================================================================

def generate_cos_aware_sigmas(total_steps, sigma_max=1.0, sigma_min=0.0,
                               exploration_frac=0.25,
                               convergence_frac=0.50,
                               consolidation_frac=0.25,
                               exploration_spacing="linear",
                               convergence_spacing="cosine",
                               consolidation_spacing="linear",
                               device="cpu"):
    """
    Generate a non-uniform sigma schedule that concentrates steps
    in the convergence phase where reasoning sensitivity is highest.

    The paper's noise perturbation experiments show peak sensitivity
    at steps ~20-30 (of 50). This schedule places more steps there.

    Args:
        total_steps: Total number of denoising steps.
        sigma_max/min: Noise level range.
        exploration_frac: Fraction of steps for exploration (coarse spacing).
        convergence_frac: Fraction of steps for convergence (dense spacing).
        consolidation_frac: Fraction of steps for consolidation (moderate).
        *_spacing: "linear" or "cosine" spacing within each phase.

    Returns:
        Tensor of sigma values [total_steps + 1], descending, ending at sigma_min.
    """
    # Normalize fractions
    total_frac = exploration_frac + convergence_frac + consolidation_frac
    exploration_frac /= total_frac
    convergence_frac /= total_frac
    consolidation_frac /= total_frac

    n_explore = max(1, int(total_steps * exploration_frac))
    n_consolidate = max(1, int(total_steps * consolidation_frac))
    n_converge = total_steps - n_explore - n_consolidate

    # Phase sigma boundaries
    # Exploration covers high noise: [sigma_max, boundary_1]
    # Convergence covers mid noise: [boundary_1, boundary_2]
    # Consolidation covers low noise: [boundary_2, sigma_min]
    sigma_range = sigma_max - sigma_min
    boundary_1 = sigma_max - 0.30 * sigma_range   # ~0.70 of sigma_max
    boundary_2 = sigma_min + 0.20 * sigma_range    # ~0.20 of sigma_max

    def make_spacing(n, start, end, method):
        if n <= 0:
            return torch.tensor([], device=device)
        if method == "cosine":
            t = torch.linspace(0, math.pi / 2, n + 1, device=device)
            # Cosine gives denser spacing at the start (transition from exploration)
            return start + (end - start) * torch.sin(t)
        elif method == "cosine_reverse":
            t = torch.linspace(0, math.pi / 2, n + 1, device=device)
            return start + (end - start) * (1.0 - torch.cos(t))
        else:  # linear
            return torch.linspace(start, end, n + 1, device=device)

    s_explore = make_spacing(n_explore, sigma_max, boundary_1, exploration_spacing)
    s_converge = make_spacing(n_converge, boundary_1, boundary_2, convergence_spacing)
    s_consolidate = make_spacing(n_consolidate, boundary_2, sigma_min, consolidation_spacing)

    # Concatenate, removing duplicate boundary points
    schedule = torch.cat([
        s_explore[:-1],      # All except last (= boundary_1)
        s_converge[:-1],     # All except last (= boundary_2)
        s_consolidate,       # All including final sigma_min
    ])

    # Ensure monotonically decreasing and ends at sigma_min
    schedule = torch.sort(schedule, descending=True).values
    if schedule[-1] != sigma_min:
        schedule[-1] = sigma_min

    return schedule


# =============================================================================
# AGPR Injection Schedule (for Zac's v7 image conditioning work)
# =============================================================================

def agpr_injection_weight(phase, step_fraction):
    """
    Compute AGPR cross-attention injection weight based on phase.

    The paper's "Perception before Action" finding means reference
    image conditioning should be strongest during exploration (when
    the model grounds WHAT exists) and decay through convergence
    (when it resolves HOW things move).

    Args:
        phase: Current phase string.
        step_fraction: Current step / total steps (0 = start, 1 = end).

    Returns:
        float: Injection weight in [0, 1].
    """
    if phase == 'exploration':
        return 1.0
    elif phase == 'convergence':
        # Cosine decay within convergence phase
        # Map convergence fraction (roughly 0.3-0.75) to [0, pi]
        progress = max(0.0, min(1.0, (step_fraction - 0.3) / 0.45))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        # Minimal injection during consolidation
        return 0.05


def perturbation_layers_for_phase(phase, total_blocks=48):
    """
    Select which DiT blocks to skip during unconditional forward pass
    (perturbation guidance) based on current phase.

    The paper's layer specialization finding:
    - Layers 0-9: Global structure / background (perception)
    - Layers 10-29: Semantic reasoning (the "reasoning window")
    - Layers 30+: Latent consolidation

    During convergence: safe to perturb consolidation layers.
    During consolidation: safe to perturb perception layers (redundant).
    During exploration: don't perturb anything.
    """
    if phase == 'exploration':
        return []
    elif phase == 'convergence':
        start = int(total_blocks * 0.75)
        return list(range(start, total_blocks))
    else:
        end = int(total_blocks * 0.25)
        return list(range(0, end))