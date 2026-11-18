"""
ComfyUI Nodes for CoS-PAFS
===========================
Node definitions that integrate the CoS-PAFS solver into ComfyUI workflows.

Nodes:
    CoSPAFS_Sampler           - Main sampler node (outputs SAMPLER for SamplerCustomAdvanced)
    CoSPAFS_Sigmas            - Non-uniform sigma schedule concentrating steps in convergence
    CoSPAFS_CFGGuider         - Phase-adaptive CFG guider (outputs GUIDER for SamplerCustomAdvanced)
    CoSPAFS_GuidanceSchedule  - Phase-aware guidance parameter schedule (legacy, prefer CFGGuider)
    CoSPAFS_AGPRSchedule      - AGPR injection weight schedule for image conditioning
    CoSPAFS_PerturbationSchedule - Phase-aware perturbation guidance layer schedule
"""

import torch
import math
import comfy.samplers
import torch.nn.functional as F

from .ecc_controller import ECCController

from .cos_pafs_sampler import (
    sample_cos_pafs,
    generate_cos_aware_sigmas,
    agpr_injection_weight,
    perturbation_layers_for_phase,
    PhaseDetector,
)

import csv
import os
import json
import hashlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
TELEMETRY_DIR = SCRIPT_DIR / "dm_telemetry"

# --- Model Hook Registry ---
# Maps model class names to their transformer block paths and attention attributes.
# Allows hooking architectures like SD3.5 (JointBlock) where attention is nested
# inside sub-modules (e.g. context_block.attn) rather than directly on the block.
_MODEL_REGISTRY = {}
_REGISTRY_PATH = SCRIPT_DIR / "model_registry.json"
if _REGISTRY_PATH.exists():
    try:
        with open(_REGISTRY_PATH, 'r') as f:
            _MODEL_REGISTRY = json.load(f)
        # Strip metadata keys
        _MODEL_REGISTRY = {k: v for k, v in _MODEL_REGISTRY.items()
                           if not k.startswith('_')}
    except Exception as e:
        print(f"[CoS-EMAG] Warning: Could not load model registry: {e}")


def _resolve_dotted_attr(obj, dotted_path):
    """Resolve a dotted attribute path like 'context_block.attn' on an object.
    Returns the module if found, None otherwise."""
    parts = dotted_path.split('.')
    current = obj
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def _linearize_unet(dm):
    """R3: transformer blocks of a ComfyUI UNetModel (SD1.5 / SDXL) in forward order:
    input_blocks -> middle_block -> output_blocks, each SpatialTransformer's
    transformer_blocks in turn. Shared by the EMAG and Passive guiders."""
    found = []
    def walk(module):
        for name, child in module.named_children():
            if hasattr(child, 'attn1') or hasattr(child, 'attn2'):
                found.append(child)
            elif hasattr(child, 'transformer_blocks'):
                found.extend(tb for tb in child.transformer_blocks
                             if hasattr(tb, 'attn1') or hasattr(tb, 'attn2'))
            elif (name in ('input_blocks', 'output_blocks', 'middle_block')
                  or isinstance(child, torch.nn.ModuleList)):
                walk(child)
            elif any(hasattr(gc, 'transformer_blocks') or hasattr(gc, 'attn1')
                     for gc in child.children()):
                walk(child)
    walk(dm)
    return found


def _as_block_seq(seq):
    """R3: list of block modules. list() on an nn.ModuleDict returns its key
    strings (Cosmos-1 GeneralDIT 'blocks'), which silently registered 0 hooks."""
    if isinstance(seq, (torch.nn.ModuleDict, dict)):
        return list(seq.values())
    return list(seq)

def _same_width(mvecs, keys):
    """R3: True if the stored mean vectors under keys all have one shape."""
    shapes = {tuple(mvecs[k].shape) for k in keys}
    return len(shapes) == 1

def _resolve_block_list(dm, block_path):
    """R3: resolve a registry block_path to a list of blocks, or None.
    Accepts a dotted path ('decoder.layers', ACE-Step 1.5), a composite
    ('double_blocks + single_blocks', Flux/Hunyuan/Chroma) and UNET_LINEARIZE.
    The R2 code used hasattr(dm, block_path), which fails on all three."""
    if not block_path or block_path == 'NEEDS_DISCOVERY':
        return None
    if block_path == 'UNET_LINEARIZE':
        return _linearize_unet(dm) or None
    blocks = []
    for part in block_path.split('+'):
        seq = _resolve_dotted_attr(dm, part.strip())
        if seq is None:
            return None
        blocks.extend(_as_block_seq(seq))
    return blocks or None

# =============================================================================
# Sampler Node
# =============================================================================

class CoSPAFS_Sampler:
    """
    Chain-of-Steps Phase-Adaptive Flow Solver.

    A reasoning-aware sampler that adapts its numerical strategy based on
    the current phase of the denoising trajectory. Designed for flow-matching
    video DiTs (LTX-2.3, Wan2.2, etc.) but works with any flow-matching model.

    Phases:
    - Exploration: Low-order + controlled stochastic perturbation (preserve hypothesis diversity)
    - Convergence: High-order RF-Solver-2 + Adams-Bashforth (maximum precision where it matters)
    - Consolidation: Momentum-smoothed Euler/Heun (stable texture refinement)

    Use with SamplerCustomAdvanced or any node that accepts a SAMPLER input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phase_mode": (["hybrid", "fixed", "auto"], {
                    "default": "hybrid",
                    "tooltip": "Phase detection mode. 'hybrid' (recommended) uses fixed guardrails + auto-detection. 'fixed' uses step fractions only. 'auto' uses velocity statistics only.",
                }),
            },
            "optional": {
                # Exploration phase
                "exploration_noise": ("FLOAT", {
                    "default": 0.015, "min": 0.0, "max": 0.10, "step": 0.001,
                    "tooltip": "Stochastic perturbation scale during exploration. Preserves multi-path hypothesis diversity. 0 = pure Euler. Default 0.015 is subtle.",
                }),
                # Convergence phase
                "taylor_order": ("INT", {
                    "default": 2, "min": 1, "max": 3, "step": 1,
                    "tooltip": "Taylor expansion order for RF-Solver in convergence phase. 2 is recommended. 3 adds accuracy at slight risk of instability.",
                }),
                "ab_blend_min": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum Taylor weight in Taylor/Adams-Bashforth blend. Higher = more Taylor, less AB.",
                }),
                "ab_blend_max": ("FLOAT", {
                    "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Maximum Taylor weight in blend. The actual blend is adaptive based on velocity smoothness.",
                }),
                # Consolidation phase
                "consolidation_momentum": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA momentum for velocity smoothing in consolidation. Higher = smoother but slower to adapt. Prevents texture flickering.",
                }),
                "use_heun_corrector": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Heun predictor-corrector in consolidation. Better quality but costs 1 extra NFE per consolidation step. Disable for VRAM-constrained setups.",
                }),
                # Phase boundaries (for fixed/hybrid modes)
                "exploration_end": ("FLOAT", {
                    "default": 0.30, "min": 0.10, "max": 0.50, "step": 0.05,
                    "tooltip": "Step fraction where exploration phase ends (fixed/hybrid modes). 0.30 = first 30% of steps.",
                }),
                "convergence_end": ("FLOAT", {
                    "default": 0.75, "min": 0.50, "max": 0.95, "step": 0.05,
                    "tooltip": "Step fraction where convergence phase ends. 0.75 = convergence occupies steps 30%-75%.",
                }),
                # Diagnostics
                "print_phases": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print phase assignments and transitions to console for debugging.",
                }),
                # Sigma lying test (experimental) ==========================================
                "sigma_lie_mode": (["none", "fixed", "reversed", "shifted", "proportional"], {
                    "default": "none",
                    "tooltip": "EXPERIMENTAL: Feed wrong sigma to model while using real sigma for ODE math. 'fixed' = constant value, 'reversed' = opposite schedule, 'shifted' = 50% ahead of actual.",
                }),
                "sigma_lie_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Sigma value for 'fixed' lie mode. 0.5 = always midpoint, 0.9 = always high noise, 0.1 = always low noise.",
                }),
                # Gate softening ==============================================
                "gate_softening_mode": (["none", "uniform", "phase_adaptive"], {
                    "default": "none",
                    "tooltip": "Timestep gate softening. Compresses effective sigma range toward an anchor to reduce timestep embedding dominance. 'uniform' = same blend every step. 'phase_adaptive' = soften more during exploration, less during convergence.",
                }),
                "gate_blend_factor": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.50, "step": 0.01,
                    "tooltip": "Base blend factor. 0.10 = 10% toward anchor. Start at 0.05, sweep to 0.30. Higher = more softening.",
                }),
                "gate_anchor_sigma": ("FLOAT", {
                    "default": 0.50, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Anchor sigma to blend toward. 0.5 = schedule midpoint. This is the 'neutral' mode the model is pulled toward.",
                }),
                "gate_explore_mult": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Phase-adaptive: multiplier on blend during exploration. >1.0 = soften MORE (free up latent comprehension for hypothesis diversity).",
                }),
                "gate_converge_mult": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Phase-adaptive: multiplier on blend during convergence. <1.0 = soften LESS (need precise mode control for commitment).",
                }),
                "gate_consolidate_mult": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Phase-adaptive: multiplier on blend during consolidation. 0.0 = no softening (texture refinement needs honest timestep).",
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = "Chain-of-Steps Phase-Adaptive Flow Solver. A reasoning-aware sampler that uses different numerical strategies at different points in the denoising trajectory."

    def get_sampler(self, phase_mode="hybrid",
                    exploration_noise=0.015,
                    taylor_order=2,
                    ab_blend_min=0.30,
                    ab_blend_max=0.70,
                    consolidation_momentum=0.85,
                    use_heun_corrector=False,
                    exploration_end=0.30,
                    convergence_end=0.75,
                    print_phases=False,
                    sigma_lie_mode="none",
                    sigma_lie_value=0.5,
                    gate_softening_mode="none",
                    gate_blend_factor=0.10,
                    gate_anchor_sigma=0.50,
                    gate_explore_mult=1.5,
                    gate_converge_mult=0.5,
                    gate_consolidate_mult=0.0):

        sampler = comfy.samplers.KSAMPLER(
            sample_cos_pafs,
            extra_options={
                "exploration_noise": exploration_noise,
                "taylor_order": taylor_order,
                "ab_blend_min": ab_blend_min,
                "ab_blend_max": ab_blend_max,
                "consolidation_momentum": consolidation_momentum,
                "use_heun_corrector": use_heun_corrector,
                "phase_mode": phase_mode,
                "exploration_end": exploration_end,
                "convergence_end": convergence_end,
                "print_phases": print_phases,
                "sigma_lie_mode": sigma_lie_mode,
                "sigma_lie_value": sigma_lie_value,
                "gate_softening_mode": gate_softening_mode,
                "gate_blend_factor": gate_blend_factor,
                "gate_anchor_sigma": gate_anchor_sigma,
                "gate_explore_mult": gate_explore_mult,
                "gate_converge_mult": gate_converge_mult,
                "gate_consolidate_mult": gate_consolidate_mult,
            },
        )
        return (sampler,)


# =============================================================================
# Sigma Schedule Node
# =============================================================================

class CoSPAFS_Sigmas:
    """
    Non-uniform sigma schedule for CoS-PAFS.

    Concentrates denoising steps in the convergence phase where the paper
    shows peak noise sensitivity (steps ~20-30 of 50). This gives more
    precision where it matters most without increasing total step count.

    The schedule divides the sigma range into three regions with
    independently configurable density and spacing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 40, "min": 4, "max": 200, "step": 1,
                    "tooltip": "Total number of denoising steps.",
                }),
                "sigma_max": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01,
                    "tooltip": "Maximum sigma (noise level at start). 1.0 for standard flow matching.",
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001,
                    "tooltip": "Minimum sigma. 0.0 for full denoising.",
                }),
            },
            "optional": {
                "exploration_frac": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 0.50, "step": 0.05,
                    "tooltip": "Fraction of steps allocated to exploration (coarse spacing).",
                }),
                "convergence_frac": ("FLOAT", {
                    "default": 0.50, "min": 0.20, "max": 0.80, "step": 0.05,
                    "tooltip": "Fraction of steps for convergence (dense spacing). Higher = more precision in reasoning phase.",
                }),
                "consolidation_frac": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 0.50, "step": 0.05,
                    "tooltip": "Fraction of steps for consolidation (moderate spacing).",
                }),
                "exploration_spacing": (["linear", "cosine", "cosine_reverse", "beta"], {
                    "default": "linear",
                    "tooltip": "Step spacing within exploration phase.",
                }),
                "convergence_spacing": (["linear", "cosine", "cosine_reverse", "beta"], {
                    "default": "cosine",
                    "tooltip": "Step spacing within convergence phase. Cosine gives denser steps at phase boundaries.",
                }),
                "consolidation_spacing": (["linear", "cosine", "cosine_reverse", "beta"], {
                    "default": "linear",
                    "tooltip": "Step spacing within consolidation phase.",
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "generate"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = "Generate a non-uniform sigma schedule that concentrates steps in the convergence phase for maximum reasoning precision."

    def generate(self, steps, sigma_max=1.0, sigma_min=0.0,
                 exploration_frac=0.25, convergence_frac=0.50,
                 consolidation_frac=0.25,
                 exploration_spacing="linear",
                 convergence_spacing="cosine",
                 consolidation_spacing="linear"):

        sigmas = generate_cos_aware_sigmas(
            total_steps=steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            exploration_frac=exploration_frac,
            convergence_frac=convergence_frac,
            consolidation_frac=consolidation_frac,
            exploration_spacing=exploration_spacing,
            convergence_spacing=convergence_spacing,
            consolidation_spacing=consolidation_spacing,
        )

        return (sigmas,)


# =============================================================================
# Guidance Schedule Node
# =============================================================================

class CoSPAFS_GuidanceSchedule:
    """
    Phase-aware guidance parameter schedule.

    Outputs per-step parameter values for CFG scale, Enhance-A-Video weight,
    FDTG high-frequency weight, and perturbation guidance layer sets —
    adapted to the current reasoning phase.

    Connect to nodes that accept step-varying parameters, or use the
    output tensors to manually modulate guidance in a custom workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 40, "min": 4, "max": 200,
                }),
                "base_cfg": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Base CFG scale. Phase multipliers are applied on top of this.",
                }),
            },
            "optional": {
                "exploration_cfg_mult": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "CFG multiplier during exploration. <1.0 = reduced guidance to allow hypothesis diversity.",
                }),
                "convergence_cfg_mult": ("FLOAT", {
                    "default": 1.3, "min": 0.5, "max": 3.0, "step": 0.05,
                    "tooltip": "CFG multiplier during convergence. >1.0 = boosted guidance to force commitment.",
                }),
                "consolidation_cfg_mult": ("FLOAT", {
                    "default": 0.9, "min": 0.3, "max": 2.0, "step": 0.05,
                    "tooltip": "CFG multiplier during consolidation. Slightly reduced — reasoning is done.",
                }),
                "exploration_end": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.50, "step": 0.05}),
                "convergence_end": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 0.95, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING",)
    RETURN_NAMES = ("cfg_schedule", "fdtg_hf_schedule", "phase_map",)
    OUTPUT_IS_LIST = (True, True, False,)
    FUNCTION = "generate"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = "Generate per-step guidance parameters adapted to the CoS reasoning phases."

    def generate(self, steps, base_cfg=3.0,
                 exploration_cfg_mult=0.6,
                 convergence_cfg_mult=1.3,
                 consolidation_cfg_mult=0.9,
                 exploration_end=0.30,
                 convergence_end=0.75):

        cfg_values = []
        fdtg_hf_values = []
        phases = []

        for i in range(steps):
            frac = i / max(steps - 1, 1)

            if frac < exploration_end:
                phase = 'exploration'
                cfg = base_cfg * exploration_cfg_mult
                fdtg_hf = 0.2  # Low HF — focus on structure
            elif frac < convergence_end:
                phase = 'convergence'
                cfg = base_cfg * convergence_cfg_mult
                # Ramp FDTG HF from low to medium through convergence
                progress = (frac - exploration_end) / (convergence_end - exploration_end)
                fdtg_hf = 0.2 + 0.4 * progress
            else:
                phase = 'consolidation'
                cfg = base_cfg * consolidation_cfg_mult
                fdtg_hf = 0.8  # High HF — texture time

            cfg_values.append(round(cfg, 3))
            fdtg_hf_values.append(round(fdtg_hf, 3))
            phases.append(phase)

        # Create a human-readable phase map string
        phase_str = " | ".join([
            f"Steps 0-{int(steps * exploration_end)}: EXPLORE",
            f"Steps {int(steps * exploration_end)}-{int(steps * convergence_end)}: CONVERGE",
            f"Steps {int(steps * convergence_end)}-{steps}: CONSOLIDATE",
        ])

        return (cfg_values, fdtg_hf_values, phase_str,)


# =============================================================================
# AGPR Injection Schedule Node
# =============================================================================

class CoSPAFS_AGPRSchedule:
    """
    Phase-aware injection weight schedule for AGPR reference image conditioning.

    Based on the paper's "Perception before Action" finding:
    - Exploration: FULL injection weight (model is grounding WHAT exists)
    - Convergence: Cosine decay (model transitions to resolving HOW things move)
    - Consolidation: Near-zero (don't fight texture generation)

    Output is a list of per-step weights [0.0, 1.0] that can be used
    to modulate AGPR cross-attention strength in your conditioning pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 40, "min": 4, "max": 200,
                }),
            },
            "optional": {
                "exploration_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "AGPR weight during exploration. 1.0 = full injection. >1.0 = boosted.",
                }),
                "consolidation_weight": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "AGPR weight during consolidation. Near-zero recommended.",
                }),
                "decay_shape": (["cosine", "linear", "exponential"], {
                    "default": "cosine",
                    "tooltip": "Shape of weight decay through convergence phase.",
                }),
                "exploration_end": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.50, "step": 0.05}),
                "convergence_end": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 0.95, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("agpr_weights",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = "Per-step AGPR injection weights based on Perception-before-Action phase scheduling."

    def generate(self, steps,
                 exploration_weight=1.0,
                 consolidation_weight=0.05,
                 decay_shape="cosine",
                 exploration_end=0.30,
                 convergence_end=0.75):

        weights = []
        for i in range(steps):
            frac = i / max(steps - 1, 1)

            if frac < exploration_end:
                w = exploration_weight
            elif frac < convergence_end:
                # Decay from exploration_weight to consolidation_weight
                progress = (frac - exploration_end) / (convergence_end - exploration_end)
                if decay_shape == "cosine":
                    t = 0.5 * (1.0 + math.cos(math.pi * progress))
                elif decay_shape == "exponential":
                    t = math.exp(-3.0 * progress)
                else:  # linear
                    t = 1.0 - progress
                w = consolidation_weight + (exploration_weight - consolidation_weight) * t
            else:
                w = consolidation_weight

            weights.append(round(w, 4))

        return (weights,)


# =============================================================================
# Perturbation Layer Schedule Node
# =============================================================================

class CoSPAFS_PerturbationSchedule:
    """
    Phase-aware perturbation guidance layer schedule.

    Outputs per-step lists of DiT block indices to skip during the
    unconditional forward pass (perturbation guidance).

    The paper's layer specialization finding tells us which layers
    are safe to perturb at each phase:
    - Exploration: No perturbation (let the model explore freely)
    - Convergence: Perturb late/consolidation layers (reasoning is in mid layers)
    - Consolidation: Perturb early/perception layers (redundant at this point)

    Output format: comma-separated block indices per step, suitable for
    use with perturbation guidance nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 4, "max": 200}),
                "total_blocks": ("INT", {
                    "default": 48, "min": 8, "max": 128,
                    "tooltip": "Total DiT blocks. LTX-2.3 22B = 48. LTX-2 19B = 48. Wan2.2 14B = 40.",
                }),
            },
            "optional": {
                "exploration_end": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.50, "step": 0.05}),
                "convergence_end": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 0.95, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_schedule",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = "Per-step perturbation guidance layer indices based on DiT layer specialization findings."

    def generate(self, steps, total_blocks=48,
                 exploration_end=0.30, convergence_end=0.75):

        schedules = []
        for i in range(steps):
            frac = i / max(steps - 1, 1)

            if frac < exploration_end:
                phase = 'exploration'
            elif frac < convergence_end:
                phase = 'convergence'
            else:
                phase = 'consolidation'

            layers = perturbation_layers_for_phase(phase, total_blocks)
            schedules.append(",".join(str(l) for l in layers) if layers else "none")

        return (schedules,)


# =============================================================================
# Phase-Adaptive CFG Guider
# =============================================================================

class Guider_CoSPAFS(comfy.samplers.CFGGuider):
    """
    Phase-adaptive CFG guider with smooth hermite transitions.

    Unlike standard CFGGuider which applies a fixed CFG value at every
    denoising step, this guider varies CFG based on the current sigma:

    - Exploration (early, high sigma): Reduced CFG allows the model to
      maintain hypothesis diversity. Strong guidance here collapses
      superposition prematurely — the extra-limbs problem.
    - Convergence (middle): Boosted CFG forces commitment to a specific
      trajectory. This is where the model needs the strongest push from
      the text conditioning.
    - Consolidation (late, low sigma): Moderate CFG for texture refinement.
      Reasoning is done; heavy guidance here causes oversaturation.

    Phase transitions use Hermite smoothstep to prevent CFG discontinuities
    that would create visible artifacts in the output.
    """

    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self._phase_cfg = None
        self._sigma_bounds = (1.0, 0.0)
        self._transition_width = 0.05
        self._log_steps = False
        self._call_count = 0

    def set_phase_cfg(self, base_cfg, explore_mult, converge_mult, consolidate_mult,
                       explore_end, converge_end, sigma_max, sigma_min,
                       transition_width=0.05, log_steps=False):
        self._phase_cfg = {
            'base': base_cfg,
            'explore_mult': explore_mult,
            'converge_mult': converge_mult,
            'consolidate_mult': consolidate_mult,
            'explore_end': explore_end,
            'converge_end': converge_end,
        }
        self._sigma_bounds = (float(sigma_max), float(sigma_min))
        self._transition_width = transition_width
        self._log_steps = log_steps
        self._call_count = 0

    def _sigma_to_frac(self, sigma):
        """Map current sigma to schedule fraction [0=start/noisy, 1=end/clean]."""
        s_max, s_min = self._sigma_bounds
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.mean().item()
        return max(0.0, min(1.0, (s_max - sigma) / max(s_max - s_min, 1e-8)))

    def _smoothstep(self, x, edge0, edge1):
        """Hermite smoothstep for artifact-free phase blending."""
        t = max(0.0, min(1.0, (x - edge0) / max(edge1 - edge0, 1e-8)))
        return t * t * (3.0 - 2.0 * t)

    def _get_phase_cfg(self, sigma):
        """Compute phase-adaptive CFG for current sigma level."""
        if self._phase_cfg is None:
            return self.cfg

        frac = self._sigma_to_frac(sigma)
        p = self._phase_cfg
        tw = self._transition_width

        cfg_e = p['base'] * p['explore_mult']
        cfg_c = p['base'] * p['converge_mult']
        cfg_s = p['base'] * p['consolidate_mult']

        e_end = p['explore_end']
        c_end = p['converge_end']

        # Smooth 3-phase blending with hermite transitions
        if frac < e_end - tw:
            cfg_val = cfg_e
            phase = 'exploration'
        elif frac < e_end + tw:
            blend = self._smoothstep(frac, e_end - tw, e_end + tw)
            cfg_val = cfg_e * (1.0 - blend) + cfg_c * blend
            phase = 'explore->converge'
        elif frac < c_end - tw:
            cfg_val = cfg_c
            phase = 'convergence'
        elif frac < c_end + tw:
            blend = self._smoothstep(frac, c_end - tw, c_end + tw)
            cfg_val = cfg_c * (1.0 - blend) + cfg_s * blend
            phase = 'converge->consolidate'
        else:
            cfg_val = cfg_s
            phase = 'consolidation'

        if self._log_steps:
            self._call_count += 1
            if self._call_count <= 3 or self._call_count % 50 == 0:
                s_val = sigma.mean().item() if isinstance(sigma, torch.Tensor) else sigma
                print(f"[CoS-PAFS Guider] call {self._call_count}: "
                      f"sigma={s_val:.4f}, frac={frac:.3f}, "
                      f"phase={phase}, cfg={cfg_val:.2f}")

        return cfg_val

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # Dynamically update CFG before parent runs cond/uncond
        self.cfg = self._get_phase_cfg(timestep)
        return super().predict_noise(x, timestep, model_options, seed)


class CoSPAFS_CFGGuider:
    """
    Phase-adaptive CFG guider for CoS-PAFS.

    Replaces the standard CFG Guider node in SamplerCustomAdvanced workflows.
    Varies classifier-free guidance strength across the denoising trajectory
    using smooth hermite transitions between phases.

    Phase rationale:
    - Exploration: LOWER CFG — let the model maintain multiple hypotheses.
      High CFG here causes premature superposition collapse (extra limbs).
    - Convergence: HIGHER CFG — push the model to commit. This is where
      text conditioning steers the trajectory to the right answer.
    - Consolidation: MODERATE CFG — reasoning is done, just refine texture.
      Over-guidance here causes oversaturation.

    Connects to SamplerCustomAdvanced's GUIDER input. Use alongside the
    CoS-PAFS Sampler and CoS-PAFS Sigmas nodes.

    IMPORTANT: Set sigma_max to match your sigma schedule. 1.0 for
    flow-matching models (LTX-2), higher for EPS models (SD1.5 with
    CoS-PAFS Sigmas at sigma_max=5 → set this to 5). Or connect your
    SIGMAS to auto-detect the range.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "base_cfg": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Base CFG scale. Phase multipliers scale on top. SD1.5 with CoS-PAFS Sigmas at sigma_max=5: try 30-40 as base.",
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS", {
                    "tooltip": "Connect your SIGMAS here to auto-detect sigma range. Overrides sigma_max/sigma_min when connected.",
                }),
                "exploration_cfg_mult": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "CFG multiplier during exploration. <1.0 = reduced guidance for hypothesis diversity. Lower values extend superposition window.",
                }),
                "convergence_cfg_mult": ("FLOAT", {
                    "default": 1.3, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "CFG multiplier during convergence. >1.0 = boosted guidance for commitment. This is where text steering matters most.",
                }),
                "consolidation_cfg_mult": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "CFG multiplier during consolidation. Slightly reduced — reasoning is done, strong guidance causes oversaturation.",
                }),
                "exploration_end": ("FLOAT", {
                    "default": 0.30, "min": 0.05, "max": 0.50, "step": 0.05,
                    "tooltip": "Schedule fraction where exploration ends. Matches CoS-PAFS Sampler phase boundaries.",
                }),
                "convergence_end": ("FLOAT", {
                    "default": 0.75, "min": 0.50, "max": 0.95, "step": 0.05,
                    "tooltip": "Schedule fraction where convergence ends.",
                }),
                "sigma_max": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01,
                    "tooltip": "Maximum sigma in your schedule (overridden if SIGMAS connected). 1.0 for LTX-2, 5.0 for SD1.5 with CoS-PAFS Sigmas at 5.",
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Minimum sigma (overridden if SIGMAS connected).",
                }),
                "transition_width": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.20, "step": 0.01,
                    "tooltip": "Width of smooth blending between phases (fraction of schedule). 0.0 = hard transition, 0.05 = smooth hermite blend.",
                }),
                "log_phases": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print CFG values and phase assignments to console.",
                }),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = "Phase-adaptive CFG guider. Varies guidance strength across denoising phases with smooth transitions. Replaces standard CFG Guider for SamplerCustomAdvanced."

    def get_guider(self, model, positive, negative, base_cfg=3.0,
                    sigmas=None,
                    exploration_cfg_mult=0.6,
                    convergence_cfg_mult=1.3,
                    consolidation_cfg_mult=0.9,
                    exploration_end=0.30,
                    convergence_end=0.75,
                    sigma_max=1.0,
                    sigma_min=0.0,
                    transition_width=0.05,
                    log_phases=False):

        # Auto-detect sigma range from connected sigmas
        if sigmas is not None and len(sigmas) > 1:
            sigma_max = float(sigmas[0])
            # sigmas[-1] is typically 0.0 (final step), use second-to-last
            # for the effective minimum, but keep 0.0 as floor
            sigma_min = float(sigmas[-1])
            if log_phases:
                print(f"[CoS-PAFS Guider] Auto-detected sigma range: "
                      f"{sigma_max:.4f} -> {sigma_min:.4f} ({len(sigmas)-1} steps)")

        guider = Guider_CoSPAFS(model)
        guider.set_conds(positive=positive, negative=negative)
        guider.set_cfg(base_cfg)  # fallback if phase logic fails

        guider.set_phase_cfg(
            base_cfg=base_cfg,
            explore_mult=exploration_cfg_mult,
            converge_mult=convergence_cfg_mult,
            consolidate_mult=consolidation_cfg_mult,
            explore_end=exploration_end,
            converge_end=convergence_end,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            transition_width=transition_width,
            log_steps=log_phases,
        )

        return (guider,)

"""
CoS-PAFS EMAG Guider — Diagnostic Build
=========================================
Phase-adaptive CFG + Exponential Moving Average attention perturbation guidance
with transition telemetry for phase boundary research.

This is the standard CoS-PAFS EMAG Guider with an added diagnostic layer that
measures three signals at each denoising step:

1. VELOCITY — cosine similarity between consecutive steps' noise predictions.
   Tracks how much the output changes per step. High velocity = the system is
   actively exploring or committing. Low velocity = the system is settled.
   A characteristic velocity pattern around the phase boundary would indicate
   a natural commitment signal.

2. PROMPT INFLUENCE — cosine similarity between unconditional and conditional
   predictions. Measures how strongly the prompt is steering the output at
   each step. Low similarity = strong prompt influence (cond diverges from
   uncond). High similarity = weak prompt influence (cond ≈ uncond).

3. SUSCEPTIBILITY — cosine similarity between standard conditional and
   EMAG-perturbed conditional predictions. Measures how much a small
   perturbation (the EMA attention injection) changes the output. Low
   similarity = high susceptibility (system is sensitive to perturbation,
   still exploring). High similarity = low susceptibility (system has
   committed, perturbation can't move it). This is the criticality-analog
   signal — the diffusion equivalent of the brain's branching parameter.
   Only available during EMAG-active steps (exploration + convergence).

All three signals are computed from quantities already available in the
guidance pipeline. Zero additional NFEs. Zero impact on output when
telemetry is disabled.

Theoretical motivation: The critical brain hypothesis predicts that
cognitive systems at a phase boundary show maximum susceptibility to
perturbation, and that susceptibility drops as the system commits. If
diffusion sampling exhibits the same dynamics, the susceptibility signal
should show a characteristic transition during the exploration-to-convergence
boundary that correlates with output quality. This diagnostic build provides
the instrumentation to test that prediction.

References:
- CoS-PAFS: Chain-of-Steps Phase-Adaptive Flow Solver (Zac, FHDC 2026)
- EMAG: Exponential Moving Average Guidance (paper Eq. 12, 15, 16)
- PSTF v2.0: Phase-Structured Teaching Framework (Zac & Claude, 2026)
- Hengen & Shew (2025): "Is criticality a unified setpoint of brain function?"
  Neuron 113(16), 2582-2598.e2
"""


# =============================================================================
# Implementation
# =============================================================================

class CoSPAFS_EMAGGuiderImpl(comfy.samplers.CFGGuider):
    """
    Phase-adaptive CFG + EMAG guidance with optional transition telemetry.
    
    Inherits from CFGGuider to maintain full SamplerCustomAdvanced compatibility.
    EMA state persists on model_patcher across guider recreations and resets
    automatically when a new generation is detected (sigma increases).
    """
    
    _EMA_STATE_KEY = '_cos_emag_ema_state'
    _EMA_STEP_KEY = '_cos_emag_step_counter'
    _active_instance = None  # Class-level: track last instance for hook cleanup

    def __init__(self, model_patcher):
        # Clean up previous instance's hooks BEFORE they become orphaned
        if CoSPAFS_EMAGGuiderImpl._active_instance is not None:
            try:
                CoSPAFS_EMAGGuiderImpl._active_instance._remove_emag_hooks()
            except Exception:
                pass
        CoSPAFS_EMAGGuiderImpl._active_instance = self
        super().__init__(model_patcher)
        
        # Phase-adaptive CFG config (set via set_phase_config)
        self._phase_cfg = None
        self._explore_end = 0.46
        self._converge_end = 0.85
        self._transition_width = 0.05
        self._sigma_bounds = (1.0, 0.0)
        
        # EMAG config (set via set_emag_config)
        self._emag_scale = 1.75
        self._emag_explore_mult = 1.5
        self._emag_converge_mult = 1.0
        self._emag_consolidate_mult = 0.0
        self._ema_decay = 0.9
        self._adaptive_layers = True
        self._perturb_img_to_text = True
        self._emag_start_pct = 1.0
        self._emag_end_pct = 0.0
        self._layer_override = None
        
        # Transient state
        self._hook_handles = []
        self._log_steps = False
        self._call_count = 0
        self.total_steps = None
        self._sigmas_schedule = None  # Real sigmas for step-based frac
        self._generation_active = False  # Track if we're mid-generation
        
        # Telemetry state
        self._telemetry_enabled = False
        self._prev_noise_pred = None
        self._telemetry_log = []
        self._hooks_registered_count = 0  # Track successful hook registrations
        self._registry_config = None  # Matched model_registry.json entry
        self._hook_mode = "auto"  # auto/attn/block — controls measurement hook type
        self._effective_hook_mode = None  # What was actually used (for CSV metadata)

        # ECC v2 closed-loop controller
        self._ecc = None  # Set via set_ecc_config
        self._ecc_enabled = False

        # CSV telemetry logging
        self._csv_session = None
        self._csv_path = None
        self._csv_gen_idx = 0
        self._csv_initialized = False
        self._csv_prompt_text = ""
        self._csv_prompt_hash = 0

    # -----------------------------------------------------------------
    # Cleanup — prevent hook accumulation across generations
    # -----------------------------------------------------------------
    
    def __del__(self):
        try:
            self._remove_emag_hooks()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Configuration (called by the node's get_guider)
    # -----------------------------------------------------------------
    
    def set_phase_config(self, base_cfg, explore_mult, converge_mult, consolidate_mult,
                          explore_end, converge_end, sigma_max, sigma_min,
                          transition_width=0.05, log_steps=False):
        self._phase_cfg = {
            'base': base_cfg,
            'explore_mult': explore_mult,
            'converge_mult': converge_mult,
            'consolidate_mult': consolidate_mult,
        }
        self._explore_end = explore_end
        self._converge_end = converge_end
        self._sigma_bounds = (float(sigma_max), float(sigma_min))
        self._transition_width = transition_width
        self._log_steps = log_steps
        self._call_count = 0

    def set_emag_config(self, emag_scale, ema_decay, 
                         emag_explore_mult, emag_converge_mult, emag_consolidate_mult,
                         adaptive_layers, perturb_img_to_text,
                         start_percent, end_percent, layer_override=None):
        self._emag_scale = emag_scale
        self._ema_decay = ema_decay
        self._emag_explore_mult = emag_explore_mult
        self._emag_converge_mult = emag_converge_mult
        self._emag_consolidate_mult = emag_consolidate_mult
        self._adaptive_layers = adaptive_layers
        self._perturb_img_to_text = perturb_img_to_text
        self._emag_start_pct = start_percent
        self._emag_end_pct = end_percent
        # Parse layer override: "0,1,2,3" or "8-11" or empty for default
        self._layer_override = None
        if layer_override and layer_override.strip():
            try:
                parts = layer_override.strip().split(',')
                layers = []
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = part.split('-')
                        layers.extend(range(int(start), int(end) + 1))
                    else:
                        layers.append(int(part))
                self._layer_override = sorted(set(layers))
            except Exception as e:
                print(f"[CoS-EMAG] Invalid layer_override '{layer_override}': {e}")
                self._layer_override = None

    def set_telemetry(self, enabled):
        self._telemetry_enabled = enabled

    def set_sigmas_schedule(self, sigmas):
        """Store the real sigmas schedule for step-based phase computation.
        This allows frac to be computed from step position in the known schedule,
        immune to gate softening or sigma lying that may alter the timestep
        the guider receives."""
        if sigmas is not None and len(sigmas) > 1:
            self._sigmas_schedule = sigmas.clone()
            self.total_steps = len(sigmas) - 1

    # -----------------------------------------------------------------
    # Phase-Adaptive Math
    # -----------------------------------------------------------------
    
    def _sigma_to_frac(self, sigma):
        """Map current sigma to schedule fraction [0=start/noisy, 1=end/clean]."""
        s_max, s_min = self._sigma_bounds
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.mean().item()
        return max(0.0, min(1.0, (s_max - sigma) / max(s_max - s_min, 1e-8)))

    def _step_to_frac(self, step):
        """Map current step index to schedule fraction [0=start, 1=end].
        Uses step counter against known total — immune to gate softening."""
        if self.total_steps is not None and self.total_steps > 0:
            return max(0.0, min(1.0, step / self.total_steps))
        return None

    def _get_frac(self, sigma, step=None):
        """Get schedule fraction, preferring step-based when available.
        Falls back to sigma-based for backward compatibility."""
        if step is not None:
            step_frac = self._step_to_frac(step)
            if step_frac is not None:
                return step_frac
        return self._sigma_to_frac(sigma)

    def _smoothstep(self, x, edge0, edge1):
        """Hermite smoothstep for artifact-free phase blending."""
        t = max(0.0, min(1.0, (x - edge0) / max(edge1 - edge0, 1e-8)))
        return t * t * (3.0 - 2.0 * t)

    def _interpolate_phase(self, frac, explore_val, converge_val, consolidate_val):
        """
        Hermite-smoothstep interpolation of a value across the three phases.
        Used for both CFG and EMAG scale phase profiles.
        """
        tw = self._transition_width
        e_end = self._explore_end
        c_end = self._converge_end

        if frac < e_end - tw:
            return explore_val
        elif frac < e_end + tw:
            blend = self._smoothstep(frac, e_end - tw, e_end + tw)
            return explore_val * (1.0 - blend) + converge_val * blend
        elif frac < c_end - tw:
            return converge_val
        elif frac < c_end + tw:
            blend = self._smoothstep(frac, c_end - tw, c_end + tw)
            return converge_val * (1.0 - blend) + consolidate_val * blend
        else:
            return consolidate_val

    def _get_phase_cfg(self, sigma, step=None):
        """Compute phase-adaptive CFG for current sigma/step."""
        if self._phase_cfg is None:
            return self.cfg
        p = self._phase_cfg
        frac = self._get_frac(sigma, step)
        cfg_val = p['base'] * self._interpolate_phase(
            frac, p['explore_mult'], p['converge_mult'], p['consolidate_mult']
        )
        return cfg_val

    def _get_phase_emag_scale(self, sigma, step=None):
        """Compute phase-adaptive EMAG scale for current sigma/step."""
        frac = self._get_frac(sigma, step)
        return self._emag_scale * self._interpolate_phase(
            frac,
            self._emag_explore_mult,
            self._emag_converge_mult,
            self._emag_consolidate_mult,
        )

    # -----------------------------------------------------------------
    # Persistent EMA State (survives guider recreation)
    # -----------------------------------------------------------------
    
    def _get_persistent_ema(self):
        if not hasattr(self.model_patcher, self._EMA_STATE_KEY):
            setattr(self.model_patcher, self._EMA_STATE_KEY, {})
        return getattr(self.model_patcher, self._EMA_STATE_KEY)
    
    def _get_persistent_step(self):
        if not hasattr(self.model_patcher, self._EMA_STEP_KEY):
            setattr(self.model_patcher, self._EMA_STEP_KEY, 0)
        return getattr(self.model_patcher, self._EMA_STEP_KEY)
    
    def _set_persistent_step(self, step):
        setattr(self.model_patcher, self._EMA_STEP_KEY, step)
    
    def _detect_new_generation(self, timestep):
        """Detect new generation using step counter against known total.
        
        Previous approach compared timesteps, which failed when gate softening
        caused non-monotonic effective sigma sequences. This approach uses the
        step counter: if we've reached or exceeded total_steps, the next call
        must be a new generation. Also detects first-ever call.
        
        Falls back to timestep comparison only when no sigmas schedule is
        available (backward compat with non-CoS-PAFS workflows).
        """
        current_step = self._get_persistent_step()
        
        # If we have a known total, use step counter (immune to gate softening)
        if self.total_steps is not None:
            if not self._generation_active:
                # First call ever or after a completed generation
                if self._log_steps:
                    print(f"[CoS-EMAG] New generation detected (first call or post-completion).")
                self._generation_active = True
                if self._csv_session:
                    self._csv_gen_idx += 1
                    if self._log_steps:
                        print(f"[CoS-CSV] New generation → gen_idx={self._csv_gen_idx}")
                return True
            elif current_step >= self.total_steps:
                # We've completed a full generation, next call is new
                if self._log_steps:
                    print(f"[CoS-EMAG] New generation detected "
                          f"(step {current_step} >= total {self.total_steps}).")
                if self._csv_session:
                    self._csv_gen_idx += 1
                    if self._log_steps:
                        print(f"[CoS-CSV] New generation → gen_idx={self._csv_gen_idx}")
                return True
            return False
        
        # Fallback: timestep comparison (for workflows without sigmas)
        ts_val = timestep.item() if isinstance(timestep, torch.Tensor) else float(timestep)
        last_ts_key = '_cos_emag_last_timestep'
        
        if not hasattr(self.model_patcher, last_ts_key):
            setattr(self.model_patcher, last_ts_key, ts_val)
            self._generation_active = True
            return True
        
        last_ts = getattr(self.model_patcher, last_ts_key, 0.0)
        setattr(self.model_patcher, last_ts_key, ts_val)
        
        if ts_val > last_ts * 1.5 + 0.5:
            # Large jump up — likely new generation (use wider margin)
            if self._log_steps:
                print(f"[CoS-EMAG] New generation detected (fallback: "
                      f"ts {ts_val:.4f} >> last {last_ts:.4f}).")
            self._generation_active = True
            return True
        
        return False

    # -----------------------------------------------------------------
    # EMA Hook Machinery
    # -----------------------------------------------------------------
    
    def _find_transformer_blocks(self, model):
        """Find transformer blocks across architectures.
        
        Resolution order:
        1. Model registry lookup by class name → known block_path + attn_attrs
        2. Auto-detect DiT/MMDiT: transformer_blocks, joint_blocks, blocks, layers
        3. UNet linearization: recursive extraction from input/mid/output blocks
        
        Returns a list of blocks that have attention modules, or None.
        Sets self._registry_config when a registry match provides attn paths.
        """
        dm = model.diffusion_model if hasattr(model, 'diffusion_model') else model
        dm_class = type(dm).__name__
        self._registry_config = None  # Reset each time
        
        # --- Step 1: Registry lookup ---
        reg_entry = _MODEL_REGISTRY.get(dm_class, None)
        if reg_entry is not None:
            block_path = reg_entry.get('block_path', 'NEEDS_DISCOVERY')
            attn_cfg = reg_entry.get('attn_attrs', {})
            self_attn_paths = attn_cfg.get('self_attn', [])
            
            blocks = _resolve_block_list(dm, block_path)
            if blocks is not None:
                if len(blocks) > 0:
                    # Validate: check that at least one attn path resolves on block[0]
                    resolved = False
                    for attn_path in self_attn_paths:
                        if attn_path == 'NEEDS_DISCOVERY':
                            continue
                        if attn_path == '__block__':
                            # Block-level hook — always valid
                            resolved = True
                            break
                        module = _resolve_dotted_attr(blocks[0], attn_path)
                        if module is not None:
                            resolved = True
                            break
                    
                    # Also accept if block has standard flat attrs (registry may
                    # omit self_attn paths for models with standard naming)
                    if not resolved:
                        for flat in ('attn1', 'attn', 'self_attn', 'attention'):
                            if hasattr(blocks[0], flat):
                                resolved = True
                                break
                    
                    if resolved:
                        self._registry_config = reg_entry
                        if self._log_steps:
                            print(f"[CoS-EMAG] Registry match: {dm_class} → "
                                  f"{len(blocks)} blocks via '{block_path}', "
                                  f"attn paths: {self_attn_paths}")
                        return list(blocks)
                    else:
                        if self._log_steps:
                            b0 = blocks[0]
                            children = [n for n, _ in b0.named_children()]
                            print(f"[CoS-EMAG] Registry entry for {dm_class} found "
                                  f"{len(blocks)} blocks via '{block_path}' but "
                                  f"attn paths {self_attn_paths} did not resolve. "
                                  f"Block children: {children[:10]}")
        
        # --- Step 2: Auto-detect DiT/MMDiT (flat structure) ---
        for attr in ('transformer_blocks', 'joint_blocks', 'blocks', 'layers'):
            if hasattr(dm, attr):
                blocks = _as_block_seq(getattr(dm, attr))
                if len(blocks) > 0:
                    # In block hook mode, accept any block list — we hook the
                    # block itself, no attn sub-module needed
                    if self._hook_mode == 'block':
                        if self._log_steps:
                            print(f"[CoS-EMAG] Found {len(blocks)} blocks "
                                  f"via '{attr}' (hook_mode=block, no attn "
                                  f"validation needed)")
                        return list(blocks)
                    # Check standard flat attention attrs
                    if (hasattr(blocks[0], 'attn1') or hasattr(blocks[0], 'attn')
                            or hasattr(blocks[0], 'self_attn')
                            or hasattr(blocks[0], 'attention')):
                        if self._log_steps:
                            print(f"[CoS-EMAG] Found {len(blocks)} DiT/MMDiT "
                                  f"transformer blocks via '{attr}'")
                        return list(blocks)
                    else:
                        if self._log_steps:
                            b0 = blocks[0]
                            attn_like = [a for a in dir(b0) if 'attn' in a.lower()]
                            children = [n for n, _ in b0.named_children()]
                            print(f"[CoS-EMAG DEBUG] Found {len(blocks)} blocks "
                                  f"via '{attr}' but no attn1/attn/self_attn")
                            print(f"[CoS-EMAG DEBUG] block[0] type: "
                                  f"{type(b0).__name__}")
                            print(f"[CoS-EMAG DEBUG] attn-like attrs: {attn_like}")
                            print(f"[CoS-EMAG DEBUG] child modules: {children[:20]}")
        
        # --- Step 3: UNet linearization ---
        unet_blocks = _linearize_unet(dm)
        
        if unet_blocks:
            if self._log_steps:
                print(f"[CoS-EMAG] Found {len(unet_blocks)} UNet transformer "
                      f"blocks (SD1.5/SDXL architecture)")
            return unet_blocks
        
        if self._log_steps:
            print(f"[CoS-EMAG] WARNING: No transformer blocks found! "
                  f"EMAG perturbation will have no effect.")
            print(f"[CoS-EMAG] diffusion_model type: {dm_class}")
            attrs = [a for a in dir(dm)
                     if not a.startswith('_')
                     and isinstance(getattr(dm, a, None), torch.nn.Module)]
            print(f"[CoS-EMAG] diffusion_model attrs: {attrs[:15]}")
        
        return None

    def _select_layers_adaptive(self, blocks):
        """Adaptive layer selection (EMAG paper Section 4.3).
        Perturbs mid-to-deep layers. Adapted for variable block counts."""
        n = len(blocks)
        if n <= 4:
            # Very few blocks — perturb all
            return list(range(n))
        elif n <= 12:
            # Small-medium model — perturb middle third
            start = max(0, n // 3)
            end = min(n, 2 * n // 3 + 1)
            return list(range(start, end))
        elif n <= 20:
            # SD1.5 range (~16 blocks typically) — perturb middle-to-deep
            start = n // 3
            end = min(n, start + n // 3 + 1)
            return list(range(start, end))
        else:
            # Large model (SDXL, DiT) — original EMAG paper strategy
            return list(range(12, min(n, 16)))

    def _hook_attention_modules(self, block, layer_idx, ema_attention):
        """Hook self-attention and optionally cross-attention on a block.
        
        Respects self._hook_mode:
        - 'block': force block-level measurement hook (passive, no EMAG)
        - 'attn': force attention-module hooks (skip __block__)
        - 'auto': use registry (including __block__), then auto-detect
        """
        hooked_self = False
        hooked_cross = False
        
        # --- Forced block mode: hook the block itself ---
        if self._hook_mode == 'block':
            handle = block.register_forward_hook(
                self._make_measurement_hook(layer_idx)
            )
            self._hook_handles.append(handle)
            if self._effective_hook_mode is None:
                self._effective_hook_mode = 'block'
            return  # Done — block hook covers everything
        
        # --- Registry path resolution (auto and attn modes) ---
        if self._registry_config is not None:
            attn_cfg = self._registry_config.get('attn_attrs', {})
            
            # Self-attention from registry
            for attn_path in attn_cfg.get('self_attn', []):
                if attn_path == 'NEEDS_DISCOVERY':
                    continue
                if attn_path == '__block__':
                    if self._hook_mode == 'attn':
                        # Skip __block__ in forced attn mode — try other paths
                        continue
                    # Auto mode: hook the block itself (measurement-only)
                    handle = block.register_forward_hook(
                        self._make_measurement_hook(layer_idx)
                    )
                    self._hook_handles.append(handle)
                    hooked_self = True
                    hooked_cross = True
                    if self._effective_hook_mode is None:
                        self._effective_hook_mode = 'block'
                    break
                module = _resolve_dotted_attr(block, attn_path)
                if module is not None:
                    handle = module.register_forward_hook(
                        self._make_emag_hook(layer_idx, 'self', ema_attention)
                    )
                    self._hook_handles.append(handle)
                    hooked_self = True
                    if self._effective_hook_mode is None:
                        self._effective_hook_mode = 'attn'
                    break
            
            # Cross-attention from registry
            if not hooked_cross and self._perturb_img_to_text:
                for attn_path in attn_cfg.get('cross_attn', []):
                    if attn_path == 'NEEDS_DISCOVERY':
                        continue
                    module = _resolve_dotted_attr(block, attn_path)
                    if module is not None:
                        handle = module.register_forward_hook(
                            self._make_emag_hook(layer_idx, 'cross', ema_attention)
                        )
                        self._hook_handles.append(handle)
                        hooked_cross = True
                        break
        
        # --- Flat auto-detect fallback (standard architectures) ---
        if not hooked_self:
            for attr_name in ('attn1', 'attn', 'self_attn', 'attention'):
                if hasattr(block, attr_name):
                    module = getattr(block, attr_name)
                    handle = module.register_forward_hook(
                        self._make_emag_hook(layer_idx, 'self', ema_attention)
                    )
                    self._hook_handles.append(handle)
                    hooked_self = True
                    if self._effective_hook_mode is None:
                        self._effective_hook_mode = 'attn'
                    break
        
        if not hooked_cross and self._perturb_img_to_text:
            for attr_name in ('attn2', 'cross_attn'):
                if hasattr(block, attr_name):
                    module = getattr(block, attr_name)
                    handle = module.register_forward_hook(
                        self._make_emag_hook(layer_idx, 'cross', ema_attention)
                    )
                    self._hook_handles.append(handle)
                    hooked_cross = True
                    break

        # --- MLP hook registration (dual-hook telemetry, v1.5) ---
        # If registry has mlp_attrs.mlp paths and the architecture is marked
        # dual_hook_supported, register a passive MLP measurement hook.
        if self._registry_config is not None:
            mlp_cfg = self._registry_config.get('mlp_attrs', {})
            mlp_paths = mlp_cfg.get('mlp', []) if isinstance(mlp_cfg, dict) else []
            dual_supported = self._registry_config.get('dual_hook_supported', False)
            if dual_supported and mlp_paths:
                mlp_resolved = False
                for mlp_path in mlp_paths:
                    if mlp_path == 'NEEDS_DISCOVERY':
                        continue
                    mlp_module = _resolve_dotted_attr(block, mlp_path)
                    if mlp_module is not None:
                        handle = mlp_module.register_forward_hook(
                            self._make_mlp_measurement_hook(layer_idx)
                        )
                        self._hook_handles.append(handle)
                        mlp_resolved = True
                        if not getattr(self, '_dual_hook_active', False):
                            self._dual_hook_active = True
                            print(f"[CoS-O2] dual_hook ACTIVE: MLP path '{mlp_path}' resolved on layer {layer_idx}")
                        break
                if not mlp_resolved and not getattr(self, '_dual_hook_warning_printed', False):
                    print(f"[CoS-O2] dual_hook requested but no MLP paths resolved on layer {layer_idx}; falling back to single-hook")
                    self._dual_hook_warning_printed = True

    def _make_measurement_hook(self, layer_idx):
        """
        Create a measurement-only hook for block-level g_micro extraction.
        
        Used for architectures like MMDiT (SD3.5) where attention sub-modules'
        forward() methods aren't called during inference (joint attention is
        computed at the block level). Hooks the block itself, extracts g_micro
        from the block output, and returns the output UNMODIFIED.
        
        No EMAG perturbation — pure passive measurement.
        """
        def hook(module, input, output):
            # Handle tuple outputs (e.g. JointBlock returns (context, x))
            # Some elements may be None (e.g. context stream for later blocks)
            if isinstance(output, tuple):
                # Try elements in reverse — latent/x stream is typically last
                activation = None
                for elem in reversed(output):
                    if elem is not None and isinstance(elem, torch.Tensor):
                        activation = elem
                        break
                if activation is None:
                    return output  # All elements None — skip
            else:
                activation = output
            
            if activation is None:
                return output
            
            if not getattr(self, '_telemetry_enabled', False):
                return output  # No measurement, pass through
            
            if not hasattr(self, '_micro_g_buffer'):
                self._micro_g_buffer = []
            
            try:
                if not getattr(self, '_micro_shape_printed', False):
                    if isinstance(output, tuple):
                        # Find which index we selected
                        for i_rev, elem in enumerate(reversed(output)):
                            if elem is activation:
                                idx = len(output) - 1 - i_rev
                                break
                        stream_info = f" (tuple[{idx}] of {len(output)} elements)"
                    else:
                        stream_info = ""
                    print(f"[CoS-O2] block output shape={activation.shape}{stream_info}")
                    self._micro_shape_printed = True
                    self._last_g_micro = None
                
                # Flatten batch and sequence, keep feature dim [N, D]
                X = activation.reshape(-1, activation.shape[-1]).float()
                
                # Eigenvalues of Covariance X^T X = Squared Singular Values of X
                cov = (X.T @ X) / X.shape[0]
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals = torch.clamp(eigvals, min=1e-12)
                singvals = torch.sqrt(eigvals)
                
                # Participation Ratio → g_micro
                pr = (torch.sum(singvals) ** 2) / torch.sum(eigvals)
                g_micro = (pr / X.shape[-1]).item()
                self._micro_g_buffer.append(g_micro)
                
                # Per-layer tracking under tuple key (layer_idx, 0) for sub_pos=attn
                # This harmonizes with the MLP hook which uses (layer_idx, 1).
                key = (layer_idx, 0)
                if not hasattr(self, '_micro_g_per_layer'):
                    self._micro_g_per_layer = {}
                if key not in self._micro_g_per_layer:
                    self._micro_g_per_layer[key] = []
                self._micro_g_per_layer[key].append(g_micro)
                
                # Store mean activation vector for phi_micro computation
                mean_vec = X.mean(dim=0).detach().cpu()
                if not hasattr(self, '_micro_mean_vectors'):
                    self._micro_mean_vectors = {}
                self._micro_mean_vectors[key] = mean_vec
                
                # Spectral gap and entropy
                sorted_sv = torch.sort(singvals, descending=True).values
                spectral_gap = (sorted_sv[0] / sorted_sv[1]).item() if sorted_sv[1] > 1e-12 else float('inf')
                p = singvals / torch.sum(singvals)
                sv_entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
                
                if not hasattr(self, '_micro_gap_buffer'):
                    self._micro_gap_buffer = []
                    self._micro_entropy_buffer = []
                self._micro_gap_buffer.append(spectral_gap)
                self._micro_entropy_buffer.append(sv_entropy)
                
                if not hasattr(self, '_micro_gap_per_layer'):
                    self._micro_gap_per_layer = {}
                    self._micro_ent_per_layer = {}
                    self._micro_seqlen_per_layer = {}
                    self._micro_hdim_per_layer = {}
                if key not in self._micro_gap_per_layer:
                    self._micro_gap_per_layer[key] = []
                    self._micro_ent_per_layer[key] = []
                self._micro_gap_per_layer[key].append(spectral_gap)
                self._micro_ent_per_layer[key].append(sv_entropy)
                self._micro_seqlen_per_layer[key] = X.shape[0]
                self._micro_hdim_per_layer[key] = X.shape[-1]
                
            except Exception as e:
                if not getattr(self, '_micro_g_error_printed', False):
                    shape_str = getattr(activation, 'shape', 'None') if activation is not None else 'None'
                    print(f"[CoS-O2] block g_micro extraction failed: {e} "
                          f"| shape={shape_str}")
                    self._micro_g_error_printed = True
            
            return output  # Always return unmodified
        
        return hook

    def _make_mlp_measurement_hook(self, layer_idx):
        """
        Create a passive measurement hook for MLP/FFN output.
        
        Captures g_micro and the mean activation vector under the tuple key
        (layer_idx, 1) — sub_pos=1 designates MLP. Used together with attn
        hooks (sub_pos=0) to produce a sequence in which consecutive captures
        alternate operation type, enabling sublattice-aware phi_micro telemetry.
        
        No EMAG perturbation — pure passive measurement.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                mlp_output = output[0]
            else:
                mlp_output = output

            if not getattr(self, '_telemetry_enabled', False):
                return output

            if not hasattr(self, '_micro_g_buffer'):
                self._micro_g_buffer = []

            try:
                if not getattr(self, '_micro_mlp_shape_printed', False):
                    print(f"[CoS-O2-MLP] mlp shape={mlp_output.shape} layer={layer_idx}")
                    self._micro_mlp_shape_printed = True

                X = mlp_output.reshape(-1, mlp_output.shape[-1]).float()
                cov = (X.T @ X) / X.shape[0]
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals = torch.clamp(eigvals, min=1e-12)
                singvals = torch.sqrt(eigvals)
                pr = (torch.sum(singvals) ** 2) / torch.sum(eigvals)
                g_micro = (pr / X.shape[-1]).item()
                self._micro_g_buffer.append(g_micro)

                key = (layer_idx, 1)  # sub_pos=1 for MLP
                if not hasattr(self, '_micro_g_per_layer'):
                    self._micro_g_per_layer = {}
                if key not in self._micro_g_per_layer:
                    self._micro_g_per_layer[key] = []
                self._micro_g_per_layer[key].append(g_micro)

                mean_vec = X.mean(dim=0).detach().cpu()
                if not hasattr(self, '_micro_mean_vectors'):
                    self._micro_mean_vectors = {}
                self._micro_mean_vectors[key] = mean_vec

                sorted_sv = torch.sort(singvals, descending=True).values
                spectral_gap = (sorted_sv[0] / sorted_sv[1]).item() if sorted_sv[1] > 1e-12 else float('inf')
                p = singvals / torch.sum(singvals)
                sv_entropy = -torch.sum(p * torch.log(p + 1e-12)).item()

                if not hasattr(self, '_micro_gap_per_layer'):
                    self._micro_gap_per_layer = {}
                    self._micro_ent_per_layer = {}
                    self._micro_seqlen_per_layer = {}
                    self._micro_hdim_per_layer = {}
                if key not in self._micro_gap_per_layer:
                    self._micro_gap_per_layer[key] = []
                    self._micro_ent_per_layer[key] = []
                self._micro_gap_per_layer[key].append(spectral_gap)
                self._micro_ent_per_layer[key].append(sv_entropy)
                self._micro_seqlen_per_layer[key] = X.shape[0]
                self._micro_hdim_per_layer[key] = X.shape[-1]

            except Exception as e:
                if not getattr(self, '_micro_mlp_error_printed', False):
                    print(f"[CoS-O2-MLP] g_micro extraction failed: {e} | shape={mlp_output.shape}")
                    self._micro_mlp_error_printed = True

            return output  # Pass through unmodified

        return hook

    def _make_emag_hook(self, layer_idx, attn_type, ema_attention):
        """
        Create EMA perturbation hook (EMAG Eq. 12).
        First call: stores current attention as EMA, returns original.
        Subsequent: returns OLD EMA (perturbation), updates EMA state.
        PLUS: Extracts Microscopic Coupling (g_micro) via SVD Participation Ratio.
        """
        decay = self._ema_decay
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None
                
            # --- BEGIN NEW: Microscopic Coupling Extraction (O2) ---
            if getattr(self, '_telemetry_enabled', False):
                if not hasattr(self, '_micro_g_buffer'):
                    self._micro_g_buffer = []
                try:
                    if not getattr(self, '_micro_shape_printed', False):
                        print(f"[CoS-O2] attn shape={attn_output.shape}")
                        self._micro_shape_printed = True
                        self._last_g_micro = None
                    # Flatten batch and sequence, keep feature dim [N, D]
                    X = attn_output.reshape(-1, attn_output.shape[-1]).float()
                    
                    # Eigenvalues of Covariance X^T X = Squared Singular Values of X
                    # This [D, D] matrix math prevents OOM on massive sequence lengths
                    cov = (X.T @ X) / X.shape[0]
                    eigvals = torch.linalg.eigvalsh(cov)
                    
                    # Clamp to prevent negative values from floating-point errors
                    eigvals = torch.clamp(eigvals, min=1e-12)
                    singvals = torch.sqrt(eigvals)
                    
                    # Participation Ratio = (Sum(s))^2 / Sum(s^2)
                    pr = (torch.sum(singvals) ** 2) / torch.sum(eigvals)
                    
                    # Normalize by dimension D to get coupling constant g ∈ [0, 1]
                    g_micro = (pr / X.shape[-1]).item()
                    self._micro_g_buffer.append(g_micro)
                    # Per-layer tracking under tuple key (layer_idx, 0) for sub_pos=attn
                    # This harmonizes with the MLP hook which uses (layer_idx, 1).
                    key = (layer_idx, 0)
                    if not hasattr(self, '_micro_g_per_layer'):
                        self._micro_g_per_layer = {}
                    if key not in self._micro_g_per_layer:
                        self._micro_g_per_layer[key] = []
                    self._micro_g_per_layer[key].append(g_micro)
                    
                    # Store mean activation vector for phi_micro computation
                    mean_vec = X.mean(dim=0).detach().cpu()
                    if not hasattr(self, '_micro_mean_vectors'):
                        self._micro_mean_vectors = {}
                    self._micro_mean_vectors[key] = mean_vec
                    
                    # Spectral gap: ratio of largest to second-largest singular value
                    sorted_sv = torch.sort(singvals, descending=True).values
                    spectral_gap = (sorted_sv[0] / sorted_sv[1]).item() if sorted_sv[1] > 1e-12 else float('inf')
                    
                    # Singular value entropy: -Σ p_i log(p_i)
                    p = singvals / torch.sum(singvals)
                    sv_entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
                    
                    if not hasattr(self, '_micro_gap_buffer'):
                        self._micro_gap_buffer = []
                        self._micro_entropy_buffer = []
                    self._micro_gap_buffer.append(spectral_gap)
                    self._micro_entropy_buffer.append(sv_entropy)
                    
                    # Per-layer gap and entropy for CSV (tuple-keyed)
                    if not hasattr(self, '_micro_gap_per_layer'):
                        self._micro_gap_per_layer = {}
                        self._micro_ent_per_layer = {}
                        self._micro_seqlen_per_layer = {}
                        self._micro_hdim_per_layer = {}
                    if key not in self._micro_gap_per_layer:
                        self._micro_gap_per_layer[key] = []
                        self._micro_ent_per_layer[key] = []
                    self._micro_gap_per_layer[key].append(spectral_gap)
                    self._micro_ent_per_layer[key].append(sv_entropy)
                    self._micro_seqlen_per_layer[key] = X.shape[0]
                    self._micro_hdim_per_layer[key] = X.shape[-1]
                except Exception as e:
                    if not getattr(self, '_micro_g_error_printed', False):
                        print(f"[CoS-O2] g_micro extraction failed: {e} | shape={attn_output.shape}")
                        self._micro_g_error_printed = True
            # --- END NEW ---
            
            key = f"layer_{layer_idx}_{attn_type}"
            
            if key in ema_attention:
                ema_attn = ema_attention[key]
                if ema_attn.shape == attn_output.shape:
                    old_ema = ema_attn
                    ema_attention[key] = decay * ema_attn + (1.0 - decay) * attn_output.detach()
                    if rest is not None:
                        return (old_ema,) + rest
                    return old_ema
                else:
                    ema_attention[key] = attn_output.detach().clone()
                    return output
            
            ema_attention[key] = attn_output.detach().clone()
            return output
        
        return hook

    def _register_emag_hooks(self, ema_attention):
        """Register EMA perturbation hooks on transformer blocks."""
        self._remove_emag_hooks()
        self._effective_hook_mode = None  # Reset — will be set by first hook
        try:
            model = self.model_patcher.model
            blocks = self._find_transformer_blocks(model)
            if blocks is None:
                if self._log_steps:
                    print(f"[CoS-EMAG] No blocks found — EMAG perturbation disabled")
                self._hooks_registered_count = 0
                return
            
            layers = (self._layer_override if self._layer_override is not None
                      else (self._select_layers_adaptive(blocks) 
                            if self._adaptive_layers 
                            else list(range(len(blocks)))))
            
            hooks_before = len(self._hook_handles)
            for idx in layers:
                if idx < len(blocks):
                    self._hook_attention_modules(blocks[idx], idx, ema_attention)
            
            self._hooks_registered_count = len(self._hook_handles) - hooks_before
            
            # Log hook count on first registration only
            if self._call_count <= 1 and self._log_steps:
                effective = self._effective_hook_mode or 'unknown'
                print(f"[CoS-EMAG] Registered {self._hooks_registered_count} hooks on "
                      f"{len(layers)} layers (of {len(blocks)} total blocks). "
                      f"hook_mode={self._hook_mode} → effective={effective}. "
                      f"Layers: {layers}")
        except Exception as e:
            print(f"[CoS-EMAG] Hook registration failed: {e}")
            import traceback
            traceback.print_exc()
            self._hooks_registered_count = 0

    def _remove_emag_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except:
                pass
        self._hook_handles.clear()

    # -----------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a, b):
        """
        Cosine similarity between two tensors, flattened to 1D.
        Returns a float in [-1, 1]. Higher = more similar.
        """
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    

    def _record_telemetry(self, step, frac, phase, sigma_val, phase_cfg, phase_emag_scale,
                           emag_active, noise_pred, uncond_pred=None,
                           cond_pred_standard=None, cond_pred_perturbed=None,
                           g_micro=None, phi_micro=None): # <-- NEW PARAMETERS
        """
        Compute and record telemetry signals for this step.
        
        Signals:
          velocity        — cos_sim(noise_pred[t], noise_pred[t-1])
          prompt_influence — cos_sim(uncond_pred, cond_pred_standard)
          susceptibility   — cos_sim(cond_pred_standard, cond_pred_perturbed)
        
        velocity is available every step. prompt_influence and susceptibility
        are only available when the component predictions are provided
        (EMAG-active steps or instrumented 2-pass steps).
        """
        entry = {
            'step': step,
            'frac': frac,
            'phase': phase,
            'sigma': sigma_val,
            'cfg': phase_cfg,
            'emag_scale': phase_emag_scale,
            'emag_active': emag_active,
            'velocity': None,
            'prompt_influence': None,
            'susceptibility': None,
            'g_micro': g_micro, # <-- NEW ENTRY
            'phi_micro': phi_micro, # <-- NEW ENTRY
        }

        # Signal 1: Velocity (step-to-step similarity of final noise prediction)
        if self._prev_noise_pred is not None:
            if self._prev_noise_pred.shape == noise_pred.shape:
                entry['velocity'] = self._cosine_sim(self._prev_noise_pred, noise_pred)
        self._prev_noise_pred = noise_pred.detach().clone()

        # Signal 2: Prompt influence (uncond vs standard cond)
        if uncond_pred is not None and cond_pred_standard is not None:
            entry['prompt_influence'] = self._cosine_sim(uncond_pred, cond_pred_standard)

        # Signal 3: Susceptibility (standard cond vs perturbed cond)
        # This is the criticality-analog signal. Only available when EMAG is active.
        if cond_pred_standard is not None and cond_pred_perturbed is not None:
            entry['susceptibility'] = self._cosine_sim(cond_pred_standard, cond_pred_perturbed)

        self._telemetry_log.append(entry)

        # Per-step console output
        if self._log_steps:
            v_str = f"{entry['velocity']:.4f}" if entry['velocity'] is not None else "---"
            p_str = f"{entry['prompt_influence']:.4f}" if entry['prompt_influence'] is not None else "---"
            s_str = f"{entry['susceptibility']:.4f}" if entry['susceptibility'] is not None else "---"
            g_str = f"{entry['g_micro']:.4f}" if entry['g_micro'] is not None else "---" # <-- NEW
            
            print(f"[CoS-TELEM] step={step:3d} frac={frac:.3f} [{phase:13s}] "
                  f"σ={sigma_val:.4f} cfg={phase_cfg:.2f} emag={phase_emag_scale:.3f} "
                  f"| vel={v_str}  prompt={p_str}  suscept={s_str}  g_micro={g_str}") # <-- UPDATED

    def _print_telemetry_summary(self):
        """
        Print a per-phase summary of telemetry signals at end of generation.
        Highlights the transition zone around the exploration-convergence boundary.
        """
        if not self._telemetry_log:
            return

        print("\n" + "=" * 90)
        print("[CoS-TELEM] GENERATION TELEMETRY SUMMARY")
        print("=" * 90)

        # Group by phase
        phases = {'exploration': [], 'E→C transition': [], 'convergence': [],
                  'C→S transition': [], 'consolidation': []}

        tw = self._transition_width
        e_end = self._explore_end
        c_end = self._converge_end

        for entry in self._telemetry_log:
            frac = entry['frac']
            if frac < e_end - tw:
                phases['exploration'].append(entry)
            elif frac < e_end + tw:
                phases['E→C transition'].append(entry)
            elif frac < c_end - tw:
                phases['convergence'].append(entry)
            elif frac < c_end + tw:
                phases['C→S transition'].append(entry)
            else:
                phases['consolidation'].append(entry)

        for phase_name, entries in phases.items():
            if not entries:
                continue

            velocities = [e['velocity'] for e in entries if e['velocity'] is not None]
            prompts = [e['prompt_influence'] for e in entries if e['prompt_influence'] is not None]
            suscepts = [e['susceptibility'] for e in entries if e['susceptibility'] is not None]

            n = len(entries)
            frac_range = f"{entries[0]['frac']:.3f}–{entries[-1]['frac']:.3f}"

            print(f"\n  {phase_name} ({n} steps, frac {frac_range})")
            print(f"  {'─' * 70}")

            if velocities:
                v_mean = sum(velocities) / len(velocities)
                v_min = min(velocities)
                v_max = max(velocities)
                # Delta: difference between first and last velocity in this phase
                v_delta = velocities[-1] - velocities[0] if len(velocities) > 1 else 0.0
                print(f"    velocity:       mean={v_mean:.4f}  "
                      f"min={v_min:.4f}  max={v_max:.4f}  "
                      f"Δ(last-first)={v_delta:+.4f}")

            if prompts:
                p_mean = sum(prompts) / len(prompts)
                p_min = min(prompts)
                p_max = max(prompts)
                p_delta = prompts[-1] - prompts[0] if len(prompts) > 1 else 0.0
                print(f"    prompt_influence: mean={p_mean:.4f}  "
                      f"min={p_min:.4f}  max={p_max:.4f}  "
                      f"Δ(last-first)={p_delta:+.4f}")

            if suscepts:
                s_mean = sum(suscepts) / len(suscepts)
                s_min = min(suscepts)
                s_max = max(suscepts)
                s_delta = suscepts[-1] - suscepts[0] if len(suscepts) > 1 else 0.0
                print(f"    susceptibility:  mean={s_mean:.4f}  "
                      f"min={s_min:.4f}  max={s_max:.4f}  "
                      f"Δ(last-first)={s_delta:+.4f}")
            elif phase_name in ('exploration', 'E→C transition'):
                print(f"    susceptibility:  (EMAG inactive — no perturbation signal)")

        # Highlight the transition specifically
        transition_entries = phases.get('E→C transition', [])
        if transition_entries:
            suscepts = [e['susceptibility'] for e in transition_entries if e['susceptibility'] is not None]
            velocities = [e['velocity'] for e in transition_entries if e['velocity'] is not None]
            if suscepts:
                print(f"\n  ► E→C TRANSITION DETAIL (the critical boundary)")
                print(f"  {'─' * 70}")
                for e in transition_entries:
                    v_str = f"{e['velocity']:.4f}" if e['velocity'] is not None else "  --- "
                    s_str = f"{e['susceptibility']:.4f}" if e['susceptibility'] is not None else "  --- "
                    p_str = f"{e['prompt_influence']:.4f}" if e['prompt_influence'] is not None else "  --- "
                    print(f"    step={e['step']:3d}  frac={e['frac']:.3f}  "
                          f"vel={v_str}  suscept={s_str}  prompt={p_str}")

        # Overall trajectory summary
        all_suscepts = [e['susceptibility'] for e in self._telemetry_log
                        if e['susceptibility'] is not None]
        if len(all_suscepts) >= 4:
            first_quarter = all_suscepts[:len(all_suscepts)//4]
            last_quarter = all_suscepts[-(len(all_suscepts)//4):]
            early_mean = sum(first_quarter) / len(first_quarter)
            late_mean = sum(last_quarter) / len(last_quarter)
            print(f"\n  ► SUSCEPTIBILITY TRAJECTORY")
            print(f"  {'─' * 70}")
            print(f"    early mean (first 25%): {early_mean:.4f}")
            print(f"    late mean  (last  25%): {late_mean:.4f}")
            print(f"    Δ: {late_mean - early_mean:+.4f}  "
                  f"({'decreasing — commitment signal' if late_mean > early_mean else 'increasing — unexpected, investigate'})")
            # Note: higher cosine similarity = standard and perturbed are MORE similar
            # = perturbation has LESS effect = system is LESS susceptible = MORE committed
            # So increasing similarity over time = decreasing susceptibility = commitment

        print("\n" + "=" * 90)
        print(f"[CoS-TELEM] {len(self._telemetry_log)} steps recorded. "
              f"Susceptibility is cos_sim(cond_standard, cond_perturbed):")
        print(f"            Higher = more committed (perturbation has less effect)")
        print(f"            Lower  = more exploratory (perturbation shifts output)")
        print("=" * 90 + "\n")

    def set_ecc_config(self, enabled=False, adaptive_boundaries=False,
                       nu=1.0, lambda_cost=0.1, f0=2.0,
                       exploration_noise=0.015, convergence_stabilization=0.005,
                       consolidation_damping=0.002):
        self._ecc_enabled = enabled
        if enabled:
            self._ecc = ECCController(
                explore_end=self._explore_end,
                converge_end=self._converge_end,
                transition_width=self._transition_width,
                exploration_noise=exploration_noise,
                convergence_stabilization=convergence_stabilization,
                consolidation_damping=consolidation_damping,
                nu=nu,
                lambda_cost=lambda_cost,
                f0=f0,
                adaptive_boundaries=adaptive_boundaries,
                enabled=True,
            )
        else:
            self._ecc = None

    def set_hook_mode(self, mode="auto"):
        """Configure measurement hook type.
        
        Args:
            mode: 'auto' = use model registry (block for MMDiT, attn for others)
                  'attn' = force attention-module hooks
                  'block' = force block-level measurement hooks
        """
        self._hook_mode = mode if mode in ('auto', 'attn', 'block') else 'auto'

    def set_csv_config(self, session_name="", prompt_text=""):
        """Configure cumulative CSV telemetry logging.
        
        Args:
            session_name: If non-empty, writes to dm_telemetry/{session_name}.csv
                         and auto-increments gen_idx across queues.
                         If empty, no CSV logging.
            prompt_text: The prompt text for this generation (metadata).
        """
        if not session_name or not session_name.strip():
            self._csv_session = None
            return
            
        self._csv_session = session_name.strip()
        self._csv_prompt_text = prompt_text.strip() if prompt_text else ""
        self._csv_prompt_hash = int(hashlib.md5(
            self._csv_prompt_text.encode()).hexdigest()[:8], 16) if self._csv_prompt_text else 0
        
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        self._csv_path = TELEMETRY_DIR / f"{self._csv_session}.csv"
        
        # Determine gen_idx: read max from existing file, or start at 1
        if self._csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self._csv_path, usecols=['gen_idx'])
                self._csv_gen_idx = int(df['gen_idx'].max()) + 1
            except Exception:
                # Fallback: count lines
                try:
                    with open(self._csv_path, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        if last_line:
                            self._csv_gen_idx = int(last_line.split(',')[0]) + 1
                        else:
                            self._csv_gen_idx = 1
                    else:
                        self._csv_gen_idx = 1
                except Exception:
                    self._csv_gen_idx = 1
        else:
            self._csv_gen_idx = 1
        
        self._csv_initialized = False
        if self._log_steps:
            print(f"[CoS-CSV] Session '{self._csv_session}' → {self._csv_path}")
            print(f"[CoS-CSV] Starting at gen_idx={self._csv_gen_idx}")

    def _ensure_csv_header(self):
        """Write CSV header if file is new."""
        if self._csv_initialized or self._csv_path is None:
            return
        if not self._csv_path.exists():
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'gen_idx', 'token_idx', 'token_type', 'layer', 'submodule',
                    'g_micro', 'phi_micro', 'phi_micro_skip', 'spectral_gap', 'sv_entropy',
                    'sigma', 'seq_len', 'hidden_dim',
                    'hook_mode',
                    'prompt_hash', 'prompt_len', 'gen_len', 'prompt_text'
                ])
        self._csv_initialized = True

    def _write_csv_step(self, step, sigma, per_layer_data):
        """Write per-layer CSV rows for one denoising step.
        
        Args:
            step: Current denoising step index
            sigma: Current noise level
            per_layer_data: dict of {layer_idx: {'g': float, 'gap': float, 'ent': float, 
                                                  'seq_len': int, 'hidden_dim': int}}
        """
        if self._csv_path is None or not per_layer_data:
            return
        
        self._ensure_csv_header()
        
        total_steps = self.total_steps or 0
        prompt_len = len(self._csv_prompt_text.split()) if self._csv_prompt_text else 0
        
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            sorted_keys = sorted(per_layer_data.keys())
            first_key = sorted_keys[0] if sorted_keys else None
            for key in sorted_keys:
                d = per_layer_data[key]

                # Support both tuple (dual-hook) and integer (legacy) keys
                if isinstance(key, tuple):
                    layer_idx, sub_pos = key
                    submodule = 'attn' if sub_pos == 0 else 'mlp'
                else:
                    layer_idx = key
                    submodule = 'attn'

                pt = self._csv_prompt_text if (step == 0 and key == first_key) else ""
                writer.writerow([
                    self._csv_gen_idx,
                    step,
                    'step',  # DMs don't have prefill/generated distinction
                    layer_idx,
                    submodule,
                    f"{d['g']:.6f}",
                    f"{d.get('phi', 0.0):.6f}",
                    f"{d.get('phi_skip', 0.0):.6f}",
                    f"{d['gap']:.4f}" if d['gap'] is not None else "",
                    f"{d['ent']:.4f}" if d['ent'] is not None else "",
                    f"{sigma:.6f}",
                    d.get('seq_len', 0),
                    d.get('hidden_dim', 0),
                    self._effective_hook_mode or self._hook_mode,
                    self._csv_prompt_hash,
                    prompt_len,
                    total_steps,
                    pt,
                ])

    # -----------------------------------------------------------------
    # Core: predict_noise
    # -----------------------------------------------------------------
    
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """
        Phase-adaptive CFG + EMAG guidance with optional transition telemetry.
        
        When EMAG is active (3 NFEs per step):
            1. Register EMA hooks
            2. Run perturbed [uncond, cond] — hooks intercept attention (2 evals)
            3. Remove hooks  
            4. Run standard [cond] — clean attention (1 eval)
            5. Eq. 15: emag_guidance = cond_std + emag_scale * (cond_std - cond_pert)
            6. Eq. 16: noise = uncond + cfg * (emag_guidance - uncond)
        
        When EMAG is inactive (2 NFEs per step):
            Standard CFG with phase-adaptive cfg value.
            (When telemetry enabled: manual 2-pass for signal access, same NFEs.)
        """
        # --- Generation detection & state management ---
        is_new_gen = self._detect_new_generation(timestep)
        if is_new_gen:
            # Print previous generation's telemetry summary before reset
            if self._telemetry_enabled and self._telemetry_log:
                self._print_telemetry_summary()
                # ECC v2 summary (prints c-theorem verification)
                if self._ecc is not None and self._ecc._telemetry_log:
                    self._ecc.print_summary()
            self._get_persistent_ema().clear()
            self._set_persistent_step(0)
            self._call_count = 0
            self._prev_noise_pred = None
            self._telemetry_log = []
            # Reset ECC for new generation
            if self._ecc is not None:
                # Print previous generation's ECC summary before reset
                if self._ecc._telemetry_log:
                    self._ecc.print_summary()
                self._ecc.reset()
        
        ema_attention = self._get_persistent_ema()
        current_step = self._get_persistent_step()
        
        # Try to detect total_steps from model_options if not set by sigmas
        if self.total_steps is None and 'sigmas' in model_options:
            self.total_steps = len(model_options['sigmas']) - 1
        
        # --- Compute phase-adaptive values using STEP-BASED frac ---
        # This is immune to gate softening — frac comes from step counter,
        # not from the potentially-modified timestep the guider receives.
        phase_cfg = self._get_phase_cfg(timestep, step=current_step)
        phase_emag_scale = self._get_phase_emag_scale(timestep, step=current_step)
        frac = self._get_frac(timestep, step=current_step)
        
        # Real sigma for display (from schedule if available, else from timestep)
        if self._sigmas_schedule is not None and current_step < len(self._sigmas_schedule):
            real_sigma = float(self._sigmas_schedule[current_step])
        else:
            real_sigma = timestep.mean().item() if isinstance(timestep, torch.Tensor) else float(timestep)
        
        received_sigma = timestep.mean().item() if isinstance(timestep, torch.Tensor) else float(timestep)
        
        phase = ('exploration' if frac < self._explore_end 
                 else 'convergence' if frac < self._converge_end 
                 else 'consolidation')
        
        # --- Determine if EMAG should fire ---
        emag_active = phase_emag_scale > 1e-4
        
        # Hard gate from start/end percent
        if emag_active and self.total_steps is not None and self.total_steps > 0:
            start_step = int(self._emag_start_pct * self.total_steps)
            end_step = int(self._emag_end_pct * self.total_steps)
            if not (end_step <= current_step <= start_step):
                emag_active = False
        
        # --- Logging (original, non-telemetry) ---
        self._call_count += 1
        if self._log_steps and not self._telemetry_enabled:
            if self._call_count <= 3 or self._call_count % 25 == 0:
                print(f"[CoS-EMAG] step {current_step}/{self.total_steps}: "
                      f"real_σ={real_sigma:.4f}, recv_σ={received_sigma:.4f}, "
                      f"frac={frac:.3f}, phase={phase}, cfg={phase_cfg:.2f}, "
                      f"emag={phase_emag_scale:.3f}, active={emag_active}")
        
        # --- Execute guidance ---
        if not emag_active:
            if self._telemetry_enabled:
                # Instrumented 2-pass: compute uncond and cond separately for
                # telemetry access. Same 2 NFEs as super().predict_noise().
                negative_cond = self.conds.get("negative", None)
                positive_cond = self.conds.get("positive", None)
                
                out = comfy.samplers.calc_cond_batch(
                    self.inner_model,
                    [negative_cond, positive_cond],
                    x, timestep, model_options
                )
                uncond_pred = out[0]
                cond_pred = out[1]
                noise_pred = uncond_pred + phase_cfg * (cond_pred - uncond_pred)
                
                self._record_telemetry(
                    step=current_step, frac=frac, phase=phase,
                    sigma_val=real_sigma, phase_cfg=phase_cfg,
                    phase_emag_scale=phase_emag_scale, emag_active=False,
                    noise_pred=noise_pred, uncond_pred=uncond_pred,
                    cond_pred_standard=cond_pred, cond_pred_perturbed=None,
                )

                # --- ECC v2 (non-EMAG path) ---
                if self._ecc_enabled and self._ecc is not None:
                    cos_sim_val = None
                    if self._telemetry_log:
                        cos_sim_val = self._telemetry_log[-1].get('velocity', None)

                    ecc_output = self._ecc.step(
                        step=current_step,
                        total_steps=self.total_steps or 50,
                        sigma=real_sigma,
                        susceptibility=None,  # No EMAG, no susceptibility
                        cos_sim=cos_sim_val,
                    )

                    # Optional: let ECC modulate EMAG scale
                    # phi_magnitude from the Pontryagin costate acts as a multiplier !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # on the perturbation strength. Larger Φ near criticality means
                    # more perturbation to maintain exploration; smaller Φ away from
                    # criticality means less perturbation as the system commits.
                    # if ecc_output is not None and ecc_output['phi_magnitude'] > 0: 
                    #     # Scale EMAG by normalized Φ magnitude
                    #     # This is gentle — caps the modulation at ±30% of current scale
                    #     phi_norm = min(ecc_output['phi_magnitude'] / 0.02, 1.0)  # normalize
                    #     ecc_emag_mult = 0.7 + 0.6 * phi_norm  # range [0.7, 1.3]
                    #     phase_emag_scale *= ecc_emag_mult

                    #     self._set_persistent_step(current_step + 1)
                    #     return noise_pred
                    # else:
                    #     # Original 2-pass path (zero overhead when telemetry off)
                    #     self.cfg = phase_cfg
                    #     result = super().predict_noise(x, timestep, model_options, seed)
                    #     self._set_persistent_step(current_step + 1)
                    #     return result

                # CRITICAL: must return here to skip the EMAG 3-pass below.
                # Without this return, the function falls through to the EMAG path
                # and runs a perturbed forward pass even when emag_active=False,
                # which is what was happening before 2026-05-04.
                self._set_persistent_step(current_step + 1)
                return noise_pred
            else:
                # Telemetry disabled, emag inactive: use vanilla path
                self.cfg = phase_cfg
                result = super().predict_noise(x, timestep, model_options, seed)
                self._set_persistent_step(current_step + 1)
                return result
        
        # --- EMAG 3-pass guidance ---
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        
        _prev_g = getattr(self, '_last_g_micro', None)

        # Pass 1+2: Perturbed predictions (hooks active)
        self._micro_g_buffer = []
        self._micro_g_per_layer = {}
        self._micro_mean_vectors = {}  # phi_micro buffer
        
        # Pass 1+2: Perturbed predictions (hooks active)
        self._register_emag_hooks(ema_attention)
        try:
            out_perturbed = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond, positive_cond],
                x, timestep, model_options
            )
            uncond_pred = out_perturbed[0]
            cond_pred_perturbed = out_perturbed[1]
        finally:
            self._remove_emag_hooks()
            
        # --- NEW: Harvest the Microscopic Coupling ---
        current_g_micro = None
        if getattr(self, '_micro_g_buffer', []):
            current_g_micro = sum(self._micro_g_buffer) / len(self._micro_g_buffer)
        self._micro_g_buffer = [] # Reset for safety
        current_spectral_gap = None
        current_sv_entropy = None
        if getattr(self, '_micro_gap_buffer', []):
            current_spectral_gap = sum(self._micro_gap_buffer) / len(self._micro_gap_buffer)
            current_sv_entropy = sum(self._micro_entropy_buffer) / len(self._micro_entropy_buffer)
        self._micro_gap_buffer = []
        self._micro_entropy_buffer = []

        current_beta_g = None
        if current_g_micro is not None and _prev_g is not None:
            current_beta_g = current_g_micro - _prev_g  # dg/dt with dt=1 step
        if current_g_micro is not None:
            self._last_g_micro = current_g_micro

        if self._log_steps and getattr(self, '_micro_g_per_layer', {}):
            layer_strs = []
            for lid in sorted(self._micro_g_per_layer.keys()):
                vals = self._micro_g_per_layer[lid]
                avg = sum(vals) / len(vals)
                # Format tuple keys as "L0a" (attn) / "L0m" (mlp); int keys as "L0"
                if isinstance(lid, tuple):
                    block_idx, sub_pos = lid
                    suffix = 'a' if sub_pos == 0 else 'm'
                    layer_strs.append(f"L{block_idx}{suffix}={avg:.4f}")
                else:
                    layer_strs.append(f"L{lid}={avg:.4f}")
            print(f"[CoS-O2-layers] {' '.join(layer_strs)}")
        self._micro_g_per_layer_snapshot = dict(self._micro_g_per_layer) if getattr(self, '_micro_g_per_layer', None) else {}
        self._micro_g_per_layer = {}
        
        # --- Compute phi_micro (stride-1) and phi_micro_skip (stride-2) from stored mean vectors ---
        # Keys may be ints (legacy single-hook) or tuples (lid, sub_pos) (dual-hook).
        # Sorted iteration gives the natural alternation under dual-hook:
        #   (0,0) → (0,1) → (1,0) → (1,1) → ... so stride-1 crosses sub_pos
        #   (cross-type / inter-sublattice) and stride-2 stays within sub_pos
        #   (same-type / intra-sublattice).
        phi_micro_per_layer = {}
        phi_skip_per_layer = {}
        _mvecs = getattr(self, '_micro_mean_vectors', {})
        if len(_mvecs) >= 3:
            _sorted_keys = sorted(_mvecs.keys())
            for _i in range(len(_sorted_keys)):
                _key = _sorted_keys[_i]
                phi_val = 0.0
                phi_skip_val = 0.0

                # R3: UNet levels change width; a cosine across a width change is undefined (nan)
                if _i >= 2 and not _same_width(_mvecs, _sorted_keys[_i - 2:_i + 1]):
                    phi_val = float('nan')
                # phi_micro: stride-1, needs i >= 2
                elif _i >= 2:
                    _prev_prev = _sorted_keys[_i - 2]
                    _prev = _sorted_keys[_i - 1]
                    _delta_prev = _mvecs[_prev] - _mvecs[_prev_prev]
                    _delta_curr = _mvecs[_key] - _mvecs[_prev]
                    _np = _delta_prev.norm()
                    _nc = _delta_curr.norm()
                    if _np > 1e-8 and _nc > 1e-8:
                        phi_val = (_delta_prev @ _delta_curr / (_np * _nc)).item()

                # phi_micro_skip: stride-2 (delta(i-2) vs delta(i)), needs i >= 3
                if _i >= 3 and not _same_width(_mvecs, _sorted_keys[_i - 3:_i + 1]):
                    phi_skip_val = float('nan')
                elif _i >= 3:
                    _two_back_prev = _sorted_keys[_i - 3]
                    _two_back_curr = _sorted_keys[_i - 2]
                    _delta_two_back = _mvecs[_two_back_curr] - _mvecs[_two_back_prev]
                    _prev = _sorted_keys[_i - 1]
                    _delta_curr = _mvecs[_key] - _mvecs[_prev]
                    _n2 = _delta_two_back.norm()
                    _nc = _delta_curr.norm()
                    if _n2 > 1e-8 and _nc > 1e-8:
                        phi_skip_val = (_delta_two_back @ _delta_curr / (_n2 * _nc)).item()

                phi_micro_per_layer[_key] = phi_val
                phi_skip_per_layer[_key] = phi_skip_val
        self._micro_mean_vectors = {}  # Reset for next step
        
        # --- CSV per-layer output ---
        if self._csv_session and getattr(self, '_micro_g_per_layer_snapshot', None):
            per_layer_data = {}
            for key in self._micro_g_per_layer_snapshot:
                g_vals = self._micro_g_per_layer_snapshot[key]
                gap_vals = getattr(self, '_micro_gap_per_layer', {}).get(key, [])
                ent_vals = getattr(self, '_micro_ent_per_layer', {}).get(key, [])
                per_layer_data[key] = {
                    'g': sum(g_vals) / len(g_vals) if g_vals else 0.0,
                    'phi': phi_micro_per_layer.get(key, 0.0),
                    'phi_skip': phi_skip_per_layer.get(key, 0.0),
                    'gap': sum(gap_vals) / len(gap_vals) if gap_vals else None,
                    'ent': sum(ent_vals) / len(ent_vals) if ent_vals else None,
                    'seq_len': getattr(self, '_micro_seqlen_per_layer', {}).get(key, 0),
                    'hidden_dim': getattr(self, '_micro_hdim_per_layer', {}).get(key, 0),
                }
            self._write_csv_step(current_step, real_sigma, per_layer_data)
        
        # Clean up per-layer buffers
        self._micro_gap_per_layer = {}
        self._micro_ent_per_layer = {}
        self._micro_seqlen_per_layer = {}
        self._micro_hdim_per_layer = {}

        if self._log_steps and current_spectral_gap is not None:
            beta_str = f"{current_beta_g:.6f}" if current_beta_g is not None else "---"
            print(f"[CoS-O2] g={current_g_micro:.4f} β(g)={beta_str} gap={current_spectral_gap:.2f} ent={current_sv_entropy:.3f}")
        
        # Pass 3: Standard conditional prediction (no hooks)
        out_standard = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],
            x, timestep, model_options
        )
        cond_pred_standard = out_standard[0]
        
        
        # Eq. 15: EMAG enhancement
        # ε̃(x_t,c)' = ε'_θ(x_t,c) + w_e · (ε_θ(x_t,c) - ε'_θ(x_t,c))
        emag_guidance = (cond_pred_standard + 
                         phase_emag_scale * (cond_pred_standard - cond_pred_perturbed))
        
        # Eq. 16: CFG with EMAG-enhanced conditioning
        # ε̃(x_t,c) = ε_θ(x_t,∅) + w · (ε̃(x_t,c)' - ε_θ(x_t,∅))
        noise_pred = uncond_pred + phase_cfg * (emag_guidance - uncond_pred)
        
        # --- Telemetry (zero additional NFEs — all quantities already computed) ---
        if self._telemetry_enabled:
            self._record_telemetry(
                step=current_step, frac=frac, phase=phase,
                sigma_val=real_sigma, phase_cfg=phase_cfg,
                phase_emag_scale=phase_emag_scale, emag_active=True,
                noise_pred=noise_pred, uncond_pred=uncond_pred,
                cond_pred_standard=cond_pred_standard,
                cond_pred_perturbed=cond_pred_perturbed,
                g_micro=current_g_micro,
                phi_micro=sum(v for v in phi_micro_per_layer.values() if v != 0.0) / max(len([v for v in phi_micro_per_layer.values() if v != 0.0]), 1) if phi_micro_per_layer else None,
            )

            # --- ECC v2 closed-loop controller step ---
            ecc_output = None
            if self._ecc_enabled and self._ecc is not None:
                # Extract susceptibility from telemetry we just recorded
                suscept_val = None
                cos_sim_val = None
                if self._telemetry_log:
                    latest = self._telemetry_log[-1]
                    suscept_val = latest.get('susceptibility', None)
                    cos_sim_val = latest.get('velocity', None)

                ecc_output = self._ecc.step(
                    step=current_step,
                    total_steps=self.total_steps or 50,
                    sigma=real_sigma,
                    susceptibility=suscept_val,
                    cos_sim=cos_sim_val,
                )

                # Log ECC output alongside existing telemetry
                if self._log_steps and current_step <= 3 or (current_step % 10 == 0):
                    print(f"[ECC v2] step={current_step:3d} "
                          f"β={ecc_output['beta']:.6f}({ecc_output['beta_source'][:5]}) "
                          f"F={ecc_output['f_beta']:.3f} "
                          f"Φ={ecc_output['phi_magnitude']:.6f} "
                          f"J={ecc_output['j_avg']:.4f} "
                          f"bounds=[{ecc_output['adaptive_explore_end']:.3f},"
                          f"{ecc_output['adaptive_converge_end']:.3f}]")
        
        self._set_persistent_step(current_step + 1)
        return noise_pred


# =============================================================================
# Node Definition
# =============================================================================

class CoSPAFS_EMAGGuider:
    """
    CoS-PAFS EMAG Guider (Diagnostic Build) — Phase-Adaptive CFG + EMA 
    Attention Perturbation with Transition Telemetry.
    
    Combines CoS-PAFS's phase-scheduled guidance with EMAG's attention
    perturbation for improved prompt adherence and structural quality.
    
    Both CFG and EMAG scale are phase-adaptive:
    - Exploration: reduced CFG + boosted EMAG (hypothesis diversity)
    - Convergence: peak CFG + standard EMAG (structural commitment)
    - Consolidation: low CFG + EMAG off (texture refinement, saves NFEs)
    
    DIAGNOSTIC MODE (log_telemetry=True):
    Measures three signals per step at zero additional NFE cost:
    - Velocity: step-to-step output change rate
    - Prompt Influence: how strongly the prompt steers output
    - Susceptibility: how much EMAG perturbation affects output
      (criticality-analog signal — drops as system commits)
    
    Prints per-phase summary and transition detail at end of generation.
    
    Drop-in replacement for CoS-PAFS CFG Guider or standalone EMAG Guider.
    Connects to SamplerCustomAdvanced's GUIDER input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "base_cfg": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Base CFG scale. Phase multipliers scale on top. "
                               "SD1.5 w/ CoS-PAFS: ~30. SD3.5: ~5.5. LTX-2: ~1.0.",
                }),
                "emag_scale": ("FLOAT", {
                    "default": 1.75, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Base EMAG perturbation scale (w_e). Phase multipliers "
                               "scale on top. Paper recommends 1.75 for conditional.",
                }),
            },
            "optional": {
                # --- Sigma range ---
                "sigmas": ("SIGMAS", {
                    "tooltip": "Connect SIGMAS to auto-detect sigma range. "
                               "Overrides sigma_max/sigma_min.",
                }),
                # --- Phase-adaptive CFG ---
                "exploration_cfg_mult": ("FLOAT", {
                    "default": 1.52, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "CFG multiplier during exploration. CoS-PAFS v5 universal: 1.52.",
                }),
                "convergence_cfg_mult": ("FLOAT", {
                    "default": 1.27, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "CFG multiplier during convergence. CoS-PAFS v5 universal: 1.27.",
                }),
                "consolidation_cfg_mult": ("FLOAT", {
                    "default": 0.51, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "CFG multiplier during consolidation. CoS-PAFS v5 universal: 0.51.",
                }),
                # --- Phase-adaptive EMAG ---
                "emag_explore_mult": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "EMAG scale multiplier during exploration. Boosted: EMA "
                               "perturbation diversifies hypotheses. Higher = more diversity.",
                }),
                "emag_converge_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "EMAG scale multiplier during convergence. Standard: "
                               "perturbation accelerates structural commitment.",
                }),
                "emag_consolidate_mult": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "EMAG scale multiplier during consolidation. 0.0 = off "
                               "(saves 1 NFE/step, signal is negligible anyway).",
                }),
                # --- EMAG core params ---
                "ema_decay": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 0.999, "step": 0.001,
                    "tooltip": "EMA decay factor (λ in Eq. 12). Higher = more temporal smoothing.",
                }),
                "adaptive_layers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Adaptive layer selection (EMAG paper §4.3). "
                               "Perturbs mid-to-deep layers only.",
                }),
                "layer_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override which layers to hook. Comma-separated indices "
                               "or ranges: '0,1,2,3' or '8-11' or '20-27'. "
                               "Empty = use adaptive selection. For Gap B sweep.",
                }),
                "perturb_img_to_text": ("BOOLEAN", {
                    "default": True,
                    "label_on": "EMAG (Full)",
                    "label_off": "EMAG-I (Img Only)",
                    "tooltip": "Full EMAG perturbs self+cross attention. "
                               "EMAG-I perturbs self-attention only.",
                }),
                # --- Phase boundaries ---
                "exploration_end": ("FLOAT", {
                    "default": 0.46, "min": 0.05, "max": 0.60, "step": 0.01,
                    "tooltip": "Schedule fraction where exploration ends. v5 universal: 0.46.",
                }),
                "convergence_end": ("FLOAT", {
                    "default": 0.85, "min": 0.50, "max": 0.95, "step": 0.01,
                    "tooltip": "Schedule fraction where convergence ends. v5 universal: 0.85.",
                }),
                "transition_width": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.20, "step": 0.01,
                    "tooltip": "Hermite smoothstep blend width at phase boundaries.",
                }),
                # --- Sigma bounds (manual override) ---
                "sigma_max": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01,
                    "tooltip": "Max sigma (overridden if SIGMAS connected). "
                               "1.0 for flow models, ~8.0 for SD1.5 w/ CoS-PAFS.",
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                }),
                # --- EMAG hard gate (safety valve) ---
                "emag_start_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Hard gate: EMAG on from this percent (1.0 = beginning). "
                               "Phase multipliers handle scheduling; this is a safety override.",
                }),
                "emag_end_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Hard gate: EMAG off after this percent (0.0 = never). "
                               "Phase multipliers handle scheduling; this is a safety override.",
                }),
                # --- Diagnostics ---
                "log_phases": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print phase assignments, CFG and EMAG values to console.",
                }),
                "log_telemetry": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "DIAGNOSTIC: Measure velocity, prompt influence, and "
                               "susceptibility (criticality-analog signal) at each step. "
                               "Prints per-step signals and per-phase summary. "
                               "Zero additional NFEs. No impact on output.",
                }),
                "csv_session": ("STRING", {
                    "default": "",
                    "tooltip": "Session name for cumulative CSV logging. "
                               "If set, writes per-layer g_micro, spectral_gap, "
                               "sv_entropy to dm_telemetry/{name}.csv. "
                               "gen_idx auto-increments across queues. "
                               "Leave empty to disable CSV logging.",
                }),
                "hook_mode": (["auto", "attn", "block"], {
                    "default": "auto",
                    "tooltip": "Which module to hook for g_micro measurement. "
                               "'auto': use model registry (block for MMDiT, attn for others). "
                               "'attn': force attention-module hooks (standard, higher signal specificity). "
                               "'block': force block-level hooks (measurement-only, includes FFN+residual). "
                               "IMPORTANT: record this in your experiment log — hook type affects g_micro values. "
                               "Use the same hook_mode for all models in a comparison group.",
                }),
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The prompt being used (for CSV metadata). "
                               "Connect your prompt text here or type it.",
                }),
                # --- ECC v2 Closed-Loop Controller ---
                "enable_ecc": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable ECC v2 closed-loop controller. Feeds susceptibility "
                               "back into RG-derived phase controller. Computes F(β), J[Φ], "
                               "and c-theorem verification data. Zero impact on output when "
                               "disabled. When enabled, prints additional summary at end.",
                }),
                "ecc_adaptive_boundaries": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ECC: Detect phase boundaries from susceptibility gradient "
                               "instead of using fixed fractions. Experimental.",
                }),
                "ecc_nu": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "ECC: Critical exponent ν for F(β) = F₀ - ν·ln|β|. "
                               "Controls how steeply feature richness peaks at criticality.",
                }),
                "ecc_lambda": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "ECC: Control cost penalty λ in J[Φ]. Higher = less Φ. "
                               "Prevents controller from over-perturbing.",
                }),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = ("CoS-PAFS EMAG Guider (Diagnostic): Phase-adaptive CFG + EMA attention "
                   "perturbation with transition telemetry. Measures susceptibility "
                   "(criticality-analog signal) at zero additional NFE cost. "
                   "Enable log_telemetry for per-step signals and phase boundary analysis.")

    def get_guider(self, model, positive, negative, base_cfg=3.0, emag_scale=1.75,
                    sigmas=None,
                    exploration_cfg_mult=1.52, convergence_cfg_mult=1.27,
                    consolidation_cfg_mult=0.51,
                    emag_explore_mult=1.5, emag_converge_mult=1.0,
                    emag_consolidate_mult=0.0,
                    ema_decay=0.9, adaptive_layers=True, layer_override="",
                    perturb_img_to_text=True,
                    exploration_end=0.46, convergence_end=0.85,
                    transition_width=0.05,
                    sigma_max=1.0, sigma_min=0.0,
                    emag_start_percent=1.0, emag_end_percent=0.0,
                    log_phases=False, log_telemetry=False,
                    enable_ecc=False, ecc_adaptive_boundaries=False,
                    ecc_nu=1.0, ecc_lambda=0.1,
                    csv_session="", prompt_text="",
                    hook_mode="auto"):

        # Auto-detect sigma range from connected sigmas
        if sigmas is not None and len(sigmas) > 1:
            sigma_max = float(sigmas[0])
            sigma_min = float(sigmas[-1])
            if log_phases or log_telemetry:
                print(f"[CoS-EMAG Guider] Auto-detected sigma range: "
                      f"{sigma_max:.4f} → {sigma_min:.4f} ({len(sigmas)-1} steps)")

        guider = CoSPAFS_EMAGGuiderImpl(model)
        guider.set_conds(positive=positive, negative=negative)
        guider.set_cfg(base_cfg)  # fallback
        
        # Store real sigmas schedule FIRST — this sets total_steps and
        # enables step-based frac computation immune to gate softening
        if sigmas is not None:
            guider.set_sigmas_schedule(sigmas)
        
        guider.set_phase_config(
            base_cfg=base_cfg,
            explore_mult=exploration_cfg_mult,
            converge_mult=convergence_cfg_mult,
            consolidate_mult=consolidation_cfg_mult,
            explore_end=exploration_end,
            converge_end=convergence_end,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            transition_width=transition_width,
            log_steps=log_phases or log_telemetry,
        )
        
        guider.set_emag_config(
            emag_scale=emag_scale,
            ema_decay=ema_decay,
            emag_explore_mult=emag_explore_mult,
            emag_converge_mult=emag_converge_mult,
            emag_consolidate_mult=emag_consolidate_mult,
            adaptive_layers=adaptive_layers,
            perturb_img_to_text=perturb_img_to_text,
            start_percent=emag_start_percent,
            end_percent=emag_end_percent,
            layer_override=layer_override,
        )

        guider.set_telemetry(log_telemetry)

        guider.set_ecc_config(
            enabled=enable_ecc,
            adaptive_boundaries=ecc_adaptive_boundaries,
            nu=ecc_nu,
            lambda_cost=ecc_lambda,
            exploration_noise=0.015,
            convergence_stabilization=0.005,
            consolidation_damping=0.002,
        )

        guider.set_csv_config(session_name=csv_session, prompt_text=prompt_text)
        guider.set_hook_mode(hook_mode)

        return (guider,)



# =============================================================================
# Node Registration
# =============================================================================

# =============================================================================
# Passive Telemetry Guider — pure measurement, no perturbation
# =============================================================================

def _parse_layer_filter(spec, n_blocks):
    """Parse a layer-filter spec string into a set of layer indices.
    
    spec: '' or 'all' → returns None (meaning 'no filter, capture all')
          '0-15'      → layers 0..15 inclusive
          '0,5,10'    → those specific indices
          '0-5,10-15,20' → mixed
    
    Out-of-range indices are silently dropped.
    """
    spec = (spec or "").strip().lower()
    if not spec or spec == "all":
        return None
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    lo, hi = hi, lo
                for i in range(lo, hi + 1):
                    if 0 <= i < n_blocks:
                        indices.add(i)
            except ValueError:
                continue
        else:
            try:
                i = int(part)
                if 0 <= i < n_blocks:
                    indices.add(i)
            except ValueError:
                continue
    if not indices:
        print(f"[CoS-Passive] WARNING: layer_range '{spec}' matches no layer in "
              f"[0, {n_blocks}); capturing ALL layers instead. Check the node's "
              f"widget values.")
        return None
    return indices


class CoSPAFS_PassiveTelemetryGuiderImpl(comfy.samplers.CFGGuider):
    """Pure passive telemetry guider — no perturbation, just measurement.
    
    Standard 2-pass CFG forward (uncond + cond) via super().predict_noise(),
    with passive forward hooks on each block's attention output and (when
    architecture supports it) MLP output. Captures g_micro / phi_micro /
    phi_micro_skip / spectral_gap / sv_entropy per (layer, submodule) per step.
    
    No EMA substitution, no phase scheduling, no extra forward passes. Output
    image is identical to a standard CFGGuider at the same cfg value.
    
    CSV schema matches CoSPAFS_EMAGGuider so the live_analyzer plays nicely.
    """
    
    _active_instance = None  # Class-level: track last instance for hook cleanup

    def __init__(self, model_patcher):
        # Clean up previous instance's hooks BEFORE they become orphaned
        if CoSPAFS_PassiveTelemetryGuiderImpl._active_instance is not None:
            try:
                CoSPAFS_PassiveTelemetryGuiderImpl._active_instance._remove_hooks()
                CoSPAFS_PassiveTelemetryGuiderImpl._active_instance._remove_root_hook()
            except Exception:
                pass
        CoSPAFS_PassiveTelemetryGuiderImpl._active_instance = self
        
        super().__init__(model_patcher)
        self._cfg_value = 1.0
        
        # R2: per-CFG-branch capture. When True, every forward's activation batch is
        # split by ComfyUI's cond_or_uncond list and statistics are computed per branch,
        # so pooled (one batched forward) and separate forwards give identical rows.
        # R3 fix: splitting is ALWAYS on. Round 3 captures came back with
        # hook_mode=passive/branch=all because the flag arrived False from the
        # workflow; the widget is now kept only so saved workflows load.
        self._split_branches = True
        self._split_requested = True   # what the workflow sent (for the warning)
        self._cur_branches = None      # cond_or_uncond from model_function_wrapper
        self._root_branches = None     # cond_or_uncond from transformer_options (2nd channel)
        self._root_hook_handles = []
        self._last_seed = None
        self._gen_steps = -1           # total steps of the running sample (from sample_sigmas)
        # Per-gen branch diagnostics (printed after step 0 of every generation)
        self._diag = {'wrapper_calls': 0, 'root_calls': 0, 'forwards': [], 'unattributed': 0}
        
        # Capture switches
        self._capture_g_micro = True
        self._capture_phi = True
        self._capture_spectral_gap = True
        self._capture_sv_entropy = True
        
        # Layer filter
        self._layer_filter_spec = ""
        
        # Hook state
        self._hook_handles = []
        self._registry_config = None
        self._dual_hook_active = False
        self._hooks_registered_this_gen = False
        
        # Per-step capture buffers (reset each step)
        self._micro_g_per_layer = {}
        self._micro_mean_vectors = {}
        self._micro_gap_per_layer = {}
        self._micro_ent_per_layer = {}
        self._micro_seqlen_per_layer = {}
        self._micro_hdim_per_layer = {}
        
        # Step / generation tracking
        self._current_step = 0
        self._last_timestep_seen = None
        
        # CSV state
        self._csv_session = None
        self._csv_path = None
        self._csv_gen_idx = 0
        self._csv_initialized = False
        self._csv_prompt_text = ""
        self._csv_prompt_hash = 0
        
        # Logging
        self._log_steps = False
        self._shape_printed_attn = False
        self._shape_printed_mlp = False
        self._shape_printed_block = False
        self._error_attn_printed = False
        self._error_mlp_printed = False
        self._error_block_printed = False
        self._first_step_logged = False

    # -----------------------------------------------------------------
    # Cleanup — prevent hook accumulation across generations
    # -----------------------------------------------------------------
    
    def __del__(self):
        try:
            self._remove_hooks()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------
    
    def set_cfg_value(self, cfg):
        self._cfg_value = float(cfg)
        self.set_cfg(float(cfg))

    def set_capture_switches(self, g_micro=True, phi=True,
                              spectral_gap=True, sv_entropy=True):
        self._capture_g_micro = bool(g_micro)
        self._capture_phi = bool(phi)
        self._capture_spectral_gap = bool(spectral_gap)
        self._capture_sv_entropy = bool(sv_entropy)

    def set_branch_split(self, enabled):
        self._split_requested = bool(enabled)
        self._split_branches = True
        if not self._split_requested:
            print("[CoS-Passive] WARNING: split_cfg_branches arrived as False and is IGNORED "
                  "(per-branch capture is always on). If you did not switch it off, this "
                  "node's saved widget values are shifted: delete and re-add the node.")

    def set_layer_filter(self, spec):
        self._layer_filter_spec = spec or ""

    def set_logging(self, enabled):
        self._log_steps = bool(enabled)

    def set_csv_config(self, session_name="", prompt_text=""):
        if not session_name or not session_name.strip():
            self._csv_session = None
            self._csv_path = None
            return
        self._csv_session = session_name.strip()
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        self._csv_path = TELEMETRY_DIR / f"{self._csv_session}.csv"
        self._csv_prompt_text = prompt_text or ""
        # R2: stable across processes (Python's hash() is salted per process)
        self._csv_prompt_hash = int(hashlib.md5(
            (prompt_text or "").encode()).hexdigest()[:8], 16)
        # Look up max gen_idx if file exists, so we append
        self._csv_gen_idx = 0
        if self._csv_path.exists() and self._csv_path.stat().st_size > 0:
            try:
                with open(self._csv_path, 'r') as f:
                    last_line = ""
                    for line in f:
                        if line.strip():
                            last_line = line
                    if last_line and ',' in last_line:
                        try:
                            self._csv_gen_idx = int(last_line.split(',')[0])
                        except ValueError:
                            self._csv_gen_idx = 0
            except Exception:
                self._csv_gen_idx = 0
        if self._log_steps:
            print(f"[CoS-Passive CSV] Session '{self._csv_session}' → {self._csv_path}")
            print(f"[CoS-Passive CSV] Next gen_idx will be {self._csv_gen_idx + 1}")

    # -----------------------------------------------------------------
    # Hook factories — passive only, never modify output
    # -----------------------------------------------------------------
    
    def _make_attn_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0] if (len(output) > 0 and output[0] is not None) else None
                if act is None:
                    return output
            else:
                act = output
            self._capture_one(layer_idx, sub_pos=0, activation=act, label='attn')
            return output  # passive: never modify
        return hook
    
    def _make_mlp_hook(self, layer_idx):
        def hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self._capture_one(layer_idx, sub_pos=1, activation=act, label='mlp')
            return output
        return hook
    
    def _make_block_hook(self, layer_idx):
        """Block-level hook for joint-attention architectures (e.g. MMDiT)."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = None
                for elem in reversed(output):
                    if elem is not None and isinstance(elem, torch.Tensor):
                        act = elem
                        break
                if act is None:
                    return output
            else:
                act = output
            self._capture_one(layer_idx, sub_pos=0, activation=act, label='block')
            return output
        return hook
    
    def _capture_one(self, layer_idx, sub_pos, activation, label):
        """R2: split the activation batch by CFG branch, then capture each chunk.

        ComfyUI concatenates conds along dim 0 in cond_or_uncond order
        (0 = cond/positive, 1 = uncond/negative), each block of size batch_size.
        Keys become (layer, sub_pos, branch) with branch in {'cond','uncond'};
        'pooled' marks a forward whose batch could not be attributed.
        """
        branches = self._cur_branches or self._root_branches
        if not self._split_branches:
            branches = None
        bdim = ((self._registry_config or {}).get('batch_dim') or {}).get(label)
        if bdim and branches and len(branches) > 1 and activation.dim() > bdim:
            # R3: sequence-first layouts (Cosmos-1 GeneralDIT: attn (THW, B, D),
            # MLP (T, H, W, B, D)) carry the CFG batch on another axis.
            activation = activation.movedim(bdim, 0)
        if branches and len(branches) == 1:
            # R3: one branch in this forward owns the whole activation, whatever its
            # layout (MiniMax H3 packs one sample into a 2-D [tokens, D] tensor,
            # which the batch split below would wrongly cut along tokens).
            name = 'cond' if branches[0] == 0 else 'uncond'
            self._capture_core(layer_idx, sub_pos, activation, label,
                               key=(layer_idx, sub_pos, name))
        elif (branches and activation.dim() >= 3
                and activation.shape[0] % len(branches) == 0):
            chunks = activation.chunk(len(branches), dim=0)
            for b, chunk in zip(branches, chunks):
                name = 'cond' if b == 0 else 'uncond'
                self._capture_core(layer_idx, sub_pos, chunk, label,
                                   key=(layer_idx, sub_pos, name))
        elif self._split_branches:
            self._diag['unattributed'] += 1
            self._capture_core(layer_idx, sub_pos, activation, label,
                               key=(layer_idx, sub_pos, 'pooled'))
        else:
            self._capture_core(layer_idx, sub_pos, activation, label)

    def _capture_core(self, layer_idx, sub_pos, activation, label, key=None):
        """Compute and store enabled metrics for one capture point."""
        # Early exit if nothing to capture
        if not (self._capture_g_micro or self._capture_phi
                or self._capture_spectral_gap or self._capture_sv_entropy):
            return
        try:
            # Diagnostic shape print on first capture per submodule type
            if self._log_steps:
                if label == 'attn' and not self._shape_printed_attn:
                    print(f"[CoS-Passive] attn shape={tuple(activation.shape)} layer={layer_idx}")
                    self._shape_printed_attn = True
                elif label == 'mlp' and not self._shape_printed_mlp:
                    print(f"[CoS-Passive] mlp shape={tuple(activation.shape)} layer={layer_idx}")
                    self._shape_printed_mlp = True
                elif label == 'block' and not self._shape_printed_block:
                    print(f"[CoS-Passive] block shape={tuple(activation.shape)} layer={layer_idx}")
                    self._shape_printed_block = True
            
            X = activation.reshape(-1, activation.shape[-1]).float()
            if X.shape[0] < 2:
                return
            
            if key is None:
                key = (layer_idx, sub_pos)
            
            # Eigendecomposition (shared between g_micro / spectral_gap / sv_entropy)
            need_eigh = (self._capture_g_micro or self._capture_spectral_gap
                          or self._capture_sv_entropy)
            singvals = None
            if need_eigh:
                cov = (X.T @ X) / X.shape[0]
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals = torch.clamp(eigvals, min=1e-12)
                singvals = torch.sqrt(eigvals)
            
            if self._capture_g_micro and singvals is not None:
                pr = (torch.sum(singvals) ** 2) / torch.sum(eigvals)
                g_micro = (pr / X.shape[-1]).item()
                if key not in self._micro_g_per_layer:
                    self._micro_g_per_layer[key] = []
                self._micro_g_per_layer[key].append(g_micro)
            
            if (self._capture_spectral_gap or self._capture_sv_entropy) and singvals is not None:
                sorted_sv = torch.sort(singvals, descending=True).values
                if self._capture_spectral_gap:
                    spectral_gap = (sorted_sv[0] / sorted_sv[1]).item() if sorted_sv[1] > 1e-12 else float('inf')
                    if key not in self._micro_gap_per_layer:
                        self._micro_gap_per_layer[key] = []
                    self._micro_gap_per_layer[key].append(spectral_gap)
                if self._capture_sv_entropy:
                    p = singvals / torch.sum(singvals)
                    sv_entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
                    if key not in self._micro_ent_per_layer:
                        self._micro_ent_per_layer[key] = []
                    self._micro_ent_per_layer[key].append(sv_entropy)
            
            if self._capture_phi:
                # Mean activation vector (overwrites — last pass wins for phi)
                self._micro_mean_vectors[key] = X.mean(dim=0).detach().cpu()
            
            self._micro_seqlen_per_layer[key] = X.shape[0]
            self._micro_hdim_per_layer[key] = X.shape[-1]
        
        except Exception as e:
            if label == 'attn' and not self._error_attn_printed:
                print(f"[CoS-Passive] attn capture failed at layer {layer_idx}: {e}")
                self._error_attn_printed = True
            elif label == 'mlp' and not self._error_mlp_printed:
                print(f"[CoS-Passive] mlp capture failed at layer {layer_idx}: {e}")
                self._error_mlp_printed = True
            elif label == 'block' and not self._error_block_printed:
                print(f"[CoS-Passive] block capture failed at layer {layer_idx}: {e}")
                self._error_block_printed = True

    # -----------------------------------------------------------------
    # Hook lifecycle
    # -----------------------------------------------------------------
    
    def _find_blocks(self, model):
        """Resolve transformer blocks via registry first, then common attrs."""
        dm = model.diffusion_model if hasattr(model, 'diffusion_model') else model
        dm_class = type(dm).__name__
        self._registry_config = None
        
        reg = _MODEL_REGISTRY.get(dm_class)
        if reg is not None:
            block_path = reg.get('block_path', 'NEEDS_DISCOVERY')
            blocks = _resolve_block_list(dm, block_path)
            if blocks:
                self._registry_config = reg
                if self._log_steps:
                    print(f"[CoS-Passive] Registry match: {dm_class} → "
                          f"{len(blocks)} blocks via '{block_path}'")
                return blocks
            print(f"[CoS-Passive] Registry entry {dm_class}: block_path "
                  f"'{block_path}' did not resolve; trying auto-detect")
        
        for attr in ('transformer_blocks', 'joint_blocks', 'blocks', 'layers'):
            if hasattr(dm, attr):
                blocks = _as_block_seq(getattr(dm, attr))
                if len(blocks) > 0:
                    if self._log_steps:
                        print(f"[CoS-Passive] Auto-detected blocks via '{attr}': {len(blocks)}")
                    return blocks
        # R3: UNet (SD1.5 / SDXL) has no flat block list; R2 returned [] here.
        blocks = _linearize_unet(dm)
        if blocks:
            print(f"[CoS-Passive] {dm_class}: {len(blocks)} UNet transformer blocks "
                  f"(linearized); not in the registry, so attn1 / ff fallbacks are used")
            return blocks
        print(f"[CoS-Passive] *** No transformer blocks found for {dm_class}; "
              f"no telemetry will be written. Children: "
              f"{[n for n, _ in dm.named_children()][:12]}")
        return []
    
    def _register_hooks(self, blocks):
        """Register passive hooks per block, respecting layer filter and registry."""
        self._remove_hooks()
        n_blocks = len(blocks)
        layer_filter = _parse_layer_filter(self._layer_filter_spec, n_blocks)
        
        # Pull paths from registry (if any)
        attn_paths = []
        mlp_paths = []
        block_level_only = False
        dual_supported = False
        if self._registry_config is not None:
            attn_cfg = self._registry_config.get('attn_attrs', {})
            attn_paths = [p for p in attn_cfg.get('self_attn', [])
                          if p and p != 'NEEDS_DISCOVERY']
            mlp_cfg = self._registry_config.get('mlp_attrs', {})
            mlp_paths = [p for p in mlp_cfg.get('mlp', [])
                          if p and p != 'NEEDS_DISCOVERY']
            dual_supported = bool(self._registry_config.get('dual_hook_supported', False))
            if '__block__' in attn_paths:
                block_level_only = True
        
        n_attn = 0
        n_mlp = 0
        n_block = 0
        for layer_idx, block in enumerate(blocks):
            if layer_filter is not None and layer_idx not in layer_filter:
                continue
            
            # Block-level fallback (rare — for joint-attention architectures
            # where attn submodule .forward() isn't called)
            if block_level_only:
                handle = block.register_forward_hook(self._make_block_hook(layer_idx))
                self._hook_handles.append(handle)
                n_block += 1
                continue
            
            # Attention hook
            attn_module = None
            for path in attn_paths:
                attn_module = _resolve_dotted_attr(block, path)
                if attn_module is not None:
                    break
            if attn_module is None:
                # Fallback: common attribute names
                for attr in ('attn1', 'self_attn', 'attn', 'attention'):
                    if hasattr(block, attr):
                        attn_module = getattr(block, attr)
                        break
            if attn_module is not None:
                handle = attn_module.register_forward_hook(self._make_attn_hook(layer_idx))
                self._hook_handles.append(handle)
                n_attn += 1
            
            # MLP hook (only if registry says dual_hook_supported, or as a free attempt)
            mlp_module = None
            if dual_supported or not mlp_paths:  # if registry has no mlp_paths, try common attrs as a free attempt
                for path in mlp_paths:
                    mlp_module = _resolve_dotted_attr(block, path)
                    if mlp_module is not None:
                        break
                if mlp_module is None:
                    for attr in ('mlp', 'feed_forward', 'ffn', 'ff'):
                        if hasattr(block, attr):
                            mlp_module = getattr(block, attr)
                            break
            if mlp_module is not None:
                handle = mlp_module.register_forward_hook(self._make_mlp_hook(layer_idx))
                self._hook_handles.append(handle)
                n_mlp += 1
        
        self._dual_hook_active = (n_mlp > 0 and n_attn > 0)
        if self._log_steps:
            filter_str = ("all" if layer_filter is None
                          else f"{len(layer_filter)} of {n_blocks}")
            print(f"[CoS-Passive] Registered {n_attn} attn + {n_mlp} mlp + "
                  f"{n_block} block hooks (filter: {filter_str}, "
                  f"dual_hook={self._dual_hook_active})")
    
    def _register_root_hook(self):
        """Second branch channel: read transformer_options['cond_or_uncond'] from the
        diffusion model's own call, so attribution survives a wrapper that is dropped
        or replaced by another node."""
        self._remove_root_hook()
        try:
            dm = getattr(self.model_patcher.model, 'diffusion_model', None)
            if dm is None:
                return
            guider = self
            def pre(module, args, kwargs):
                to = kwargs.get('transformer_options', None) if kwargs else None
                cu = to.get('cond_or_uncond', None) if isinstance(to, dict) else None
                guider._root_branches = list(cu) if cu is not None else None
                if cu is not None:
                    guider._diag['root_calls'] += 1
            def post(module, args, kwargs, output):
                guider._root_branches = None
            self._root_hook_handles.append(dm.register_forward_pre_hook(pre, with_kwargs=True))
            self._root_hook_handles.append(dm.register_forward_hook(post, with_kwargs=True))
        except Exception as e:
            print(f"[CoS-Passive] root branch hook unavailable ({e}); wrapper channel only")

    def _remove_root_hook(self):
        for h in self._root_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._root_hook_handles = []
        self._root_branches = None

    def _branch_check(self):
        """Printed after step 0 of every generation, whatever log_telemetry says."""
        keys = list(self._micro_g_per_layer.keys()) or list(self._micro_mean_vectors.keys())
        got = sorted({k[2] for k in keys if isinstance(k, tuple) and len(k) == 3})
        layers = sorted({k[0] for k in keys if isinstance(k, tuple)})
        need = ['cond'] if abs(self._cfg_value - 1.0) < 1e-6 else ['cond', 'uncond']
        fw = self._diag['forwards']
        regime = ('pooled' if any(f and len(f) > 1 for f in fw)
                  else 'separate' if len(fw) > 1 else 'single')
        ok = all(b in got for b in need) and 'pooled' not in got and len(keys) > 0
        msg = (f"[CoS-Passive] branch check gen {self._csv_gen_idx}: cfg={self._cfg_value:g} "
               f"branches={','.join(got) or 'none'} layers={len(layers)} "
               f"forwards={len(fw)} ({regime}) wrapper_calls={self._diag['wrapper_calls']} "
               f"root_calls={self._diag['root_calls']} seed={self._last_seed} "
               f"steps={self._gen_steps}")
        print(msg + ("  OK" if ok else ""))
        if not ok:
            print("[CoS-Passive] *** BRANCH CHECK FAILED *** expected branches "
                  f"{need}; unattributed batches={self._diag['unattributed']}. "
                  "Rows are still written; send this console line with the CSV.")

    def sample(self, *args, **kwargs):
        """Remove every hook when a sample ends, so guiders that share one model in
        a multi-arm workflow never leave hooks firing on each other's runs."""
        try:
            return super().sample(*args, **kwargs)
        finally:
            self._remove_hooks()
            self._remove_root_hook()
            self._last_timestep_seen = None

    def _remove_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []
    
    # -----------------------------------------------------------------
    # Generation tracking
    # -----------------------------------------------------------------
    
    def _on_step(self, timestep):
        """Detect new-generation transitions. Returns (current_step, is_new_gen)."""
        if isinstance(timestep, torch.Tensor):
            ts_val = float(timestep.flatten()[0].item())
        else:
            ts_val = float(timestep)
        
        is_new = False
        if self._last_timestep_seen is None or ts_val > self._last_timestep_seen + 1e-3:
            # First call ever, OR timestep increased = new generation
            is_new = True
            self._current_step = 0
            if self._csv_session:
                self._csv_gen_idx += 1
                if self._log_steps:
                    print(f"[CoS-Passive CSV] New generation → gen_idx={self._csv_gen_idx}")
        else:
            self._current_step += 1
        self._last_timestep_seen = ts_val
        return self._current_step, is_new

    # -----------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------
    
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        step, is_new = self._on_step(timestep)
        
        # Reset capture buffers for this step
        self._micro_g_per_layer = {}
        self._micro_mean_vectors = {}
        self._micro_gap_per_layer = {}
        self._micro_ent_per_layer = {}
        
        # On new generation: re-register hooks. (Hooks persist across steps within
        # a generation; we re-register on each new generation in case the model
        # changed underneath us.)
        if is_new:
            self._remove_hooks()
            try:
                model = self.model_patcher.model
                blocks = self._find_blocks(model)
                if blocks:
                    self._register_hooks(blocks)
                else:
                    if self._log_steps:
                        print(f"[CoS-Passive] No transformer blocks found — telemetry disabled")
            except Exception as e:
                print(f"[CoS-Passive] Hook setup failed: {e}")
                import traceback
                traceback.print_exc()
        
        if is_new:
            self._register_root_hook()
            self._diag = {'wrapper_calls': 0, 'root_calls': 0, 'forwards': [], 'unattributed': 0}
            try:
                ss = model_options.get('transformer_options', {}).get('sample_sigmas', None)
                self._gen_steps = int(len(ss) - 1) if ss is not None else -1
            except Exception:
                self._gen_steps = -1
        self._last_seed = seed
        
        # R2: wrap the model call so hooks know which branch each batch row is.
        # Chains any wrapper already present (e.g. from other nodes).
        if self._split_branches:
            prev_wrapper = model_options.get('model_function_wrapper', None)
            guider = self
            def _branch_wrapper(apply_model, args):
                cu = args.get('cond_or_uncond', None)
                if cu is None:
                    cu = (args.get('c', {}).get('transformer_options', {})
                          .get('cond_or_uncond', None))
                guider._cur_branches = list(cu) if cu is not None else None
                guider._diag['wrapper_calls'] += 1
                guider._diag['forwards'].append(tuple(cu) if cu is not None else None)
                try:
                    if prev_wrapper is not None:
                        return prev_wrapper(apply_model, args)
                    return apply_model(args['input'], args['timestep'], **args['c'])
                finally:
                    guider._cur_branches = None
            model_options = dict(model_options)
            model_options['model_function_wrapper'] = _branch_wrapper
        
        # R3: some architectures need a transformer option for consistent capture
        # (Qwen-Image 2.1's prefix K/V cache runs text rows once, then image rows
        # only, so hooks would see different sequences per step). The registry
        # entry's 'transformer_options_set' is merged in for this guider's calls.
        tos = (self._registry_config or {}).get('transformer_options_set')
        if tos:
            model_options = dict(model_options)
            to = dict(model_options.get('transformer_options', {}))
            to.update(tos)
            model_options['transformer_options'] = to
            if is_new:
                print(f"[CoS-Passive] registry transformer options applied: {tos}")
        
        # Standard 2-pass CFG forward — clean, no perturbation
        try:
            self.cfg = self._cfg_value
            noise_pred = super().predict_noise(x, timestep, model_options, seed)
        except Exception:
            self._remove_hooks()
            raise
        
        # Compute phi from captured mean vectors (if phi capture enabled and
        # we have at least 3 entries).
        phi_per = {}
        skip_per = {}
        if self._capture_phi and len(self._micro_mean_vectors) >= 3:
            # R2: phi is computed along (layer, sub) order *within* each branch
            groups = {}
            for k in self._micro_mean_vectors.keys():
                groups.setdefault(k[2] if (isinstance(k, tuple) and len(k) == 3) else None, []).append(k)
            sorted_keys_all = [sorted(v) for v in groups.values()]
        else:
            sorted_keys_all = []
        for sorted_keys in sorted_keys_all:
            for i in range(len(sorted_keys)):
                key = sorted_keys[i]
                phi_val = 0.0
                phi_skip_val = 0.0
                mv = self._micro_mean_vectors
                if i >= 2 and not _same_width(mv, sorted_keys[i - 2:i + 1]):
                    # R3: UNet levels change width (SD1.5 320/640/1280), so a step
                    # across a level boundary has no direction cosine. Left blank.
                    phi_val = None
                elif i >= 2:
                    pp = sorted_keys[i - 2]
                    p = sorted_keys[i - 1]
                    dp = self._micro_mean_vectors[p] - self._micro_mean_vectors[pp]
                    dc = self._micro_mean_vectors[key] - self._micro_mean_vectors[p]
                    np_, nc = dp.norm(), dc.norm()
                    if np_ > 1e-8 and nc > 1e-8:
                        phi_val = (dp @ dc / (np_ * nc)).item()
                if i >= 3 and not _same_width(mv, sorted_keys[i - 3:i + 1]):
                    phi_skip_val = None
                elif i >= 3:
                    tbp = sorted_keys[i - 3]
                    tbc = sorted_keys[i - 2]
                    p = sorted_keys[i - 1]
                    d2b = self._micro_mean_vectors[tbc] - self._micro_mean_vectors[tbp]
                    dc = self._micro_mean_vectors[key] - self._micro_mean_vectors[p]
                    n2, nc = d2b.norm(), dc.norm()
                    if n2 > 1e-8 and nc > 1e-8:
                        phi_skip_val = (d2b @ dc / (n2 * nc)).item()
                phi_per[key] = phi_val
                skip_per[key] = phi_skip_val
        
        # Per-step console summary (g_micro layer-by-layer, like the EMAG guider)
        if self._log_steps and self._micro_g_per_layer:
            layer_strs = []
            for lid in sorted(self._micro_g_per_layer.keys()):
                vals = self._micro_g_per_layer[lid]
                avg = sum(vals) / len(vals)
                if isinstance(lid, tuple):
                    block_idx, sub_pos = lid[0], lid[1]
                    suffix = 'a' if sub_pos == 0 else 'm'
                    if len(lid) == 3:
                        suffix += {'cond': 'c', 'uncond': 'u'}.get(lid[2], 'p')
                    layer_strs.append(f"L{block_idx}{suffix}={avg:.4f}")
                else:
                    layer_strs.append(f"L{lid}={avg:.4f}")
            # Print only first step of each gen + every 25 steps to limit noise
            if step == 0 or step % 25 == 0:
                print(f"[CoS-Passive] step {step} g: {' '.join(layer_strs)}")
        
        if step == 0:
            self._branch_check()
        
        # Write CSV row(s) for this step
        if self._csv_session and self._csv_path is not None:
            try:
                self._write_csv_step(step, timestep, phi_per, skip_per)
            except Exception as e:
                print(f"[CoS-Passive CSV] Write failed at step {step}: {e}")
        
        return noise_pred

    def _write_csv_step(self, step, timestep, phi_per, skip_per):
        """Write one row per (layer, submodule) pair to the CSV."""
        # Initialize header if file doesn't exist or is empty
        if not self._csv_initialized:
            file_exists = self._csv_path.exists() and self._csv_path.stat().st_size > 0
            if not file_exists:
                with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'gen_idx', 'token_idx', 'token_type', 'layer', 'submodule',
                        'g_micro', 'phi_micro', 'phi_micro_skip', 'spectral_gap', 'sv_entropy',
                        'sigma', 'seq_len', 'hidden_dim',
                        'hook_mode',
                        'prompt_hash', 'prompt_len', 'gen_len', 'prompt_text',
                        'branch', 'prompt_influence', 'seed',
                    ])
            self._csv_initialized = True
        
        # Sigma / timestep value
        if isinstance(timestep, torch.Tensor):
            sigma_val = float(timestep.flatten()[0].item())
        else:
            sigma_val = float(timestep)
        
        # Union of all keys captured this step (some metrics may be selectively disabled)
        all_keys = set()
        all_keys.update(self._micro_g_per_layer.keys())
        all_keys.update(self._micro_mean_vectors.keys())
        all_keys.update(self._micro_gap_per_layer.keys())
        all_keys.update(self._micro_ent_per_layer.keys())
        all_keys.update(self._micro_seqlen_per_layer.keys())
        if not all_keys:
            return
        
        sorted_keys = sorted(all_keys)
        first_key = sorted_keys[0]
        prompt_len = len(self._csv_prompt_text)
        
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for key in sorted_keys:
                branch = 'all'
                if isinstance(key, tuple):
                    layer_idx, sub_pos = key[0], key[1]
                    if len(key) == 3:
                        branch = key[2]
                    submodule = 'attn' if sub_pos == 0 else 'mlp'
                else:
                    layer_idx = key
                    submodule = 'attn'
                
                # R2: prompt influence = cos(mean_cond, mean_uncond) at this (layer, sub)
                pinf = None
                if isinstance(key, tuple) and len(key) == 3:
                    mc = self._micro_mean_vectors.get((key[0], key[1], 'cond'))
                    mu = self._micro_mean_vectors.get((key[0], key[1], 'uncond'))
                    if mc is not None and mu is not None:
                        den = (mc.norm() * mu.norm()).item()
                        if den > 1e-12:
                            pinf = (mc @ mu).item() / den
                
                g_vals = self._micro_g_per_layer.get(key, [])
                gap_vals = self._micro_gap_per_layer.get(key, [])
                ent_vals = self._micro_ent_per_layer.get(key, [])
                
                g = sum(g_vals) / len(g_vals) if g_vals else None
                gap = sum(gap_vals) / len(gap_vals) if gap_vals else None
                ent = sum(ent_vals) / len(ent_vals) if ent_vals else None
                phi = phi_per.get(key, None)
                phi_skip = skip_per.get(key, None)
                
                pt = self._csv_prompt_text if (step == 0 and key == first_key) else ""
                
                writer.writerow([
                    self._csv_gen_idx,
                    step,
                    'step',
                    layer_idx,
                    submodule,
                    f"{g:.6f}" if g is not None else "",
                    f"{phi:.6f}" if phi is not None else "",
                    f"{phi_skip:.6f}" if phi_skip is not None else "",
                    f"{gap:.4f}" if gap is not None else "",
                    f"{ent:.4f}" if ent is not None else "",
                    f"{sigma_val:.6f}",
                    self._micro_seqlen_per_layer.get(key, 0),
                    self._micro_hdim_per_layer.get(key, 0),
                    'passive-branch' if self._split_branches else 'passive',
                    self._csv_prompt_hash,
                    prompt_len,
                    self._gen_steps,  # R3: total steps from sample_sigmas (-1 if unavailable)
                    pt,
                    branch,
                    f"{pinf:.6f}" if pinf is not None else "",
                    self._last_seed if self._last_seed is not None else "",
                ])


class CoSPAFS_PassiveTelemetryGuider:
    """ComfyUI node: Passive Telemetry Guider — clean CFG forward + measurement only.
    
    Drop-in replacement for the standard CFGGuider. Image output is identical
    to a vanilla CFG forward at the same cfg value. Adds passive measurement
    hooks that capture g_micro, phi_micro (stride-1), phi_micro_skip (stride-2),
    spectral_gap, and sv_entropy on each block's attention output and (when
    architecture supports it) MLP output.
    
    Per-metric switches let you disable capture you don't need (cheaper).
    Layer-range filter restricts capture to specific block indices.
    
    No EMAG, no perturbation, no phase scheduling. CSV schema matches the EMAG
    guider so the live_analyzer plays nicely with both.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Standard CFG scale. Applied uniformly across all steps. "
                               "Output is identical to a vanilla CFGGuider at the same value.",
                }),
            },
            "optional": {
                # --- Capture switches ---
                "capture_g_micro": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Capture g_micro (Participation Ratio / D). Costs one "
                               "[D x D] eigendecomposition per (layer, submodule) per step.",
                }),
                "capture_phi_micro": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Capture phi_micro (stride-1, cross-type) and "
                               "phi_micro_skip (stride-2, same-type). Stores the mean "
                               "activation vector and computes cosines after the forward pass. "
                               "Most useful for sublattice / two-sector analysis.",
                }),
                "capture_spectral_gap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Capture spectral gap (largest sv / second largest sv). "
                               "Free if g_micro is also enabled (shares the eigendecomposition).",
                }),
                "capture_sv_entropy": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Capture singular-value entropy. Free if g_micro is enabled.",
                }),
                # --- Layer filter ---
                "layer_range": ("STRING", {
                    "default": "",
                    "tooltip": "Layers to capture. Empty = all layers. "
                               "Examples: '0-15' (range, inclusive), '0,5,10' (specific), "
                               "'0-5,10-15,20' (mixed). Indices outside [0, n_blocks) are dropped.",
                }),
                # --- Logging / CSV ---
                "log_telemetry": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-step telemetry summary to console. "
                               "Includes hook registration info on the first step.",
                }),
                "csv_session": ("STRING", {
                    "default": "",
                    "tooltip": "CSV session name. Empty = no CSV output. "
                               "File saved to dm_telemetry/<name>.csv. "
                               "Existing files are appended to (gen_idx auto-increments).",
                }),
                "prompt_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Prompt text for CSV metadata. Stored once per generation "
                               "(first row only) for live_analyzer prompt grouping.",
                }),
                # --- R2: per-branch capture ---
                # Kept LAST on purpose: ComfyUI restores widget values by position,
                # so a new widget anywhere else shifts saved workflows (layer_range
                # then receives a boolean and the filter matches 0 blocks).
                "split_cfg_branches": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Per-branch rows (column 'branch' = cond/uncond) are ALWAYS "
                               "written since the R3 fix. This switch is kept only so saved "
                               "workflows load; a False value is ignored and logged.",
                }),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/cos-pafs"
    DESCRIPTION = ("Passive Telemetry Guider — vanilla CFG forward with passive measurement "
                   "hooks. No perturbation, no EMAG. Capture switches and layer-range filter "
                   "control what is recorded. CSV schema matches the EMAG guider.")

    def get_guider(self, model, positive, negative, cfg=3.5,
                   capture_g_micro=True, capture_phi_micro=True,
                   capture_spectral_gap=True, capture_sv_entropy=True,
                   layer_range="",
                   log_telemetry=False,
                   csv_session="", prompt_text="",
                   split_cfg_branches=True):
        # R3: catch widget values restored into the wrong slots
        shifted = []
        if isinstance(layer_range, bool) or str(layer_range).strip().lower() in ("true", "false"):
            shifted.append(f"layer_range={layer_range!r}")
            layer_range = ""
        if not isinstance(log_telemetry, bool):
            shifted.append(f"log_telemetry={log_telemetry!r}")
            log_telemetry = bool(log_telemetry) and str(log_telemetry).lower() not in ("false", "0")
        if str(csv_session).strip().lower() in ("true", "false"):
            shifted.append(f"csv_session={csv_session!r}")
        if shifted:
            print("[CoS-Passive] WARNING: widget values look shifted (" + ", ".join(shifted) +
                  "). Delete and re-add the Passive Telemetry Guider node in this workflow.")
        guider = CoSPAFS_PassiveTelemetryGuiderImpl(model)
        guider.set_branch_split(split_cfg_branches)
        guider.set_conds(positive=positive, negative=negative)
        guider.set_logging(log_telemetry)  # set logging FIRST so set_csv_config can print
        guider.set_cfg_value(cfg)
        guider.set_capture_switches(
            g_micro=capture_g_micro,
            phi=capture_phi_micro,
            spectral_gap=capture_spectral_gap,
            sv_entropy=capture_sv_entropy,
        )
        guider.set_layer_filter(layer_range)
        guider.set_csv_config(session_name=csv_session, prompt_text=prompt_text)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "CoSPAFS_Sampler": CoSPAFS_Sampler,
    "CoSPAFS_Sigmas": CoSPAFS_Sigmas,
    "CoSPAFS_CFGGuider": CoSPAFS_CFGGuider,
    "CoSPAFS_GuidanceSchedule": CoSPAFS_GuidanceSchedule,
    "CoSPAFS_AGPRSchedule": CoSPAFS_AGPRSchedule,
    "CoSPAFS_PerturbationSchedule": CoSPAFS_PerturbationSchedule,
    "CoSPAFS_EMAGGuider": CoSPAFS_EMAGGuider,
    "CoSPAFS_PassiveTelemetryGuider": CoSPAFS_PassiveTelemetryGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoSPAFS_Sampler": "CoS-PAFS Sampler",
    "CoSPAFS_Sigmas": "CoS-PAFS Sigmas",
    "CoSPAFS_CFGGuider": "CoS-PAFS CFG Guider",
    "CoSPAFS_GuidanceSchedule": "CoS-PAFS Guidance Schedule",
    "CoSPAFS_AGPRSchedule": "CoS-PAFS AGPR Schedule",
    "CoSPAFS_PerturbationSchedule": "CoS-PAFS Perturbation Schedule",
    "CoSPAFS_EMAGGuider": "CoS-PAFS EMAG Guider (Diagnostic)",
    "CoSPAFS_PassiveTelemetryGuider": "CoS-PAFS Passive Telemetry Guider",
}