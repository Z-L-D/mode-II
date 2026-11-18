"""
ComfyUI-CoS-PAFS
================
Chain-of-Steps Phase-Adaptive Flow Solver

A reasoning-aware sampler for flow-matching video diffusion models,
based on findings from "Demystifying Video Reasoning" (Wang et al., 2026).

Nodes:
    CoS-PAFS Sampler              - Phase-adaptive ODE solver
    CoS-PAFS Sigmas               - Non-uniform sigma schedule
    CoS-PAFS Guidance Schedule    - Phase-aware CFG/FDTG scheduling
    CoS-PAFS AGPR Schedule        - Reference image conditioning schedule
    CoS-PAFS Perturbation Schedule - Step-adaptive perturbation guidance layers

Installation:
    Place this directory in ComfyUI/custom_nodes/comfyui-cos-pafs/
    No additional dependencies required beyond standard ComfyUI.

Usage:
    The CoS-PAFS Sampler node outputs a SAMPLER object compatible with
    SamplerCustomAdvanced. Pair with CoS-PAFS Sigmas for the optimized
    sigma schedule, or use any standard sigmas node.

    For LTX-2.3: Use with steps=30-50, sigma_max=1.0, sigma_min=0.0.
    Default parameters are tuned for LTX-2.3 22B flow matching.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = None  # No custom JS frontend needed
