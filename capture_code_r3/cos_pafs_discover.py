#!/usr/bin/env python3
"""
cos_pafs_discover.py — Model Hook Discovery & Registry Tool
════════════════════════════════════════════════════════════

Probes a loaded model (or loads one from ComfyUI's model directory) to
identify transformer blocks, attention attribute names, hidden dimensions,
and generates a registry entry for model_registry.json.

Usage from ComfyUI's Python environment:

    # If model is already loaded (e.g., from ComfyUI model loader):
    python cos_pafs_discover.py --from-comfy "SD3.5-medium"

    # Probe a safetensors file directly:
    python cos_pafs_discover.py --model "G:/models/sd3.5-medium.safetensors" --arch sd3

    # Just dump the module tree of whatever is loaded:
    python cos_pafs_discover.py --dump-tree --model "path/to/model"

    # Update registry from discovery results:
    python cos_pafs_discover.py --model "path/to/model" --update-registry

Output: Prints discovery report and optionally updates model_registry.json.

Can also be imported and called from within ComfyUI nodes:
    from .cos_pafs_discover import discover_from_model, update_registry
"""

import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent.resolve()
REGISTRY_PATH = SCRIPT_DIR / "model_registry.json"

# ─────────────────────────────────────────────────────────────────────
# Known attention attribute names across architectures
# ─────────────────────────────────────────────────────────────────────

KNOWN_SELF_ATTN_NAMES = [
    'attn1', 'attn', 'self_attn', 'self_attention', 'attention',
    'spatial_attn', 'temporal_attn',
]

KNOWN_CROSS_ATTN_NAMES = [
    'attn2', 'cross_attn', 'cross_attention', 'context_attn',
]

KNOWN_BLOCK_CONTAINER_NAMES = [
    'transformer_blocks', 'joint_blocks', 'blocks', 'layers',
    'single_transformer_blocks', 'double_transformer_blocks',
    'temporal_transformer_blocks',
]


# ─────────────────────────────────────────────────────────────────────
# Core Discovery
# ─────────────────────────────────────────────────────────────────────

def discover_from_model(model: nn.Module, model_name: str = "unknown") -> Dict:
    """
    Probe a model and return a complete discovery report.

    Args:
        model: The diffusion model (or model.diffusion_model).
               Can be the full pipeline model or just the transformer/unet.
        model_name: Human-readable name for the report.

    Returns:
        Dictionary with discovery results ready for registry insertion.
    """
    # Unwrap diffusion_model if present
    dm = model.diffusion_model if hasattr(model, 'diffusion_model') else model

    report = {
        'model_name': model_name,
        'model_class': type(dm).__name__,
        'total_params': sum(p.numel() for p in dm.parameters()),
        'block_containers': [],
        'recommended_config': None,
    }

    # Find all block containers (ModuleLists with >4 same-type children)
    containers = _find_block_containers(dm)

    for path, module_list, count, child_type in containers:
        container_info = {
            'path': path,
            'count': count,
            'child_type': child_type,
            'self_attn_attr': None,
            'cross_attn_attr': None,
            'hidden_dim': None,
            'head_dim': None,
            'num_heads': None,
            'has_feedforward': False,
            'block_heterogeneous': False,
        }

        # Probe first block for attention attributes
        first_block = module_list[0]
        last_block = module_list[-1]

        # Self-attention (supports nested paths like "x_block.attn")
        for attr in KNOWN_SELF_ATTN_NAMES:
            # Check direct attribute first
            if hasattr(first_block, attr):
                container_info['self_attn_attr'] = attr
                attn_mod = getattr(first_block, attr)
                dims = _extract_attn_dims(attn_mod)
                if dims:
                    container_info['hidden_dim'] = dims.get('hidden_dim')
                    container_info['head_dim'] = dims.get('head_dim')
                    container_info['num_heads'] = dims.get('num_heads')
                break
            # Check nested: look in immediate children
            nested = _find_nested_attr(first_block, attr)
            if nested:
                container_info['self_attn_attr'] = nested
                attn_mod = _resolve_attr_path(first_block, nested)
                dims = _extract_attn_dims(attn_mod)
                if dims:
                    container_info['hidden_dim'] = dims.get('hidden_dim')
                    container_info['head_dim'] = dims.get('head_dim')
                    container_info['num_heads'] = dims.get('num_heads')
                break

        # Cross-attention (supports nested paths)
        for attr in KNOWN_CROSS_ATTN_NAMES:
            if hasattr(first_block, attr):
                container_info['cross_attn_attr'] = attr
                break
            nested = _find_nested_attr(first_block, attr)
            if nested:
                container_info['cross_attn_attr'] = nested
                break

        # If still no known names, deep-scan for anything attention-like
        if container_info['self_attn_attr'] is None:
            attn_attrs = _scan_for_attention(first_block)
            if attn_attrs:
                container_info['self_attn_attr'] = attn_attrs[0]
                if len(attn_attrs) > 1:
                    container_info['cross_attn_attr'] = attn_attrs[1]

        # Check for feedforward
        for attr in ('ff', 'mlp', 'feedforward', 'ff_context', 'feed_forward'):
            if hasattr(first_block, attr):
                container_info['has_feedforward'] = True
                break
            # Also check nested
            nested_ff = _find_nested_attr(first_block, attr)
            if nested_ff:
                container_info['has_feedforward'] = True
                break

        # Check heterogeneity (first vs last block)
        first_children = set(name for name, _ in first_block.named_children())
        last_children = set(name for name, _ in last_block.named_children())
        container_info['block_heterogeneous'] = (first_children != last_children)

        # If hidden_dim not found from attention, try other methods
        if container_info['hidden_dim'] is None:
            container_info['hidden_dim'] = _detect_hidden_dim_broad(first_block)

        report['block_containers'].append(container_info)

    # Generate recommended config
    if report['block_containers']:
        best = _select_best_container(report['block_containers'])
        report['recommended_config'] = _generate_registry_entry(
            report['model_class'], model_name, best
        )

    return report


def _find_block_containers(model: nn.Module) -> List[Tuple[str, nn.ModuleList, int, str]]:
    """Find all ModuleLists that look like transformer block sequences."""
    containers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 3:
            child_types = [type(c).__name__ for c in module]
            unique_types = set(child_types)
            if len(unique_types) <= 3:  # allow some heterogeneity
                containers.append((
                    name,
                    module,
                    len(module),
                    child_types[0],
                ))
    return containers


def _resolve_attr_path(module: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted attribute path like 'x_block.attn'."""
    parts = path.split('.')
    for part in parts:
        module = getattr(module, part)
    return module


def _find_nested_attr(block: nn.Module, attr_name: str) -> Optional[str]:
    """Check if attr_name exists inside any immediate child of block."""
    for child_name, child_mod in block.named_children():
        if hasattr(child_mod, attr_name):
            return f"{child_name}.{attr_name}"
    return None


def _scan_for_attention(block: nn.Module) -> List[str]:
    """Scan a block for any attention-like submodules, including nested."""
    attn_attrs = []

    # Direct children first
    for name, child in block.named_children():
        name_lower = name.lower()
        is_attn = (
            'attn' in name_lower or
            'attention' in name_lower or
            type(child).__name__.lower().startswith('attention') or
            type(child).__name__.lower().startswith('multihead')
        )
        if is_attn:
            has_proj = any(
                'to_q' in n or 'to_k' in n or 'q_proj' in n or 'k_proj' in n
                or 'query' in n or 'key' in n or 'in_proj' in n
                for n, _ in child.named_modules()
            )
            if has_proj:
                attn_attrs.append(name)

    # Nested children (e.g., JointBlock -> x_block -> attn)
    for child_name, child_mod in block.named_children():
        for name, grandchild in child_mod.named_children():
            name_lower = name.lower()
            is_attn = (
                'attn' in name_lower or
                'attention' in name_lower or
                type(grandchild).__name__.lower().startswith('attention') or
                type(grandchild).__name__.lower().startswith('multihead')
            )
            if is_attn:
                has_proj = any(
                    'to_q' in n or 'to_k' in n or 'q_proj' in n or 'k_proj' in n
                    or 'query' in n or 'key' in n or 'in_proj' in n
                    for n, _ in grandchild.named_modules()
                )
                if has_proj:
                    attn_attrs.append(f"{child_name}.{name}")

    return attn_attrs


def _extract_attn_dims(attn_module: nn.Module) -> Optional[Dict]:
    """Extract hidden_dim, head_dim, num_heads from an attention module."""
    dims = {}

    # Try common attributes first
    for attr in ('hidden_size', 'embed_dim', 'inner_dim', 'd_model', 'dim'):
        if hasattr(attn_module, attr):
            val = getattr(attn_module, attr)
            if isinstance(val, int) and val > 32:
                dims['hidden_dim'] = val
                break

    for attr in ('head_dim', 'd_head', 'head_size'):
        if hasattr(attn_module, attr):
            val = getattr(attn_module, attr)
            if isinstance(val, int) and val > 0:
                dims['head_dim'] = val

    for attr in ('num_heads', 'n_heads', 'heads', 'num_attention_heads'):
        if hasattr(attn_module, attr):
            val = getattr(attn_module, attr)
            if isinstance(val, int) and val > 0:
                dims['num_heads'] = val

    # Infer from projection weights
    if 'hidden_dim' not in dims:
        for name, param in attn_module.named_parameters():
            if ('to_q' in name or 'q_proj' in name or 'query' in name) and 'weight' in name:
                dims['hidden_dim'] = param.shape[-1]  # input dim
                if 'head_dim' not in dims and 'num_heads' in dims:
                    dims['head_dim'] = param.shape[0] // dims['num_heads']
                break

    return dims if dims else None


def _detect_hidden_dim_broad(block: nn.Module) -> Optional[int]:
    """Try to detect hidden dimension from any Linear layer in the block."""
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            # Prefer layers that look like projections (square or near-square)
            in_f, out_f = mod.in_features, mod.out_features
            if in_f == out_f and in_f > 64:
                return in_f
    # Fallback: first Linear's input features
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and mod.in_features > 64:
            return mod.in_features
    return None


def _select_best_container(containers: List[Dict]) -> Dict:
    """Select the most likely primary transformer block container."""
    # Prefer: has self_attn, has feedforward, most blocks
    scored = []
    for c in containers:
        score = 0
        if c['self_attn_attr']:
            score += 10
        if c['cross_attn_attr']:
            score += 5
        if c['has_feedforward']:
            score += 3
        score += c['count'] * 0.1  # slight preference for more blocks
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _generate_registry_entry(model_class: str, model_name: str, 
                              container: Dict) -> Dict:
    """Generate a registry-ready JSON entry from discovery results."""
    attn_attrs = {}
    if container['self_attn_attr']:
        attn_attrs['self_attn'] = [container['self_attn_attr']]
    if container['cross_attn_attr']:
        attn_attrs['cross_attn'] = [container['cross_attn_attr']]

    return {
        'description': f"{model_name} ({container['child_type']})",
        'models': [model_name],
        'block_path': container['path'],
        'block_count': container['count'],
        'hidden_dim': container['hidden_dim'],
        'attn_attrs': attn_attrs,
        'notes': f"Auto-discovered. head_dim={container.get('head_dim')}, "
                 f"num_heads={container.get('num_heads')}, "
                 f"heterogeneous={container['block_heterogeneous']}",
        'verified': False,
    }


# ─────────────────────────────────────────────────────────────────────
# Registry Management
# ─────────────────────────────────────────────────────────────────────

def load_registry(path: Path = REGISTRY_PATH) -> Dict:
    """Load the model hook registry."""
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_registry(registry: Dict, path: Path = REGISTRY_PATH):
    """Save the model hook registry."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)
    print(f"Registry saved: {path}")


def update_registry(model_class: str, entry: Dict, path: Path = REGISTRY_PATH):
    """Update or add a registry entry."""
    registry = load_registry(path)

    if model_class in registry:
        # Merge: update fields but preserve manually-set values
        existing = registry[model_class]
        for key, val in entry.items():
            if key == 'models':
                # Merge model lists
                existing_models = set(existing.get('models', []))
                existing_models.update(val)
                existing['models'] = sorted(existing_models)
            elif val is not None and val != "NEEDS_DISCOVERY":
                existing[key] = val
        existing['verified'] = False  # mark for human review
    else:
        registry[model_class] = entry

    save_registry(registry, path)
    return registry


def lookup_model(model: nn.Module, registry_path: Path = REGISTRY_PATH) -> Optional[Dict]:
    """
    Look up a model in the registry by its class name.
    Returns the registry entry or None if not found.

    Call this from the EMAG guider to get hook configuration.
    """
    dm = model.diffusion_model if hasattr(model, 'diffusion_model') else model
    model_class = type(dm).__name__

    registry = load_registry(registry_path)

    # Direct class name match
    if model_class in registry:
        entry = registry[model_class]
        if entry.get('block_path') and entry['block_path'] != 'NEEDS_DISCOVERY':
            return entry

    # Try partial matches (e.g., "UNet2DConditionModel" matches both SD15 and SDXL entries)
    for key, entry in registry.items():
        if key.startswith('_'):
            continue
        if model_class in key or key in model_class:
            if entry.get('block_path') and entry['block_path'] != 'NEEDS_DISCOVERY':
                return entry

    return None


# ─────────────────────────────────────────────────────────────────────
# Pretty Printing
# ─────────────────────────────────────────────────────────────────────

def print_discovery_report(report: Dict):
    """Print a human-readable discovery report."""
    print("=" * 70)
    print(f"DISCOVERY REPORT: {report['model_name']}")
    print(f"  Class: {report['model_class']}")
    print(f"  Parameters: {report['total_params']:,}")
    print("=" * 70)

    for i, container in enumerate(report['block_containers']):
        print(f"\n  Block Container #{i+1}: {container['path']}")
        print(f"    Blocks: {container['count']} × {container['child_type']}")
        print(f"    Self-attention: {container['self_attn_attr'] or 'NOT FOUND'}")
        print(f"    Cross-attention: {container['cross_attn_attr'] or 'NOT FOUND'}")
        print(f"    Hidden dim: {container['hidden_dim'] or 'NOT FOUND'}")
        print(f"    Head dim: {container.get('head_dim') or 'unknown'}")
        print(f"    Num heads: {container.get('num_heads') or 'unknown'}")
        print(f"    Has feedforward: {container['has_feedforward']}")
        print(f"    Heterogeneous: {container['block_heterogeneous']}")

        if container['hidden_dim'] and container['count']:
            dn = container['hidden_dim'] / container['count']
            nd = container['count'] / container['hidden_dim']
            print(f"    D/N: {dn:.1f}")
            print(f"    N/D: {nd:.6f}")

    if report['recommended_config']:
        print(f"\n  RECOMMENDED REGISTRY ENTRY:")
        print(f"  {'-' * 60}")
        print(f'  Key: "{report["model_class"]}"')
        print(json.dumps(report['recommended_config'], indent=4))
    else:
        print(f"\n  [!] No suitable block container found.")
        print(f"  Manual inspection required.")

    print("=" * 70)


def print_module_tree(model: nn.Module, max_depth: int = 4):
    """Print the module tree for manual inspection."""
    dm = model.diffusion_model if hasattr(model, 'diffusion_model') else model
    print(f"Module tree for {type(dm).__name__}:")
    print("=" * 70)
    _walk_tree(dm, "", 0, max_depth)


def _walk_tree(module: nn.Module, prefix: str, depth: int, max_depth: int):
    """Recursive module tree printer."""
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        n_params = sum(p.numel() for p in child.parameters())
        n_children = sum(1 for _ in child.children())

        indent = "  " * depth
        type_name = type(child).__name__

        # Highlight attention modules
        is_attn = any(a in name.lower() for a in ['attn', 'attention'])
        marker = " ◄ ATTENTION" if is_attn else ""

        # Highlight ModuleLists
        if isinstance(child, nn.ModuleList):
            marker = f" ◄ ModuleList[{len(child)}]"

        info = f"{indent}{path} ({type_name}) [{n_params:,} params]{marker}"
        print(info)

        if depth < max_depth:
            _walk_tree(child, path, depth + 1, max_depth)


# ─────────────────────────────────────────────────────────────────────
# Integration with EMAG Guider
# ─────────────────────────────────────────────────────────────────────

def resolve_hooks_from_registry(model: nn.Module, log: bool = False) -> Optional[Dict]:
    """
    Look up model in registry and return resolved hook targets.

    Returns dict with:
        'blocks': list of (name, module) pairs
        'self_attn_attr': str
        'cross_attn_attr': str or None
        'hidden_dim': int
        'block_count': int

    Or None if model not in registry / not verified.

    Call this from _find_transformer_blocks as a first attempt before
    falling back to auto-detection.
    """
    entry = lookup_model(model)
    if entry is None:
        if log:
            print(f"[CoS-REGISTRY] Model not in registry. Run cos_pafs_discover.py")
        return None

    dm = model.diffusion_model if hasattr(model, 'diffusion_model') else model
    block_path = entry.get('block_path')

    if not block_path or block_path == 'NEEDS_DISCOVERY':
        if log:
            print(f"[CoS-REGISTRY] Model registered but needs discovery")
        return None

    if block_path == 'UNET_LINEARIZE':
        if log:
            print(f"[CoS-REGISTRY] UNet model — using linearization")
        return None  # Let the existing UNet path handle it

    # Resolve the block path
    try:
        parts = block_path.split('.')
        target = dm
        for part in parts:
            target = getattr(target, part)

        blocks = [(f"{block_path}.{i}", target[i]) for i in range(len(target))]

        attn_attrs = entry.get('attn_attrs', {})

        result = {
            'blocks': blocks,
            'self_attn_attr': attn_attrs.get('self_attn', [None])[0],
            'cross_attn_attr': attn_attrs.get('cross_attn', [None])[0],
            'hidden_dim': entry.get('hidden_dim'),
            'block_count': len(blocks),
        }

        if log:
            print(f"[CoS-REGISTRY] Resolved {len(blocks)} blocks via "
                  f"'{block_path}', self_attn='{result['self_attn_attr']}', "
                  f"cross_attn='{result['cross_attn_attr']}', "
                  f"D={result['hidden_dim']}")

        return result

    except (AttributeError, IndexError, TypeError) as e:
        if log:
            print(f"[CoS-REGISTRY] Failed to resolve '{block_path}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CoS-PAFS Model Hook Discovery & Registry Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover from a ComfyUI-loaded model (run in ComfyUI's Python):
  python cos_pafs_discover.py --from-comfy "SD3.5 Medium"

  # Discover and update registry:
  python cos_pafs_discover.py --model "model.safetensors" --arch sd3 --update-registry

  # List current registry:
  python cos_pafs_discover.py --list-registry

  # Dump module tree:
  python cos_pafs_discover.py --dump-tree --model "model.safetensors" --arch sd3
        """
    )

    parser.add_argument('--model', type=str, help='Model path (safetensors or directory)')
    parser.add_argument('--arch', type=str, help='Architecture hint for loading')
    parser.add_argument('--dump-tree', action='store_true', help='Print full module tree')
    parser.add_argument('--update-registry', action='store_true', 
                        help='Update model_registry.json with discovery results')
    parser.add_argument('--list-registry', action='store_true',
                        help='List current registry entries')
    parser.add_argument('--tree-depth', type=int, default=4,
                        help='Module tree depth (default: 4)')

    args = parser.parse_args()

    if args.list_registry:
        registry = load_registry()
        print(f"\nModel Hook Registry ({REGISTRY_PATH})")
        print("=" * 70)
        for key, entry in registry.items():
            if key.startswith('_'):
                continue
            verified = "✓" if entry.get('verified') else "?"
            block_path = entry.get('block_path', 'unknown')
            hdim = entry.get('hidden_dim', '?')
            count = entry.get('block_count', '?')
            status = "READY" if block_path not in ('NEEDS_DISCOVERY', 'UNET_LINEARIZE', None) else block_path
            print(f"\n  [{verified}] {key}")
            print(f"      {entry.get('description', '')}")
            print(f"      blocks: {block_path} ({count}), D={hdim}, status={status}")
            models = entry.get('models', [])
            if models:
                print(f"      models: {', '.join(models)}")
        print()
        return

    if not args.model:
        parser.error("--model required (or use --list-registry)")

    # Load model
    print(f"Loading model: {args.model}")
    try:
        import diffusers
        # Try loading just the transformer/unet component
        if args.arch == 'sd3':
            from diffusers import SD3Transformer2DModel
            model = SD3Transformer2DModel.from_single_file(args.model, torch_dtype=torch.float16)
        else:
            # Generic loading attempt
            from diffusers import AutoModel
            model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    except Exception as e:
        print(f"Failed to load via diffusers: {e}")
        print("Try loading the model in ComfyUI and using the discover function directly.")
        return

    # Run discovery
    report = discover_from_model(model, model_name=Path(args.model).stem)
    print_discovery_report(report)

    if args.dump_tree:
        print("\n")
        print_module_tree(model, max_depth=args.tree_depth)

    if args.update_registry and report['recommended_config']:
        update_registry(report['model_class'], report['recommended_config'])
        print(f"\nRegistry updated with entry for '{report['model_class']}'")


if __name__ == '__main__':
    main()
