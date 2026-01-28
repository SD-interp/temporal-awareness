#!/usr/bin/env python
"""
Activation patching for intertemporal preference analysis.

Workflow:
1. Position sweep: Patch all layers at each position
2. Expand regions: Include in-between positions around significant positions
3. Full sweep: Granular layer x position patching

Usage:
    python scripts/circuits/activation_patching_intertemporal.py
    python scripts/circuits/activation_patching_intertemporal.py --no-components  # Skip MLP/attention
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import ModelRunner
from src.models.intervention_utils import patch
from src.data import (
    load_pref_data_with_prompts,
    build_prompt_pairs,
    find_preference_data,
    get_preference_data_id,
)
from src.common.io import ensure_dir, save_json, get_timestamp
from src.analysis import build_position_mapping, create_metric, find_section_markers, get_token_labels
from src.common.positions_schema import PositionsFile, PositionSpec
from src.viz import plot_layer_position_heatmap
from src.profiler import P

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "activation_patching"
THRESHOLD = 0.05

# Component name mappings (defined once)
COMP_DISPLAY = {"resid_post": "Residual", "attn_out": "Attention", "mlp_out": "MLP"}
COMP_SHORT = {"resid_post": "resid", "attn_out": "attn", "mlp_out": "mlp"}


@dataclass
class PatchingContext:
    """Shared context for patching operations."""
    runner: ModelRunner
    clean_text: str
    corrupted_text: str
    metric: object
    pos_mapping: dict[int, int]
    seq_len: int
    token_labels: list[str]
    section_markers: dict[str, int]
    layers: list[int]


@dataclass
class Results:
    """Patching results."""
    position_sweeps: dict[str, np.ndarray] = field(default_factory=dict)
    full_sweeps: dict[str, np.ndarray] = field(default_factory=dict)
    positions: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    markers: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Calculation
# =============================================================================


def position_sweep(ctx: PatchingContext, component: str) -> np.ndarray:
    """Patch all layers at each position. Returns 1D recovery array."""
    all_layers = list(range(ctx.runner.n_layers))
    hooks = [f"blocks.{l}.hook_{component}" for l in all_layers]
    _, cache = ctx.runner.run_with_cache(ctx.clean_text, names_filter=lambda n: n in hooks)

    results = np.zeros(ctx.seq_len)
    for pos in range(ctx.seq_len):
        corr_pos = ctx.pos_mapping.get(pos, pos)
        interventions = []
        for layer in all_layers:
            act = cache[f"blocks.{layer}.hook_{component}"]
            if pos < act.shape[1]:
                interventions.append(patch(
                    layer=layer,
                    values=act[0, pos].detach().cpu().numpy(),
                    positions=[corr_pos],
                    component=component,
                ))
        if interventions:
            results[pos] = ctx.metric(ctx.runner.forward_with_intervention(ctx.corrupted_text, interventions))
    return results


def full_sweep(ctx: PatchingContext, positions: list[int], component: str) -> np.ndarray:
    """Layer x position patching. Returns 2D array [layers, positions]."""
    hooks = [f"blocks.{l}.hook_{component}" for l in ctx.layers]
    _, cache = ctx.runner.run_with_cache(ctx.clean_text, names_filter=lambda n: n in hooks)

    results = np.zeros((len(ctx.layers), len(positions)))
    for li, layer in enumerate(ctx.layers):
        act = cache[f"blocks.{layer}.hook_{component}"]
        for pi, pos in enumerate(positions):
            if pos < act.shape[1]:
                corr_pos = ctx.pos_mapping.get(pos, pos)
                intervention = patch(layer=layer, values=act[0, pos].detach().cpu().numpy(),
                                     positions=[corr_pos], component=component)
                results[li, pi] = ctx.metric(ctx.runner.forward_with_intervention(ctx.corrupted_text, intervention))
        print(f"  Layer {layer}: max={results[li].max():.3f}")
    return results


def expand_positions(positions: list[int], max_pos: int, max_gap: int = 10) -> list[int]:
    """Expand positions to include in-between (if gap < max_gap)."""
    if len(positions) < 2:
        return positions
    result = set(positions)
    sorted_pos = sorted(positions)
    for i in range(len(sorted_pos) - 1):
        gap = sorted_pos[i + 1] - sorted_pos[i]
        if 1 < gap < max_gap:
            result.update(range(sorted_pos[i] + 1, min(sorted_pos[i + 1], max_pos)))
    return sorted(result)


def remap_markers(markers: dict[str, int], positions: list[int]) -> dict[str, int]:
    """Remap section markers to position indices."""
    result = {}
    for name, pos in markers.items():
        if pos in positions:
            result[name] = positions.index(pos)
        else:
            candidates = [i for i, p in enumerate(positions) if p <= pos]
            if candidates:
                result[name] = max(candidates)
    return result


# =============================================================================
# Visualization
# =============================================================================


def save_heatmap(matrix: np.ndarray, layers: list[int], labels: list[str],
                 markers: dict[str, int], path: Path, title: str) -> None:
    """Save heatmap with standard settings."""
    plot_layer_position_heatmap(matrix, layers, labels, path, title=title,
                                cbar_label="Recovery", vmin=0.0, vmax=1.0,
                                section_markers=markers)


def save_positions_json(results: Results, component: str, ctx: PatchingContext,
                        threshold: float, model: str, dataset_id: str, output_dir: Path) -> None:
    """Save positions.json."""
    full = results.full_sweeps[component]
    positions = []
    for li, layer in enumerate(ctx.layers):
        for pi, pos in enumerate(results.positions):
            val = float(full[li, pi])
            if not np.isnan(val) and val > threshold:
                section = None
                for sec, sec_pos in ctx.section_markers.items():
                    if pos >= sec_pos:
                        section = sec.replace("before_", "")
                positions.append(PositionSpec(position=pos, token=ctx.token_labels[pos],
                                              score=val, layer=layer, section=section))

    PositionsFile(model=model, method="activation_patching",
                  positions=sorted(positions, key=lambda p: p.score, reverse=True),
                  dataset_id=dataset_id, threshold=threshold, component=component
                  ).save(output_dir / f"positions_{COMP_SHORT[component]}.json")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preference-data", type=str)
    p.add_argument("--max-pairs", type=int, default=2)
    p.add_argument("--no-components", action="store_true", help="Skip MLP/attention analysis")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    with P("total"):
        # Load data
        pref_dir = PROJECT_ROOT / "out" / "preference_data"
        data_dir = PROJECT_ROOT / "out" / "datasets"

        pref_id = args.preference_data
        if not pref_id:
            recent = find_preference_data(pref_dir)
            if not recent:
                return print("No preference data found") or 1
            pref_id = get_preference_data_id(recent)

        with P("load"):
            pref_data = load_pref_data_with_prompts(pref_id, pref_dir, data_dir)
            pairs = build_prompt_pairs(pref_data, args.max_pairs, include_response=True)
            print(f"Loaded {len(pref_data.preferences)} samples, {len(pairs)} pairs")

        if not pairs:
            return print("No pairs!") or 1

        with P("model"):
            runner = ModelRunner(pref_data.model)
            print(f"Model: {runner.n_layers} layers")

        run_dir = args.output / get_timestamp()
        ensure_dir(run_dir)

        # Setup
        clean_text, corrupted_text, clean_sample, corrupted_sample = pairs[0]
        with P("setup"):
            pos_mapping, seq_len, _ = build_position_mapping(
                runner, clean_text, corrupted_text,
                [clean_sample.short_term_label, clean_sample.long_term_label],
                [corrupted_sample.short_term_label, corrupted_sample.long_term_label])
            token_labels = get_token_labels(runner, clean_text)
            section_markers = find_section_markers(runner, clean_text,
                                                   clean_sample.short_term_label,
                                                   clean_sample.long_term_label)
            metric = create_metric(runner, clean_sample, corrupted_sample, clean_text, corrupted_text)
            print(f"Seq len: {seq_len}, markers: {section_markers}")
            print(f"Metric: clean={metric.clean_val:.3f}, corr={metric.corr_val:.3f}")

        # 12 evenly spaced layers
        layers = [int(i * (runner.n_layers - 1) / 11) for i in range(12)]

        ctx = PatchingContext(runner=runner, clean_text=clean_text, corrupted_text=corrupted_text,
                              metric=metric, pos_mapping=pos_mapping, seq_len=seq_len,
                              token_labels=token_labels, section_markers=section_markers, layers=layers)

        components = ["resid_post"] + ([] if args.no_components else ["attn_out", "mlp_out"])
        results = Results()

        # Phase 1: Position sweeps
        print("\n=== Position Sweeps ===")
        for comp in components:
            print(f"{COMP_DISPLAY[comp]}...")
            with P(f"pos_sweep_{comp}"):
                results.position_sweeps[comp] = position_sweep(ctx, comp)
            above = np.sum(results.position_sweeps[comp] > args.threshold)
            print(f"  Max: {results.position_sweeps[comp].max():.3f}, {above} above threshold")

        # Phase 2: Position selection
        print("\n=== Position Selection ===")
        significant = np.where(results.position_sweeps["resid_post"] > args.threshold)[0].tolist()
        if not significant:
            return print("No positions above threshold!") or 1
        results.positions = expand_positions(significant, seq_len)
        results.labels = [token_labels[i] for i in results.positions]
        results.markers = remap_markers(section_markers, results.positions)
        print(f"Significant: {len(significant)}, expanded: {len(results.positions)}")

        # Phase 3: Full sweeps
        print("\n=== Full Sweeps ===")
        for comp in components:
            print(f"{COMP_DISPLAY[comp]}...")
            with P(f"full_sweep_{comp}"):
                results.full_sweeps[comp] = full_sweep(ctx, results.positions, comp)

        # Phase 4: Save
        print("\n=== Saving ===")
        for comp in components:
            short = COMP_SHORT[comp]
            # Position sweep as broadcast matrix
            pos_vals = np.array([results.position_sweeps[comp][i] for i in results.positions])
            broadcast = np.tile(pos_vals, (len(layers), 1))
            if comp != "resid_post":
                for pi, pos in enumerate(results.positions):
                    if results.position_sweeps[comp][pos] <= args.threshold:
                        broadcast[:, pi] = np.nan
            save_heatmap(broadcast, layers, results.labels, results.markers,
                         run_dir / f"position_sweep{'_' + short if comp != 'resid_post' else ''}.png",
                         f"Position Sweep ({COMP_DISPLAY[comp]})")

            # Full sweep
            full = results.full_sweeps[comp].copy()
            if comp != "resid_post":
                for pi, pos in enumerate(results.positions):
                    if results.position_sweeps[comp][pos] <= args.threshold:
                        full[:, pi] = np.nan
            save_heatmap(full, layers, results.labels, results.markers,
                         run_dir / f"heatmap_{short}.png", f"Activation Patching ({COMP_DISPLAY[comp]})")

            save_positions_json(results, comp, ctx, args.threshold, pref_data.model, pref_data.dataset_id, run_dir)

        save_json({"timestamp": get_timestamp(), "model": pref_data.model,
                   "dataset_id": pref_data.dataset_id, "section_markers": section_markers,
                   "components": components, "threshold": args.threshold,
                   "n_significant": len(significant), "n_expanded": len(results.positions)},
                  run_dir / "metadata.json")

        print(f"\nSaved: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
