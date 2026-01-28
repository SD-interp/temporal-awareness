#!/usr/bin/env python
"""
Unified intertemporal preference experiment.

Runs the full analysis pipeline:
1. Generate preference data (or load existing)
2. Activation patching to find important positions
3. Attribution patching for circuit analysis
4. Compute steering vectors via contrastive analysis
5. Evaluate steering vectors

Usage:
    # Quick test with minimal config
    uv run python scripts/experiments/intertemporal_experiment.py --small

    # Full pipeline
    uv run python scripts/experiments/intertemporal_experiment.py

    # Use existing preference data
    uv run python scripts/experiments/intertemporal_experiment.py --preference-data <id>

    # Skip slow steps
    uv run python scripts/experiments/intertemporal_experiment.py --skip-attribution --skip-steering-eval
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import ensure_dir, save_json, get_timestamp
from src.data import (
    generate_preference_data,
    load_pref_data_with_prompts,
    find_preference_data,
    get_preference_data_id,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL,
)
from src.models import ModelRunner
from src.experiments import (
    run_activation_patching,
    run_attribution_patching,
    compute_steering_vector,
    apply_steering,
)
from src.viz import plot_layer_position_heatmap
from src.profiler import P

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "experiments"

# Small config for quick testing (20 samples for meaningful steering vectors)
SMALL_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": {
        "name": "test_minimal",
        "context": {
            "reward_unit": "dollars",
            "role": "you",
            "situation": "Choose between options.",
        },
        "options": {
            "short_term": {
                "reward_range": [100, 500],
                "time_range": [[1, "days"], [30, "days"]],
                "reward_steps": [2, "linear"],
                "time_steps": [2, "linear"],
            },
            "long_term": {
                "reward_range": [800, 2000],
                "time_range": [[1, "months"], [6, "months"]],
                "reward_steps": [2, "linear"],
                "time_steps": [2, "linear"],
            },
        },
        "time_horizons": [None],
        "add_formatting_variations": False,
    },
    "max_samples": 20,
    "max_pairs": 1,
    "ig_steps": 3,
    "position_threshold": 0.03,
    "contrastive_max_samples": 20,
    "top_n_positions": 1,
    "steering_strengths": [-1.0, 0.0, 1.0],
    "test_prompts": ["Choose: $100 now or $300 in 3 months?"],
}

# Normal config for real experiments
NORMAL_CONFIG = {
    "model": DEFAULT_MODEL,
    "dataset_config": DEFAULT_DATASET_CONFIG,
    "max_samples": 50,
    "max_pairs": 3,
    "ig_steps": 10,
    "position_threshold": 0.05,
    "contrastive_max_samples": 200,
    "top_n_positions": 1,
    "steering_strengths": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "test_prompts": [
        "You have two options: receive $100 today, or receive $150 in one year. Which do you prefer?",
        "Would you rather have a small reward now or a larger reward later?",
    ],
}


def get_memory_usage() -> dict:
    """Get current memory usage stats."""
    stats = {}
    if torch.cuda.is_available():
        stats["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    if hasattr(torch.mps, "current_allocated_memory"):
        try:
            stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass
    return stats


def log_memory(stage: str):
    """Log memory usage at a stage."""
    mem = get_memory_usage()
    if mem:
        mem_str = ", ".join(f"{k}={v:.2f}" for k, v in mem.items())
        print(f"  [Memory @ {stage}] {mem_str}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full intertemporal preference experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use minimal config for quick testing"
    )
    parser.add_argument(
        "--preference-data", type=str,
        help="Use existing preference data instead of generating new"
    )
    parser.add_argument(
        "--skip-attribution", action="store_true",
        help="Skip attribution patching (faster)"
    )
    parser.add_argument(
        "--skip-steering-eval", action="store_true",
        help="Skip steering vector evaluation"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SMALL_CONFIG if args.small else NORMAL_CONFIG

    print(f"Config: {'SMALL (test)' if args.small else 'NORMAL'}")
    print(f"Model: {config['model']}")
    print(f"Max samples: {config['max_samples']}")

    with P("total"):
        ts = get_timestamp()
        run_dir = args.output / ts

        # Create organized subdirectories
        viz_dir = run_dir / "viz"
        data_dir = run_dir / "data"
        ensure_dir(viz_dir)
        ensure_dir(data_dir)

        print(f"Output: {run_dir}")

        # Save config immediately
        save_json({
            "config": config,
            "args": {
                "small": args.small,
                "skip_attribution": args.skip_attribution,
                "skip_steering_eval": args.skip_steering_eval,
            },
            "timestamp": ts,
        }, run_dir / "config.json")

        try:
            # Step 1: Get preference data
            print("\n" + "=" * 60)
            print("STEP 1: PREFERENCE DATA")
            print("=" * 60)

            pref_dir = PROJECT_ROOT / "out" / "preference_data"
            datasets_dir = PROJECT_ROOT / "out" / "datasets"

            if args.preference_data:
                with P("load_data"):
                    pref_data = load_pref_data_with_prompts(args.preference_data, pref_dir, datasets_dir)
                print(f"Loaded existing: {args.preference_data}")
            else:
                with P("generate_data"):
                    pref_data = generate_preference_data(
                        model=config["model"],
                        dataset_config=config["dataset_config"],
                        max_samples=config["max_samples"],
                        verbose=True,
                    )

            print(f"Model: {pref_data.model}")
            print(f"Samples: {len(pref_data.preferences)}")

            # Save preference summary
            pref_summary = {
                "model": pref_data.model,
                "dataset_id": pref_data.dataset_id,
                "n_samples": len(pref_data.preferences),
                "choices": {
                    "short_term": sum(1 for p in pref_data.preferences if p.choice == "short_term"),
                    "long_term": sum(1 for p in pref_data.preferences if p.choice == "long_term"),
                },
            }
            save_json(pref_summary, data_dir / "preference_summary.json")
            log_memory("after_data")

            # Step 2: Load model
            print("\n" + "=" * 60)
            print("STEP 2: LOAD MODEL")
            print("=" * 60)

            with P("load_model"):
                runner = ModelRunner(pref_data.model)
                print(f"Model loaded: {runner.n_layers} layers, d_model={runner.d_model}")
            log_memory("after_model_load")

            # Step 3: Activation patching
            print("\n" + "=" * 60)
            print("STEP 3: ACTIVATION PATCHING")
            print("=" * 60)

            with P("activation_patching"):
                pos_sweep, full_sweep, filtered_pos, token_labels, section_markers = run_activation_patching(
                    runner, pref_data,
                    max_pairs=config["max_pairs"],
                    threshold=config["position_threshold"],
                )

            print(f"Position sweep: max={pos_sweep.max():.3f}, argmax={np.argmax(pos_sweep)}")
            print(f"Filtered positions: {len(filtered_pos)} (threshold={config['position_threshold']})")
            print(f"Full sweep shape: {full_sweep.shape}")

            # Compute layer indices used
            n_layers_sample = min(12, runner.n_layers)
            if n_layers_sample > 1:
                layer_indices = [int(i * (runner.n_layers - 1) / (n_layers_sample - 1)) for i in range(n_layers_sample)]
            else:
                layer_indices = [0]
            filtered_labels = [token_labels[i] if i < len(token_labels) else f"pos{i}" for i in filtered_pos]

            # Save activation patching results
            np.save(data_dir / "position_sweep.npy", pos_sweep)
            np.save(data_dir / "full_sweep.npy", full_sweep)
            save_json({
                "filtered_positions": filtered_pos,
                "layer_indices": layer_indices,
                "token_labels": token_labels,
                "section_markers": section_markers,
                "position_sweep_max": float(pos_sweep.max()),
                "position_sweep_argmax": int(np.argmax(pos_sweep)),
            }, data_dir / "activation_patching.json")

            # Plot heatmap
            if len(filtered_pos) > 0 and full_sweep.size > 0:
                plot_layer_position_heatmap(
                    full_sweep, layer_indices, filtered_labels,
                    viz_dir / "activation_patching.png",
                    title="Activation Patching",
                    cbar_label="Recovery", vmin=0.0, vmax=1.0,
                    section_markers={k: filtered_pos.index(v) for k, v in section_markers.items() if v in filtered_pos},
                )
                print(f"  Saved: {viz_dir / 'activation_patching.png'}")

            # Find best position/layer
            best_pos_idx = int(np.argmax(pos_sweep))
            if best_pos_idx in filtered_pos:
                col_idx = filtered_pos.index(best_pos_idx)
                best_layer_idx = int(np.argmax(full_sweep[:, col_idx]))
            else:
                best_layer_idx = int(np.argmax(full_sweep[:, 0])) if full_sweep.size > 0 else 0
            best_layer = layer_indices[best_layer_idx] if layer_indices else 0
            best_token = token_labels[best_pos_idx] if best_pos_idx < len(token_labels) else f"pos{best_pos_idx}"

            print(f"Best position: {best_pos_idx} ({best_token})")
            print(f"Best layer: {best_layer}")
            log_memory("after_act_patching")

            # Step 4: Attribution patching (optional)
            attribution_results = None
            if not args.skip_attribution:
                print("\n" + "=" * 60)
                print("STEP 4: ATTRIBUTION PATCHING")
                print("=" * 60)

                with P("attribution_patching"):
                    attribution_results, attr_labels, attr_markers = run_attribution_patching(
                        runner, pref_data,
                        max_pairs=config["max_pairs"],
                        ig_steps=config["ig_steps"],
                    )

                # Save attribution results
                layers = list(range(runner.n_layers))
                for key, scores in attribution_results.items():
                    np.save(data_dir / f"attribution_{key}.npy", scores)

                    # Plot heatmap
                    plot_layer_position_heatmap(
                        scores, layers, attr_labels,
                        viz_dir / f"attribution_{key}.png",
                        title=f"Attribution: {key}",
                        cbar_label="Attribution",
                        cmap="RdBu_r",
                    )
                    print(f"  {key}: max={scores.max():.4f}, min={scores.min():.4f}")
                    print(f"    Saved: {viz_dir / f'attribution_{key}.png'}")

                save_json({
                    "methods": list(attribution_results.keys()),
                    "token_labels": attr_labels,
                    "section_markers": attr_markers,
                }, data_dir / "attribution_metadata.json")
                log_memory("after_attr_patching")
            else:
                print("\n[Skipping attribution patching]")

            # Step 5: Compute steering vector
            print("\n" + "=" * 60)
            print("STEP 5: STEERING VECTOR")
            print("=" * 60)

            with P("steering_vector"):
                direction, stats = compute_steering_vector(
                    runner, pref_data,
                    layer=best_layer,
                    position=best_pos_idx,
                    max_samples=config["contrastive_max_samples"],
                )

            print(f"Layer: {stats['layer']}, Position: {stats['position']}")
            print(f"Direction norm: {stats['direction_norm']:.4f}")
            print(f"Samples: class0={stats['n_class0']}, class1={stats['n_class1']}")

            # Save steering vector
            steering_data = {
                "type": "steering_vector",
                "model": pref_data.model,
                "layer": best_layer,
                "position": best_pos_idx,
                "token": best_token,
                "direction": direction.tolist(),
                **stats,
            }
            save_json(steering_data, data_dir / "steering_vector.json")
            print(f"  Saved: {data_dir / 'steering_vector.json'}")
            log_memory("after_steering")

            # Step 6: Evaluate steering (optional)
            if not args.skip_steering_eval:
                print("\n" + "=" * 60)
                print("STEP 6: STEERING EVALUATION")
                print("=" * 60)

                with P("steering_eval"):
                    eval_results = []
                    for prompt in config["test_prompts"]:
                        print(f"\nPrompt: {prompt[:60]}...")
                        prompt_results = {"prompt": prompt, "responses": []}

                        for strength in config["steering_strengths"]:
                            try:
                                response = apply_steering(
                                    runner, prompt, direction,
                                    layer=best_layer,
                                    strength=strength,
                                    max_new_tokens=50,
                                )
                                prompt_results["responses"].append({
                                    "strength": strength,
                                    "response": response,
                                })
                                # Truncate for display
                                display = response.replace('\n', ' ')[:60]
                                print(f"  strength={strength:+.1f}: {display}...")
                            except Exception as e:
                                print(f"  strength={strength:+.1f}: ERROR - {e}")
                                prompt_results["responses"].append({
                                    "strength": strength,
                                    "error": str(e),
                                })

                        eval_results.append(prompt_results)

                    save_json(eval_results, data_dir / "steering_eval.json")
                    print(f"\n  Saved: {data_dir / 'steering_eval.json'}")
            else:
                print("\n[Skipping steering evaluation]")

            # Save final metadata
            metadata = {
                "timestamp": ts,
                "model": pref_data.model,
                "dataset_id": pref_data.dataset_id,
                "config": config,
                "results": {
                    "best_position": best_pos_idx,
                    "best_layer": best_layer,
                    "position_token": best_token,
                    "steering_norm": stats["direction_norm"],
                    "position_sweep_max": float(pos_sweep.max()),
                    "n_filtered_positions": len(filtered_pos),
                },
                "skipped": {
                    "attribution": args.skip_attribution,
                    "steering_eval": args.skip_steering_eval,
                },
            }
            save_json(metadata, run_dir / "metadata.json")

            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n{'=' * 60}")
            print(f"COMPLETE")
            print(f"{'=' * 60}")
            print(f"Output directory: {run_dir}")
            print(f"  viz/  - Heatmaps and plots")
            print(f"  data/ - JSON and numpy arrays")

        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()
            save_json({
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, run_dir / "error.json")
            return 1

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
