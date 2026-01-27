"""Factory functions for creating interventions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from .interventions import Intervention, Target

if TYPE_CHECKING:
    from .model_runner import ModelRunner


def steering(
    layer: int,
    direction: Union[np.ndarray, list],
    strength: float = 1.0,
    positions: Optional[Union[int, list[int]]] = None,
    neurons: Optional[Union[int, list[int]]] = None,
    pattern: Optional[str] = None,
    component: str = "resid_post",
    normalize: bool = True,
) -> Intervention:
    """Add direction to activations (mode=add)."""
    direction = np.array(direction, dtype=np.float32).flatten()
    if normalize and len(direction) > 0:
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

    return Intervention(
        layer=layer,
        mode="add",
        values=direction,
        target=_target(positions, neurons, pattern),
        component=component,
        strength=strength,
    )


def ablation(
    layer: int,
    values: Optional[Union[np.ndarray, list, float]] = None,
    positions: Optional[Union[int, list[int]]] = None,
    neurons: Optional[Union[int, list[int]]] = None,
    pattern: Optional[str] = None,
    component: str = "resid_post",
) -> Intervention:
    """Set activations to fixed values (mode=set). Default: zero."""
    if values is None:
        values = np.array([0.0], dtype=np.float32)
    elif isinstance(values, (int, float)):
        values = np.array([float(values)], dtype=np.float32)
    else:
        values = np.array(values, dtype=np.float32)

    return Intervention(
        layer=layer,
        mode="set",
        values=values,
        target=_target(positions, neurons, pattern),
        component=component,
        strength=1.0,
    )


def patch(
    layer: int,
    values: Union[np.ndarray, list],
    positions: Optional[Union[int, list[int]]] = None,
    neurons: Optional[Union[int, list[int]]] = None,
    pattern: Optional[str] = None,
    component: str = "resid_post",
) -> Intervention:
    """Replace activations with cached values (mode=set)."""
    return Intervention(
        layer=layer,
        mode="set",
        values=np.array(values, dtype=np.float32),
        target=_target(positions, neurons, pattern),
        component=component,
        strength=1.0,
    )


def scale(
    layer: int,
    factor: float,
    positions: Optional[Union[int, list[int]]] = None,
    neurons: Optional[Union[int, list[int]]] = None,
    pattern: Optional[str] = None,
    component: str = "resid_post",
) -> Intervention:
    """Multiply activations by factor (mode=mul)."""
    return Intervention(
        layer=layer,
        mode="mul",
        values=np.array([factor], dtype=np.float32),
        target=_target(positions, neurons, pattern),
        component=component,
        strength=1.0,
    )


def _target(positions=None, neurons=None, pattern=None) -> Target:
    if pattern is not None:
        return Target.on_pattern(pattern)
    if positions is not None:
        return Target.at_positions(positions)
    if neurons is not None:
        return Target.at_neurons(neurons)
    return Target.all()


def compute_mean_activations(
    runner: "ModelRunner",
    layer: int,
    prompts: Union[str, list[str]],
    component: str = "resid_post",
) -> np.ndarray:
    """Compute mean activations across prompts."""
    if isinstance(prompts, str):
        prompts = [prompts]

    hook_name = f"blocks.{layer}.hook_{component}"
    means = []

    for prompt in prompts:
        _, cache = runner.run_with_cache(prompt, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]
        if isinstance(acts, torch.Tensor):
            acts = acts.detach().cpu().numpy()
        means.append(acts.mean(axis=(0, 1)))

    return np.mean(means, axis=0).astype(np.float32)


def get_activations(
    runner: "ModelRunner",
    layer: int,
    prompt: str,
    component: str = "resid_post",
) -> np.ndarray:
    """Get activations [seq_len, d_model] for a prompt."""
    hook_name = f"blocks.{layer}.hook_{component}"
    _, cache = runner.run_with_cache(prompt, names_filter=lambda n: n == hook_name)
    acts = cache[hook_name]
    if isinstance(acts, torch.Tensor):
        acts = acts.detach().cpu().numpy()
    return acts[0].astype(np.float32)


def random_direction(d_model: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random unit direction vector."""
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.randn(d_model).astype(np.float32)
    return vec / np.linalg.norm(vec)
