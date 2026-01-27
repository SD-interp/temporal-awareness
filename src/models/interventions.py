"""Activation interventions for modifying model behavior during inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch

from ..common.schema_utils import SchemaClass

Mode = Literal["add", "set", "mul"]
Axis = Literal["all", "position", "neuron", "pattern"]


@dataclass
class Target(SchemaClass):
    """Where to apply an intervention."""

    axis: Axis = "all"
    positions: Optional[list[int]] = None
    neurons: Optional[list[int]] = None
    pattern: Optional[str] = None

    def __post_init__(self):
        if self.axis == "position" and not self.positions:
            raise ValueError("positions required for axis='position'")
        if self.axis == "neuron" and not self.neurons:
            raise ValueError("neurons required for axis='neuron'")
        if self.axis == "pattern" and not self.pattern:
            raise ValueError("pattern required for axis='pattern'")
        super().__post_init__()

    @classmethod
    def all(cls) -> Target:
        return cls(axis="all")

    @classmethod
    def at_positions(cls, positions: Union[int, list[int]]) -> Target:
        if isinstance(positions, int):
            positions = [positions]
        return cls(axis="position", positions=positions)

    @classmethod
    def at_neurons(cls, neurons: Union[int, list[int]]) -> Target:
        if isinstance(neurons, int):
            neurons = [neurons]
        return cls(axis="neuron", neurons=neurons)

    @classmethod
    def on_pattern(cls, pattern: str) -> Target:
        return cls(axis="pattern", pattern=pattern)


@dataclass
class Intervention(SchemaClass):
    """Intervention config: layer, mode (add/set/mul), values, target."""

    layer: int
    mode: Mode
    values: np.ndarray
    target: Target = field(default_factory=Target.all)
    component: str = "resid_post"
    strength: float = 1.0

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)
        super().__post_init__()

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_{self.component}"

    @property
    def scaled_values(self) -> np.ndarray:
        return self.values * self.strength


class PatternMatcher:
    """Tracks generated tokens and signals when pattern is matched."""

    def __init__(self, pattern: str, tokenizer):
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.generated_text = ""
        self._triggered = False

    def update(self, token_ids: torch.Tensor):
        if self._triggered:
            return
        self.generated_text += self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if self.pattern in self.generated_text:
            self._triggered = True

    def should_apply(self) -> bool:
        return self._triggered

    def mark_applied(self):
        self._triggered = False


def create_intervention_hook(
    config: Intervention,
    dtype: torch.dtype,
    device: str,
    tokenizer=None,
) -> tuple[Callable, Optional[PatternMatcher]]:
    """Create a forward hook for the intervention."""
    values = torch.tensor(config.scaled_values, dtype=dtype, device=device)
    target = config.target
    mode = config.mode

    if target.axis == "pattern":
        if tokenizer is None:
            raise ValueError("tokenizer required for pattern targeting")
        matcher = PatternMatcher(target.pattern, tokenizer)

        def hook(act, hook=None):
            if matcher.should_apply():
                act = _apply(act, values, mode)
                matcher.mark_applied()
            return act

        return hook, matcher

    if target.axis == "all":
        return lambda act, hook=None: _apply(act, values, mode), None

    if target.axis == "position":
        positions = target.positions

        def hook(act, hook=None):
            for pos in positions:
                if pos < act.shape[1]:
                    act[:, pos] = _apply_slice(act[:, pos], values, mode)
            return act

        return hook, None

    if target.axis == "neuron":
        neurons = target.neurons
        v = values.item() if values.numel() == 1 else values

        def hook(act, hook=None):
            for n in neurons:
                if n < act.shape[-1]:
                    val = v[n] if hasattr(v, "__getitem__") and n < len(v) else v
                    act[:, :, n] = _apply_scalar(act[:, :, n], val, mode)
            return act

        return hook, None

    raise ValueError(f"Unknown axis: {target.axis}")


def _apply(act: torch.Tensor, values: torch.Tensor, mode: Mode) -> torch.Tensor:
    if mode == "add":
        return act + values
    if mode == "mul":
        return act * values
    # set
    if values.dim() == 1:
        return values.expand_as(act)
    seq = min(act.shape[1], values.shape[0])
    result = act.clone()
    result[:, :seq] = values[:seq].unsqueeze(0).expand(act.shape[0], -1, -1)
    return result


def _apply_slice(act: torch.Tensor, values: torch.Tensor, mode: Mode) -> torch.Tensor:
    v = values[-1] if values.dim() > 1 else values
    if mode == "add":
        return act + v
    if mode == "mul":
        return act * v
    return v.expand_as(act)


def _apply_scalar(act: torch.Tensor, value, mode: Mode) -> torch.Tensor:
    if mode == "add":
        return act + value
    if mode == "mul":
        return act * value
    v = value if isinstance(value, float) else value.item()
    return torch.full_like(act, v)
