"""Tests for model interventions.

Creates a toy model with known weights to verify interventions have exact effects.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.interventions import (
    Intervention,
    Target,
    PatternMatcher,
    create_intervention_hook,
)
from src.models.intervention_utils import steering, ablation, patch, scale


# =============================================================================
# Toy Model - fully defined weights for exact verification
# =============================================================================


class ToyTransformer(nn.Module):
    """Minimal transformer with hooks for testing interventions.

    Architecture: embed -> block0 (linear) -> block1 (linear) -> unembed
    All weights are identity/ones for predictable outputs.
    """

    def __init__(self, d_model: int = 4, n_layers: int = 2, vocab_size: int = 10):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Simple embeddings: token i -> [i, i, i, i]
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight.data = torch.arange(vocab_size).float().unsqueeze(1).expand(-1, d_model)

        # Identity layers - output = input (so we can track exact values)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        # Unembed: sum across d_model -> logits
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.unembed.weight.data = torch.ones(vocab_size, d_model)

        self._hooks = {}

    def register_intervention_hook(self, name: str, hook_fn):
        """Register hook by name (e.g., 'blocks.0.hook_resid_post')."""
        self._hooks[name] = hook_fn

    def clear_hooks(self):
        self._hooks = {}

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass returning (logits, cache of activations)."""
        cache = {}
        x = self.embed(input_ids)  # [batch, seq, d_model]

        for i, block in enumerate(self.blocks):
            x = block(x)
            hook_name = f"blocks.{i}.hook_resid_post"
            cache[hook_name] = x.clone()

            # Apply intervention hook if registered
            if hook_name in self._hooks:
                x = self._hooks[hook_name](x)

        logits = self.unembed(x)  # [batch, seq, vocab]
        return logits, cache


class FakeTokenizer:
    """Tokenizer for pattern matching tests."""

    def __init__(self):
        self.eos_token_id = 0
        # Maps: 'I'->1, ' '->2, 's'->3, 'e'->4, 'l'->5, 'c'->6, 't'->7, ':'->8
        self._vocab = {c: i for i, c in enumerate("I select:")}

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self._vocab.items()}
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(inv.get(i, "?") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self._vocab.get(c, 0) for c in text]


# Test constants
D_MODEL = 4
DTYPE = torch.float32
DEVICE = "cpu"


def make_direction(d_model: int = D_MODEL) -> np.ndarray:
    """Unit direction vector [1/sqrt(d), 1/sqrt(d), ...]."""
    vec = np.ones(d_model)
    return vec / np.linalg.norm(vec)


# =============================================================================
# Target Tests
# =============================================================================


class TestTarget:
    def test_all(self):
        t = Target.all()
        assert t.axis == "all"

    def test_at_positions_single(self):
        t = Target.at_positions(3)
        assert t.axis == "position"
        assert t.positions == [3]

    def test_at_positions_multiple(self):
        t = Target.at_positions([1, 3, 5])
        assert t.positions == [1, 3, 5]

    def test_at_neurons(self):
        t = Target.at_neurons([0, 2])
        assert t.axis == "neuron"
        assert t.neurons == [0, 2]

    def test_on_pattern(self):
        t = Target.on_pattern("I select:")
        assert t.axis == "pattern"
        assert t.pattern == "I select:"

    def test_validation(self):
        with pytest.raises(ValueError):
            Target(axis="position")  # missing positions
        with pytest.raises(ValueError):
            Target(axis="neuron")  # missing neurons
        with pytest.raises(ValueError):
            Target(axis="pattern")  # missing pattern


# =============================================================================
# Intervention Tests with Toy Model
# =============================================================================


class TestSteeringWithModel:
    """Test steering interventions produce exact expected outputs."""

    def test_apply_to_all_changes_all_positions(self):
        """Steering with all positions adds direction to every position."""
        model = ToyTransformer(d_model=D_MODEL)
        input_ids = torch.tensor([[1, 2, 3]])  # 3 tokens

        # Get baseline output
        base_logits, base_cache = model(input_ids)

        # Apply steering
        direction = make_direction(D_MODEL)
        strength = 2.0
        intervention = steering(layer=0, direction=direction, strength=strength)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)
        model.register_intervention_hook("blocks.0.hook_resid_post", hook)

        steered_logits, _ = model(input_ids)
        model.clear_hooks()

        # Verify: all positions shifted, logits changed
        # Logits should increase because activations increased
        assert (steered_logits > base_logits).all()

    def test_apply_to_position_selective(self):
        """Steering with specific positions only affects those positions."""
        model = ToyTransformer(d_model=D_MODEL)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5 tokens

        direction = make_direction(D_MODEL)
        intervention = steering(
            layer=0,
            direction=direction,
            strength=10.0,
            positions=2,
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        # Track which positions are modified
        activation = torch.zeros(1, 5, D_MODEL)
        result = hook(activation.clone())

        # Position 2 modified, others unchanged
        assert result[0, 2].sum() > 0
        assert result[0, 0].sum() == 0
        assert result[0, 1].sum() == 0


class TestAblationWithModel:
    """Test ablation interventions produce exact expected outputs."""

    def test_zero_ablation_zeros_activations(self):
        """Zero ablation sets all activations to zero."""
        model = ToyTransformer(d_model=D_MODEL)
        input_ids = torch.tensor([[1, 2, 3]])

        intervention = ablation(layer=0)  # defaults to zero ablation
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)
        model.register_intervention_hook("blocks.0.hook_resid_post", hook)

        logits, _ = model(input_ids)
        model.clear_hooks()

        # With zero activations, unembed produces zeros
        assert torch.allclose(logits, torch.zeros_like(logits))

    def test_mean_ablation_sets_to_mean(self):
        """Mean ablation sets activations to provided mean values."""
        mean_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        intervention = ablation(layer=0, values=mean_val)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        # All positions should have mean values
        expected = torch.tensor(mean_val)
        for pos in range(3):
            assert torch.allclose(result[0, pos], expected)

    def test_ablation_apply_to_position(self):
        """Position-targeted ablation only affects specified positions."""
        intervention = ablation(
            layer=0,
            values=0,
            positions=[1, 3],
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.ones(1, 5, D_MODEL)
        result = hook(activation)

        # Positions 1 and 3 zeroed, others untouched
        assert result[0, 1].sum() == 0
        assert result[0, 3].sum() == 0
        assert result[0, 0].sum() == D_MODEL
        assert result[0, 2].sum() == D_MODEL


class TestPatchingWithModel:
    """Test activation patching with exact values."""

    def test_patching_replaces_activations(self):
        """Patching replaces activations with cached values."""
        patch_vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)

        intervention = patch(layer=0, values=patch_vals)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        expected = torch.tensor(patch_vals)
        assert torch.allclose(result[0], expected)

    def test_patching_position_targeted(self):
        """Position-targeted patching only replaces specified positions."""
        patch_vals = np.array([[100, 100, 100, 100]], dtype=np.float32)

        intervention = patch(
            layer=0,
            values=patch_vals,
            positions=0,
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.ones(1, 3, D_MODEL)
        result = hook(activation)

        # Only position 0 patched
        assert result[0, 0].sum() == 400
        assert result[0, 1].sum() == D_MODEL


class TestPatternMatching:
    """Test pattern-based interventions."""

    def test_pattern_matcher_basic(self):
        """Pattern matcher detects pattern in generated text."""
        tokenizer = FakeTokenizer()
        matcher = PatternMatcher("I select:", tokenizer)

        assert not matcher.should_apply()

        # Simulate generation
        for char in "I select:":
            ids = tokenizer.encode(char)
            matcher.update(torch.tensor(ids))

        assert matcher.should_apply()

    def test_steering_with_pattern_deferred(self):
        """Pattern steering only applies when pattern matches."""
        tokenizer = FakeTokenizer()
        intervention = steering(
            layer=0,
            direction=make_direction(),
            strength=10.0,
            pattern="test",
        )

        hook, matcher = create_intervention_hook(intervention, DTYPE, DEVICE, tokenizer)

        assert matcher is not None
        # Hook exists but pattern not matched yet
        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)
        assert torch.allclose(result, activation)  # No change


# =============================================================================
# Hook Creation Tests
# =============================================================================


class TestHookCreation:
    """Test create_intervention_hook produces correct hooks."""

    def test_steering_hook(self):
        steer = steering(layer=0, direction=make_direction(D_MODEL), strength=10.0)
        hook, matcher = create_intervention_hook(steer, DTYPE, DEVICE)

        assert matcher is None  # No pattern
        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)
        assert result.sum() > 0

    def test_ablation_hook(self):
        ablate = ablation(layer=0)
        hook, _ = create_intervention_hook(ablate, DTYPE, DEVICE)

        activation = torch.ones(1, 3, D_MODEL)
        result = hook(activation)
        assert result.sum() == 0

    def test_patching_hook(self):
        patch_vals = np.ones((3, D_MODEL), dtype=np.float32) * 5
        patch_intervention = patch(layer=0, values=patch_vals)
        hook, _ = create_intervention_hook(patch_intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)
        assert torch.allclose(result, torch.tensor(patch_vals).unsqueeze(0))


class TestConfigValidation:
    """Test validation in Intervention configs."""

    def test_intervention_requires_layer_and_mode(self):
        with pytest.raises(TypeError):
            Intervention(mode="add", values=np.array([1.0]))  # missing layer

    def test_target_validation(self):
        with pytest.raises(ValueError):
            Target(axis="position")  # positions required
        with pytest.raises(ValueError):
            Target(axis="pattern")  # pattern required

    def test_component_in_hook_name(self):
        intervention = steering(layer=5, direction=make_direction(), component="attn_out")
        assert intervention.hook_name == "blocks.5.hook_attn_out"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_steering_zero_strength(self):
        """Zero strength has no effect."""
        steer = steering(layer=0, direction=make_direction(), strength=0.0)
        hook, _ = create_intervention_hook(steer, DTYPE, DEVICE)

        activation = torch.ones(1, 3, D_MODEL)
        result = hook(activation)
        assert torch.allclose(result, activation)

    def test_position_out_of_bounds(self):
        """Out-of-bounds positions are skipped."""
        intervention = steering(
            layer=0,
            direction=make_direction(),
            strength=10.0,
            positions=100,  # way beyond seq_len=3
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.ones(1, 3, D_MODEL)
        result = hook(activation)
        # Out of bounds position skipped, activation unchanged
        assert torch.allclose(result, activation)

    def test_multiple_positions(self):
        """Multiple positions all modified."""
        intervention = steering(
            layer=0,
            direction=make_direction(),
            strength=10.0,
            positions=[0, 2, 4],
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 5, D_MODEL)
        result = hook(activation)

        assert result[0, 0].sum() > 0
        assert result[0, 2].sum() > 0
        assert result[0, 4].sum() > 0
        assert result[0, 1].sum() == 0
        assert result[0, 3].sum() == 0

    def test_mean_ablation_single_position(self):
        """Mean ablation at single position."""
        mean_val = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        intervention = ablation(
            layer=0,
            values=mean_val,
            positions=1,
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        assert result[0, 1].sum() == 20  # 4 * 5
        assert result[0, 0].sum() == 0
        assert result[0, 2].sum() == 0


class TestSteeringMagnitude:
    """Test steering behavior with various strengths."""

    def test_increasing_strength_increases_effect(self):
        """Higher strength produces larger activation change."""
        intervention = steering(layer=0, direction=make_direction(), strength=100.0)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        # Strength 100 should produce large values
        assert result.abs().max() > 10

    def test_negative_strength(self):
        """Negative strength subtracts direction."""
        intervention = steering(layer=0, direction=make_direction(), strength=-2.0)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        # Should be negative
        assert (result < 0).all()


class TestScaling:
    """Test scaling interventions."""

    def test_scale_doubles_activations(self):
        """Scaling by 2 doubles activations."""
        intervention = scale(layer=0, factor=2.0)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.ones(1, 3, D_MODEL)
        result = hook(activation)
        assert torch.allclose(result, activation * 2)

    def test_scale_zeros_activations(self):
        """Scaling by 0 zeros activations."""
        intervention = scale(layer=0, factor=0.0)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.ones(1, 3, D_MODEL)
        result = hook(activation)
        assert result.sum() == 0


class TestPatchingEdgeCases:
    """Edge cases for patching."""

    def test_patching_full_sequence(self):
        """Patching replaces entire sequence."""
        cached_act = np.random.randn(5, D_MODEL).astype(np.float32)
        intervention = patch(layer=0, values=cached_act)
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 5, D_MODEL)
        result = hook(activation)

        expected = torch.tensor(cached_act)
        assert torch.allclose(result[0], expected)

    def test_patching_single_position(self):
        """Patching at single position."""
        cached_act = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        intervention = patch(
            layer=0,
            values=cached_act,
            positions=0,
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        assert result[0, 0].sum() == 10  # 1+2+3+4
        assert result[0, 1].sum() == 0
        assert result[0, 2].sum() == 0


class TestInterventionProperty:
    """Test Intervention class properties."""

    def test_hook_name(self):
        intervention = steering(layer=5, direction=make_direction())
        assert intervention.hook_name == "blocks.5.hook_resid_post"

    def test_scaled_values(self):
        direction = make_direction()
        intervention = steering(layer=0, direction=direction, strength=10.0)
        expected = direction * 10.0
        np.testing.assert_allclose(intervention.scaled_values, expected, rtol=1e-5)


class TestNeuronTargeting:
    """Test neuron-level targeting."""

    def test_steering_specific_neurons(self):
        """Steering can target specific neurons."""
        intervention = steering(
            layer=0,
            direction=make_direction(),
            strength=10.0,
            neurons=[0, 2],
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        # Only neurons 0 and 2 should be modified
        assert result[0, 0, 0] != 0  # neuron 0
        assert result[0, 0, 1] == 0  # neuron 1
        assert result[0, 0, 2] != 0  # neuron 2
        assert result[0, 0, 3] == 0  # neuron 3

    def test_ablation_specific_neurons(self):
        """Ablation can target specific neurons."""
        intervention = ablation(
            layer=0,
            values=99.0,
            neurons=[1],
        )
        hook, _ = create_intervention_hook(intervention, DTYPE, DEVICE)

        activation = torch.zeros(1, 3, D_MODEL)
        result = hook(activation)

        # Only neuron 1 should be set to 99
        assert result[0, 0, 1] == 99.0
        assert result[0, 0, 0] == 0
        assert result[0, 0, 2] == 0
