"""Tests for intervention JSON loading and helpers."""

import pytest
import numpy as np

from src.models.intervention_utils import (
    steering,
    ablation,
    patch,
    scale,
    compute_mean_activations,
    get_activations,
    random_direction,
)
from src.models.intervention_loader import (
    load_intervention,
    load_intervention_json,
    load_intervention_from_dict,
    list_sample_interventions,
)
from src.models.interventions import Intervention


TEST_MODEL = "gpt2"


# =============================================================================
# Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test intervention_utils helper functions."""

    def test_steering_creates_add_intervention(self):
        """steering() creates add mode intervention."""
        direction = np.random.randn(768)
        i = steering(layer=5, direction=direction, strength=50.0)
        assert i.mode == "add"
        assert i.strength == 50.0
        assert i.target.axis == "all"

    def test_steering_normalizes_direction(self):
        """Direction is normalized to unit length."""
        direction = np.array([3.0, 4.0])
        i = steering(layer=5, direction=direction)
        norm = np.linalg.norm(i.values)
        assert abs(norm - 1.0) < 1e-5

    def test_steering_with_positions(self):
        """steering() with position targeting."""
        i = steering(layer=5, direction=[1, 0], positions=[1, 2, 3])
        assert i.target.axis == "position"
        assert i.target.positions == [1, 2, 3]

    def test_ablation_zero_default(self):
        """ablation() defaults to zero."""
        i = ablation(layer=3)
        assert i.mode == "set"
        assert i.values[0] == 0

    def test_ablation_with_values(self):
        """ablation() with custom values."""
        values = np.array([1.0, 2.0, 3.0])
        i = ablation(layer=3, values=values)
        np.testing.assert_array_equal(i.values, values)

    def test_ablation_with_positions(self):
        """ablation() with position targeting."""
        i = ablation(layer=3, positions=[5, 6, 7])
        assert i.target.axis == "position"

    def test_patch_creates_set_intervention(self):
        """patch() creates set mode intervention."""
        values = np.random.randn(10, 768)
        i = patch(layer=4, values=values)
        assert i.mode == "set"
        assert i.values.shape == (10, 768)

    def test_scale_creates_mul_intervention(self):
        """scale() creates mul mode intervention."""
        i = scale(layer=5, factor=0.5)
        assert i.mode == "mul"
        assert i.values[0] == 0.5

    def test_scale_with_neurons(self):
        """scale() with neuron targeting."""
        i = scale(layer=5, factor=2.0, neurons=[0, 50, 100])
        assert i.target.axis == "neuron"
        assert i.target.neurons == [0, 50, 100]

    def test_random_direction(self):
        """random_direction() creates unit vector."""
        d = random_direction(768)
        assert d.shape == (768,)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-5

    def test_random_direction_with_seed(self):
        """random_direction() is reproducible with seed."""
        d1 = random_direction(768, seed=42)
        d2 = random_direction(768, seed=42)
        np.testing.assert_array_equal(d1, d2)


# =============================================================================
# JSON Loading
# =============================================================================


class TestJSONLoading:
    """Test intervention loading from JSON."""

    @pytest.fixture(scope="class")
    def runner(self):
        from src.models.model_runner import ModelRunner, ModelBackend
        runner = ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)
        yield runner

    def test_list_sample_interventions(self):
        """Sample directory has expected configs."""
        samples = list_sample_interventions()
        assert len(samples) >= 10
        assert "steer_all" in samples
        assert "zero_all" in samples
        assert "scale_all" in samples

    @pytest.mark.parametrize("config_name", list_sample_interventions())
    def test_json_is_valid(self, config_name):
        """Each JSON file parses correctly."""
        data = load_intervention_json(config_name)
        assert "layer" in data
        assert "mode" in data

    @pytest.mark.parametrize("config_name", list_sample_interventions())
    def test_config_loads(self, config_name, runner):
        """Each config creates valid Intervention."""
        intervention = load_intervention(config_name, runner)
        assert isinstance(intervention, Intervention)
        assert 0 <= intervention.layer < runner.n_layers

    def test_steer_all_loads(self, runner):
        """steer_all.json loads correctly."""
        i = load_intervention("steer_all", runner)
        assert i.mode == "add"
        assert i.target.axis == "all"
        assert i.strength == 50.0
        assert len(i.values) == runner.d_model

    def test_zero_all_loads(self, runner):
        """zero_all.json loads correctly."""
        i = load_intervention("zero_all", runner)
        assert i.mode == "set"
        assert i.values[0] == 0

    def test_mean_all_loads(self, runner):
        """mean_all.json computes mean from model."""
        i = load_intervention("mean_all", runner)
        assert i.mode == "set"
        assert len(i.values) == runner.d_model

    def test_scale_all_loads(self, runner):
        """scale_all.json loads correctly."""
        i = load_intervention("scale_all", runner)
        assert i.mode == "mul"
        assert i.values[0] == 0.5

    def test_position_target_loads(self, runner):
        """Position targeting loads correctly."""
        i = load_intervention("steer_positions", runner)
        assert i.target.axis == "position"
        assert i.target.positions == [2, 4, 6]

    def test_neuron_target_loads(self, runner):
        """Neuron targeting loads correctly."""
        i = load_intervention("steer_neurons", runner)
        assert i.target.axis == "neuron"
        assert i.target.neurons == [0, 100, 200]

    def test_pattern_target_loads(self, runner):
        """Pattern targeting loads correctly."""
        i = load_intervention("steer_pattern", runner)
        assert i.target.axis == "pattern"
        assert i.target.pattern == "answer:"

    def test_load_from_dict(self, runner):
        """load_intervention_from_dict works."""
        data = {"layer": 5, "mode": "add", "values": "random", "strength": 10.0}
        i = load_intervention_from_dict(data, runner)
        assert i.layer == 5
        assert i.mode == "add"
        assert i.strength == 10.0


# =============================================================================
# Activation Helpers
# =============================================================================


class TestActivationHelpers:
    """Test activation computation helpers."""

    @pytest.fixture(scope="class")
    def runner(self):
        from src.models.model_runner import ModelRunner, ModelBackend
        runner = ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)
        yield runner

    def test_compute_mean_activations(self, runner):
        """compute_mean_activations returns correct shape."""
        mean = compute_mean_activations(runner, layer=5, prompts="Hello world")
        assert mean.shape == (runner.d_model,)
        assert mean.dtype == np.float32

    def test_compute_mean_multiple_prompts(self, runner):
        """Mean over multiple prompts."""
        prompts = ["Hello world", "Goodbye world", "Testing testing"]
        mean = compute_mean_activations(runner, layer=5, prompts=prompts)
        assert mean.shape == (runner.d_model,)

    def test_get_activations(self, runner):
        """get_activations returns correct shape."""
        acts = get_activations(runner, layer=5, prompt="Hello world")
        assert acts.ndim == 2  # (seq_len, d_model)
        assert acts.shape[1] == runner.d_model
        assert acts.dtype == np.float32


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def runner(self):
        from src.models.model_runner import ModelRunner, ModelBackend
        runner = ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)
        yield runner

    def test_nonexistent_config_raises(self, runner):
        """Loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_intervention("nonexistent_config", runner)

    def test_layer_clamped_to_valid_range(self, runner):
        """Layer numbers clamped to valid range."""
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"layer": 999, "mode": "add", "values": "random"}, f)
            temp_path = f.name

        i = load_intervention(temp_path, runner)
        assert i.layer < runner.n_layers

        import os
        os.unlink(temp_path)

    def test_invalid_values_spec_raises(self, runner):
        """Invalid values specification raises error."""
        with pytest.raises(ValueError):
            data = {"layer": 5, "mode": "add", "values": "invalid_spec"}
            load_intervention_from_dict(data, runner)

    def test_empty_values_list(self, runner):
        """Empty values list handled."""
        data = {"layer": 5, "mode": "set", "values": []}
        i = load_intervention_from_dict(data, runner)
        assert len(i.values) == 0


