"""
Model runner for inference with intervention support.

Supports TransformerLens and nnsight backends.

Example:
    runner = ModelRunner("Qwen/Qwen2.5-7B-Instruct")
    output = runner.generate("What is 2+2?")

    # With intervention
    from src.models.intervention_utils import steering
    intervention = steering(layer=26, direction=probe.direction, strength=100.0)
    output = runner.generate("What is 2+2?", intervention=intervention)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from .interventions import Intervention, create_intervention_hook


class ModelBackend(Enum):
    TRANSFORMERLENS = "transformerlens"
    NNSIGHT = "nnsight"
    PYVENE = "pyvene"


@dataclass
class LabelProbsOutput:
    """Probabilities for two label options."""

    prob1: float
    prob2: float


class ModelRunner:
    """Model runner for inference with intervention support."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: ModelBackend = ModelBackend.TRANSFORMERLENS,
    ):
        self.model_name = model_name
        self.backend = backend

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
        self.dtype = dtype

        if backend == ModelBackend.TRANSFORMERLENS:
            self._init_transformerlens()
        elif backend == ModelBackend.NNSIGHT:
            self._init_nnsight()
        elif backend == ModelBackend.PYVENE:
            self._init_pyvene()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._is_chat_model = self._detect_chat_model()
        print(f"Model loaded: {model_name} (chat={self._is_chat_model})")
        print(f"  n_layers={self.n_layers}, d_model={self.d_model}\n")

    def _init_transformerlens(self) -> None:
        from transformer_lens import HookedTransformer

        print(f"Loading {self.model_name} on {self.device} (TransformerLens)...")
        self.model = HookedTransformer.from_pretrained(
            self.model_name, device=self.device, dtype=self.dtype
        )
        self.model.eval()
        self._backend = _TransformerLensBackend(self)

    def _init_nnsight(self) -> None:
        from nnsight import LanguageModel

        print(f"Loading {self.model_name} on {self.device} (nnsight)...")
        self.model = LanguageModel(
            self.model_name, device_map=self.device, torch_dtype=self.dtype
        )
        self._backend = _NNsightBackend(self)

    def _init_pyvene(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} on {self.device} (pyvene)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = _PyveneBackend(self)

    def _detect_chat_model(self) -> bool:
        name = self.model_name.lower()
        return any(x in name for x in ["instruct", "chat", "-it", "rlhf"])

    def _apply_chat_template(self, prompt: str) -> str:
        if not self._is_chat_model:
            return prompt
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    @property
    def tokenizer(self):
        return self._backend.get_tokenizer()

    @property
    def n_layers(self) -> int:
        return self._backend.get_n_layers()

    @property
    def d_model(self) -> int:
        return self._backend.get_d_model()

    def tokenize(self, text: str) -> torch.Tensor:
        return self._backend.tokenize(text)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._backend.decode(token_ids)

    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str | list[str]:
        """Generate text, optionally with intervention."""
        if isinstance(prompt, str):
            return self._generate_single(
                prompt, max_new_tokens, temperature, intervention, past_kv_cache
            )

        return [
            self._generate_single(
                p, max_new_tokens, temperature, intervention, past_kv_cache
            )
            for p in prompt
        ]

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        formatted = self._apply_chat_template(prompt)
        return self._backend.generate(
            formatted, max_new_tokens, temperature, intervention, past_kv_cache
        )

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        return self._backend.generate_from_cache(
            prefill_logits, frozen_kv_cache, max_new_tokens, temperature
        )

    def get_label_probs(
        self,
        prompt: str | list[str],
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple | list[tuple]:
        """Get probabilities for two label options."""
        if isinstance(prompt, str):
            return self._get_label_probs_single(
                prompt, choice_prefix, labels, past_kv_cache
            )
        return [
            self._get_label_probs_single(p, choice_prefix, labels, past_kv_cache)
            for p in prompt
        ]

    def _get_label_probs_single(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple:
        """Get probabilities for two label options.

        Handles both simple labels (a/b, 1/2) and complex multi-token labels
        (OPTION_ONE/OPTION_TWO) by finding where they diverge and looking at
        the appropriate token position.
        """
        tokenizer = self.tokenizer
        label1, label2 = labels

        # Tokenize both labels with space prefix (model outputs space after "I select:")
        ids1 = tokenizer.encode(" " + label1, add_special_tokens=False)
        ids2 = tokenizer.encode(" " + label2, add_special_tokens=False)

        # Find first position where tokens differ
        diverge_pos = 0
        for i in range(min(len(ids1), len(ids2))):
            if ids1[i] != ids2[i]:
                diverge_pos = i
                break
        else:
            # One is prefix of the other
            diverge_pos = min(len(ids1), len(ids2))

        # Get the diverging tokens
        tok1 = ids1[diverge_pos] if diverge_pos < len(ids1) else None
        tok2 = ids2[diverge_pos] if diverge_pos < len(ids2) else None

        # Always use full prompt context (kv_cache doesn't help when we need to extend)
        base_text = self._apply_chat_template(prompt) + choice_prefix

        # If labels diverge at first token, look at next token after choice_prefix
        if diverge_pos == 0:
            probs = self._backend.get_next_token_probs_by_id(
                base_text, [tok1, tok2], past_kv_cache
            )
            return (probs.get(tok1, 0.0), probs.get(tok2, 0.0))

        # Labels have common prefix - need to condition on it
        common_prefix = tokenizer.decode(ids1[:diverge_pos])
        extended_text = base_text + common_prefix
        probs = self._backend.get_next_token_probs_by_id(
            extended_text, [tok1, tok2], past_kv_cache
        )
        return (probs.get(tok1, 0.0), probs.get(tok2, 0.0))

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache."""
        formatted = self._apply_chat_template(prompt)
        input_ids = self.tokenize(formatted)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

    def init_kv_cache(self):
        return self._backend.init_kv_cache()


class _BackendBase(ABC):
    def __init__(self, runner: ModelRunner):
        self.runner = runner

    @abstractmethod
    def get_tokenizer(self): ...
    @abstractmethod
    def get_n_layers(self) -> int: ...
    @abstractmethod
    def get_d_model(self) -> int: ...
    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor: ...
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str: ...
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str: ...
    @abstractmethod
    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]: ...
    @abstractmethod
    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]: ...
    @abstractmethod
    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]: ...

    @abstractmethod
    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    def init_kv_cache(self): ...


class _TransformerLensBackend(_BackendBase):
    def get_tokenizer(self):
        return self.runner.model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner.model.cfg.n_layers

    def get_d_model(self) -> int:
        return self.runner.model.cfg.d_model

    def tokenize(self, text: str) -> torch.Tensor:
        return self.runner.model.to_tokens(text, prepend_bos=True)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.runner.model.to_string(token_ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        # If we have a frozen kv_cache, use custom generation loop
        if past_kv_cache is not None:
            return self._generate_with_cache(
                input_ids, max_new_tokens, temperature, past_kv_cache
            )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "stop_at_eos": True,
            "verbose": False,
            "use_past_kv_cache": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            if intervention is not None:
                hook, _ = create_intervention_hook(
                    intervention,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                    tokenizer=self.get_tokenizer(),
                )
                with self.runner.model.hooks(
                    fwd_hooks=[(intervention.hook_name, hook)]
                ):
                    output_ids = self.runner.model.generate(input_ids, **gen_kwargs)
            else:
                output_ids = self.runner.model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any,
    ) -> str:
        """Generate using frozen kv_cache - only pass new tokens each step."""
        import copy

        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        # Unfreeze a copy of the cache for generation
        kv = copy.deepcopy(past_kv_cache)
        kv.unfreeze()

        # Get first token logits from prefill (cache already has prompt processed)
        # We need to do one forward pass with NO new tokens to get the next-token logits
        # TransformerLens doesn't support empty input, so we use the logits from prefill
        # Actually we need to get the logits that were computed during prefill
        # Since we don't have them, we need to recompute with the full prompt
        logits = self.runner.model(input_ids, past_kv_cache=kv)
        next_logits = logits[0, -1, :]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax().unsqueeze(0)

                generated_ids.append(next_token.item())

                if next_token.item() == eos_token_id:
                    break

                # Pass only the new token
                step_logits = self.runner.model(
                    next_token.unsqueeze(0), past_kv_cache=kv
                )
                next_logits = step_logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            logits = self.runner.model(input_ids, past_kv_cache=past_kv_cache)
        probs = torch.softmax(logits[0, -1, :], dim=-1)

        result = {}
        tokenizer = self.get_tokenizer()
        for token_str in target_tokens:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            result[token_str] = probs[ids[0]].item() if ids else 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            logits = self.runner.model(input_ids, past_kv_cache=past_kv_cache)
        probs = torch.softmax(logits[0, -1, :], dim=-1)

        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            return self.runner.model.run_with_cache(
                input_ids, names_filter=names_filter, past_kv_cache=past_kv_cache
            )

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        import copy

        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        # Copy and unfreeze the cache for generation
        kv = copy.deepcopy(frozen_kv_cache)
        kv.unfreeze()

        # Start from the prefill logits (already computed in Step 1)
        next_logits = prefill_logits[0, -1, :]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax().unsqueeze(0)

                generated_ids.append(next_token.item())

                if next_token.item() == eos_token_id:
                    break

                # Pass only the new token - kv cache supplies the prefix context
                step_logits = self.runner.model(
                    next_token.unsqueeze(0), past_kv_cache=kv
                )
                next_logits = step_logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def init_kv_cache(self):
        from transformer_lens.past_key_value_caching import (
            HookedTransformerKeyValueCache,
        )

        return HookedTransformerKeyValueCache.init_cache(
            self.runner.model.cfg,
            device=self.runner.device,
            batch_size=1,
        )


class _NNsightBackend(_BackendBase):
    def __init__(self, runner):
        super().__init__(runner)
        # Detect model architecture for layer access
        # GPT2: model.transformer.h[i], LLaMA/Mistral: model.model.layers[i]
        if hasattr(self.runner.model, "transformer"):
            self._layers = self.runner.model.transformer.h
        elif hasattr(self.runner.model, "model") and hasattr(
            self.runner.model.model, "layers"
        ):
            self._layers = self.runner.model.model.layers
        else:
            raise ValueError(f"Unknown model architecture: {type(self.runner.model)}")

    def get_tokenizer(self):
        return self.runner.model.tokenizer

    def get_n_layers(self) -> int:
        return self.runner.model.config.num_hidden_layers

    def get_d_model(self) -> int:
        return self.runner.model.config.hidden_size

    def tokenize(self, text: str) -> torch.Tensor:
        # Prepend BOS token to match TransformerLens behavior
        tokenizer = self.get_tokenizer()
        ids = tokenizer(text, return_tensors="pt").input_ids
        bos_id = tokenizer.bos_token_id
        if bos_id is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos_id):
            bos = torch.tensor([[bos_id]], dtype=ids.dtype)
            ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        # Note: Do NOT use torch.no_grad() - it interferes with nnsight's tracing
        generated = input_ids.clone()

        # Prepare steering if needed
        steering_layer = None
        steering_direction = None
        if intervention is not None and isinstance(intervention, Intervention) and intervention.mode == "add":
            steering_layer = self._layers[intervention.layer]
            steering_direction = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

        for _ in range(max_new_tokens):
            with self.runner.model.trace(generated):
                if steering_layer is not None:
                    steering_layer.output[0][:, :, :] += steering_direction
                logits = self.runner.model.lm_head.output.save()

            # nnsight 0.5.15 returns tensors directly (no .value)
            if temperature > 0:
                probs = torch.softmax(logits[0, -1, :].detach() / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
            else:
                next_token = logits[0, -1, :].detach().argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        # Note: Do NOT use torch.no_grad() - it interferes with nnsight's tracing
        with self.runner.model.trace(input_ids):
            logits = self.runner.model.lm_head.output.save()

        # nnsight 0.5.15 returns tensors directly (no .value)
        probs = torch.softmax(logits[0, -1, :].detach(), dim=-1)
        result = {}
        tokenizer = self.get_tokenizer()
        for token_str in target_tokens:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            result[token_str] = probs[ids[0]].item() if ids else 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.tokenize(prompt)
        # Note: Do NOT use torch.no_grad() - it interferes with nnsight's tracing
        with self.runner.model.trace(input_ids):
            logits = self.runner.model.lm_head.output.save()

        # nnsight 0.5.15 returns tensors directly (no .value)
        probs = torch.softmax(logits[0, -1, :].detach(), dim=-1)
        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        cache = {}
        with self.runner.model.trace(input_ids):
            for i, layer in enumerate(self._layers):
                name = f"blocks.{i}.hook_resid_post"
                if names_filter is None or names_filter(name):
                    cache[name] = layer.output[0].save()
            logits = self.runner.model.lm_head.output.save()
        # nnsight 0.5.15 returns tensors directly (no .value)
        return logits, {k: v for k, v in cache.items()}

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Not implemented for nnsight backend."""
        raise NotImplementedError(
            "generate_from_cache not supported for nnsight backend"
        )

    def init_kv_cache(self):
        pass


class _PyveneBackend(_BackendBase):
    """Backend using pyvene for interventions."""

    def __init__(self, runner):
        super().__init__(runner)
        # Detect model architecture for layer access
        if hasattr(self.runner.model, "transformer"):
            # GPT2 style: model.transformer.h[i]
            self._layers_attr = "transformer.h"
            self._layers = self.runner.model.transformer.h
            self._n_layers = len(self._layers)
            self._d_model = self.runner.model.config.n_embd
        elif hasattr(self.runner.model, "gpt_neox"):
            # Pythia/GPT-NeoX style: model.gpt_neox.layers[i]
            self._layers_attr = "gpt_neox.layers"
            self._layers = self.runner.model.gpt_neox.layers
            self._n_layers = len(self._layers)
            self._d_model = self.runner.model.config.hidden_size
        elif hasattr(self.runner.model, "model") and hasattr(
            self.runner.model.model, "layers"
        ):
            # LLaMA/Mistral style: model.model.layers[i]
            self._layers_attr = "model.layers"
            self._layers = self.runner.model.model.layers
            self._n_layers = len(self._layers)
            self._d_model = self.runner.model.config.hidden_size
        else:
            raise ValueError(f"Unknown model architecture: {type(self.runner.model)}")

    def get_tokenizer(self):
        return self.runner._tokenizer

    def get_n_layers(self) -> int:
        return self._n_layers

    def get_d_model(self) -> int:
        return self._d_model

    def tokenize(self, text: str) -> torch.Tensor:
        tokenizer = self.get_tokenizer()
        ids = tokenizer(text, return_tensors="pt").input_ids
        # Prepend BOS token to match TransformerLens behavior
        bos_id = tokenizer.bos_token_id
        if bos_id is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos_id):
            bos = torch.tensor([[bos_id]], dtype=ids.dtype)
            ids = torch.cat([bos, ids], dim=1)
        return ids.to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.get_tokenizer().decode(token_ids, skip_special_tokens=True)

    def _get_component_name(self, layer: int, component: str = "block_output") -> str:
        """Get pyvene component name for a layer."""
        # Map our naming to pyvene's expected format
        if self._layers_attr == "transformer.h":
            # GPT2: use h[layer] style
            if component == "block_output":
                return f"transformer.h[{layer}]"
            elif component == "mlp_output":
                return f"transformer.h[{layer}].mlp"
        elif self._layers_attr == "gpt_neox.layers":
            # Pythia/GPT-NeoX: use gpt_neox.layers[layer] style
            if component == "block_output":
                return f"gpt_neox.layers[{layer}]"
            elif component == "mlp_output":
                return f"gpt_neox.layers[{layer}].mlp"
        else:
            # LLaMA style: use model.layers[layer] style
            if component == "block_output":
                return f"model.layers[{layer}]"
            elif component == "mlp_output":
                return f"model.layers[{layer}].mlp"
        return f"transformer.h[{layer}]"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        if intervention is not None and isinstance(intervention, Intervention) and intervention.mode == "add":
            # Use direct PyTorch hooks for steering (pyvene's subspace projection
            # doesn't work well for direct addition to hidden states)
            direction = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

            # Get the layer to hook
            layer_module = self._layers[intervention.layer]

            # Create steering hook
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    steered = hidden + direction.unsqueeze(0).unsqueeze(0)
                    return (steered,) + output[1:]
                else:
                    return output + direction.unsqueeze(0).unsqueeze(0)

            # Generate token by token with intervention
            generated = input_ids.clone()
            eos_id = self.get_tokenizer().eos_token_id

            for _ in range(max_new_tokens):
                # Register hook for this forward pass
                hook = layer_module.register_forward_hook(steering_hook)

                with torch.no_grad():
                    outputs = self.runner.model(generated)
                    logits = outputs.logits

                hook.remove()

                if temperature > 0:
                    probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).unsqueeze(0)
                else:
                    next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == eos_id:
                    break
        else:
            # No intervention - use standard HF generate
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = self.runner.model.generate(input_ids, **gen_kwargs)
            generated = output_ids

        return self.decode(generated[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            outputs = self.runner.model(input_ids)
            logits = outputs.logits

        probs = torch.softmax(logits[0, -1, :], dim=-1)
        result = {}
        tokenizer = self.get_tokenizer()
        for token_str in target_tokens:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            result[token_str] = probs[ids[0]].item() if ids else 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            outputs = self.runner.model(input_ids)
            logits = outputs.logits

        probs = torch.softmax(logits[0, -1, :], dim=-1)
        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        import pyvene as pv

        cache = {}

        # Determine which layers to collect
        layers_to_collect = []
        for i in range(self._n_layers):
            name = f"blocks.{i}.hook_resid_post"
            if names_filter is None or names_filter(name):
                layers_to_collect.append((i, name))

        if layers_to_collect:
            # Create collect intervention config for all requested layers
            # Use pyvene's abstract component name "block_output"
            representations = [
                pv.RepresentationConfig(
                    layer=layer_idx,
                    component="block_output",
                    unit="pos",
                    max_number_of_units=input_ids.shape[1],
                    intervention_link_key=idx,
                )
                for idx, (layer_idx, _) in enumerate(layers_to_collect)
            ]

            config = pv.IntervenableConfig(
                model_type=type(self.runner.model),
                representations=representations,
                intervention_types=pv.CollectIntervention,
            )
            intervenable = pv.IntervenableModel(config, self.runner.model)
            intervenable.set_device(self.runner.device)

            # Run and collect
            # pyvene returns ((intervened_output, [collected_tensors]), original_output)
            result, original_output = intervenable(
                {"input_ids": input_ids},
                output_original_output=True,
            )
            # result is (intervened_output, list_of_collected_tensors)
            # result[1] contains the actual collected activation tensors
            collected_tensors = result[1]

            # Map collected activations to cache
            for idx, (layer_idx, name) in enumerate(layers_to_collect):
                if idx < len(collected_tensors):
                    cache[name] = collected_tensors[idx]

            logits = original_output.logits
        else:
            # No collection needed, just forward pass
            with torch.no_grad():
                outputs = self.runner.model(input_ids)
            logits = outputs.logits

        return logits, cache

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen KV cache."""
        # pyvene uses HF models which support past_key_values
        eos_token_id = self.get_tokenizer().eos_token_id
        generated_ids = []

        next_logits = prefill_logits[0, -1, :]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax().unsqueeze(0)

                generated_ids.append(next_token.item())

                if next_token.item() == eos_token_id:
                    break

                # Forward with KV cache
                outputs = self.runner.model(
                    next_token.unsqueeze(0),
                    past_key_values=frozen_kv_cache,
                    use_cache=True,
                )
                next_logits = outputs.logits[0, -1, :]

        return self.decode(torch.tensor(generated_ids))

    def init_kv_cache(self):
        """Initialize a KV cache wrapper for HF models."""
        # Return a simple wrapper that stores past_key_values
        class HFKVCache:
            def __init__(self):
                self.past_key_values = None
                self._frozen = False

            def freeze(self):
                self._frozen = True

            def unfreeze(self):
                self._frozen = False

            @property
            def frozen(self):
                return self._frozen

        return HFKVCache()
