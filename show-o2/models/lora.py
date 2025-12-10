# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoRA (Low-Rank Adaptation) implementation for Qwen2 LLM backbone in Show-o2.

This module provides LoRA fine-tuning capabilities specifically for the LLM backbone,
allowing efficient parameter-efficient fine-tuning while keeping other components frozen.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration class for LoRA parameters.

    Args:
        r: Rank of the low-rank decomposition. Default is 8.
        lora_alpha: Scaling factor for LoRA. The actual scaling is lora_alpha/r. Default is 16.
        lora_dropout: Dropout probability for LoRA layers. Default is 0.0.
        target_modules: List of module names to apply LoRA.
            Default targets attention projections: ["q_proj", "k_proj", "v_proj", "o_proj"].
            Can also include MLP layers: ["gate_proj", "up_proj", "down_proj"].
        modules_to_save: List of modules to save in full (not as LoRA). Default is empty.
        bias: Whether to train bias parameters. Options: "none", "all", "lora_only". Default is "none".
        use_rslora: Whether to use Rank-Stabilized LoRA (RS-LoRA) scaling. Default is False.
        init_lora_weights: How to initialize LoRA weights. Options: True (default), "gaussian", False.
        layers_to_transform: Specific layer indices to apply LoRA. None means all layers.
        layers_pattern: Regex pattern to match layer names for LoRA application.
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    modules_to_save: List[str] = field(default_factory=list)
    bias: str = "none"  # "none", "all", "lora_only"
    use_rslora: bool = False
    init_lora_weights: Union[bool, str] = True
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None

    def __post_init__(self):
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"bias must be one of 'none', 'all', 'lora_only', got {self.bias}")
        if self.r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {self.r}")


class LoRALinear(nn.Module):
    """
    LoRA layer implementation that wraps a linear layer with low-rank adapters.

    The output is computed as: output = original_output + (x @ A @ B) * scaling
    where A is (in_features, r) and B is (r, out_features).

    Args:
        original_layer: The original nn.Linear layer to wrap.
        r: Rank of the low-rank decomposition.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability.
        use_rslora: Whether to use RS-LoRA scaling.
        init_lora_weights: How to initialize weights.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        init_lora_weights: Union[bool, str] = True,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze the original layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # Create LoRA layers
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Scaling factor
        if use_rslora:
            # RS-LoRA: scale by sqrt(r) instead of r
            self.scaling = lora_alpha / math.sqrt(r)
        else:
            self.scaling = lora_alpha / r

        # Initialize weights
        self._init_weights(init_lora_weights)

        # Flag to enable/disable LoRA
        self.lora_enabled = True

        # For merged weights mode
        self.merged = False

    def _init_weights(self, init_lora_weights: Union[bool, str]):
        """Initialize LoRA weights."""
        if init_lora_weights is False:
            return

        if init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r)
        else:
            # Default: Kaiming uniform for A, zeros for B
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # B is always initialized to zeros so that LoRA starts as identity
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional LoRA adaptation."""
        # Original linear output
        result = self.original_layer(x)

        if self.lora_enabled and not self.merged:
            # LoRA adaptation: result += (x @ A @ B) * scaling
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))
            result = result + lora_output * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into the original layer for inference efficiency."""
        if self.merged:
            return

        with torch.no_grad():
            # W' = W + (B @ A) * scaling
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original_layer.weight.add_(delta_weight)

        self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights from the original layer."""
        if not self.merged:
            return

        with torch.no_grad():
            # W = W' - (B @ A) * scaling
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original_layer.weight.sub_(delta_weight)

        self.merged = False

    def enable_lora(self):
        """Enable LoRA adaptation."""
        self.lora_enabled = True

    def disable_lora(self):
        """Disable LoRA adaptation (use original weights only)."""
        self.lora_enabled = False

    @property
    def weight(self):
        """Return the effective weight (for compatibility)."""
        return self.original_layer.weight

    @property
    def bias(self):
        """Return the bias (for compatibility)."""
        return self.original_layer.bias


def _find_target_modules(
    model: nn.Module,
    target_modules: List[str],
    layers_to_transform: Optional[List[int]] = None,
    layers_pattern: Optional[str] = None,
) -> Dict[str, nn.Linear]:
    """
    Find all target modules in the model that match the criteria.

    Args:
        model: The model to search.
        target_modules: List of module name patterns to match.
        layers_to_transform: Specific layer indices to apply LoRA.
        layers_pattern: Regex pattern to extract layer index from module name.

    Returns:
        Dictionary mapping full module paths to Linear layers.
    """
    target_dict = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check if module name matches any target pattern
        module_name = name.split('.')[-1]
        if module_name not in target_modules:
            continue

        # Check layer index if specified
        if layers_to_transform is not None:
            # Try to extract layer index from name
            if layers_pattern:
                match = re.search(layers_pattern, name)
            else:
                # Default pattern for transformer layers
                match = re.search(r'layers\.(\d+)', name)

            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layers_to_transform:
                    continue

        target_dict[name] = module

    return target_dict


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    prefix: str = "showo.model",
) -> Dict[str, LoRALinear]:
    """
    Inject LoRA layers into the target modules of the model.

    This function replaces target Linear layers with LoRALinear layers,
    which wrap the original layer and add low-rank adapters.

    Args:
        model: The model to inject LoRA into.
        config: LoRA configuration.
        prefix: Module name prefix to search within (for targeting LLM backbone only).

    Returns:
        Dictionary mapping module paths to injected LoRALinear layers.
    """
    # Find the submodule to inject LoRA into
    target_model = model
    if prefix:
        for part in prefix.split('.'):
            if hasattr(target_model, part):
                target_model = getattr(target_model, part)
            else:
                raise ValueError(f"Could not find module {prefix} in model")

    # Find all target modules
    target_modules = _find_target_modules(
        target_model,
        config.target_modules,
        config.layers_to_transform,
        config.layers_pattern,
    )

    if not target_modules:
        raise ValueError(
            f"No target modules found matching {config.target_modules} in {prefix}. "
            "Check your target_modules configuration."
        )

    injected_modules = {}

    for name, linear_layer in target_modules.items():
        # Create LoRA layer
        lora_layer = LoRALinear(
            original_layer=linear_layer,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
        )

        # Replace the module in the model
        parts = name.split('.')
        parent = target_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

        full_name = f"{prefix}.{name}" if prefix else name
        injected_modules[full_name] = lora_layer

    print(f"Injected LoRA into {len(injected_modules)} modules:")
    for name in sorted(injected_modules.keys())[:10]:  # Show first 10
        print(f"  - {name}")
    if len(injected_modules) > 10:
        print(f"  ... and {len(injected_modules) - 10} more")

    return injected_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.

    Args:
        model: The model containing LoRA layers.

    Returns:
        List of LoRA parameters (lora_A and lora_B weights).
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A.weight, module.lora_B.weight])
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA weights.

    Args:
        model: The model containing LoRA layers.

    Returns:
        State dict with only LoRA parameters.
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
):
    """
    Load LoRA weights from a state dict.

    Args:
        model: The model containing LoRA layers.
        state_dict: State dict with LoRA parameters.
        strict: Whether to raise error for missing/unexpected keys.
    """
    model_state = model.state_dict()

    missing_keys = []
    unexpected_keys = []

    for key, value in state_dict.items():
        if key in model_state:
            model_state[key].copy_(value)
        else:
            unexpected_keys.append(key)

    if strict:
        # Check for missing LoRA keys
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                for lora_key in [f"{name}.lora_A.weight", f"{name}.lora_B.weight"]:
                    if lora_key not in state_dict:
                        missing_keys.append(lora_key)

        if missing_keys:
            raise RuntimeError(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")


def save_lora_weights(model: nn.Module, path: str):
    """
    Save LoRA weights to a file.

    Args:
        model: The model containing LoRA layers.
        path: Path to save the weights.
    """
    state_dict = get_lora_state_dict(model)
    torch.save(state_dict, path)
    print(f"Saved LoRA weights to {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """
    Load LoRA weights from a file.

    Args:
        model: The model containing LoRA layers.
        path: Path to load the weights from.
        strict: Whether to raise error for missing/unexpected keys.
    """
    state_dict = torch.load(path, map_location='cpu')
    load_lora_state_dict(model, state_dict, strict=strict)
    print(f"Loaded LoRA weights from {path}")


def merge_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights into original layers for inference.

    Args:
        model: The model containing LoRA layers.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()


def unmerge_lora_weights(model: nn.Module):
    """
    Unmerge all LoRA weights from original layers.

    Args:
        model: The model containing LoRA layers.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()


def enable_lora(model: nn.Module):
    """Enable LoRA adaptation in all LoRA layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enable_lora()


def disable_lora(model: nn.Module):
    """Disable LoRA adaptation in all LoRA layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.disable_lora()


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count LoRA and total parameters in the model.

    Args:
        model: The model to count parameters for.

    Returns:
        Dictionary with parameter counts.
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
        'lora_percentage': 100 * lora_params / total_params if total_params > 0 else 0,
    }


def print_lora_summary(model: nn.Module):
    """Print a summary of LoRA configuration and parameters."""
    counts = count_lora_parameters(model)

    print("\n" + "=" * 60)
    print("LoRA Summary")
    print("=" * 60)
    print(f"Total parameters:     {counts['total_params']:,}")
    print(f"Trainable parameters: {counts['trainable_params']:,} ({counts['trainable_percentage']:.2f}%)")
    print(f"LoRA parameters:      {counts['lora_params']:,} ({counts['lora_percentage']:.2f}%)")
    print("=" * 60 + "\n")


def prepare_model_for_lora_training(
    model: nn.Module,
    config: LoRAConfig,
    llm_backbone_prefix: str = "showo.model",
) -> nn.Module:
    """
    Prepare a model for LoRA training by:
    1. Freezing all parameters
    2. Injecting LoRA into target modules
    3. Setting up trainable parameters

    Args:
        model: The model to prepare.
        config: LoRA configuration.
        llm_backbone_prefix: Prefix for the LLM backbone modules.

    Returns:
        The prepared model with LoRA injected.
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into LLM backbone
    injected_modules = inject_lora(model, config, prefix=llm_backbone_prefix)

    # Handle bias training
    if config.bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif config.bias == "lora_only":
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                if module.original_layer.bias is not None:
                    module.original_layer.bias.requires_grad = True

    # Handle modules_to_save (train full modules)
    for module_name in config.modules_to_save:
        for name, param in model.named_parameters():
            if module_name in name:
                param.requires_grad = True

    # Print summary
    print_lora_summary(model)

    return model
