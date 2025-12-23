#!/usr/bin/env python3
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
Simplified script to load the Show-o2 model and print its trainable parameters.

This script provides a minimal setup to load the Showo2Qwen2_5 model and display 
detailed information about its trainable parameters.

Usage:
    python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml

    Or if you want to load from a local checkpoint:
    python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml model_path=/path/to/checkpoint.bin

Requirements:
    - torch
    - transformers
    - All dependencies listed in requirements.txt or build_env.sh
"""

import os
import sys

# Add show-o2 directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from models import Showo2Qwen2_5
    from models.misc import get_text_tokenizer
    from utils import get_config, path_to_llm_name, load_state_dict
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies first:")
    print("  bash build_env.sh")
    print("or:")
    print("  pip install torch transformers einops timm")
    sys.exit(1)


def format_number(num):
    """Format large numbers for better readability."""
    if num >= 1e9:
        return f"{num/1e9:.3f}B"
    elif num >= 1e6:
        return f"{num/1e6:.3f}M"
    elif num >= 1e3:
        return f"{num/1e3:.3f}K"
    else:
        return str(num)


def count_parameters(model):
    """Count parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def get_module_parameters(model):
    """Get parameter counts organized by top-level module."""
    module_stats = {}
    
    for name, param in model.named_parameters():
        # Extract top-level module name
        if '.' in name:
            module_name = name.split('.')[0]
        else:
            module_name = name
        
        if module_name not in module_stats:
            module_stats[module_name] = {
                'trainable': 0,
                'frozen': 0,
                'total': 0
            }
        
        param_count = param.numel()
        module_stats[module_name]['total'] += param_count
        
        if param.requires_grad:
            module_stats[module_name]['trainable'] += param_count
        else:
            module_stats[module_name]['frozen'] += param_count
    
    return module_stats


def print_parameter_summary(model):
    """Print a comprehensive summary of model parameters."""
    
    print("\n" + "=" * 90)
    print(" " * 25 + "SHOW-O2 MODEL PARAMETER SUMMARY")
    print("=" * 90)
    
    # Overall statistics
    total_params, trainable_params, frozen_params = count_parameters(model)
    
    print("\n📊 Overall Statistics:")
    print("-" * 90)
    print(f"  {'Total Parameters:':<30} {format_number(total_params):>12} ({total_params:>15,})")
    print(f"  {'Trainable Parameters:':<30} {format_number(trainable_params):>12} ({trainable_params:>15,})")
    print(f"  {'Frozen Parameters:':<30} {format_number(frozen_params):>12} ({frozen_params:>15,})")
    
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0
    print(f"  {'Trainable Percentage:':<30} {trainable_pct:>12.2f}%")
    
    # Module-level breakdown
    print("\n📋 Parameter Breakdown by Module:")
    print("-" * 90)
    
    module_stats = get_module_parameters(model)
    
    # Print header
    header = f"{'Module Name':<35} {'Trainable':>12} {'Frozen':>12} {'Total':>12} {'% Train':>10}"
    print(header)
    print("-" * 90)
    
    # Sort modules by total parameters (descending)
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for module_name, stats in sorted_modules:
        trainable = stats['trainable']
        frozen = stats['frozen']
        total = stats['total']
        pct = 100.0 * trainable / total if total > 0 else 0
        
        print(f"{module_name:<35} {format_number(trainable):>12} "
              f"{format_number(frozen):>12} {format_number(total):>12} {pct:>9.1f}%")
    
    print("-" * 90)
    
    # Additional detailed statistics
    print("\n🔍 Detailed Layer Information:")
    print("-" * 90)
    
    # Count different types of parameters
    layer_types = {}
    for name, param in model.named_parameters():
        # Categorize by layer type
        if 'weight' in name:
            layer_type = 'Weights'
        elif 'bias' in name:
            layer_type = 'Biases'
        elif 'norm' in name.lower():
            layer_type = 'Normalization'
        elif 'embed' in name.lower():
            layer_type = 'Embeddings'
        else:
            layer_type = 'Other'
        
        if layer_type not in layer_types:
            layer_types[layer_type] = {'count': 0, 'params': 0, 'trainable': 0}
        
        layer_types[layer_type]['count'] += 1
        layer_types[layer_type]['params'] += param.numel()
        if param.requires_grad:
            layer_types[layer_type]['trainable'] += param.numel()
    
    for layer_type, stats in sorted(layer_types.items()):
        print(f"  {layer_type:<20}: {stats['count']:>5} layers, "
              f"{format_number(stats['params']):>10} params "
              f"({format_number(stats['trainable']):>10} trainable)")
    
    print("\n" + "=" * 90)
    print()


def main():
    """Main function to load the model and print parameter information."""
    
    print("\n🚀 Loading Show-o2 Model...\n")
    
    # Parse configuration
    try:
        config = get_config()
        print(f"✓ Configuration loaded from: {config.config}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("\nUsage: python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    if config.model.weight_type == "bfloat16":
        weight_type = torch.bfloat16
    elif config.model.weight_type == "float32":
        weight_type = torch.float32
    else:
        weight_type = torch.float32
    
    print(f"✓ Using weight type: {weight_type}")
    
    # Get tokenizer
    try:
        print(f"\n📝 Initializing tokenizer from: {config.model.showo.llm_model_path}")
        llm_model_path = config.model.showo.llm_model_path
        llm_name = path_to_llm_name.get(llm_model_path, 'qwen2_5')
        text_tokenizer, showo_token_ids = get_text_tokenizer(
            llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=llm_name
        )
        config.model.showo.llm_vocab_size = len(text_tokenizer)
        print(f"✓ Tokenizer initialized with vocabulary size: {len(text_tokenizer)}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load tokenizer: {e}")
        print("   Using default vocabulary size from config")
    
    # Load model
    try:
        print(f"\n🔨 Loading model architecture...")
        
        if config.model.showo.load_from_showo:
            print(f"   Loading from pretrained: {config.model.showo.pretrained_model_path}")
            model = Showo2Qwen2_5.from_pretrained(
                config.model.showo.pretrained_model_path,
                use_safetensors=False
            )
        else:
            print(f"   Initializing from config...")
            model = Showo2Qwen2_5(**config.model.showo)
            
            # Load checkpoint if specified
            model_path = getattr(config, 'model_path', None)
            if model_path:
                print(f"   Loading weights from: {model_path}")
                state_dict = load_state_dict(model_path)
                model.load_state_dict(state_dict)
        
        model.to(device)
        model.to(weight_type)
        model.eval()
        
        print(f"✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure you have:")
        print("   1. Internet connection (if loading from HuggingFace)")
        print("   2. Required model weights downloaded")
        print("   3. All dependencies installed (bash build_env.sh)")
        sys.exit(1)
    
    # Print parameter information
    print_parameter_summary(model)
    
    print("✅ Analysis complete!\n")


if __name__ == '__main__':
    main()
