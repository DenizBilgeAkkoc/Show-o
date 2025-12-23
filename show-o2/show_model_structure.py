#!/usr/bin/env python3
# coding=utf-8
"""
Simple script to demonstrate Show-o2 model structure and parameter counting.

This script shows how to load the model and count trainable parameters
without downloading large pretrained weights.

Usage:
    python show_model_structure.py
"""

import os
import sys

# Add show-o2 directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError:
    print("❌ Error: PyTorch not installed")
    print("   Please run: pip install torch")
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


def print_model_summary(model, model_name="Show-o2"):
    """Print model parameter summary."""
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    
    total, trainable, frozen = count_parameters(model)
    
    print("\n📊 Overall Statistics:")
    print("-" * 80)
    print(f"Total Parameters:      {format_number(total):>12}  ({total:,})")
    print(f"Trainable Parameters:  {format_number(trainable):>12}  ({trainable:,})")
    print(f"Frozen Parameters:     {format_number(frozen):>12}  ({frozen:,})")
    
    if total > 0:
        print(f"Trainable Percentage:  {100.0 * trainable / total:>12.2f}%")
    
    print("\n📋 Module Breakdown:")
    print("-" * 80)
    
    # Group by module
    module_params = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0] if '.' in name else name
        if module_name not in module_params:
            module_params[module_name] = {'total': 0, 'trainable': 0}
        
        param_count = param.numel()
        module_params[module_name]['total'] += param_count
        if param.requires_grad:
            module_params[module_name]['trainable'] += param_count
    
    # Sort by total parameters
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print(f"{'Module':<30} {'Trainable':>12} {'Total':>12} {'%':>8}")
    print("-" * 80)
    
    for module_name, stats in sorted_modules:
        trainable = stats['trainable']
        total = stats['total']
        pct = 100.0 * trainable / total if total > 0 else 0
        print(f"{module_name:<30} {format_number(trainable):>12} {format_number(total):>12} {pct:>7.1f}%")
    
    print("=" * 80 + "\n")


def demo_basic_model():
    """Demonstrate with a basic model structure."""
    print("\n🔍 DEMO: Basic Model Parameter Counting")
    print("\nThis demo shows how parameter counting works with a simple model:")
    
    # Simple demo model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(100, 50)
            self.layer2 = torch.nn.Linear(50, 10)
            self.embedding = torch.nn.Embedding(1000, 100)
        
        def forward(self, x):
            return self.layer2(torch.relu(self.layer1(x)))
    
    model = SimpleModel()
    print_model_summary(model, "Simple Demo")


def load_show_o2_model():
    """Attempt to load the Show-o2 model."""
    print("\n🚀 Attempting to load Show-o2 model...\n")
    
    try:
        from models import Showo2Qwen2_5
        print("✓ Model class imported successfully")
        
        # Try to load with minimal config
        print("\nOption 1: To load from HuggingFace pretrained:")
        print("  model = Showo2Qwen2_5.from_pretrained('showlab/show-o2-1.5B', use_safetensors=False)")
        
        print("\nOption 2: To load with config file:")
        print("  python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml")
        
        print("\n💡 The model has the following main components:")
        print("   - showo: LLM backbone (Qwen2.5)")
        print("   - image_embedder_und: Patch embedding for understanding")
        print("   - image_embedder_gen: Patch embedding for generation")
        print("   - und_trans: Understanding transformer (from SigLIP)")
        print("   - fusion_proj: Fusion projection layer")
        print("   - diffusion_head_a: Diffusion attention blocks")
        print("   - diffusion_head_b: Final diffusion layer")
        print("   - time_embed: Timestep embedding")
        
    except ImportError as e:
        print(f"⚠️  Could not import Show-o2 model: {e}")
        print("\nThis is expected if dependencies are not fully installed.")
        print("\nTo use Show-o2:")
        print("  1. Install all dependencies: bash build_env.sh")
        print("  2. Run: python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml")


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("Show-o2 Model Parameter Analysis Tool")
    print("=" * 80)
    
    # Show basic demo
    demo_basic_model()
    
    # Try to load Show-o2
    load_show_o2_model()
    
    print("\n" + "=" * 80)
    print("📚 For more information, see PRINT_PARAMS_README.md")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
