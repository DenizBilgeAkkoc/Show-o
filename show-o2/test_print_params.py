#!/usr/bin/env python3
# coding=utf-8
"""
Test script to demonstrate loading Show-o2 and printing trainable parameters.

This creates a minimal model instance to show parameter counting without 
downloading large pretrained weights.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models import Showo2Qwen2_5


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


def print_model_parameters(model, model_name="Show-o2"):
    """Print comprehensive parameter information."""
    print("\n" + "=" * 90)
    print(f"{model_name.upper()} - TRAINABLE PARAMETERS ANALYSIS")
    print("=" * 90)
    
    # Overall statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n📊 Overall Statistics:")
    print("-" * 90)
    print(f"  Total Parameters:      {format_number(total_params):>12}  ({total_params:,})")
    print(f"  Trainable Parameters:  {format_number(trainable_params):>12}  ({trainable_params:,})")
    print(f"  Frozen Parameters:     {format_number(frozen_params):>12}  ({frozen_params:,})")
    
    if total_params > 0:
        trainable_pct = 100.0 * trainable_params / total_params
        print(f"  Trainable Percentage:  {trainable_pct:>12.2f}%")
    
    # Module breakdown
    print("\n📋 Parameter Breakdown by Top-Level Module:")
    print("-" * 90)
    
    module_stats = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0] if '.' in name else name
        if module_name not in module_stats:
            module_stats[module_name] = {'total': 0, 'trainable': 0, 'frozen': 0}
        
        param_count = param.numel()
        module_stats[module_name]['total'] += param_count
        if param.requires_grad:
            module_stats[module_name]['trainable'] += param_count
        else:
            module_stats[module_name]['frozen'] += param_count
    
    # Sort by total parameters
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print(f"{'Module Name':<30} {'Trainable':>12} {'Frozen':>12} {'Total':>12} {'%':>8}")
    print("-" * 90)
    
    for module_name, stats in sorted_modules:
        trainable = stats['trainable']
        frozen = stats['frozen']
        total = stats['total']
        pct = 100.0 * trainable / total if total > 0 else 0
        
        print(f"{module_name:<30} {format_number(trainable):>12} {format_number(frozen):>12} "
              f"{format_number(total):>12} {pct:>7.1f}%")
    
    print("=" * 90 + "\n")


def main():
    """Main function to demonstrate model loading and parameter analysis."""
    
    print("\n🚀 Show-o2 Model - Trainable Parameters Analysis Demo\n")
    
    # Option 1: Load from HuggingFace (requires download and internet)
    print("=" * 90)
    print("OPTION 1: Load from HuggingFace Pretrained Model")
    print("=" * 90)
    print("\nThis would download the full pretrained model (~2-8GB depending on model size):")
    print("  model = Showo2Qwen2_5.from_pretrained('showlab/show-o2-1.5B', use_safetensors=False)")
    print("\nNote: Skipping this option in the demo to avoid large downloads.")
    
    # Option 2: Initialize with minimal config (faster, no download)
    print("\n" + "=" * 90)
    print("OPTION 2: Initialize Model Architecture (No Pretrained Weights)")
    print("=" * 90)
    print("\nInitializing model architecture with minimal config...")
    print("Note: This creates the model structure without loading pretrained weights.\n")
    
    try:
        # Minimal configuration to create model structure
        model_config = {
            'llm_vocab_size': 151936,
            'llm_model_path': 'Qwen/Qwen2.5-1.5B-Instruct',
            'load_from_showo': False,  # Don't load pretrained weights
            'image_latent_dim': 16,
            'image_latent_height': 27,
            'image_latent_width': 27,
            'video_latent_height': 27,
            'video_latent_width': 27,
            'patch_size': 2,
            'hidden_size': 1536,
            'clip_latent_dim': 1152,
            'num_diffusion_layers': 10,
            'add_time_embeds': True,
            'add_qk_norm': True,
            'clip_pretrained_model_path': "google/siglip-so400m-patch14-384",
        }
        
        print("Creating Show-o2 model architecture...")
        print(f"  - LLM: {model_config['llm_model_path']}")
        print(f"  - Hidden size: {model_config['hidden_size']}")
        print(f"  - Image latent: {model_config['image_latent_height']}x{model_config['image_latent_width']}")
        print(f"  - Diffusion layers: {model_config['num_diffusion_layers']}")
        
        # This will download the base LLM and vision models
        print("\n⚠️  Note: This will download base models (Qwen2.5-1.5B and SigLIP)...")
        print("Press Ctrl+C to cancel, or wait for downloads to complete...\n")
        
        model = Showo2Qwen2_5(**model_config)
        model.eval()
        
        print("✓ Model architecture created successfully!\n")
        
        # Print parameter analysis
        print_model_parameters(model, "Show-o2 (1.5B variant)")
        
        print("\n💡 To analyze the full pretrained model, use:")
        print("   python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        print("\nTo run a simpler demo without downloads:")
        print("  python show_model_structure.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tips:")
        print("  - Ensure you have internet connection")
        print("  - Check that all dependencies are installed: bash build_env.sh")
        print("  - Try the simple demo: python show_model_structure.py")


if __name__ == '__main__':
    main()
