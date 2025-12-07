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
Script to load the Show-o2 model and print its trainable parameters.
This script loads the Showo2Qwen2_5 model and displays detailed information 
about its trainable parameters.
"""

import os
import sys
import torch

# Try to import tabulate, but make it optional
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Add show-o2 directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Showo2Qwen2_5
from models.misc import get_text_tokenizer
from utils import get_config, path_to_llm_name


def count_parameters(model):
    """
    Count the total number of parameters and trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params, frozen_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return total_params, trainable_params, frozen_params


def get_parameter_info_by_module(model):
    """
    Get detailed parameter information organized by module.
    
    Args:
        model: PyTorch model
        
    Returns:
        list: List of tuples containing (module_name, trainable_params, frozen_params, total_params)
    """
    module_info = {}
    
    for name, param in model.named_parameters():
        # Get the top-level module name
        module_name = name.split('.')[0] if '.' in name else name
        
        if module_name not in module_info:
            module_info[module_name] = {
                'trainable': 0,
                'frozen': 0,
                'total': 0
            }
        
        param_count = param.numel()
        module_info[module_name]['total'] += param_count
        
        if param.requires_grad:
            module_info[module_name]['trainable'] += param_count
        else:
            module_info[module_name]['frozen'] += param_count
    
    # Convert to list of tuples for display
    result = []
    for module_name, counts in sorted(module_info.items()):
        result.append((
            module_name,
            counts['trainable'],
            counts['frozen'],
            counts['total']
        ))
    
    return result


def format_number(num):
    """Format large numbers for better readability."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_model_info(model):
    """
    Print comprehensive information about the model's trainable parameters.
    
    Args:
        model: PyTorch model
    """
    print("=" * 80)
    print("SHOW-O2 MODEL TRAINABLE PARAMETERS SUMMARY")
    print("=" * 80)
    print()
    
    # Get overall statistics
    total_params, trainable_params, frozen_params = count_parameters(model)
    
    print("Overall Statistics:")
    print("-" * 80)
    print(f"Total Parameters:      {format_number(total_params):>15} ({total_params:,})")
    print(f"Trainable Parameters:  {format_number(trainable_params):>15} ({trainable_params:,})")
    print(f"Frozen Parameters:     {format_number(frozen_params):>15} ({frozen_params:,})")
    print(f"Trainable Percentage:  {100 * trainable_params / total_params:>15.2f}%")
    print()
    
    # Get module-level information
    module_info = get_parameter_info_by_module(model)
    
    print("Parameter Breakdown by Module:")
    print("-" * 80)
    
    # Prepare table data
    table_data = []
    for module_name, trainable, frozen, total in module_info:
        table_data.append([
            module_name,
            format_number(trainable),
            format_number(frozen),
            format_number(total),
            f"{100 * trainable / total:.1f}%" if total > 0 else "0%"
        ])
    
    # Print table
    headers = ["Module Name", "Trainable", "Frozen", "Total", "% Trainable"]
    
    if HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        # Simple table format without tabulate
        # Print header
        col_widths = [30, 15, 15, 15, 15]
        header_line = " | ".join([headers[i].ljust(col_widths[i]) for i in range(len(headers))])
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in table_data:
            row_line = " | ".join([str(row[i]).ljust(col_widths[i]) for i in range(len(row))])
            print(row_line)
    
    print()
    
    print("=" * 80)
    print()


def main():
    """Main function to load model and print trainable parameters."""
    
    print("Loading Show-o2 model...")
    print()
    
    # Get configuration
    # Use a demo config file if available, otherwise use default parameters
    config_file = "configs/showo2_1.5b_demo_432x432.yaml"
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found. Using default parameters.")
        config = None
    else:
        config = get_config()
    
    if config is not None:
        # Load model with config
        text_tokenizer, showo_token_ids = get_text_tokenizer(
            config.model.showo.llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=path_to_llm_name[config.model.showo.llm_model_path]
        )
        config.model.showo.llm_vocab_size = len(text_tokenizer)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print()
        
        if config.model.showo.load_from_showo:
            print(f"Loading from pretrained: {config.model.showo.pretrained_model_path}")
            model = Showo2Qwen2_5.from_pretrained(
                config.model.showo.pretrained_model_path,
                use_safetensors=False
            ).to(device)
        else:
            print("Initializing model from config...")
            model = Showo2Qwen2_5(**config.model.showo).to(device)
        
        print("Model loaded successfully!")
        print()
    else:
        # Load with minimal default parameters for demonstration
        print("Loading model with default parameters...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_kwargs = {
            'llm_vocab_size': 151936,
            'llm_model_path': 'Qwen/Qwen2.5-1.5B-Instruct',
            'load_from_showo': True,
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
        
        try:
            model = Showo2Qwen2_5(**model_kwargs).to(device)
            print("Model initialized successfully!")
            print()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nNote: The model may require downloading pretrained weights from HuggingFace.")
            print("Please ensure you have internet connection and HuggingFace access tokens if needed.")
            return
    
    # Print trainable parameters information
    print_model_info(model)
    
    print("Script completed successfully!")


if __name__ == '__main__':
    main()
