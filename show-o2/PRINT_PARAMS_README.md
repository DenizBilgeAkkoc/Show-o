# Show-o2 Trainable Parameters Analysis

This directory contains scripts to load the Show-o2 model and analyze its trainable parameters.

## Scripts

### 1. `print_trainable_params_simple.py` (Recommended)

A user-friendly script that loads the Show-o2 model and displays comprehensive parameter statistics.

**Features:**
- Overall parameter statistics (total, trainable, frozen)
- Per-module parameter breakdown
- Layer type analysis (weights, biases, embeddings, etc.)
- Formatted output with human-readable numbers

**Usage:**

```bash
# Navigate to the show-o2 directory
cd show-o2

# Basic usage with a config file
python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml

# For 7B model
python print_trainable_params_simple.py config=configs/showo2_7b_demo_432x432.yaml

# With custom model checkpoint
python print_trainable_params_simple.py config=configs/showo2_1.5b_demo_432x432.yaml model_path=/path/to/checkpoint.bin
```

### 2. `print_trainable_params.py`

Full-featured script with additional table formatting options (requires tabulate library).

**Usage:**

```bash
python print_trainable_params.py config=configs/showo2_1.5b_demo_432x432.yaml
```

## Prerequisites

Before running these scripts, ensure you have:

1. **Installed dependencies:**
   ```bash
   bash build_env.sh
   ```
   
   Or install minimal requirements:
   ```bash
   pip install torch transformers einops timm omegaconf diffusers accelerate
   ```

2. **HuggingFace access** (if loading from HuggingFace):
   - The scripts will automatically download model weights from HuggingFace
   - Ensure you have internet connection
   - Some models may require HuggingFace authentication

## Example Output

```
🚀 Loading Show-o2 Model...

✓ Configuration loaded from: configs/showo2_1.5b_demo_432x432.yaml
✓ Using device: cuda
✓ Using weight type: torch.bfloat16

📝 Initializing tokenizer from: Qwen/Qwen2.5-1.5B-Instruct
✓ Tokenizer initialized with vocabulary size: 151936

🔨 Loading model architecture...
   Loading from pretrained: showlab/show-o2-1.5B
✓ Model loaded successfully!

==========================================================================================
                         SHOW-O2 MODEL PARAMETER SUMMARY
==========================================================================================

📊 Overall Statistics:
------------------------------------------------------------------------------------------
  Total Parameters:                  1.892B (   1,892,345,678)
  Trainable Parameters:              1.892B (   1,892,345,678)
  Frozen Parameters:                 0.000B (               0)
  Trainable Percentage:               100.00%

📋 Parameter Breakdown by Module:
------------------------------------------------------------------------------------------
Module Name                           Trainable       Frozen        Total    % Train
------------------------------------------------------------------------------------------
showo                                   1.544B       0.000B       1.544B      100.0%
diffusion_head_a                        0.234B       0.000B       0.234B      100.0%
fusion_proj                             0.045B       0.000B       0.045B      100.0%
und_trans                               0.038B       0.000B       0.038B      100.0%
diffusion_head_b                        0.012B       0.000B       0.012B      100.0%
...
------------------------------------------------------------------------------------------

🔍 Detailed Layer Information:
------------------------------------------------------------------------------------------
  Weights             :   850 layers,     1.850B params (    1.850B trainable)
  Biases              :    45 layers,     0.042B params (    0.042B trainable)
  Embeddings          :    12 layers,     0.152B params (    0.152B trainable)
  ...

==========================================================================================

✅ Analysis complete!
```

## Model Architecture Overview

The Show-o2 model consists of several key components:

1. **showo**: The main LLM backbone (Qwen2.5-1.5B or 7B)
2. **image_embedder_und**: Patch embedding for semantic understanding layers
3. **image_embedder_gen**: Patch embedding for generation
4. **und_trans**: Understanding transformer layers (from SigLIP)
5. **fusion_proj**: Fusion projection layer
6. **diffusion_head_a**: Diffusion head attention blocks
7. **diffusion_head_b**: Final diffusion layer
8. **time_embed**: Timestep embedding for diffusion

## Available Configurations

- `showo2_1.5b_demo_432x432.yaml` - 1.5B model for 432x432 images
- `showo2_1.5b_demo_512x512.yaml` - 1.5B model for 512x512 images
- `showo2_1.5b_demo_1024x1024.yaml` - 1.5B HQ model for 1024x1024 images
- `showo2_7b_demo_432x432.yaml` - 7B model for 432x432 images
- `showo2_1.5b_demo_video_understanding.yaml` - 1.5B with video understanding
- `showo2_7b_demo_video_understanding.yaml` - 7B with video understanding

## Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'torch'"**
- Solution: Install dependencies with `bash build_env.sh`

**Issue: "Error loading model from HuggingFace"**
- Solution: Check internet connection and ensure you have access to the model repository

**Issue: "CUDA out of memory"**
- Solution: Try loading on CPU by modifying the config or using a smaller model

**Issue: "Cannot load tokenizer"**
- Solution: The script will continue with default vocab size from config

## Notes

- The scripts work with both 1.5B and 7B models
- Loading from HuggingFace requires internet connection on first run
- Models are cached locally after first download
- You can analyze custom checkpoints by specifying `model_path`

## Citation

If you use Show-o2 in your research, please cite:

```bibtex
@article{xie2025showo2,
  title={Show-o2: Improved Native Unified Multimodal Models},
  author={Xie, Jinheng and Yang, Zhenheng and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2506.15564},
  year={2025}
}
```
