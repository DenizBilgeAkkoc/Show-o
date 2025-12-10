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
LoRA Fine-tuning Script for Show-o2 Model.

This script demonstrates how to fine-tune the LLM backbone of Show-o2 using LoRA.
Only the LoRA parameters in the Qwen2 LLM backbone are trained.

Example usage:
    # Single GPU
    python train_lora.py --config configs/showo2_lora_finetune.yaml

    # Multi-GPU with accelerate
    accelerate launch --config_file accelerate_configs/default.yaml \
        train_lora.py --config configs/showo2_lora_finetune.yaml
"""

import os
import logging
import math
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from models import Showo2Qwen2_5
from models.lora import LoRAConfig
from models.lr_schedulers import get_scheduler
from models.misc import get_text_tokenizer, get_weight_type

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__, log_level="INFO")


def get_config():
    """Load configuration from command line argument."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, unknown = parser.parse_known_args()

    config = OmegaConf.load(args.config)

    # Override with command line arguments
    cli_conf = OmegaConf.from_cli(unknown)
    config = OmegaConf.merge(config, cli_conf)

    return config


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.get("enable_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if config.get("wandb", {}).get("enabled", False) else None,
        project_dir=config.experiment.logging_dir,
    )

    total_batch_size_per_gpu = config.training.batch_size
    total_batch_size = total_batch_size_per_gpu * accelerator.num_processes * config.training.gradient_accumulation_steps

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # Set seed
    if config.training.get("seed") is not None:
        set_seed(config.training.seed)

    #########################
    # LOAD MODEL            #
    #########################
    logger.info("Loading models...")
    weight_type = get_weight_type(config)

    # Initialize tokenizer
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2.5"
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # Load model
    if config.model.showo.get("pretrained_model_path"):
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=config.model.showo.get("use_safetensors", True)
        ).to(accelerator.device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    #########################
    # ENABLE LoRA           #
    #########################
    logger.info("Enabling LoRA fine-tuning...")

    lora_config = config.lora
    model.enable_lora_finetuning(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.get("lora_dropout", 0.0),
        target_modules=lora_config.target_modules,
        bias=lora_config.get("bias", "none"),
        use_rslora=lora_config.get("use_rslora", False),
    )

    # Print parameter summary
    if accelerator.is_main_process:
        model.print_lora_summary()

    #########################
    # OPTIMIZER & SCHEDULER #
    #########################
    logger.info("Setting up optimizer and scheduler...")

    # Only optimize LoRA parameters
    lora_params = model.get_lora_parameters()
    optimizer_config = config.optimizer.params

    optimizer = AdamW(
        lora_params,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.get("beta1", 0.9), optimizer_config.get("beta2", 0.999)),
        weight_decay=optimizer_config.get("weight_decay", 0.0),
        eps=optimizer_config.get("epsilon", 1e-8),
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.get("warmup_steps", 0),
        num_training_steps=config.training.max_train_steps,
    )

    #########################
    # DATALOADER            #
    #########################
    # NOTE: Replace this with your actual dataloader
    # This is a placeholder to demonstrate the training loop structure
    logger.info("Setting up dataloader...")

    # Example: Create a simple dummy dataloader for demonstration
    # In practice, use the datasets from the datasets/ folder
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Return dummy data - replace with your actual data format
            return {
                "text_tokens": torch.randint(0, 1000, (256,)),
                "image_latents": torch.randn(16, 27, 27),
                "attention_mask": torch.ones(256),
                "text_labels": torch.randint(0, 1000, (256,)),
            }

    train_dataset = DummyDataset(config.training.get("num_train_samples", 10000))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=total_batch_size_per_gpu,
        shuffle=True,
        num_workers=config.dataset.get("num_workers", 4),
        pin_memory=True,
    )

    #########################
    # PREPARE FOR TRAINING  #
    #########################
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running LoRA training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per GPU = {total_batch_size_per_gpu}")
    logger.info(f"  Total batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.training.max_train_steps}")

    #########################
    # TRAINING LOOP         #
    #########################
    global_step = 0
    model.train()

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                # NOTE: Adapt this to your actual model forward call
                # This is a simplified example

                # Example forward pass (adapt based on your task):
                # outputs = model(
                #     text_tokens=batch["text_tokens"],
                #     image_latents=batch["image_latents"],
                #     attention_mask=batch["attention_mask"],
                #     text_labels=batch["text_labels"],
                #     ...
                # )
                # loss = outputs.loss  # or compute loss from outputs

                # Placeholder loss computation for demonstration
                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if config.training.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(lora_params, config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if global_step % config.experiment.get("log_every", 50) == 0:
                logger.info(
                    f"Step: {global_step}, Loss: {loss.item():.4f}, "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                )

            # Save checkpoint
            if global_step % config.experiment.get("save_every", 1000) == 0 and global_step > 0:
                if accelerator.is_main_process:
                    save_path = Path(config.experiment.output_dir) / f"lora_checkpoint_{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)

                    # Save LoRA weights
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_lora_weights(str(save_path / "lora_weights.pt"))

                    # Save optimizer state
                    torch.save(optimizer.state_dict(), save_path / "optimizer.pt")

                    logger.info(f"Saved checkpoint to {save_path}")

            global_step += 1

            if global_step >= config.training.max_train_steps:
                break

        if global_step >= config.training.max_train_steps:
            break

    #########################
    # SAVE FINAL MODEL      #
    #########################
    if accelerator.is_main_process:
        final_save_path = Path(config.experiment.output_dir) / "final_lora_weights"
        final_save_path.mkdir(parents=True, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_lora_weights(str(final_save_path / "lora_weights.pt"))

        # Optionally, save merged model
        if config.get("save_merged_model", False):
            unwrapped_model.merge_lora()
            unwrapped_model.save_pretrained(str(final_save_path / "merged_model"))

        logger.info(f"Training complete! Final weights saved to {final_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
