#!/usr/bin/env python
# coding=utf-8
"""
Test script to verify LoRA implementation works correctly.
"""

import sys
import torch
import torch.nn as nn

# Add the show-o2 directory to path
sys.path.insert(0, '/home/user/Show-o/show-o2')

def test_lora_config():
    """Test LoRAConfig creation and validation."""
    print("=" * 60)
    print("Testing LoRAConfig...")
    print("=" * 60)

    from models.lora import LoRAConfig

    # Test default config
    config = LoRAConfig()
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert "q_proj" in config.target_modules
    print("  [PASS] Default config creation")

    # Test custom config
    config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="lora_only",
        use_rslora=True,
    )
    assert config.r == 16
    assert config.use_rslora == True
    print("  [PASS] Custom config creation")

    # Test validation
    try:
        config = LoRAConfig(r=-1)
        print("  [FAIL] Should have raised error for negative r")
        return False
    except ValueError:
        print("  [PASS] Validation for negative r")

    try:
        config = LoRAConfig(bias="invalid")
        print("  [FAIL] Should have raised error for invalid bias")
        return False
    except ValueError:
        print("  [PASS] Validation for invalid bias")

    print("  All LoRAConfig tests passed!\n")
    return True


def test_lora_linear():
    """Test LoRALinear layer."""
    print("=" * 60)
    print("Testing LoRALinear...")
    print("=" * 60)

    from models.lora import LoRALinear

    # Create a simple linear layer
    original = nn.Linear(64, 128, bias=True)
    original_weight = original.weight.clone()

    # Wrap with LoRA
    lora_layer = LoRALinear(
        original_layer=original,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    # Check that original weights are frozen
    assert not original.weight.requires_grad
    print("  [PASS] Original weights frozen")

    # Check that LoRA weights are trainable
    assert lora_layer.lora_A.weight.requires_grad
    assert lora_layer.lora_B.weight.requires_grad
    print("  [PASS] LoRA weights trainable")

    # Test forward pass
    x = torch.randn(2, 10, 64)
    output = lora_layer(x)
    assert output.shape == (2, 10, 128)
    print("  [PASS] Forward pass shape")

    # Test that output is different from original (LoRA adds contribution)
    # Note: B is initialized to zeros, so initially they should be the same
    original_output = original(x)
    assert torch.allclose(output, original_output, atol=1e-6)
    print("  [PASS] Initial output matches original (B=0)")

    # Modify LoRA weights and check output changes
    with torch.no_grad():
        lora_layer.lora_B.weight.fill_(0.1)
    output_after = lora_layer(x)
    assert not torch.allclose(output_after, original_output)
    print("  [PASS] Output changes after modifying LoRA weights")

    # Test disable/enable
    lora_layer.disable_lora()
    output_disabled = lora_layer(x)
    assert torch.allclose(output_disabled, original_output, atol=1e-6)
    print("  [PASS] Disable LoRA works")

    lora_layer.enable_lora()
    output_enabled = lora_layer(x)
    assert torch.allclose(output_enabled, output_after)
    print("  [PASS] Enable LoRA works")

    # Test merge/unmerge
    lora_layer.merge_weights()
    output_merged = lora_layer(x)
    assert torch.allclose(output_merged, output_after, atol=1e-5)
    print("  [PASS] Merge weights works")

    lora_layer.unmerge_weights()
    output_unmerged = lora_layer(x)
    assert torch.allclose(output_unmerged, output_after, atol=1e-5)
    print("  [PASS] Unmerge weights works")

    print("  All LoRALinear tests passed!\n")
    return True


def test_lora_injection():
    """Test LoRA injection into a simple model."""
    print("=" * 60)
    print("Testing LoRA injection...")
    print("=" * 60)

    from models.lora import LoRAConfig, inject_lora, LoRALinear

    # Create a simple transformer-like model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.ModuleDict({
                        'q_proj': nn.Linear(64, 64),
                        'k_proj': nn.Linear(64, 64),
                        'v_proj': nn.Linear(64, 64),
                        'o_proj': nn.Linear(64, 64),
                    }),
                    'mlp': nn.ModuleDict({
                        'gate_proj': nn.Linear(64, 256),
                        'up_proj': nn.Linear(64, 256),
                        'down_proj': nn.Linear(256, 64),
                    }),
                })
                for _ in range(2)
            ])

    model = SimpleModel()

    # Count parameters before
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Inject LoRA into attention layers only
    config = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    injected = inject_lora(model, config, prefix="")

    # Check injection count
    assert len(injected) == 8  # 4 proj layers * 2 transformer layers
    print(f"  [PASS] Injected into {len(injected)} modules")

    # Check that injected modules are LoRALinear
    for name, module in model.named_modules():
        if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            if "self_attn" in name:
                assert isinstance(module, LoRALinear), f"{name} should be LoRALinear"
    print("  [PASS] All target modules converted to LoRALinear")

    # Check that MLP layers are NOT converted
    for name, module in model.named_modules():
        if any(t in name for t in ["gate_proj", "up_proj", "down_proj"]):
            assert isinstance(module, nn.Linear), f"{name} should remain nn.Linear"
    print("  [PASS] Non-target modules unchanged")

    print("  All injection tests passed!\n")
    return True


def test_prepare_model_for_training():
    """Test full model preparation for LoRA training."""
    print("=" * 60)
    print("Testing prepare_model_for_lora_training...")
    print("=" * 60)

    from models.lora import LoRAConfig, prepare_model_for_lora_training, count_lora_parameters

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.ModuleDict({
                        'q_proj': nn.Linear(64, 64),
                        'k_proj': nn.Linear(64, 64),
                        'v_proj': nn.Linear(64, 64),
                        'o_proj': nn.Linear(64, 64),
                    }),
                })
                for _ in range(2)
            ])
            self.head = nn.Linear(64, 1000)

    model = SimpleModel()

    config = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # Only Q and V
    )

    prepare_model_for_lora_training(model, config, llm_backbone_prefix="")

    # Check that only LoRA params are trainable
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]

    for name in trainable_params:
        assert "lora_" in name, f"Non-LoRA param {name} is trainable"
    print(f"  [PASS] Only LoRA parameters are trainable ({len(trainable_params)} params)")

    # Count parameters
    counts = count_lora_parameters(model)
    print(f"  [INFO] Total params: {counts['total_params']:,}")
    print(f"  [INFO] LoRA params: {counts['lora_params']:,} ({counts['lora_percentage']:.2f}%)")

    assert counts['lora_params'] > 0
    assert counts['lora_percentage'] < 5  # Should be a small percentage
    print("  [PASS] Parameter counts look correct")

    print("  All prepare_model tests passed!\n")
    return True


def test_save_load_weights():
    """Test saving and loading LoRA weights."""
    print("=" * 60)
    print("Testing save/load LoRA weights...")
    print("=" * 60)

    import tempfile
    import os
    from models.lora import LoRAConfig, prepare_model_for_lora_training, save_lora_weights, load_lora_weights

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'q_proj': nn.Linear(64, 64),
                    'v_proj': nn.Linear(64, 64),
                })
                for _ in range(2)
            ])

    model1 = SimpleModel()
    model2 = SimpleModel()

    config = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])

    prepare_model_for_lora_training(model1, config, llm_backbone_prefix="")
    prepare_model_for_lora_training(model2, config, llm_backbone_prefix="")

    # Modify model1's LoRA weights
    for module in model1.modules():
        if hasattr(module, 'lora_A'):
            with torch.no_grad():
                module.lora_A.weight.fill_(0.5)
                module.lora_B.weight.fill_(0.5)

    # Save model1's weights
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "lora_weights.pt")
        save_lora_weights(model1, save_path)
        print(f"  [PASS] Saved weights to {save_path}")

        # Load into model2
        load_lora_weights(model2, save_path)
        print("  [PASS] Loaded weights")

        # Check weights match
        for (n1, m1), (n2, m2) in zip(model1.named_modules(), model2.named_modules()):
            if hasattr(m1, 'lora_A'):
                assert torch.allclose(m1.lora_A.weight, m2.lora_A.weight)
                assert torch.allclose(m1.lora_B.weight, m2.lora_B.weight)
        print("  [PASS] Weights match after loading")

    print("  All save/load tests passed!\n")
    return True


def test_with_showo2_model():
    """Test LoRA integration with actual Showo2Qwen2_5 model structure."""
    print("=" * 60)
    print("Testing with Showo2Qwen2_5 model structure...")
    print("=" * 60)

    # This test creates a mock of the Qwen2 model structure
    # to verify LoRA injection works with the actual module paths

    from models.lora import LoRAConfig, inject_lora, count_lora_parameters

    # Create a mock model with Qwen2-like structure
    class MockQwen2Attention(nn.Module):
        def __init__(self, hidden_size=64):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

    class MockQwen2MLP(nn.Module):
        def __init__(self, hidden_size=64, intermediate_size=256):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size)
            self.up_proj = nn.Linear(hidden_size, intermediate_size)
            self.down_proj = nn.Linear(intermediate_size, hidden_size)

    class MockQwen2DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = MockQwen2Attention()
            self.mlp = MockQwen2MLP()

    class MockQwen2Model(nn.Module):
        def __init__(self, num_layers=4):
            super().__init__()
            self.embed_tokens = nn.Embedding(1000, 64)
            self.layers = nn.ModuleList([MockQwen2DecoderLayer() for _ in range(num_layers)])

    class MockQwen2ForCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MockQwen2Model()
            self.lm_head = nn.Linear(64, 1000)

    class MockShowo2(nn.Module):
        def __init__(self):
            super().__init__()
            self.showo = MockQwen2ForCausalLM()
            self.image_embedder = nn.Linear(16, 64)
            self.diffusion_head = nn.Linear(64, 16)

    model = MockShowo2()

    # Count parameters before
    total_before = sum(p.numel() for p in model.parameters())

    # Inject LoRA with the actual prefix used in Showo2
    config = LoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    injected = inject_lora(model, config, prefix="showo.model")

    # Should inject into 4 layers * 4 projections = 16 modules
    assert len(injected) == 16, f"Expected 16, got {len(injected)}"
    print(f"  [PASS] Injected into {len(injected)} modules")

    # Verify paths
    for path in injected.keys():
        assert path.startswith("showo.model.layers")
        assert any(t in path for t in ["q_proj", "k_proj", "v_proj", "o_proj"])
    print("  [PASS] All injection paths correct")

    # Freeze and check
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if hasattr(module, 'lora_A'):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        m.lora_A.weight.numel() + m.lora_B.weight.numel()
        for m in model.modules() if hasattr(m, 'lora_A')
    )

    assert trainable == lora_params
    print(f"  [PASS] LoRA params: {lora_params:,} ({100*lora_params/total_before:.2f}% of total)")

    print("  All Showo2 structure tests passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running LoRA Implementation Tests")
    print("=" * 60 + "\n")

    tests = [
        ("LoRAConfig", test_lora_config),
        ("LoRALinear", test_lora_linear),
        ("LoRA Injection", test_lora_injection),
        ("Prepare Model", test_prepare_model_for_training),
        ("Save/Load Weights", test_save_load_weights),
        ("Showo2 Structure", test_with_showo2_model),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, traceback.format_exc()))
            print(f"  [FAIL] {name}: {e}\n")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, error in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status} {name}")
        if error:
            print(f"         Error: {error[:100]}...")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
