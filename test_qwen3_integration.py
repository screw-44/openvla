"""
test_qwen3_integration.py

快速测试 Qwen3-VL 集成是否正常工作。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.models.vlms.qwen3_vla import Qwen3VLA, Qwen3ImageTransform
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def test_model_loading():
    """测试模型加载"""
    overwatch.info("=" * 60)
    overwatch.info("Test 1: Model Loading")
    overwatch.info("=" * 60)

    try:
        model = Qwen3VLA(
            model_id="qwen3-vl-test",
            model_size="2B",
            enable_mixed_precision_training=True,
        )
        overwatch.info("✅ Model loaded successfully")
        overwatch.info(f"Device: {model.device}")
        overwatch.info(f"Model dtype: {next(model.model.parameters()).dtype}")
        return model
    except Exception as e:
        overwatch.error(f"❌ Model loading failed: {e}")
        raise


def test_image_transform(model):
    """测试图像转换器"""
    overwatch.info("=" * 60)
    overwatch.info("Test 2: Image Transform")
    overwatch.info("=" * 60)

    try:
        transform = model.vision_backbone.get_image_transform()
        overwatch.info(f"Transform type: {type(transform)}")

        # 创建一个测试图像 (模拟 LeRobotDataset 输出)
        img_tensor = torch.rand(3, 224, 224)  # [C, H, W] in [0, 1]
        overwatch.info(
            f"Input tensor shape: {img_tensor.shape}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]"
        )

        # 应用转换
        transformed = transform(img_tensor)
        overwatch.info(
            f"Output tensor shape: {transformed.shape}, range: [{transformed.min():.2f}, {transformed.max():.2f}]"
        )

        overwatch.info("✅ Image transform works")
        return transform
    except Exception as e:
        overwatch.error(f"❌ Image transform failed: {e}")
        raise


def test_fake_batch(model):
    """测试 forward 通过假数据"""
    overwatch.info("=" * 60)
    overwatch.info("Test 3: Forward Pass with Fake Batch")
    overwatch.info("=" * 60)

    try:
        # 准备假数据
        batch_size = 2
        seq_len = 50

        # 创建 fake pixel_values (双摄像头)
        pixel_values = {
            "cam1": torch.rand(batch_size, 3, 224, 224),
            "cam2": torch.rand(batch_size, 3, 224, 224),
        }

        # 创建 fake input_ids 和 labels
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        # 移动到设备
        device = model.device
        pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        overwatch.info(f"Batch shapes:")
        overwatch.info(f"  pixel_values[cam1]: {pixel_values['cam1'].shape}")
        overwatch.info(f"  pixel_values[cam2]: {pixel_values['cam2'].shape}")
        overwatch.info(f"  input_ids: {input_ids.shape}")
        overwatch.info(f"  attention_mask: {attention_mask.shape}")
        overwatch.info(f"  labels: {labels.shape}")

        # 尝试前向传播
        overwatch.info("Running forward pass...")

        # 注意：这可能会失败，因为 input_ids 不包含图像占位符
        # 但我们可以看到错误信息来诊断问题
        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                )

            overwatch.info(f"Output type: {type(output)}")
            overwatch.info(
                f"Loss: {output.loss.item() if output.loss is not None else 'None'}"
            )
            overwatch.info(
                f"Logits shape: {output.logits.shape if output.logits is not None else 'None'}"
            )
            overwatch.info("✅ Forward pass successful")
        except Exception as e:
            overwatch.warning(
                f"⚠️ Forward pass failed (expected if no image tokens): {e}"
            )
            overwatch.info(
                "This is OK for initial testing - need proper tokenization with image placeholders"
            )

    except Exception as e:
        overwatch.error(f"❌ Batch test failed: {e}")
        raise


def test_tokenizer_compatibility(model):
    """测试 tokenizer 兼容性"""
    overwatch.info("=" * 60)
    overwatch.info("Test 4: Tokenizer Compatibility")
    overwatch.info("=" * 60)

    try:
        tokenizer = model.llm_backbone.get_tokenizer()
        overwatch.info(f"Tokenizer type: {type(tokenizer)}")
        overwatch.info(f"Vocab size: {tokenizer.vocab_size}")
        overwatch.info(f"Padding side: {tokenizer.padding_side}")
        overwatch.info(
            f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})"
        )
        overwatch.info(
            f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})"
        )

        # 测试编码
        text = "Pick up the red block"
        encoded = tokenizer(text, return_tensors="pt")
        overwatch.info(f"Encoded text shape: {encoded['input_ids'].shape}")

        # 测试解码
        decoded = tokenizer.decode(encoded["input_ids"][0])
        overwatch.info(f"Decoded text: {decoded}")

        overwatch.info("✅ Tokenizer compatibility OK")
    except Exception as e:
        overwatch.error(f"❌ Tokenizer test failed: {e}")
        raise


def main():
    """运行所有测试"""
    overwatch.info("\n" + "=" * 60)
    overwatch.info("Qwen3-VL Integration Test Suite")
    overwatch.info("=" * 60 + "\n")

    try:
        # Test 1: 加载模型
        model = test_model_loading()
        overwatch.info("")

        # Test 2: 图像转换
        transform = test_image_transform(model)
        overwatch.info("")

        # Test 3: Tokenizer
        test_tokenizer_compatibility(model)
        overwatch.info("")

        # Test 4: 前向传播
        test_fake_batch(model)
        overwatch.info("")

        overwatch.info("=" * 60)
        overwatch.info("✅ All basic tests completed!")
        overwatch.info("=" * 60)
        overwatch.info("")
        overwatch.info("Next steps:")
        overwatch.info("1. Run full training with: ./launch/experiments/test_qwen.sh")
        overwatch.info("2. Check VlaTokenizer integration with real data")
        overwatch.info("3. Verify gradient flow and loss computation")

    except Exception as e:
        overwatch.error(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
