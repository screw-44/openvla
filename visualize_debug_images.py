"""
可视化脚本：对比训练和推理时的图像差异
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 定义图像路径
root = Path("/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla")
train_debug_path = root / "train)_debug_cam1_cam2_concat.png"
inference_debug_path = root / "scripts/debug_cam1_cam2_concat.png"

print(f"Looking for images:")
print(f"  Train debug: {train_debug_path}")
print(f"  Inference debug: {inference_debug_path}")

# 加载图像
if train_debug_path.exists():
    train_img = cv2.imread(str(train_debug_path))
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    print(f"✓ Loaded train image: shape={train_img.shape}")
else:
    print(f"✗ Train image not found at {train_debug_path}")
    train_img = None

if inference_debug_path.exists():
    infer_img = cv2.imread(str(inference_debug_path))
    infer_img = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
    print(f"✓ Loaded inference image: shape={infer_img.shape}")
else:
    print(f"✗ Inference image not found at {inference_debug_path}")
    infer_img = None

if train_img is None or infer_img is None:
    print("\n❌ Unable to proceed - one or both images are missing")
    exit(1)

# 确保两个图像大小一致
if train_img.shape != infer_img.shape:
    print(
        f"\n⚠️ Image shapes differ: train={train_img.shape}, inference={infer_img.shape}"
    )
    # 调整到较小的尺寸
    min_h = min(train_img.shape[0], infer_img.shape[0])
    min_w = min(train_img.shape[1], infer_img.shape[1])
    train_img = train_img[:min_h, :min_w]
    infer_img = infer_img[:min_h, :min_w]
    print(f"  Resized to: {train_img.shape}")

# ===== 1. 找出训练图像中的白色像素 =====
white_mask = np.all(train_img == [255, 255, 255], axis=2)
white_regions = np.zeros_like(train_img)
white_regions[white_mask] = [255, 0, 0]  # 标记为红色

print(
    f"\n📊 White pixels in train image: {white_mask.sum()} / {white_mask.size} ({100*white_mask.sum()/white_mask.size:.2f}%)"
)

# ===== 2. 计算两个图像的差异 =====
diff = np.abs(train_img.astype(np.float32) - infer_img.astype(np.float32))
diff_gray = np.mean(diff, axis=2)
print(f"📊 Pixel difference stats:")
print(f"  Mean: {diff_gray.mean():.2f}")
print(f"  Max: {diff_gray.max():.2f}")
print(f"  Min: {diff_gray.min():.2f}")

# 二值化差异（阈值设为 10）
threshold = 10
diff_binary = (diff_gray > threshold).astype(np.uint8) * 255
print(
    f"  Pixels with diff > {threshold}: {(diff_gray > threshold).sum()} ({100*(diff_gray > threshold).sum()/diff_gray.size:.2f}%)"
)

# ===== 3. 可视化 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：原始图像
axes[0, 0].imshow(train_img)
axes[0, 0].set_title("Train Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(infer_img)
axes[0, 1].set_title("Inference Image")
axes[0, 1].axis("off")

axes[0, 2].imshow(white_regions)
axes[0, 2].set_title("White Pixels in Train (红色标记)")
axes[0, 2].axis("off")

# 第二行：差异分析
axes[1, 0].imshow(diff_gray, cmap="hot")
axes[1, 0].set_title(f"Pixel Difference (Mean={diff_gray.mean():.2f})")
axes[1, 0].colorbar = plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

axes[1, 1].imshow(diff_binary, cmap="gray")
axes[1, 1].set_title(f"Binary Diff (threshold={threshold})")
axes[1, 1].axis("off")

# 叠加差异在原始图像上
overlay = infer_img.copy()
overlay[diff_gray > threshold] = [255, 0, 0]  # 差异区域标记为红色
axes[1, 2].imshow(overlay)
axes[1, 2].set_title("Difference Overlay on Inference")
axes[1, 2].axis("off")

plt.tight_layout()
output_path = root / "image_comparison.png"
plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
print(f"\n✓ Visualization saved to: {output_path}")

plt.show()

# ===== 4. 额外统计 =====
print("\n📈 Statistical Summary:")
print(f"Train image mean RGB: {train_img.mean(axis=(0,1))}")
print(f"Inference image mean RGB: {infer_img.mean(axis=(0,1))}")
print(f"Mean difference per channel: {diff.mean(axis=(0,1))}")
