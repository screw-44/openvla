#!/usr/bin/env python3
"""
download_models.py

下载 VLA 训练所需的新模型权重到 HuggingFace 缓存中

支持的模型:
- Qwen3-VL (Qwen2-VL) 2B/7B/72B
- DistilGPT2

使用方法:
    python download_models.py --all                    # 下载所有模型
    python download_models.py --model qwen3-vl-2b      # 下载特定模型
    python download_models.py --model distilgpt2      # 下载 DistilGPT2
"""

import argparse
import os
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download

# 模型映射表
MODEL_REGISTRY = {
    "qwen3-vl-2b": {
        "hf_path": "Qwen/Qwen3-VL-2B-Instruct",
        "size": "~4.5GB",
        "description": "Qwen3-VL 2B Instruct (统一多模态模型)",
    },
    "qwen3-vl-4b": {
        "hf_path": "Qwen/Qwen3-VL-4B-Instruct",
        "size": "~10GB",
        "description": "Qwen3-VL 4B Instruct (统一多模态模型)",
    },
    "qwen3-vl-7b": {
        "hf_path": "Qwen/Qwen3-VL-7B-Instruct",
        "size": "~15GB",
        "description": "Qwen3-VL 7B Instruct (统一多模态模型)",
    },
    "distilgpt2": {
        "hf_path": "distilgpt2",
        "size": "~320MB",
        "description": "DistilGPT2 (轻量级语言模型，用于调试)",
    },
}


def download_model(model_name: str, force: bool = False) -> None:
    """
    下载指定模型到 HuggingFace 缓存
    
    使用 snapshot_download 避免立即加载模型导致的兼容性问题
    
    Args:
        model_name: 模型名称 (e.g., 'qwen3-vl-2b', 'distilgpt2')
        force: 是否强制重新下载（即使已缓存）
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_info = MODEL_REGISTRY[model_name]
    hf_path = model_info["hf_path"]
    size = model_info["size"]
    description = model_info.get("description", "")
    
    print(f"\n{'='*60}")
    print(f"📦 下载模型: {model_name}")
    print(f"   说明: {description}")
    print(f"   HF 路径: {hf_path}")
    print(f"   预计大小: {size}")
    print(f"{'='*60}\n")
    
    try:
        # 使用 snapshot_download 下载整个模型仓库
        # 这样避免了立即加载模型可能导致的版本不兼容问题
        print(f"正在下载 {hf_path}...")
        print(f"提示: 使用 snapshot_download 下载完整模型文件")
        
        cache_dir = snapshot_download(
            repo_id=hf_path,
            repo_type="model",
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # 跳过不需要的格式
            local_files_only=False,
            force_download=force,
        )
        
        print(f"✅ 模型已下载到: {cache_dir}")
        print(f"🎉 {model_name} 下载完成!\n")
        
    except Exception as e:
        print(f"\n❌ 下载 {model_name} 时出错: {e}\n")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="下载 VLA 训练所需的模型权重到 HuggingFace 缓存"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="指定要下载的模型",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有模型（警告：总大小 ~165GB）",
    )
    parser.add_argument(
        "--skip-72b",
        action="store_true",
        help="跳过 72B 模型（与 --all 一起使用）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使已缓存）",
    )
    
    args = parser.parse_args()
    
    # 确定要下载的模型列表
    if args.all:
        models_to_download = list(MODEL_REGISTRY.keys())
        if args.skip_72b:
            models_to_download = [m for m in models_to_download if m != "qwen3-vl-72b"]
            print("⚠️  跳过 Qwen3-VL 72B 模型")
    elif args.model:
        models_to_download = [args.model]
    else:
        # 默认下载小模型用于调试
        print("未指定模型，默认下载轻量级模型：")
        models_to_download = ["qwen3-vl-4b"]
    
    # 显示下载计划
    print("\n" + "="*60)
    print("📋 下载计划:")
    total_size = 0
    for model_name in models_to_download:
        info = MODEL_REGISTRY[model_name]
        print(f"  - {model_name:20s} ({info['size']})")
    print("="*60)
    
    # 确认下载
    if args.all and not args.skip_72b:
        response = input("\n⚠️  将下载所有模型（~165GB），确认继续？[y/N]: ")
        if response.lower() != 'y':
            print("已取消下载")
            return
    
    # 开始下载
    print("\n🚀 开始下载...\n")
    success_count = 0
    failed_models = []
    
    for i, model_name in enumerate(models_to_download, 1):
        print(f"\n{'#'*60}")
        print(f"# 进度: {i}/{len(models_to_download)}")
        print(f"{'#'*60}")
        
        try:
            download_model(model_name, force=args.force)
            success_count += 1
        except Exception as e:
            print(f"❌ 跳过 {model_name}")
            failed_models.append(model_name)
    
    # 总结
    print("\n" + "="*60)
    print("📊 下载总结:")
    print(f"  ✅ 成功: {success_count}/{len(models_to_download)}")
    if failed_models:
        print(f"  ❌ 失败: {', '.join(failed_models)}")
    print("="*60)
    
    # 显示 HF 缓存位置
    hf_cache = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface")
    )
    print(f"\n💾 模型已缓存到: {hf_cache}")
    print("\n✨ 现在可以运行训练了！")


if __name__ == "__main__":
    main()
