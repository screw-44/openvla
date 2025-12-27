#!/usr/bin/env python3
"""
download_models.py

下载 VLA 训练所需的新模型权重到 HuggingFace 缓存中

支持的模型:
- Qwen2.5-0.5B-Instruct (推荐用于 VLA LLM Backbone)
- Qwen2-VL 2B/7B (用于对比实验)
- DistilGPT2 (旧调试模型)

使用方法:
    python download_models.py                          # 默认下载 Qwen2.5-0.5B
    python download_models.py --model qwen2.5-0.5b     # 显式指定下载
    python download_models.py --all                    # 下载所有
"""

import argparse
import os
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download

# 模型映射表
MODEL_REGISTRY = {
    # ✅ [核心] 你的新主力 LLM Backbone
    "qwen2.5-0.5b": {
        "hf_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "size": "~1.2GB",
        "description": "Qwen2.5 0.5B Instruct (VLA 训练最佳小模型 backbone)",
    },
    # VLA 视觉塔 (Vision Backbone) - Prismatic 默认使用 SigLIP
    "siglip-so400m": {
        "hf_path": "google/siglip-so400m-patch14-384",
        "size": "~1.8GB",
        "description": "SigLIP So400M (VLA 推荐 Vision Backbone)",
    },
    # 旧调试模型
    "distilgpt2": {
        "hf_path": "distilgpt2",
        "size": "~320MB",
        "description": "DistilGPT2 (旧调试模型)",
    },
    # 其它全量 VLM (如果你想跑对比实验)
    "qwen2-vl-2b": {
        "hf_path": "Qwen/Qwen2-VL-2B-Instruct",
        "size": "~4.5GB",
        "description": "Qwen2-VL 2B Instruct (基准对比模型)",
    },
}


def download_model(model_name: str, force: bool = False) -> None:
    """
    下载指定模型到 HuggingFace 缓存
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
        print(f"正在下载 {hf_path}...")
        print(f"提示: 使用 snapshot_download 下载完整模型文件")
        
        # 核心下载逻辑
        cache_dir = snapshot_download(
            repo_id=hf_path,
            repo_type="model",
            # 排除不必要的超大文件，只下载 safetensors
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"], 
            local_files_only=False,
            force_download=force,
        )
        
        print(f"✅ 模型已下载到: {cache_dir}")
        print(f"🎉 {model_name} 下载完成!\n")
        
    except Exception as e:
        print(f"\n❌ 下载 {model_name} 时出错: {e}\n")
        print(f"建议: 请检查网络连接，或者是否配置了 HF_ENDPOINT 镜像\n")
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
        help="下载所有注册的模型",
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
    elif args.model:
        models_to_download = [args.model]
    else:
        # ✅ 默认修改为下载 Qwen2.5-0.5B
        print("未指定模型，默认下载 VLA 所需的核心组件：")
        models_to_download = ["qwen2.5-0.5b", "siglip-so400m"]
    
    # 显示下载计划
    print("\n" + "="*60)
    print("📋 下载计划:")
    for model_name in models_to_download:
        info = MODEL_REGISTRY[model_name]
        print(f"  - {model_name:20s} ({info['size']})")
    print("="*60)
    
    # 确认下载
    if args.all:
        response = input("\n⚠️  将下载所有模型，确认继续？[y/N]: ")
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
    
    # 显示 HF 缓存位置
    hf_cache = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface")
    )
    print(f"💾 模型缓存路径: {hf_cache}")
    print("="*60)


if __name__ == "__main__":
    main()