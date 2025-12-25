#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import argparse

# 确保 libero 缓存软连接指向 HF_HOME
libero_cache_src = Path("/root/.cache/libero")
libero_cache_dst = Path(
    "/inspire/ssd/project/robot-decision/hexinyu-253108100063/Software/libero"
)
if not libero_cache_src.exists() and not libero_cache_src.is_symlink():
    libero_cache_src.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(libero_cache_dst, libero_cache_src)

import hf_wrapper  # HACK： Register VLA config，所以要import，尽管没有使用。不然找不到vlaconfig
from lerobot.scripts.lerobot_eval import main as lerobot_eval_main


def setup_eval_model_link(model_path: Path):
    """
    创建软连接指向模型权重。
    注：不加载config.json，因为config.json里是VLAConfig（模型配置），
    而LeRobot期望EvalConfig（评估配置）。评估参数通过命令行传入。
    """
    assert (
        model_path.exists() and model_path.suffix == ".safetensors"
    ), f"{model_path} 不存在或不是 .safetensors 文件"
    model_path = Path(model_path).resolve()
    base_path = model_path.parent.parent

    # 创建软连接到model_path（不涉及config.json）
    eval_model_link = base_path / "model.safetensors"
    if eval_model_link.is_symlink() or eval_model_link.exists():
        eval_model_link.unlink()

    os.symlink(model_path, eval_model_link)
    print(f"✅ 软连接已创建: {eval_model_link} → {model_path}")
    return base_path


def main():
    parser = argparse.ArgumentParser(
        description="简化的 VLA 评估脚本：直接读取 config.json，用软连接链接权重"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="/inspire/hdd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla_runs/" \
        "base+b64+x7--1-distilgpt2-aff-bining/checkpoints/latest-checkpoint.safetensors",
        help="包含 config.json 的模型目录（如训练的 run 目录）",
    )
    parser.add_argument("--env_task", default="libero_10", help="环境任务")
    parser.add_argument("--n_episodes", type=int, default=1, help="评估轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument(
        "--output_dir", default="./eval_results", help="评估结果输出目录"
    )

    args = parser.parse_args()

    # 设置评估目录（创建软连接）
    base_path = setup_eval_model_link(args.model_path)

    # 直接运行 lerobot-eval（不从config.json读取，避免VLAConfig混淆EvalConfig）
    print(f"\n🚀 运行 lerobot-eval...")
    sys.argv = [
        "lerobot-eval",
        f"--policy.path={base_path}",  # 指向包含model.safetensors的目录
        "--env.type=libero",
        f"--env.task={args.env_task}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--eval.batch_size={args.batch_size}",
        "--policy.device=cuda",
        f"--env.control_mode=relative",  # 设置为 False 即使用绝对位置
        f"--output_dir={args.output_dir}",
    ]

    lerobot_eval_main()


if __name__ == "__main__":
    main()
