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

import hf_wrapper.configuration_vla 
from lerobot.scripts.lerobot_eval import main as lerobot_eval_main


def setup_eval(model_path: Path):
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

    config = base_path / "config.json"
    config_empty = {
        "type": "vla"
    }
    with open(config, "w") as f:
        json.dump(config_empty, f, indent=2)
        
    
    # 创建空的 processor 配置文件（LeRobot 要求，但 VLA 不需要任何处理）
    preprocessor_config = base_path / "policy_preprocessor.json"
    postprocessor_config = base_path / "policy_postprocessor.json"
    
    # Preprocessor 配置：添加 LeRobot 需要的占位步骤（实际不会影响 VLA 推理）
    preprocessor_empty = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}}
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cuda", "float_dtype": None}
            }
        ]
    }
    
    # Postprocessor 配置：空步骤即可
    postprocessor_empty = {
        "name": "policy_postprocessor",
        "steps": []
    }
    
    # 总是重新创建（覆盖旧文件）
    with open(preprocessor_config, "w") as f:
        json.dump(preprocessor_empty, f, indent=2)
    print(f"✅ 已创建 preprocessor 配置: {preprocessor_config.name}")
    
    with open(postprocessor_config, "w") as f:
        json.dump(postprocessor_empty, f, indent=2)
    print(f"✅ 已创建 postprocessor 配置: {postprocessor_config.name}")
    
    return base_path


def main():
    dir = "2025-12-30/16-00-49/qwen2.5-0.5b+b16+x7--1-qwen25-abs_aff_uniform_bspline"
    parser = argparse.ArgumentParser(
        description="简化的 VLA 评估脚本：直接读取 config.json，用软连接链接权重"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/output/" \
        f"{dir}/checkpoints/latest-checkpoint.safetensors", # latest-checkpoint  step-010000-epoch-00-loss=0.0934
        help="包含 config.json 的模型目录（如训练的 run 目录）",
    )
    # libero_10,libero_object,libero_spatial,libero_goal
    parser.add_argument("--env_task", default="libero_10", help="环境任务")
    parser.add_argument("--n_episodes", type=int, default=1, help="评估轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument(
        "--output_dir", default=f"./eval_results/{dir}", help="评估结果输出目录"
    )

    args = parser.parse_args()

    # 设置评估目录（创建软连接）
    base_path = setup_eval(args.model_path)

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
        f"--env.control_mode=relative",  # 设置为 relative ,absolute
        f"--output_dir={args.output_dir}",
    ]

    lerobot_eval_main()


if __name__ == "__main__":
    main()
