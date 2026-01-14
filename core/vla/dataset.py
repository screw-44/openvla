"""
用lerobot3.0数据集格式，高效率的实现dataset的读取。

使用LeRobotDatasetMetadata先过滤task，然后用LeRobotDataset加载指定的episodes。

核心功能:
1. 支持按task_ids过滤episodes
2. 支持限制每个task加载的episode数量
3. 为每个样本添加future_actions（从当前到episode结束的所有actions）
4. 可配置的处理频率(process_hz)和batch变换
"""
import torch
import json
import numpy as np
import random

from time import time
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from core.models.backbones.llm.prompting import PurePromptBuilder
from core.models.backbones.vision.base_vision import ImageTransform
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from core.vla.trajectory_compression import BaseTrajectoryCompression, BiningTrajectoryCompression
from core.vla.tokenizer import VlaTokenizer, BaseTrajectoryConverter
from core.util.overwatch import initialize_overwatch

# 不同的Dataset有不同的key映射，uniform_key
DATASET_ITEM_MAP_KEYS ={
    'HuggingFaceVLA/libero': {
        'cam1': 'observation.images.image', # 还有 observation.images.image2 (两个camera)
        'cam2': 'observation.images.image2',
        'language': 'task',
    },
}

class MyLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            repo_id: str, 
            image_transform: ImageTransform,
            tokenizer: VlaTokenizer,
            trajectory_compression: BaseTrajectoryCompression,
            real_root:Path=Path("/inspire/hdd/project/robot-decision/public/datasets/"), 
            **kwargs
        ):
        self.repo_id = repo_id
        self.tokenizer = tokenizer 
        self.traj_compress = trajectory_compression
        self.root = real_root / repo_id
        # NOTE: 完全删除掉metadata这个类，直接离线处理，拿到episode index直接在get item中过滤。

        self.overwatch = initialize_overwatch(__name__)
        self._dataset = LeRobotDataset(
            repo_id,
            root=self.root,
            episodes=None, 
            image_transforms=image_transform,
        ) # NOTE：不需要采用专门的delta_timestamps了，我们是从离线获取的，所以简化代码了这里
        self.overwatch.info(f"training dataset length:{len(self._dataset)}") #, validate dataset length:{len(self.val_dataset)}")

    @property
    def dataset(self): return self._dataset 

    def __len__(self): return len(self.dataset)
        
    def __getitem__(self, index):
        # 根据是哪一个具体的数据集，拿到对应的数据
        item = self.dataset.__getitem__(index)
        frame_index, episode_index = item['frame_index'], item['episode_index']

        # 这里扩展到了两图输入的libero的格式（目前先focus在libero上）
        uni_key_item = dict(
            cam1=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam1']],
            cam2=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam2']],
            language=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['language']],
            trajectory=self.traj_compress(frame_index, episode_index),
            state=item['observation.state'],
            dataset_names=self.repo_id
        )

        return self.tokenizer.tokenize_batch(uni_key_item)
    

if __name__ == "__main__":
    def _test_my_dataset_full():
            """
            三种模式：
            - MODE="dataset": 只测 GT labels -> decode -> reconstruct（保持你原有功能/可视化风格）
            - MODE="model"  : 只测 model generate -> decode -> reconstruct
            - MODE="both"   : 两者都测 + 打印 GT vs Pred 对比（可视化仍各自单独保存，风格不变）
            """
            from pathlib import Path
            import numpy as np
            from tqdm import tqdm

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # ==================== 配置参数（写死，最简单） ====================
            MODE = "both"  # "dataset" | "model" | "both"
            NEED_DATASET = MODE in ("dataset", "both")
            NEED_MODEL = MODE in ("model", "both")

            # 采样策略（保持你之前默认：从9900开始，每100个取一个）
            START_IDX = 0
            STRIDE = 300

            # 是否每个样本暂停（你之前是必暂停；这里默认保持一致）
            PAUSE_EACH_SAMPLE = True

            # model 推理参数
            MAX_NEW_TOKENS = 10240

            config_dict = {
                "repo_id": "HuggingFaceVLA/libero",
                "compression_method": "bspline_v3",
                "converter_type": "bspline_v3",
                "dataset_root": Path("/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA/libero"),
            }

            # 你的训练输出目录（用于加载 vla 模型）
            config_path = "/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/outputs/" \
            "2026-01-13/04-48-57/qwen2.5-0.5b+b16+x7--1-qwen25-abs_aff_uniform_bspline_v3"
            ckpt_path = Path(config_path) / "checkpoints" / "step-015000-epoch-00-loss=0.1329.safetensors" # "latest-checkpoint.safetensors" # "step-035000-epoch-02-loss=0.0147.safetensors" # "step-085000-epoch-04-loss=0.0517.safetensors"

            print("=" * 80)
            print("【数据集/模型 编码-解码测试】")
            print("=" * 80)
            print(f"MODE: {MODE}")
            print(f"配置: {config_dict}\n")

            # ==================== 加载数据集（raw） ====================
            print("📖 正在加载完整数据集...")

            full_traj_dataset = LeRobotDataset(
                config_dict["repo_id"],
                root=config_dict["dataset_root"],
                delta_timestamps={"abs_aff": []},
            )
            print(f"✓ 完整数据集大小: {len(full_traj_dataset)}")

            # ==================== 加载 VLA 数据集 + tokenizer / transform（dataset-only 默认） ====================
            print("\n📊 正在加载 VLA 数据集...")
            from transformers import AutoTokenizer
            from core.util.vla_utils import get_vla_dataset

            # dataset-only 的默认 tokenizer / prompt（保持你原逻辑）
            base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

            prompt_builder_fn = PurePromptBuilder
            image_transform = None

            # ==================== 可选：加载模型并对齐 tokenizer / prompt / transform ====================
            vla = None
            if NEED_MODEL:
                import torch
                from omegaconf import OmegaConf
                from core.models.load import load

                run_dir = Path(config_path)

                # Hydra 输出一般在 .hydra/config.yaml；有些工程也会放在根目录 config.yaml
                cfg_path_candidates = [
                    run_dir / ".hydra" / "config.yaml",
                    run_dir / "config.yaml",
                ]
                cfg_path = None
                for p in cfg_path_candidates:
                    if p.exists():
                        cfg_path = p
                        break
                if cfg_path is None:
                    raise FileNotFoundError(f"找不到 config.yaml（已尝试: {cfg_path_candidates}）")

                cfg = OmegaConf.load(cfg_path)

                vla = load(
                    vla_cfg=cfg.vla,
                    checkpoint_path=ckpt_path,
                    load_for_training=False,
                )
                vla = vla.to(device="cuda", dtype=torch.bfloat16).eval()

                # 关键：推理必须与模型 tokenizer / prompt / image_transform 对齐
                base_tokenizer = vla.llm_backbone.get_tokenizer()
                prompt_builder_fn = vla.llm_backbone.prompt_builder_fn
                image_transform = vla.vision_backbone.get_image_transform()

            vla_dataset, trajectory_converter, collator = get_vla_dataset(
                data_repo_id=config_dict["repo_id"],
                data_task_ids=None,
                trajectory_compression_method=config_dict["compression_method"],
                trajectory_converter_type=config_dict["converter_type"],
                base_tokenizer=base_tokenizer,
                prompt_builder_fn=prompt_builder_fn,
                image_transform=image_transform,
            )
            print(f"✓ VLA 数据集大小: {len(vla_dataset)}")

            # ==================== 工具函数：拆分 prompt / action token ====================
            def split_prompt_and_action(labels_np: np.ndarray):
                if (labels_np == -100).any():
                    prompt_len = int(np.where(labels_np == -100)[0][-1] + 1)
                    return prompt_len, labels_np[prompt_len:]
                return 0, labels_np

            # ==================== decode -> reconstruct -> error（不改变你核心逻辑） ====================
            def decode_and_reconstruct(token_ids: np.ndarray, abs_aff_gt: np.ndarray, frame_index: int):
                decoded_cp = trajectory_converter.decode_text_ids_to_trajectory(token_ids)

                # 你原来就是这样拿两个 bspline
                bspline, gripper_bspline = vla_dataset.traj_compress.decode_to_action(decoded_cp)

                knots = decoded_cp[:, -1]
                num_samples = knots[-1].astype(int) + 1  # ✅ 按你要求保留
                t_eval = np.arange(num_samples)

                reconstructed = np.zeros((num_samples, 7))
                reconstructed[:, :6] = bspline(t_eval)

                # ✅ 按你要求：删除 random 扰动；使用真实 current pose（前6维）
                current_pose = abs_aff_gt[frame_index][:6]
                print("current_pose is:", current_pose)
                print("bspline 0 pose: ", bspline(0))

                if False and num_samples > 20:
                    offset = vla_dataset.traj_compress.start_offset(current_pose, bspline(10))
                    print("offset:", offset)
                    L = offset.shape[0]
                    reconstructed[:L, :6] += offset

                reconstructed[:, 6] = gripper_bspline(t_eval)

                gt_segment = abs_aff_gt[frame_index : frame_index + num_samples]
                if len(reconstructed) != len(gt_segment):
                    print("⚠️ 重建长度与GT长度不匹配，进行裁剪对齐")
                    min_len = min(len(reconstructed), len(gt_segment))
                    errors = np.abs(reconstructed[:min_len] - gt_segment[:min_len])
                else:
                    errors = np.abs(reconstructed - gt_segment)

                return decoded_cp, reconstructed, gt_segment, errors, float(np.mean(errors)), float(np.max(errors)), float(np.std(errors))

            # ==================== 可视化（保持你之前代码风格一致） ====================
            def plot_like_before(decoded_cp, reconstructed, gt_segment, frame_index, episode_index, idx, stats_list, output_path: Path, title_prefix: str | None):
                # 时间轴
                gt_t = np.arange(len(gt_segment)) + frame_index
                knots_vis = decoded_cp[:, -1] + frame_index
                t_eval_vis = np.arange(len(reconstructed)) + frame_index

                fig, axes = plt.subplots(4, 2, figsize=(16, 14))
                axes = axes.flatten()
                dims_to_plot = [0, 1, 2, 3, 4, 5]
                dim_names = ["x", "y", "z", "yaw", "pitch", "roll"]

                for i, (dim, dim_name) in enumerate(zip(dims_to_plot, dim_names)):
                    ax = axes[i]
                    ax.plot(gt_t, gt_segment[:, dim], label=f"GT {dim_name}", linewidth=2, alpha=0.8, color="green")
                    ax.plot(t_eval_vis, reconstructed[:, dim], label=f"Reconstructed {dim_name}",
                            linestyle="--", linewidth=1.5, alpha=0.8, color="red")
                    ax.scatter(knots_vis, decoded_cp[:, dim], c="red", s=50, marker="x", label="Control Points", zorder=5)
                    ax.set_ylabel(dim_name)
                    ax.set_xlabel("Time (frames)")
                    ax.set_title(f"Dimension: {dim_name}")
                    ax.legend(loc="upper right")
                    ax.grid(True, alpha=0.3)

                # gripper
                axes[6].plot(gt_t, gt_segment[:, 6], label="GT gripper", linewidth=2, alpha=0.8, color="green")
                axes[6].plot(t_eval_vis, reconstructed[:, 6], label="Reconstructed gripper",
                            linestyle="--", linewidth=1.5, alpha=0.8, color="red")
                axes[6].scatter(knots_vis, decoded_cp[:, 6], c="red", s=50, marker="x", label="Control Points", zorder=5)
                axes[6].set_ylabel("gripper")
                axes[6].set_xlabel("Time (frames)")
                axes[6].set_title("Dimension: gripper")
                axes[6].legend(loc="upper right")
                axes[6].grid(True, alpha=0.3)

                # 统计信息（保持你之前“累计统计”的显示方式）
                mean_errs = np.array([s["mean_error"] for s in stats_list], dtype=np.float64)
                max_errs = np.array([s["max_error"] for s in stats_list], dtype=np.float64)
                std_errs = np.array([s["std_error"] for s in stats_list], dtype=np.float64)

                axes[7].axis("off")
                stats_text = f"""Error Statistics:
        Mean Error:  {np.mean(mean_errs):.6f} +- {np.std(mean_errs):.6f}
        Max Error:   {np.mean(max_errs):.6f} +- {np.std(max_errs):.6f}
        Std Error:   {np.mean(std_errs):.6f} +- {np.std(std_errs):.6f}
        Visualized Sample:
        Control_points: {len(decoded_cp)}
        Original Length: {len(gt_segment)}
        Compression Ratio: {len(gt_segment)/len(decoded_cp):.2f}
        Episode: {episode_index}
        Frame: {frame_index}
        Index: {idx}
    """
                axes[7].text(0.1, 0.5, stats_text, fontsize=10,
                            verticalalignment="center",
                            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
                axes[7].set_visible(True)

                title = f"Trajectory Reconstruction (GT vs Reconstructed) - Ep{episode_index}, Frame{frame_index}"
                if title_prefix:
                    title = f"{title_prefix} | " + title

                plt.suptitle(title, fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()

            # ==================== 遍历所有样本进行测试 ====================
            print("🔄 正在遍历并测试所有样本...\n")

            gt_error_stats = []
            pred_error_stats = []

            sample_count = 0
            gt_success = 0
            pred_success = 0
            skip_count = 0

            # （你之前有 filter_index 逻辑，这里保持注释，不影响）
            # filter_index = vla_dataset.filter_index

            for idx in tqdm(range(START_IDX, len(full_traj_dataset), STRIDE), desc="处理中"):
                raw_item = full_traj_dataset[idx]
                frame_index = int(raw_item["frame_index"])
                episode_index = int(raw_item["episode_index"])

                # if episode_index in filter_index:
                #     skip_count += 1
                #     print("skipping episode index:", episode_index)
                #     continue

                sample_count += 1

                # raw 轨迹（差分 -> 绝对）
                abs_aff = raw_item["abs_aff"].numpy()
                abs_aff_gt = abs_aff.copy()
                abs_aff_gt[:, :-1] = np.cumsum(abs_aff_gt[:, :-1], axis=0)

                vla_item = vla_dataset[idx]
                labels = vla_item["labels"].numpy()
                prompt_len, gt_action_ids = split_prompt_and_action(labels)

                gt_res = None
                pred_res = None

                # ========= 1) GT(dataset) 评估（保持你原逻辑/可视化） =========
                if NEED_DATASET:
                    if len(gt_action_ids) < 8:
                        print(f"[GT] skip: action token too short. idx={idx}")
                    else:
                        decoded_cp, reconstructed, gt_segment, errors, mean_e, max_e, std_e = decode_and_reconstruct(
                            gt_action_ids, abs_aff_gt, frame_index
                        )
                        gt_success += 1
                        gt_error_stats.append({
                            "sample_idx": idx,
                            "episode_idx": episode_index,
                            "frame_idx": frame_index,
                            "mean_error": mean_e,
                            "max_error": max_e,
                            "std_error": std_e,
                        })

                        # 你的输出路径保持不变
                        out_path = Path("/tmp/dataset_decode_error_analysis.png")
                        plot_like_before(
                            decoded_cp=decoded_cp,
                            reconstructed=reconstructed,
                            gt_segment=gt_segment,
                            frame_index=frame_index,
                            episode_index=episode_index,
                            idx=idx,
                            stats_list=gt_error_stats,
                            output_path=out_path,
                            title_prefix=None,  # 保持你之前标题格式
                        )
                        print(f"✓ [GT] 可视化已保存: {out_path}")
                        gt_res = (mean_e, max_e, std_e, decoded_cp, reconstructed, gt_segment)

                # ========= 2) Model 评估 =========
                if NEED_MODEL:
                    import torch
                    # print("full input ids:", vla_item["input_ids"].numpy())
                    # prompt 输入：只喂 prompt 部分（和你之前一致）
                    input_ids = vla_item["input_ids"][:prompt_len].unsqueeze(0).to("cuda")
                    attn = vla_item.get("attention_mask", None)
                    if attn is not None:
                        attn = attn[:prompt_len].unsqueeze(0).to("cuda")

                    pixel_values = vla_item.get("pixel_values", None)
                    if isinstance(pixel_values, dict):
                        pixel_values = {k: v.unsqueeze(0).to("cuda") for k, v in pixel_values.items()}
                    elif pixel_values is not None:
                        pixel_values = pixel_values.unsqueeze(0).to("cuda")

                    gen_kwargs = dict(
                        input_ids=input_ids,
                        use_cache=False,
                        do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    if attn is not None:
                        gen_kwargs["attention_mask"] = attn
                    if pixel_values is not None:
                        gen_kwargs["pixel_values"] = pixel_values

                    half_dtype = getattr(vla.llm_backbone, "half_precision_dtype", torch.bfloat16)

                    with torch.inference_mode(), torch.autocast("cuda", dtype=half_dtype):
                        from transformers.generation.utils import GenerationMixin
                        pred_ids = GenerationMixin.generate(vla, **gen_kwargs)

                    pred_ids = pred_ids[0].detach().cpu().numpy()
                    # print("model pred_ids:", pred_ids)
                    pred_action_ids = pred_ids[prompt_len:]

                    if pred_action_ids is None or len(pred_action_ids) < 8:
                        print(f"[Pred] skip: action token too short after sanitize. idx={idx}, len={0 if pred_action_ids is None else len(pred_action_ids)}")
                    else:
                        decoded_cp, reconstructed, gt_segment, errors, mean_e, max_e, std_e = decode_and_reconstruct(
                            pred_action_ids, abs_aff_gt, frame_index
                        )
                        pred_success += 1
                        pred_error_stats.append({
                            "sample_idx": idx,
                            "episode_idx": episode_index,
                            "frame_idx": frame_index,
                            "mean_error": mean_e,
                            "max_error": max_e,
                            "std_error": std_e,
                        })

                        out_path = Path("/tmp/model_decode_error_analysis.png")
                        plot_like_before(
                            decoded_cp=decoded_cp,
                            reconstructed=reconstructed,
                            gt_segment=gt_segment,
                            frame_index=frame_index,
                            episode_index=episode_index,
                            idx=idx,
                            stats_list=pred_error_stats,
                            output_path=out_path,
                            title_prefix="Pred",  # 只在模型图上加前缀，不影响你原 GT 图
                        )
                        print(f"✓ [Pred] 可视化已保存: {out_path}")
                        pred_res = (mean_e, max_e, std_e, decoded_cp, reconstructed, gt_segment, pred_action_ids)

                # ========= 3) both 模式下：打印 GT vs Pred 差距（不改你可视化） =========
                if MODE == "both" and (gt_res is not None) and (pred_res is not None):
                    gt_mean, gt_max, gt_std, _, _, _ = gt_res
                    pr_mean, pr_max, pr_std, _, _, _, pred_action_ids = pred_res

                    m = min(len(gt_action_ids), len(pred_action_ids))
                    match = float((gt_action_ids[:m] == pred_action_ids[:m]).mean()) if m > 0 else 0.0

                    print(f"[GT vs Pred] token match (prefix {m}): {match*100:.2f}% | GT_len={len(gt_action_ids)} Pred_len={len(pred_action_ids)}")
                    print(f"[GT recon]   mean={gt_mean:.6f} max={gt_max:.6f} std={gt_std:.6f}")
                    print(f"[Pred err]   mean={pr_mean:.6f} max={pr_max:.6f} std={pr_std:.6f}")

                if PAUSE_EACH_SAMPLE and (NEED_DATASET or NEED_MODEL):
                    input("按回车键继续下一个样本...")

            # ==================== 打印统计结果 ====================
            print("\n" + "=" * 80)
            print("【统计结果】")
            print("=" * 80)
            print(f"总处理样本数: {sample_count}")
            print(f"跳过的 episode: {skip_count}")

            if NEED_DATASET and len(gt_error_stats) > 0:
                mean_errs = np.array([s["mean_error"] for s in gt_error_stats])
                max_errs = np.array([s["max_error"] for s in gt_error_stats])
                std_errs = np.array([s["std_error"] for s in gt_error_stats])
                print("\n【GT(dataset) 误差统计】")
                print(f"  Mean Error: {np.mean(mean_errs):.6f} ± {np.std(mean_errs):.6f}")
                print(f"  Max Error:  {np.mean(max_errs):.6f} ± {np.std(max_errs):.6f}")
                print(f"  Std Error:  {np.mean(std_errs):.6f} ± {np.std(std_errs):.6f}")
                print(f"  成功样本数: {gt_success}")

            if NEED_MODEL and len(pred_error_stats) > 0:
                mean_errs = np.array([s["mean_error"] for s in pred_error_stats])
                max_errs = np.array([s["max_error"] for s in pred_error_stats])
                std_errs = np.array([s["std_error"] for s in pred_error_stats])
                print("\n【Model(Pred) 误差统计】")
                print(f"  Mean Error: {np.mean(mean_errs):.6f} ± {np.std(mean_errs):.6f}")
                print(f"  Max Error:  {np.mean(max_errs):.6f} ± {np.std(max_errs):.6f}")
                print(f"  Std Error:  {np.mean(std_errs):.6f} ± {np.std(std_errs):.6f}")
                print(f"  成功样本数: {pred_success}")

            print("\n✅ 测试完成！")
            return gt_error_stats, pred_error_stats

    _test_my_dataset_full()

