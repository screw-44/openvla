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

        compression_json = Path(__file__).parent.parent.parent / "assets" / "compression_results_processed.json"
        with open(compression_json, "r") as f:
            offline_compression_results = json.load(f)
        self.compression_statics = offline_compression_results["compression_statics"]  #  存储的compression statics
        sorted_by_max_error = self.compression_statics["validation_statistics"]["sorted_by_max_error"]
        self.filter_index = [item["episode_idx"] for item in sorted_by_max_error]
        self.overwatch.info(f"Filter index length:{len(self.filter_index)}, and they are:{self.filter_index}") #, validate dataset length:{len(self.val_dataset)}")

    @property
    def dataset(self): return self._dataset 

    def __len__(self): return len(self.dataset)
        
    def __getitem__(self, index):
        # 根据是哪一个具体的数据集，拿到对应的数据
        item = self.dataset.__getitem__(index)
        frame_index, episode_index = item['frame_index'], item['episode_index']
        # 过滤error过大的indexs
        while index in self.filter_index:
            index = random.randint(0, len(self.dataset)-1)
        # 这里扩展到了两图输入的libero的格式（目前先focus在libero上）
        uni_key_item = dict(
            cam1=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam1']],
            cam2=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam2']],
            language=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['language']],
            trajectory=self.traj_compress(frame_index, episode_index),
            dataset_names=self.repo_id
        )

        return self.tokenizer.tokenize_batch(uni_key_item)
    

if __name__ == "__main__":
    def _test_my_dataset_full():
        """
        测试 dataset 的端到端编码-解码过程
        读取所有数据，进行 tokenize -> decode，对比重构误差
        """
        from tqdm import tqdm
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.util.vla_utils import get_vla_dataset
        
        # ==================== 配置参数 ====================
        config_dict = {
            "repo_id": "HuggingFaceVLA/libero",
            "compression_method": "bspline_v3",
            "converter_type": "bspline_v3",
            "dataset_root": Path("/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA/libero"),
        }
        
        print("="*80)
        print("【数据集编码-解码测试】")
        print("="*80)
        print(f"配置: {config_dict}\n")
        
        # ==================== 加载数据集 ====================
        print("📖 正在加载完整数据集...")
        full_traj_dataset = LeRobotDataset(
            config_dict["repo_id"],
            root=config_dict["dataset_root"],
            delta_timestamps={"abs_aff": []}
        )
        print(f"✓ 完整数据集大小: {len(full_traj_dataset)}")
        
        # ==================== 加载 VLA 数据集和 Tokenizer ====================
        print("\n📊 正在加载 VLA 数据集...")
        from transformers import AutoTokenizer
        
        base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        
        vla_dataset, trajectory_converter, _ = get_vla_dataset(
            data_repo_id=config_dict["repo_id"],
            data_task_ids=None,
            trajectory_compression_method=config_dict["compression_method"],
            trajectory_converter_type=config_dict["converter_type"],
            base_tokenizer=base_tokenizer,
            prompt_builder_fn=PurePromptBuilder,
            image_transform=None,
        )
        print(f"✓ VLA 数据集大小: {len(vla_dataset)}")
        
        # ==================== 获取需要过滤的 episode ====================
        filter_index = vla_dataset.filter_index
        # ==================== 遍历所有样本进行测试 ====================
        print("🔄 正在遍历并测试所有样本...\n")
        
        error_stats = []
        sample_count = 0
        success_count = 0
        skip_count = 0
        
        for idx in tqdm(range(9900, len(full_traj_dataset), 100), desc="处理中"):
            raw_item = full_traj_dataset[idx]
            frame_index = int(raw_item["frame_index"])
            episode_index = int(raw_item["episode_index"])
            
            # 过滤: 跳过需要跳过的 episode
            if episode_index in filter_index:
                skip_count += 1
                print("skipping episode index:", episode_index)
                continue
            
            sample_count += 1
            
            # 从 VLA 数据集获取 token, 拿到的就是slice好的数据
            vla_item = vla_dataset[idx]
            labels = vla_item['labels'].numpy()
            
            # 提取 action token (去掉 prompt 部分的 -100)
            if (labels == -100).any():
                first_valid_idx = np.where(labels == -100)[0][-1] + 1
                token_ids = labels[first_valid_idx:]
            else:
                token_ids = labels
            # print("labels:", labels)
            # print("token_ids:", token_ids)

            # 获取原始轨迹, index从0开始
            abs_aff = raw_item["abs_aff"].numpy()
            abs_aff_gt = abs_aff.copy()
            abs_aff_gt[:, :-1] = np.cumsum(abs_aff_gt[:, :-1], axis=0)

            print(f"🔄 正在解码 episode {episode_index} 从 frame {frame_index} 开始的轨迹...")
            print(f"   总长度: {len(abs_aff_gt)}, 控制点数量: {len(token_ids)//8}")

            # 解码回控制点
            decoded_cp = trajectory_converter.decode_text_ids_to_trajectory(token_ids)
            # print("decoded_cp:", decoded_cp)
            
            # 获取当前位姿并重建轨迹
            current_pose = abs_aff_gt[frame_index]
            bspline, gripper_traj = vla_dataset.traj_compress.decode_to_action(
                decoded_cp, current_eef_pose=current_pose
            )
            
            # 重建采样轨迹
            knots = decoded_cp[:, -1]
            # print("internal knots:", knots)
            num_samples = knots[-1].astype(int) + 1
            t_eval = np.arange(num_samples)
            
            reconstructed = np.zeros((num_samples, 7))
            reconstructed[:, :6] = bspline(t_eval)
            # bspline_gripper already contains the full trajectory from decode_to_action
            reconstructed[:, 6] = gripper_traj
            
            # 对比 ground truth（注意：bspline_gripper已经从frame_index开始，所以GT也要对应）
            gt_segment = abs_aff_gt[frame_index:frame_index + num_samples]
            
            # 计算误差
            errors = np.abs(reconstructed - gt_segment)
            result = {
                'success': True,
                'mean_error': np.mean(errors),
                'max_error': np.max(errors),
                'std_error': np.std(errors),
                'reconstructed': reconstructed,
                'gt_segment': gt_segment,
                'errors': errors,
            }

            
            if result['success']:
                success_count += 1
                error_stats.append({
                    'sample_idx': idx,
                    'episode_idx': episode_index,
                    'frame_idx': frame_index,
                    'mean_error': result['mean_error'],
                    'max_error': result['max_error'],
                    'std_error': result['std_error'],
                })

                # 时间轴
                gt_t = np.arange(len(gt_segment)) + frame_index
                knots = decoded_cp[:, -1]
                t_eval = np.arange(len(reconstructed)) + frame_index
                
                # 可视化: 对比6个维度 (x, y, z, yaw, pitch, roll) + gripper，2列布局
                fig, axes = plt.subplots(4, 2, figsize=(16, 14))
                axes = axes.flatten()  # 展平成1D数组便于索引
                dims_to_plot = [0, 1, 2, 3, 4, 5]
                dim_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
                
                # 绘制6个位置/姿态维度
                for i, (dim, dim_name) in enumerate(zip(dims_to_plot, dim_names)):
                    ax = axes[i]
                    ax.plot(gt_t, gt_segment[:, dim], label=f'GT {dim_name}', linewidth=2, alpha=0.8, color='green')
                    ax.plot(t_eval, reconstructed[:, dim], label=f'Reconstructed {dim_name}', 
                            linestyle='--', linewidth=1.5, alpha=0.8, color='red')
                    ax.scatter(knots, decoded_cp[:, dim], 
                            c='red', s=50, marker='x', label='Control Points', zorder=5)
                    ax.set_ylabel(dim_name)
                    ax.set_xlabel('Time (frames)')
                    ax.set_title(f'Dimension: {dim_name}')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
                
                # gripper维度
                axes[6].plot(gt_t, gt_segment[:, 6], label='GT gripper', linewidth=2, alpha=0.8, color='green')
                axes[6].plot(t_eval, reconstructed[:, 6], label='Reconstructed gripper', 
                            linestyle='--', linewidth=1.5, alpha=0.8, color='red')
                axes[6].scatter(knots, decoded_cp[:, 6], 
                            c='red', s=50, marker='x', label='Control Points', zorder=5)
                axes[6].set_ylabel('gripper')
                axes[6].set_xlabel('Time (frames)')
                axes[6].set_title('Dimension: gripper')
                axes[6].legend(loc='upper right')
                axes[6].grid(True, alpha=0.3)
                
                # 添加误差统计信息在最后一个子图
                mean_errs = np.array([s['mean_error'] for s in error_stats])
                max_errs = np.array([s['max_error'] for s in error_stats])
                std_errs = np.array([s['std_error'] for s in error_stats])
                axes[7].axis('off')
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
                        verticalalignment='center', bbox=dict(boxstyle='round', 
                        facecolor='lightblue', alpha=0.5))
                axes[7].set_visible(True)
                
                plt.suptitle(f'Trajectory Reconstruction (GT vs Reconstructed) - Ep{episode_index}, Frame{frame_index}', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                output_path = Path("/tmp/dataset_decode_error_analysis.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ 可视化已保存: {output_path}")

                input("按回车键继续下一个样本...")
                
        
        # ==================== 打印统计结果 ====================
        print("\n" + "="*80)
        print("【统计结果】")
        print("="*80)
        print(f"总处理样本数: {sample_count}")
        print(f"成功解码: {success_count}")
        print(f"跳过的 episode: {skip_count}")
        mean_errs = np.array([s['mean_error'] for s in error_stats])
        max_errs = np.array([s['max_error'] for s in error_stats])
        std_errs = np.array([s['std_error'] for s in error_stats])
        
        print(f"\n【误差统计】")
        print(f"  Mean Error: {np.mean(mean_errs):.6f} ± {np.std(mean_errs):.6f}")
        print(f"  Max Error:  {np.mean(max_errs):.6f} ± {np.std(max_errs):.6f}")
        print(f"  Std Error:  {np.mean(std_errs):.6f} ± {np.std(std_errs):.6f}")

        
        print("\n✅ 测试完成！")
        
        return error_stats
        
    _test_my_dataset_full()

    exit()

    DATA_INDEX = 30000-9 #135122+150 # 120736是数值范围影响最大的idx

    # 3w是那个ep 108 frame 9 拟合崩溃

    import hydra
    from omegaconf import OmegaConf, DictConfig
    from scipy.interpolate import BSpline
    
    # config_path = "/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/outputs/2025-12-30/16-00-49/qwen2.5-0.5b+b16+x7--1-qwen25-abs_aff_uniform_bspline"
    config_path = "/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/outputs/2026-01-06/09-55-16/qwen2.5-0.5b+b16+x7--1-qwen25-abs_aff_uniform_bspline_v2_test_converge_on_ep0"
    @hydra.main(config_path=config_path, config_name="config", version_base=None)
    def _test_my_dataset_full_with_model(cfg: DictConfig):
        """ 
        这个文件需要从MyLeRobotdataset中进行加载dataset(用bspline absolute的配置，你需要看一下别的代码来充分思考），配置dataloader，batch=1，拿到输出后。
        然后再进行decoding，变换回原始的轨迹（bspline解压）。然后我们看一下解压回去后的轨迹和原始的数据集中的轨迹有多少的差别。你需要帮我完成这个代码，
        然后最后除了print出来具体的差距距离（control point上的差距，还有整条轨迹上的max和mean的差距，最好再给我可视化一下
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from core.util.vla_utils import get_vla_dataset
        from core.models.load import load

        dataset_cfg = cfg.dataset
        vla_cfg = cfg.vla

        vla = load(
            vla_cfg=vla_cfg,
            checkpoint_path=Path(config_path) / "checkpoints" / "latest-checkpoint.safetensors", # "latest-checkpoint.safetensors",
            load_for_training=False
        )
        vla = vla.to(dtype=torch.bfloat16, device="cuda")

        vla_dataset, trajectory_converter, collator = get_vla_dataset(
            data_repo_id=dataset_cfg.repo_id,
            data_task_ids=dataset_cfg.get_task_ids() if hasattr(dataset_cfg, 'get_task_ids') else None,
            trajectory_compression_method=vla_cfg.trajectory.compression_method,
            trajectory_converter_type=vla_cfg.trajectory.converter_type,
            trajectory_n_bins=vla_cfg.trajectory.n_bins,
            trajectory_n_dims=vla_cfg.trajectory.n_dims,
            base_tokenizer=vla.llm_backbone.get_tokenizer(),
            prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
            image_transform=vla.vision_backbone.get_image_transform(),
        )
        

        # 取一个样本并走完encode->decode流程
        # ============= 0. 原始数据集中的输出数据 ============
        raw_item = vla_dataset.dataset[DATA_INDEX]
        frame_index = int(raw_item["frame_index"].item()) if "frame_index" in raw_item else 0
        episode_index = int(raw_item["episode_index"].item()) if "episode_index" in raw_item else 0
        abs_aff = raw_item["abs_aff"].numpy()  # [T,7] 差分形式
        np.set_printoptions(suppress=True, linewidth=200, precision=6)
        abs_aff_gt = abs_aff.copy()
        abs_aff_gt[:, :-1] = np.cumsum(abs_aff_gt[:, :-1], axis=0)
        print("abs_aff_gt top 3 point:", abs_aff_gt[:21], ", shape:", abs_aff_gt.shape)


        # ========== 1. 数据集压缩后的数值， 转绝对坐标，这就是ground truth ========
        # 压缩得到控制点（与 MyLeRobotDataset 内部一致）
        my_dataset_item = vla_dataset[DATA_INDEX]
        # print(my_dataset_item['input_ids'])
        print("lables:", my_dataset_item['labels'])
        labels = my_dataset_item['labels'].numpy()
        if (labels == -100).any():
            # 找到最后一个-100的下标，取其后所有数据
            first_valid_idx = np.where(labels == -100)[0][-1] + 1
            traj_token_ids = labels[first_valid_idx:]
        else:
            traj_token_ids = labels

        # ======== 2. 进行模型进行推理，拿到pred的坐标和解码 =========
        # from transformers import GenerationMixin
        # input_ids = my_dataset_item['input_ids'][:first_valid_idx].unsqueeze(0).to("cuda")
        # pixel_values = {k: v.unsqueeze(0).to("cuda") for k, v in my_dataset_item['pixel_values'].items()}
        # # print("input ids:", input_ids)
        # # print("pixel values:", pixel_values)
        # with torch.autocast("cuda", dtype=vla.llm_backbone.half_precision_dtype):
        #     pred_ids = GenerationMixin.generate(
        #         vla,
        #         input_ids=input_ids,
        #         # attention_mask=my_dataset_item['attention_mask'].unsqueeze(0),
        #         pixel_values=pixel_values,
        #         use_cache=False,
        #         do_sample=False,  # 强制贪心解码
        #         max_new_tokens=1024,
        #     )
        # pred_ids = pred_ids.cpu().numpy()
        # print("shape:", pred_ids.shape, "pred_ids: ", pred_ids)
        
        # ========== 3. 封装解码+重建+评估+可视化函数 =========
        def decode_reconstruct_and_evaluate(
            token_ids: np.ndarray,
            name: str,
            abs_aff_gt: np.ndarray,
            frame_index: int,
            episode_index: int,
            vla_dataset,
            output_dir: Path,
        ):
            """
            解码token ids -> 控制点 -> 重建轨迹 -> 计算误差 -> 可视化
            
            Args:
                token_ids: action token ids (1D numpy array)
                name: "GT" or "Pred"
                abs_aff_gt: ground truth absolute trajectory
                frame_index: 当前帧索引
                episode_index: episode索引
                vla_dataset: dataset对象
                output_dir: 输出目录
            
            Returns:
                dict: 包含控制点、轨迹、误差、图片路径等
            """
            print(f"\n{'='*60}")
            print(f"[{name}] Decoding and Evaluating")
            print(f"{'='*60}")
            
            try:
                # 编码 -> token ids -> 解码回控制点
                decoded_control_points = vla_dataset.tokenizer.trajectory_converter.decode_text_ids_to_trajectory(token_ids)
                print(f"\n=== [{name}] Decoded Control Points ===")
                print(f"Token IDs shape: {token_ids.shape}, Control Points shape: {decoded_control_points.shape}")
                print("Top 5 control points (pos + gripper + knot):")
                for i, cp in enumerate(decoded_control_points[:5]):
                    print(f"  CP[{i}]: pos={cp[:6]}, grip={cp[6]:.3f}, knot={cp[7]:.1f}")
                
                # 用decode_to_action获取下一步动作和bspline对象
                current_pose = abs_aff_gt[frame_index]
                print("frame id: ", frame_index, ", current_pose is:", current_pose)
                _, bspline = vla_dataset.traj_compress.decode_to_action(
                    decoded_control_points, current_eef_pose=current_pose
                )
                print(f"\n=== [{name}] Next Action (t=0.1s) ===")
                
                # 重建整条轨迹：用bspline在knot时间点上采样
                knot_times = decoded_control_points[:, -1]
                num_samples = 100  # 采样点数
                t_eval = np.linspace(0, knot_times[-1], num_samples)
                reconstructed_traj = np.zeros((num_samples, 7))
                reconstructed_traj[:, :6] = bspline(t_eval)
                # gripper用线性插值
                # reconstructed_traj[:, 6] = np.interp(t_eval, knot_times, decoded_control_points[:, 6])
                # NOTE 修改成0阶的插值方法 (这里修改很重要，下面第二行也没太看懂==================================)
                indices = np.searchsorted(knot_times, t_eval, side='right') - 1
                indices = np.clip(indices, 0, len(decoded_control_points) - 1)  # 添加这行
                reconstructed_traj[:, 6] = decoded_control_points[indices, 6]

                # 提取ground truth的对应片段（从frame_index开始）
                gt_segment = abs_aff_gt[frame_index:]

                print("gt_segment:", gt_segment[:3])
                
                # 控制点误差：对比控制点在knot时刻的位置和ground truth
                print(f"\n=== [{name}] Control Point Errors ===")
                knot_indices = np.clip(knot_times.astype(int), 0, len(gt_segment) - 1)
                gt_at_knots = gt_segment[knot_indices]
                cp_pos_err = np.linalg.norm(decoded_control_points[:, :6] - gt_at_knots[:, :6], axis=1)
                cp_grip_err = np.abs(decoded_control_points[:, 6] - gt_at_knots[:, 6])
                print(f"Position L2 error -> mean: {cp_pos_err.mean():.6f}, max: {cp_pos_err.max():.6f}, std: {cp_pos_err.std():.6f}")
                print(f"Gripper abs error -> mean: {cp_grip_err.mean():.6f}, max: {cp_grip_err.max():.6f}")
                
                # 整条轨迹误差：在采样点上对比
                print(f"\n=== [{name}] Trajectory Reconstruction Errors ===")
                # 对gt也做插值到相同时间点
                gt_t = np.arange(len(gt_segment))
                gt_interp = np.zeros((num_samples, 7))
                for dim in range(6):
                    gt_interp[:, dim] = np.interp(t_eval, gt_t, gt_segment[:, dim])
                gt_interp[:, 6] = np.interp(t_eval, gt_t, gt_segment[:, 6])
                
                traj_pos_err = np.linalg.norm(reconstructed_traj[:, :6] - gt_interp[:, :6], axis=1)
                traj_grip_err = np.abs(reconstructed_traj[:, 6] - gt_interp[:, 6])
                print(f"Position L2 error -> mean: {traj_pos_err.mean():.6f}, max: {traj_pos_err.max():.6f}, std: {traj_pos_err.std():.6f}")
                print(f"Gripper abs error -> mean: {traj_grip_err.mean():.6f}, max: {traj_grip_err.max():.6f}")
                
                # 可视化: 对比6个维度 (x, y, z, yaw, pitch, roll) + gripper，2列布局
                fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
                axes = axes.flatten()  # 展平成1D数组便于索引
                dims_to_plot = [0, 1, 2, 3, 4, 5]
                dim_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
                
                for i, (dim, dim_name) in enumerate(zip(dims_to_plot, dim_names)):
                    axes[i].plot(gt_t, gt_segment[:, dim], label=f'GT {dim_name}', linewidth=2, alpha=0.8, color='green')
                    axes[i].plot(t_eval, reconstructed_traj[:, dim], label=f'Reconstructed {dim_name}', 
                                linestyle='--', linewidth=1.5, alpha=0.8, color='red')
                    axes[i].scatter(knot_times, decoded_control_points[:, dim], 
                                   c='red', s=50, marker='x', label='Control Points', zorder=5)
                    axes[i].set_ylabel(dim_name)
                    axes[i].set_xlabel('Time (frames)')
                    axes[i].legend(loc='upper right')
                    axes[i].grid(True, alpha=0.3)
                
                # gripper维度
                axes[6].plot(gt_t, gt_segment[:, 6], label='GT gripper', linewidth=2, alpha=0.8, color='green')
                axes[6].plot(t_eval, reconstructed_traj[:, 6], label='Reconstructed gripper', 
                            linestyle='--', linewidth=1.5, alpha=0.8, color='red')
                axes[6].scatter(knot_times, decoded_control_points[:, 6], 
                               c='red', s=50, marker='x', label='Control Points', zorder=5)
                axes[6].set_ylabel('gripper')
                axes[6].set_xlabel('Time (frames)')
                axes[6].legend(loc='upper right')
                axes[6].grid(True, alpha=0.3)
                
                # 隐藏最后一个空的子图
                axes[7].set_visible(False)
                
                plt.suptitle(f'[{name}] B-Spline Reconstruction (ep={episode_index}, frame={frame_index})')
                plt.tight_layout()
                out_path = output_dir / f"bspline_reconstruct_{name}.png" # _{DATA_INDEX}
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"\n[{name}] 可视化已保存: {out_path}")
                
                # 返回结果
                return {
                    'control_points': decoded_control_points,
                    'knot_times': knot_times,
                    'reconstructed_traj': reconstructed_traj,
                    'gt_segment': gt_segment,
                    'errors': {
                        'cp_pos_mean': cp_pos_err.mean(),
                        'cp_pos_max': cp_pos_err.max(),
                        'cp_pos_std': cp_pos_err.std(),
                        'cp_grip_mean': cp_grip_err.mean(),
                        'cp_grip_max': cp_grip_err.max(),
                        'traj_pos_mean': traj_pos_err.mean(),
                        'traj_pos_max': traj_pos_err.max(),
                        'traj_pos_std': traj_pos_err.std(),
                        'traj_grip_mean': traj_grip_err.mean(),
                        'traj_grip_max': traj_grip_err.max(),
                    },
                    'fig_path': out_path,
                }
                
            except Exception as e:
                print(f"\n[{name}] ERROR: Failed to decode/reconstruct: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # ========== 4. 处理 GT 和 Pred，分别调用函数 =========
        output_dir = Path("/tmp")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 处理 GT (labels)
        print("\n" + "="*80)
        print("Processing Ground Truth (GT) Labels")
        print("="*80)
        gt_results = decode_reconstruct_and_evaluate(
            token_ids=traj_token_ids,
            name="GT",
            abs_aff_gt=abs_aff_gt,
            frame_index=frame_index,
            episode_index=episode_index,
            vla_dataset=vla_dataset,
            output_dir=output_dir,
        )
        
        return 
        # 处理 Pred (model predictions)
        print("\n" + "="*80)
        print("Processing Model Predictions (Pred)")
        print("="*80)
        
        # 提取 pred_ids 中的 action token 部分（去掉 prompt）
        pred_action_ids = pred_ids[0, first_valid_idx:]  # (batch=1, 去掉prompt)
        print(f"Pred action token IDs shape: {pred_action_ids.shape} (after removing prompt and EOS)")
        
        pred_results = decode_reconstruct_and_evaluate(
            token_ids=pred_action_ids,
            name="Pred",
            abs_aff_gt=abs_aff_gt,
            frame_index=frame_index,
            episode_index=episode_index,
            vla_dataset=vla_dataset,
            output_dir=output_dir,
        )
        
        # ========== 5. 对比 GT 和 Pred =========
        print("\n" + "="*80)
        print("GT vs Pred Comparison")
        print("="*80)
        
        if gt_results and pred_results:
            print(f"\n=== Shape Comparison ===")
            print(f"GT  token IDs: {traj_token_ids.shape}, Control Points: {gt_results['control_points'].shape}")
            print(f"Pred token IDs: {pred_action_ids.shape}, Control Points: {pred_results['control_points'].shape}")
            
            print(f"\n=== Error Comparison ===")
            print(f"{'Metric':<30} {'GT':>15} {'Pred':>15}")
            print(f"{'-'*60}")
            print(f"{'CP Position Mean Error':<30} {gt_results['errors']['cp_pos_mean']:>15.6f} {pred_results['errors']['cp_pos_mean']:>15.6f}")
            print(f"{'CP Position Max Error':<30} {gt_results['errors']['cp_pos_max']:>15.6f} {pred_results['errors']['cp_pos_max']:>15.6f}")
            print(f"{'Traj Position Mean Error':<30} {gt_results['errors']['traj_pos_mean']:>15.6f} {pred_results['errors']['traj_pos_mean']:>15.6f}")
            print(f"{'Traj Position Max Error':<30} {gt_results['errors']['traj_pos_max']:>15.6f} {pred_results['errors']['traj_pos_max']:>15.6f}")
            print(f"{'CP Gripper Mean Error':<30} {gt_results['errors']['cp_grip_mean']:>15.6f} {pred_results['errors']['cp_grip_mean']:>15.6f}")
            print(f"{'Traj Gripper Mean Error':<30} {gt_results['errors']['traj_grip_mean']:>15.6f} {pred_results['errors']['traj_grip_mean']:>15.6f}")
            
            # Token-level accuracy (如果长度一致)
            if len(traj_token_ids) == len(pred_action_ids):
                token_accuracy = (traj_token_ids == pred_action_ids).mean()
                print(f"\n=== Token-Level Accuracy ===")
                print(f"Token match rate: {token_accuracy*100:.2f}%")
                print(f"Matching tokens: {(traj_token_ids == pred_action_ids).sum()}/{len(traj_token_ids)}")
            else:
                print(f"\n=== Token-Level Accuracy ===")
                print(f"Cannot compute (length mismatch: GT={len(traj_token_ids)}, Pred={len(pred_action_ids)})")
            
            print(f"\n=== Output Files ===")
            print(f"GT visualization: {gt_results['fig_path']}")
            print(f"Pred visualization: {pred_results['fig_path']}")
        else:
            print("\n⚠️  One or both evaluations failed, cannot compare.")
        
        print("\n" + "="*80)
        print("Evaluation Complete!")
        print("="*80)
    
    # 禁用Hydra输出目录
    import sys
    sys.argv.extend(['hydra.run.dir=.', 'hydra.output_subdir=null'])
    

    for i in range(0, 200000, 2000):
        DATA_INDEX = i
        _test_my_dataset_full_with_model()








#     def _test_libero_datset():
#         dataset = LeRobotDataset(
#             'HuggingFaceVLA/libero',
#             root=Path("/inspire/hdd/project/robot-decision/public/datasets/")/'HuggingFaceVLA/libero',
#             delta_timestamps={"abs_aff":[]}
#         )

#         from tqdm import tqdm
#         dim_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
#         n_bins = 512
        
#         # 第一遍：收集所有 bspline 控制点数据 (T, 6)
#         all_data = []
#         i = 0
#         with tqdm(total=len(dataset), desc="Collecting data") as pbar:
#             while i < len(dataset):
#                 abs_aff = np.asarray(dataset[i]['abs_aff'])
#                 # 转换为绝对坐标
#                 abs_aff[:, :-1] = np.cumsum(abs_aff[:, :-1], axis=0)
#                 all_data.append(abs_aff[:, :6])  # 只保留 bspline 的 6 维
#                 # 如果换成压缩后的bspline的方式，还是结果很奇怪，所以就不管了，就把极端数值当作0.1和1%忽略了。
#                 seq_len = len(abs_aff)
#                 i += seq_len
#                 pbar.update(seq_len)
        
#         # 合并所有数据
#         all_data = np.vstack(all_data)  # shape: (total_samples, 6)
        
#         # 第二遍：用分位数计算 edges
#         q = np.linspace(0, 1, n_bins + 1)
#         edges = np.quantile(all_data, q, axis=0).T  # shape: (6, n_bins+1)
        
#         # 处理重复的边界值（某些维度可能有饱和）
#         for d in range(6):
#             for j in range(1, len(edges[d])):
#                 if edges[d, j] <= edges[d, j-1]:
#                     edges[d, j] = edges[d, j-1] + 1e-8
        
#         # 第三遍：统计每个维度的 bin 分布
#         print("\n" + "="*90)
#         print("QUANTILE BIN STATISTICS (n_bins=512)")
#         print("="*90)
        
#         bin_stats = []
#         for d in range(6):
#             e = edges[d]
#             bin_counts, _ = np.histogram(all_data[:, d], bins=e)
#             non_empty_bins = np.sum(bin_counts > 0)
#             max_count = np.max(bin_counts)
#             min_count = np.min(bin_counts[bin_counts > 0]) if np.any(bin_counts > 0) else 0
            
#             bin_stats.append({
#                 'dim': dim_names[d],
#                 'range': (e[0], e[-1]),
#                 'non_empty': non_empty_bins,
#                 'max_count': max_count,
#                 'min_count': min_count,
#                 'edges': e,
#                 'bin_counts': bin_counts
#             })
            
#             print(f"{dim_names[d]:6s}: Range=[{e[0]:10.4f}, {e[-1]:10.4f}] | "
#                   f"Non-empty bins={non_empty_bins:3d}/{n_bins} | "
#                   f"Max_bin={max_count:5d} samples | Min_bin={min_count:5d} samples")
        
#         # 可视化：绘制每个维度的 bin 分布密度
#         print("\n" + "="*90)
#         print("Generating bin distribution visualization...")
#         print("="*90)
        
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
        
#         fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#         axes = axes.flatten()
        
#         for d, stat in enumerate(bin_stats):
#             ax = axes[d]
#             dim_name = stat['dim']
#             edges_d = stat['edges']
#             bin_counts = stat['bin_counts']
            
#             # bin 的中心位置和宽度
#             bin_centers = (edges_d[:-1] + edges_d[1:]) / 2.0
#             bin_widths = edges_d[1:] - edges_d[:-1]
            
#             # 绘制柱状图（仅非零 bins）
#             mask_nonzero = bin_counts > 0
#             ax.bar(bin_centers[mask_nonzero], bin_counts[mask_nonzero], 
#                    width=bin_widths[mask_nonzero], alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
            
#             # 统计信息
#             ax.set_xlabel(f'{dim_name} Value', fontsize=11, fontweight='bold')
#             ax.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
#             ax.set_title(f'{dim_name.upper()} Bin Distribution\n'
#                         f'Range: [{edges_d[0]:.2f}, {edges_d[-1]:.2f}], '
#                         f'Non-empty: {stat["non_empty"]}/512', 
#                         fontsize=12, fontweight='bold')
#             ax.grid(True, alpha=0.3, axis='y')
            
#             # 添加密度统计信息
#             max_sample = stat['max_count']
#             min_sample = stat['min_count']
#             density_ratio = max_sample / min_sample if min_sample > 0 else float('inf')
            
#             info_text = f"Max: {max_sample}\nMin: {min_sample}\nRatio: {density_ratio:.1f}x"
#             ax.text(0.98, 0.97, info_text, transform=ax.transAxes, 
#                    fontsize=10, verticalalignment='top', horizontalalignment='right',
#                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
#         plt.suptitle('Quantile Bin Distribution Across Dimensions\n'
#                     '(Demonstrating Data-Adaptive Discretization)', 
#                     fontsize=14, fontweight='bold', y=0.995)
#         plt.tight_layout()
        
#         # 保存可视化
#         viz_path = Path("./libero_bin_distribution_visualization.png")
#         plt.savefig(viz_path, dpi=150, bbox_inches='tight')
#         print(f"✓ Visualization saved to: {viz_path}")
#         plt.close()
        
#         # 输出 numpy 数组格式（可直接复制到 tokenizer）
#         print("\n" + "="*90)
#         print("COPY THIS TO TOKENIZER (edges array):")
#         print("="*90)
#         print("edges = np.array([")
#         for stat in bin_stats:
#             print(f"    # {stat['dim']}")
#             edges_str = np.array2string(stat['edges'], separator=',', max_line_width=120)
#             print(f"    {edges_str},")
#         print("])")
#         print("="*90)

#         # 最后保存一下这个edges的数据
#         np.save("./libero_dim_bin_distribution.npy", edges)
#     # 保存npy文件
#     # _test_libero_datset()

# # ======================================================================
# # FINAL STATISTICS - Max/Min per dimension across all episodes:
# # ======================================================================
# # x     :
# #   Max:    40.1116 (global idx 123368). Min:   -50.1857 (global idx 120736). Range:    90.2973
# # y     :
# #   Max:    67.2777 (global idx 120736). Min:   -30.3134 (global idx 152710). Range:    97.5911
# # z     :
# #   Max:    16.0259 (global idx 135122). Min:   -98.6786 (global idx 120736). Range:   114.7045
# # yaw   :
# #   Max:    10.8343 (global idx 9756). Min:   -11.8736 (global idx 115864). Range:    22.7079
# # pitch :
# #   Max:    17.3861 (global idx 123368). Min:   -14.0604 (global idx 250342). Range:    31.4464
# # roll  :
# #   Max:    24.5229 (global idx 137162). Min:   -24.8657 (global idx 6785). Range:    49.3886
# # Sequence length: max=505, min=75
# # Min non-zero change value: 0.0007142857
# # ======================================================================