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
import numpy as np

from time import time
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from prismatic.models.backbones.vision.base_vision import ImageTransform
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from prismatic.vla.trajectory_compression import BaseTrajectoryCompression, BiningTrajectoryCompression
from prismatic.vla.tokenizer import VlaTokenizer, BaseTrajectoryConverter
from prismatic.overwatch import initialize_overwatch



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
        self.tokenizer = tokenizer # 都在_get_item__中处理
        self.trajectory_compression = trajectory_compression

        self.root = real_root / repo_id
        # self.metadata = LeRobotDatasetMetadata(repo_id, root=self.root)

        # Initialize overwatch logger
        self.overwatch = initialize_overwatch(__name__)

        # 过滤出 task-centric的 episodes；task_ids 为 None 或 [-1] 时加载全部 episodes
        # if task_ids is None or task_ids == [-1]:
        #     self.episodes = list(self.metadata.episodes["episode_index"])
        #     self.overwatch.info(f"DATASET: Loading ALL episodes ({len(self.episodes)} total)")
        # else:
        #     self.episodes = self.get_episode_indices_for_tasks(task_ids)
        #     self.overwatch.info(f"DATASET: Loading episodes for task_ids={task_ids} ({len(self.episodes)} episodes)")
        #     if len(self.episodes) == 0: raise ValueError("No episodes found for the given task_ids; check dataset or filters")
        
        # ================== 临时修改，不用第一个episode数据来做验证模型的泛化能力
        self.episodes = sorted(list(self.metadata.episodes["episode_index"]))[1:]
        self.overwatch.info(f"DATASET: Loading ALL episodes ({len(self.episodes)} total, first five {self.episodes[:5]})")
        
        
        if self.trajectory_compression.exp_type is not None:
            delta_timestamps = {f"{self.trajectory_compression.exp_type}":[]} 
        else:
            delta_timestamps = None

        self._dataset = LeRobotDataset(
            #"HuggingFaceVLA_cus/libero_cut_zcd_20_15_lastseg_indicator",
            repo_id,
            root=self.root,
            episodes=self.episodes, 
            image_transforms=image_transform,
            delta_timestamps=delta_timestamps  # 获取从当前帧到 episode 结尾的完整 action 序列
        )

        self.overwatch.info(f"training dataset length:{len(self._dataset)}") #, validate dataset length:{len(self.val_dataset)}")

    @property
    def dataset(self):
        """动态返回训练集或验证集"""
        return self._dataset 
    
    # def get_episode_indices_for_tasks(self, task_ids: list[int]) -> list[int]:
    #     tasks = self.metadata.tasks
    #     # 不同的repo的实现是不同的，注意这里 TODO: 未来分成不同的类
    #     if self.repo_id.endswith("libero"):
    #         # 对于libero的而言，根据meta中的文本string来过滤出task_id
    #         # tasks 的 index 通常是 task_name（string），所以需要先获取对应的 task_name
    #         task_mask = tasks["task_index"].isin(task_ids)
    #         selected_task_str = tasks[task_mask].index.tolist()  # 获取选中的 task_name list
            
    #         selected_episode_metadata = self.metadata.episodes.filter(lambda x: x['tasks'][0] in selected_task_str)
    #         result = list(selected_episode_metadata["episode_index"])
            
    #         return result
    #     elif self.repo_id.endswith("pusht_image"):
    #         pass
    #     elif self.repo_id.endswith("2025-challenge-demos"):
    #         pass
    #     else: 
    #         raise NotImplementedError(f"Unknown repo_id format: {self.repo_id}")
    
    def get_trajectory_for_item(self, item, **kwargs):
        # affordance 已经从数据集中作为 tensor 字段直接获取
        if self.trajectory_compression.exp_type is not None:
            qurey_key = self.trajectory_compression.exp_type 
        else:
            qurey_key = "action"
        
        original_trajectory = item[qurey_key].numpy()
        compressed_trajectory = self.trajectory_compression(original_trajectory, **kwargs)
        return torch.Tensor(compressed_trajectory)

    def __len__(self): 
        return len(self.dataset)
        
    def __getitem__(self, index):
        # 根据是哪一个具体的数据集，拿到对应的数据
        item = self.dataset.__getitem__(index)
        frame_index, episode_index = item['frame_index'], item['episode_index']
        # 这里扩展到了两图输入的libero的格式（目前先focus在libero上）
        uni_key_item = dict(
            cam1=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam1']],
            cam2=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam2']],
            language=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['language']],
            trajectory=self.get_trajectory_for_item(item, frame_index=frame_index, episode_index=episode_index),
            dataset_names=self.repo_id
        )

        return self.tokenizer.tokenize_batch(uni_key_item)
    

if __name__ == "__main__":
    def _test_libero_datset():
        dataset = LeRobotDataset(
            #"HuggingFaceVLA_cus/libero_cut_zcd_20_15_lastseg_indicator",
            'HuggingFaceVLA/libero',
            root=Path("/inspire/hdd/project/robot-decision/public/datasets/")/'HuggingFaceVLA/libero',
            delta_timestamps={"abs_aff":[]}  # 获取从当前帧到 episode 结尾的完整 action 序列
        )

        dataloader = DataLoader(dataset, num_workers=1, shuffle=False)
        max_sequence_length = 0
        from tqdm import tqdm
        i = 0
        current_epsiode = 0
        max_value, min_value = 0, 0
        max_length, min_length = 0, 10000000
        
        min_change_value = 100
        with tqdm(total=len(dataset)) as pbar:
            while i < len(dataset):
                

                episode_data = dataset[i]
                abs_aff = episode_data['abs_aff']
                print("")
                cumulative_abs_aff = torch.cumsum(abs_aff[:, :-1], dim=0)[-1]

                i += len(abs_aff)

                max_value = max(max_value, max(cumulative_abs_aff))
                min_value = min(min_value, min(cumulative_abs_aff))

                max_length = max(max_length, len(abs_aff))
                min_length = min(min_length, len(abs_aff))

                abs_aff_sub = abs_aff[:, :-1]
                nonzero_mask = abs_aff_sub != 0
                if torch.any(nonzero_mask):
                    min_nonzero_change = torch.min(torch.abs(abs_aff_sub[nonzero_mask])).item()
                    min_change_value = min(min_nonzero_change, min_change_value)
                # tensor(1691) max_value:67.27767944335938 min_value:-98.67857360839844 min_change_value:0.0007142857066355646
                print(episode_data['episode_index'], f"max,min length:{max_length} {min_length}",f"max_value:{max_value}", f"min_value:{min_value}", f"min_change_value:{min_change_value}")
                



            # print(item['episode_index'])
            # print("item:", item)

        # for i, item in tqdm(enumerate(dataloader)):
            # print("item frame_index:", item['frame_index'])    
            # print("item abs_aff shape:", item['abs_aff'].shape)
            # print("item episode index:", item['episode_index'])
            # if i == 5:
            #     break

        print("max_sequence_length is", max_sequence_length)

    import hydra
    from omegaconf import OmegaConf, DictConfig
    from scipy.interpolate import BSpline
    
    config_path = "/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/output/2025-12-30/16-00-49/qwen2.5-0.5b+b16+x7--1-qwen25-abs_aff_uniform_bspline"
    @hydra.main(config_path=config_path, config_name="config", version_base=None)
    def _test_my_dataset_full(cfg: DictConfig):
        """ 
        这个文件需要从MyLeRobotdataset中进行加载dataset(用bspline absolute的配置，你需要看一下别的代码来充分思考），配置dataloader，batch=1，拿到输出后。
        然后再进行decoding，变换回原始的轨迹（bspline解压）。然后我们看一下解压回去后的轨迹和原始的数据集中的轨迹有多少的差别。你需要帮我完成这个代码，
        然后最后除了print出来具体的差距距离（control point上的差距，还有整条轨迹上的max和mean的差距，最好再给我可视化一下
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from prismatic.util.vla_utils import get_vla_dataset
        from prismatic.models.load import load

        dataset_cfg = cfg.dataset
        vla_cfg = cfg.vla

        vla = load(
            vla_cfg=vla_cfg,
            checkpoint_path=Path(config_path) / "checkpoints" / "latest-checkpoint.safetensors",
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
        
        data_index = 100

        # 取一个样本并走完encode->decode流程
        # ============= 0. 原始数据集中的输出数据 ============
        raw_item = vla_dataset.dataset[data_index]
        frame_index = int(raw_item["frame_index"].item()) if "frame_index" in raw_item else 0
        episode_index = int(raw_item["episode_index"].item()) if "episode_index" in raw_item else 0
        abs_aff = raw_item["abs_aff"].numpy()  # [T,7] 差分形式
        np.set_printoptions(suppress=True, linewidth=200, precision=6)
        abs_aff_gt = abs_aff.copy()
        abs_aff_gt[:, :-1] = np.cumsum(abs_aff_gt[:, :-1], axis=0)
        print("abs_aff_gt top 3 point:", abs_aff_gt[:21], ", shape:", abs_aff_gt.shape)


        # ========== 1. 数据集压缩后的数值， 转绝对坐标，这就是ground truth ========
        # 压缩得到控制点（与 MyLeRobotDataset 内部一致）
        my_dataset_item = vla_dataset[data_index]
        # print(my_dataset_item['input_ids'])
        print("lables:", my_dataset_item['labels'])
        labels = my_dataset_item['labels'].numpy()
        if (labels == -100).any():
            # 找到最后一个-100的下标，取其后所有数据
            first_valid_idx = np.where(labels == -100)[0][-1] + 1
            traj_token_ids = labels[first_valid_idx:]
        else:
            traj_token_ids = labels

        # ======== 2. 进行模型进行推理，拿到pred的坐标和解码(TODO 这里还没完成) =========
        from transformers import GenerationMixin
        input_ids = my_dataset_item['input_ids'][:first_valid_idx].unsqueeze(0).to("cuda")
        pixel_values = {k: v.unsqueeze(0).to("cuda") for k, v in my_dataset_item['pixel_values'].items()}
        # print("input ids:", input_ids)
        # print("pixel values:", pixel_values)
        with torch.autocast("cuda", dtype=vla.llm_backbone.half_precision_dtype):
            pred_ids = GenerationMixin.generate(
                vla,
                input_ids=input_ids,
                # attention_mask=my_dataset_item['attention_mask'].unsqueeze(0),
                pixel_values=pixel_values,
                use_cache=False,
                do_sample=False,  # 强制贪心解码
                max_new_tokens=1024,
            )
        pred_ids = pred_ids.cpu().numpy()
        print("shape:", pred_ids.shape, "pred_ids: ", pred_ids)
        
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
                next_action, bspline = vla_dataset.trajectory_compression.decode_to_action(
                    decoded_control_points, current_eef_pose=current_pose
                )
                print(f"\n=== [{name}] Next Action (t=0.1s) ===")
                print(f"Position: {next_action[:6]}")
                print(f"Gripper: {next_action[6]:.3f}")
                
                # 重建整条轨迹：用bspline在knot时间点上采样
                knot_times = decoded_control_points[:, -1]
                num_samples = 100  # 采样点数
                t_eval = np.linspace(knot_times[0], knot_times[-1], num_samples)
                reconstructed_traj = np.zeros((num_samples, 7))
                reconstructed_traj[:, :6] = bspline(t_eval)
                # gripper用线性插值
                reconstructed_traj[:, 6] = np.interp(t_eval, knot_times, decoded_control_points[:, 6])
                
                # 提取ground truth的对应片段（从frame_index开始）
                gt_segment = abs_aff_gt[frame_index:]
                
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
                
                # 可视化: 对比前3个维度 (x, y, z) 和 gripper
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                dims_to_plot = [0, 1, 2]  # x, y, z
                dim_names = ['x', 'y', 'z']
                
                for i, (dim, dim_name) in enumerate(zip(dims_to_plot, dim_names)):
                    axes[i].plot(gt_t, gt_segment[:, dim], label=f'GT {dim_name}', linewidth=2, alpha=0.8, color='green')
                    axes[i].plot(t_eval, reconstructed_traj[:, dim], label=f'Reconstructed {dim_name}', 
                                linestyle='--', linewidth=1.5, alpha=0.8, color='red')
                    axes[i].scatter(knot_times, decoded_control_points[:, dim], 
                                   c='red', s=50, marker='x', label='Control Points', zorder=5)
                    axes[i].set_ylabel(dim_name)
                    axes[i].legend(loc='upper right')
                    axes[i].grid(True, alpha=0.3)
                
                # gripper维度
                axes[3].plot(gt_t, gt_segment[:, 6], label='GT gripper', linewidth=2, alpha=0.8, color='green')
                axes[3].plot(t_eval, reconstructed_traj[:, 6], label='Reconstructed gripper', 
                            linestyle='--', linewidth=1.5, alpha=0.8, color='red')
                axes[3].scatter(knot_times, decoded_control_points[:, 6], 
                               c='red', s=50, marker='x', label='Control Points', zorder=5)
                axes[3].set_ylabel('gripper')
                axes[3].set_xlabel('Time (frames)')
                axes[3].legend(loc='upper right')
                axes[3].grid(True, alpha=0.3)
                
                plt.suptitle(f'[{name}] B-Spline Reconstruction (ep={episode_index}, frame={frame_index})')
                plt.tight_layout()
                out_path = output_dir / f"bspline_reconstruct_{name}.png"
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
    _test_my_dataset_full()
    