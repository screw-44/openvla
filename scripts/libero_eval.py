import os
import torch
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from termcolor import colored
from PIL import Image
from scipy.spatial.transform import Rotation as R


# LIBERO imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# HuggingFace / Transformers
from transformers.generation.utils import GenerationMixin

# 你的项目 imports
from core.models.load import load
from core.util.vla_utils import get_vla_dataset

# ✅ [修改] 1. 定义视频输出目录
VIDEO_OUTPUT_DIR = "./eval_videos"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
VIZ_OUTPUT_DIR = "/tmp/"

NUM_EPISODES = 10
NUM_TASK = 1
PRED_GAP = 150        
SMOOTH_RATIO = 1
MAX_STEPS = 150

# ==============================================================================
# 可视化函数
# ==============================================================================
def plot_predicted_trajectory(decoded_cp, reconstructed, frame_index, episode_index, pred_idx, output_path: Path):
    """绘制预测的轨迹（无GT，只显示重建轨迹和控制点）"""
    os.makedirs(output_path.parent, exist_ok=True)
    
    # 时间轴
    knots_vis = decoded_cp[:, -1] + frame_index
    t_eval_vis = np.arange(len(reconstructed)) + frame_index

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    axes = axes.flatten()
    dims_to_plot = [0, 1, 2, 3, 4, 5, 6]
    dim_names = ["x", "y", "z", "yaw", "pitch", "roll", "gripper"]

    for i, (dim, dim_name) in enumerate(zip(dims_to_plot, dim_names)):
        ax = axes[i]
        ax.plot(t_eval_vis, reconstructed[:, dim], label=f"Reconstructed {dim_name}",
                linewidth=2, alpha=0.8, color="blue")
        ax.scatter(knots_vis, decoded_cp[:, dim], c="red", s=60, marker="x", 
                   label="Control Points", zorder=5)
        ax.set_ylabel(dim_name)
        ax.set_xlabel("Time (frames)")
        ax.set_title(f"Dimension: {dim_name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # 统计信息
    axes[8].axis("off")
    stats_text = f"""Prediction Statistics:
Reconstructed Length: {len(reconstructed)}
Control Points: {len(decoded_cp)}
Episode: {episode_index}
Frame Index: {frame_index}
Prediction ID: {pred_idx}
Time Span: [{frame_index}, {frame_index + len(reconstructed) - 1}]"""
    
    axes[8].text(0.1, 0.5, stats_text, fontsize=11,
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                family="monospace")

    title = f"Predicted Trajectory - Ep{episode_index}, Frame{frame_index}, PredID{pred_idx}"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()



# 评测主类
# ==============================================================================
class LiberoEvaluator:
    def __init__(self, cfg_path, ckpt_path, config_dict, device="cuda"):
        self.device = device
        self.config_dict = config_dict
        self.pred_counter = 0  # 用于追踪预测次数
        
        print(f"Loading model from {ckpt_path}...")
        self._load_model_and_converter(cfg_path, ckpt_path)
        
        # 获取 B-Spline 处理类实例 (直接复用 dataset 中的实例)
        self.traj_compressor = self.vla_dataset.traj_compress
        
        # 强制重置平滑缓存
        self.traj_compressor._cache_smoothed_trajectoy = None

    def _load_model_and_converter(self, cfg_path, ckpt_path):
        """加载模型和配置"""
        # 1. Load Config & Model
        cfg = OmegaConf.load(cfg_path)
        self.vla = load(
            vla_cfg=cfg.vla,
            checkpoint_path=ckpt_path,
            load_for_training=False,
        ).to(self.device, dtype=torch.bfloat16).eval()

        # 2. Extract Components
        self.base_tokenizer = self.vla.llm_backbone.get_tokenizer()
        
        # [修改点] 这里去掉括号，只获取类，不要实例化
        self.prompt_builder_fn = self.vla.llm_backbone.prompt_builder_fn 
        
        self.image_transform = self.vla.vision_backbone.get_image_transform()

        # 3. Load Converter
        print("Initializing converter and dataset info...")
        self.vla_dataset, self.trajectory_converter, _ = get_vla_dataset(
            data_repo_id=self.config_dict["repo_id"],
            data_task_ids=None, 
            trajectory_compression_method=self.config_dict["compression_method"],
            trajectory_converter_type=self.config_dict["converter_type"],
            base_tokenizer=self.base_tokenizer,
            prompt_builder_fn=self.prompt_builder_fn,
            image_transform=self.image_transform,
        )

    def get_processed_images(self, obs):
        """
        处理 LIBERO 图像并返回 PIL Image 对象
        LIBERO 的 agentview 通常需要翻转
        """
        # print('obs keys are:', obs.keys())
        # 1. AgentView -> cam1
        img_agent = obs["agentview_image"]
        img_agent = img_agent[::-1, ::-1] # LIBERO 图像翻转修正
        img_agent_pil = Image.fromarray(img_agent)

        # 2. EyeInHand -> cam2
        img_wrist = obs["robot0_eye_in_hand_image"]
        img_wrist = img_wrist[::-1, ::-1] # 同样翻转以防万一，视具体环境配置而定
        img_wrist_pil = Image.fromarray(img_wrist)

        return img_agent_pil, img_wrist_pil

    def predict_and_decode(self, obs, task_description, current_pose, pred_gap, smooth_ratio=0.8):
        """
        整合了 推理(Predict) + 解码(Decode) + 平滑(Smooth) 的全流程
        """
        # ======================================================================
        # 1. 数据预处理 (完全按照你提供的 snippet 方式)
        # ======================================================================
        
        # A. 图像处理
        cam1_pil, cam2_pil = self.get_processed_images(obs)
        
        pixel_values = {}
        # cam1
        pixel_values["cam1"] = (
            self.image_transform(cam1_pil).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        )
        # cam2
        pixel_values["cam2"] = (
            self.image_transform(cam2_pil).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        )

        # B. 文本处理 (使用 PromptBuilder, 参考VlaTokenizer的实现)
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {task_description}?")
        prompt_text = prompt_builder.get_prompt()
        print("prompt text:", prompt_text)

        input_ids = (
            self.base_tokenizer(prompt_text, return_tensors="pt")
            .input_ids.to(self.device)
        )
        print("input ids:", input_ids)

        # ======================================================================
        # 2. 模型生成 (GenerationMixin)
        # ======================================================================
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_ids = GenerationMixin.generate(
                self.vla, # self.model
                input_ids=input_ids,
                pixel_values=pixel_values,
                use_cache=False,
                do_sample=False,  # 强制贪心解码
                max_new_tokens=10240,
            )
        
        # 提取 Action Tokens
        pred_ids = pred_ids.cpu().numpy()
        input_ids_len = input_ids.shape[1]
        action_ids = pred_ids[0, input_ids_len:]

        # 如果生成过短，直接返回 None
        if len(action_ids) < 5:
            return None

        # ======================================================================
        # 3. 解码与平滑 (Decode & Smooth)
        # ======================================================================
        
        # A. Token -> Control Points
        decoded_cp = self.trajectory_converter.decode_text_ids_to_trajectory(action_ids)

        # B. Control Points -> B-Spline Objects
        bspline, gripper_bspline = self.traj_compressor.decode_to_action(decoded_cp)
        
        # C. 采样 (Sample Dense Trajectory)
        knots = decoded_cp[:, -1]
        print("predicted knots:", knots)
        num_samples = int(knots[-1]) + 1 
        t_eval = np.arange(num_samples)
        
        # [N, 6]
        traj_pose = bspline(t_eval) 
        # [N, 1]
        traj_gripper = gripper_bspline(t_eval).reshape(-1, 1)

        # D. Start Offset Correction (修正起始点跳变)(这里先不添加)
        # if num_samples > 10:
        #     offset = self.traj_compressor.start_offset(current_pose[:6], traj_pose[0], decay_distance=10)
        #     L = min(len(offset), len(traj_pose))
        #     traj_pose[:L] += offset[:L]

        # 合并 pose 和 gripper -> [N, 7]
        full_traj = np.hstack([traj_pose, traj_gripper])

        # E. Trajectory Smoothing (平滑)
        smoothed_traj = self.traj_compressor.smooth_traj(
            full_traj, 
            pred_gap=PRED_GAP, 
            smooth_ratio=SMOOTH_RATIO, 
            reset=False
        )
        
        # ✅ [新增] 可视化预测轨迹
        viz_path = Path(VIZ_OUTPUT_DIR) / f"libero_eval_pred.png"
        print("Saving trajectory visualization to:", viz_path)
        plot_predicted_trajectory(decoded_cp, smoothed_traj, 
                                 frame_index=getattr(self, '_current_step', 0),
                                 episode_index=getattr(self, '_current_ep', 0),
                                 pred_idx=self.pred_counter,
                                 output_path=viz_path)
        self.pred_counter += 1

        return smoothed_traj # Absolute trajectory


def run_eval():

    # 1. 配置路径
    config_dict = {
        "repo_id": "HuggingFaceVLA/libero",
        "compression_method": "bspline_v3",
        "converter_type": "bspline_v3",
    }
    
    base_path = Path("/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/outputs/2026-01-13/04-48-57/qwen2.5-0.5b+b16+x7--1-qwen25-abs_aff_uniform_bspline_v3")
    cfg_path = base_path / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = base_path / "config.yaml"
    ckpt_path = base_path / "checkpoints" / "step-035000-epoch-02-loss=0.0147.safetensors" # step-085000-epoch-04-loss=0.0517.safetensors" # "step-055000-epoch-03-loss=0.0210.safetensors"

    # 2. 初始化 Evaluator
    evaluator = LiberoEvaluator(cfg_path, ckpt_path, config_dict)

    # 3. 初始化 LIBERO 环境
    # libero_10,libero_object,libero_spatial,libero_goal
    def init_libero_env(benchmark_name="libero_spatial", task_id=0):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[benchmark_name]()
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env_args = {
            "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            "render_gpu_device_id": 0,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        return env, task, init_states
    # 4. 评测参数
    success_count = 0

    print(f"Videos will be saved to: {VIDEO_OUTPUT_DIR}")

    for task_idx in range(NUM_TASK):
        env, task, init_states = init_libero_env(benchmark_name="libero_object", task_id=task_idx)
        print(f"\n=== Evaluating Task {task_idx}: {task.name} ===")
        for ep_idx in range(NUM_EPISODES):
            evaluator._current_ep = ep_idx  # 设置当前episode
            env.reset()
            obs = env.set_init_state(init_states[ep_idx % len(init_states)])
            evaluator.traj_compressor.smooth_traj(np.zeros((1, 7)), 0, reset=True) 
            
            done = False
            step = 0
            action_queue = []
            
            # ✅ [修改] 2. 初始化视频帧列表
            video_frames = []

            pbar = tqdm(total=MAX_STEPS, desc=f"Episode {ep_idx}", leave=False)

            while step < MAX_STEPS:
                evaluator._current_step = step  # 设置当前step
                
                # ✅ [修改] 3. 捕获并处理图像用于视频保存
                # 获取图像并翻转（Libero特性）
                img_agent = obs["agentview_image"][::-1, ::-1]
                img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                # 水平拼接 (Horizontal Concatenation)
                combined_img = np.concatenate([img_agent, img_wrist], axis=1)
                video_frames.append(combined_img)

                # -----------------------------------------------------------
                # RHC 逻辑
                # -----------------------------------------------------------
                if step % PRED_GAP == 0 or len(action_queue) == 0:
                    curr_pos = obs["robot0_eef_pos"]
                    curr_quat = obs["robot0_eef_quat"] 
                    curr_euler = R.from_quat(curr_quat).as_euler('xyz')
                    current_pose_input = np.concatenate([curr_pos, curr_euler])

                    print("current_pose_input:", current_pose_input)

                    smoothed_traj = evaluator.predict_and_decode(
                        obs,
                        task.language, 
                        current_pose_input, 
                        pred_gap=PRED_GAP, 
                        smooth_ratio=SMOOTH_RATIO
                    )

                    # 如果预测不出来的话
                    if smoothed_traj is None or len(smoothed_traj) < 2:
                        print(colored(f"Episode {ep_idx} - Step {step}: Prediction failed or too short, taking zero action.", "yellow"))
                        action_queue = [np.array([0.0, 0, 0, 0, 0, 0, -1])]
                        break
                    
                    smoothed_traj[:-1, :6] = np.diff(smoothed_traj[:, :6], axis=0)
                    smoothed_traj[-1, :6] = np.array([0.0]*6)  # 最后一个动作设为0
                    # print("predict smoothed_traj:", smoothed_traj[:2])  # 打印前5个动作以供调试
                    
                    if smoothed_traj is not None:
                        action_queue = []
                        valid_len = min(PRED_GAP * 2, len(smoothed_traj))
                        for i in range(valid_len):
                            # act = diff_traj[i].copy()
                            # act[6] = 1.0 if act[6] > 0.5 else -1.0
                            action_queue.append(smoothed_traj[i].copy())
                # 执行动作
                action = action_queue.pop(0)
                obs, reward, done, info = env.step(action)
                step += 1
                pbar.update(1)

                if done: 
                    break
            
            pbar.close()
            
            # 检查成功状态
            is_success = env.check_success()
            status_str = "Success" if is_success else "Failed"

            if is_success:
                print(colored(f"Episode {ep_idx} Success!", "green"))
                success_count += 1
            else:
                print(colored(f"Episode {ep_idx} Failed.", "red"))

            # ✅ [修改] 4. 保存视频
            if len(video_frames) > 0:
                save_path = os.path.join(VIDEO_OUTPUT_DIR, f"task_{task_idx}_ep{ep_idx}_{status_str}.mp4")
                # 转换为BGR格式用于保存
                # frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in video_frames]
                imageio.mimwrite(save_path, video_frames, fps=20, codec='libx264')
                print(f"Saved video to {save_path}")

        print(f"Success Rate: {success_count}/{NUM_EPISODES} = {success_count/NUM_EPISODES*100:.2f}%")
        env.close()

if __name__ == "__main__":
    run_eval()