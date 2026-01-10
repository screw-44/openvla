import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import BSpline
from typing import Dict, Tuple, List

# ==========================================
# 配置
# ==========================================
RESULTS_JSON_PATH = Path("compression_results.json")
VISUAL_OUTPUT_DIR = Path("visual_fig_analysis")
DATASET_ROOT = Path("/inspire/hdd/project/robot-decision/public/datasets/") / "HuggingFaceVLA/libero"
COMPRESSION_FAIL_PATH = Path("compression_fail.json")
MAX_ERROR_THRESHOLD = 0.5

# ==========================================
# 加载数据集
# ==========================================
def load_dataset():
    """加载完整数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(
        'HuggingFaceVLA/libero',
        root=DATASET_ROOT,
        delta_timestamps={"abs_aff":[]}
    )
    return dataset


def load_episode_by_id(dataset, episode_idx: int) -> Tuple[np.ndarray, str, int]:
    """
    通过 episode_idx 在数据集中找到对应的 episode 并加载
    
    Returns:
        abs_aff_absolute: (T, 7) 绝对坐标轨迹
        task_name: 任务名称
        dataset_start_index: 该 episode 在数据集中的起始索引
    """
    current_idx = 0
    while current_idx < len(dataset):
        item = dataset[current_idx]
        if int(item['episode_index']) == episode_idx:
            # 找到了目标 episode
            abs_aff = np.asarray(item['abs_aff'])
            
            # 转换为绝对坐标
            abs_aff_absolute = abs_aff.copy()
            abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
            
            task_name = item['task'] if 'task' in item else "Unknown"
            return abs_aff_absolute, task_name, current_idx
        
        # 跳过当前 episode 的所有帧
        episode_length = len(item['abs_aff'])
        current_idx += episode_length
    
    raise ValueError(f"Episode {episode_idx} 不存在于数据集中")


# ==========================================
# B-spline 重建函数
# ==========================================
def reconstruct_bspline_trajectory(control_points: List[float], knot_vector: List[float], 
                                    time_points: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    从 B-spline 控制点重建轨迹
    """
    if len(control_points) == 0:
        return np.zeros_like(time_points, dtype=float)
    
    ctrl_pts = np.array(control_points)
    knot_vec = np.array(knot_vector)
    
    try:
        bspline = BSpline(knot_vec, ctrl_pts, degree, extrapolate=False)
        return bspline(time_points)
    except Exception as e:
        print(f"  [Warning] B-spline 重建失败: {e}")
        return np.zeros_like(time_points, dtype=float)


def reconstruct_episode(episode_data: Dict) -> Tuple[np.ndarray, int, int, int]:
    """
    根据保存的 B-spline 参数重建 episode 的 7D 轨迹
    
    Returns:
        reconstructed_7d: (7, T) 重建的轨迹
        num_knots_6d: 前6维使用的 knot 数
        num_forced_knots_gripper: gripper 的强制 knot 数
        trajectory_length: 轨迹长度
    """
    trajectory_length = episode_data["trajectory_length"]
    time_points = np.arange(trajectory_length)
    bspline_data = episode_data["bspline"]
    
    control_points = bspline_data["control_points"]  # 7 个维度
    knot_vectors = bspline_data["knot_vectors"]      # 7 个维度
    num_knots = bspline_data["num_knots"]            # 前6维的内部knot数
    forced_knots = bspline_data["forced_knots"]      # gripper 的强制knot
    
    reconstructed_7d = np.zeros((7, trajectory_length))
    
    # 重建前6维 (3阶 B-spline)
    for d in range(6):
        reconstructed_7d[d] = reconstruct_bspline_trajectory(
            control_points[d], knot_vectors[d], time_points, degree=3
        )
    
    # 重建第7维 (gripper, 0阶 B-spline)
    reconstructed_7d[6] = reconstruct_bspline_trajectory(
        control_points[6], knot_vectors[6], time_points, degree=0
    )
    
    return reconstructed_7d, num_knots, len(forced_knots), trajectory_length


# ==========================================
# 计算误差
# ==========================================
def calculate_errors(original: np.ndarray, reconstructed: np.ndarray) -> Dict:
    """
    计算逐维度的误差统计
    """
    abs_errors = np.abs(original - reconstructed)
    
    stats = {
        "max": float(np.max(abs_errors)),
        "min": float(np.min(abs_errors)),
        "mean": float(np.mean(abs_errors)),
        "std": float(np.std(abs_errors)),
    }
    
    # 逐维度统计
    stats["per_dim"] = {}
    for d in range(original.shape[0]):
        dim_errors = abs_errors[d]
        stats["per_dim"][d] = {
            "max": float(np.max(dim_errors)),
            "min": float(np.min(dim_errors)),
            "mean": float(np.mean(dim_errors)),
        }
    
    return stats


# ==========================================
# 可视化
# ==========================================
def visualize_episode(original_7d: np.ndarray, reconstructed_7d: np.ndarray, 
                      episode_idx: int, task_name: str, 
                      num_knots_6d: int, num_forced_knots: int,
                      errors: Dict, save_path: Path):
    """
    可视化单个 episode：2行4列布局
    - 第1-4列：前6维的轨迹（选择关键维度）
    - 最后一列：gripper + 错误统计
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # 原始数据应该是 (T, 7)，需要转置
    if original_7d.shape[0] != 7:
        original_7d = original_7d.T
    if reconstructed_7d.shape[0] != 7:
        reconstructed_7d = reconstructed_7d.T
    
    trajectory_length = original_7d.shape[1]
    time_axis = np.arange(trajectory_length)
    
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    # 7个维度显示顺序：X, Y, Z, Gripper, Yaw, Pitch, Roll (2行4列，最后一格是统计)
    display_dims = [0, 1, 2, 6, 3, 4, 5]  # 显示7个维度：X, Y, Z, Gripper, Yaw, Pitch, Roll
    
    # 绘制7个维度
    for plot_idx, dim_idx in enumerate(display_dims):
        ax = axes[plot_idx]
        
        original_dim = original_7d[dim_idx]
        reconstructed_dim = reconstructed_7d[dim_idx]
        
        # 绘制原始数据
        ax.plot(time_axis, original_dim, 'o-', color='gray', alpha=0.6, 
                linewidth=0.8, markersize=3, label='Original')
        
        # 绘制重建数据
        color = 'red' if dim_idx < 6 else 'green'
        ax.plot(time_axis, reconstructed_dim, '-', color=color, linewidth=1.5, 
                label='Reconstructed', alpha=0.9)
        
        # 计算误差
        dim_error = np.abs(original_dim - reconstructed_dim)
        max_err = np.max(dim_error)
        mean_err = np.mean(dim_error)
        
        ax.set_title(f"{dim_names[dim_idx]}\nMax: {max_err:.6f} | Mean: {mean_err:.6f}", 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # 最后一列：错误统计信息
    ax_stats = axes[7]
    ax_stats.axis('off')
    
    # 构建统计文本
    overall_errors = errors["overall_mean_error"]
    max_error = errors["max"]
    min_error = errors["min"]
    
    stats_text = f"""Error Summary
━━━━━━━━━━━━━━━━━━━━━
Overall Mean: {overall_errors:.6f}
Max Error: {max_error:.6f}
Min Error: {min_error:.6f}

Knot Configuration
━━━━━━━━━━━━━━━━━━━━━
Internal Knots (6D): {num_knots_6d}
Forced Knots (Gripper): {num_forced_knots}
Total Knots: {num_knots_6d + num_forced_knots}

Trajectory Info
━━━━━━━━━━━━━━━━━━━━━
Length: {trajectory_length} frames
Compression Ratio: {(num_knots_6d + num_forced_knots) / trajectory_length:.2%}

Per-Dimension Errors
━━━━━━━━━━━━━━━━━━━━━
"""
    
    for d in range(7):
        dim_stat = errors["per_dim"][d]
        stats_text += f"{dim_names[d]:>8}: Max={dim_stat['max']:.5f} Mean={dim_stat['mean']:.5f}\n"
    
    ax_stats.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                  verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
    
    # 总标题
    fig.suptitle(f'B-spline Reconstruction - Episode {episode_idx}\nTask: {task_name[:100]}...', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 可视化已保存: {save_path}")


# ==========================================
# 主函数
# ==========================================
def main():
    """加载结果、重建、对比并可视化"""
    print("=" * 80)
    print("B-spline 重建与对比可视化")
    print("=" * 80)
    
    # 加载 JSON 结果
    print("\n正在加载压缩结果...")
    with open(RESULTS_JSON_PATH, 'r') as f:
        results = json.load(f)
    
    episodes = results.get("episodes", {})
    print(f"找到 {len(episodes)} 个已完成的 episode")
    
    if not episodes:
        print("没有可视化的数据，退出。")
        return

    # TODO: 过滤到从第几个episode再去计算。


    
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset()
    print(f"数据集加载完成，共 {len(dataset)} 个样本")
    
    # 记录失败的 episode
    failed_episodes = []
    
    # 遍历每个 episode 并可视化
    for ep_id_str, episode_data in sorted(episodes.items()):
        episode_idx = int(ep_id_str)
        task_name = episode_data.get("task_name", "Unknown")
        
        print(f"\n处理 Episode {episode_idx}...")
        
        try:
            # 从数据集加载原始数据
            original_7d, _, dataset_start_idx = load_episode_by_id(dataset, episode_idx)
            
            # 重建 B-spline 轨迹
            reconstructed_7d, num_knots_6d, num_forced_knots, traj_len = reconstruct_episode(episode_data)
            
            # 计算误差
            errors = calculate_errors(original_7d, reconstructed_7d.T)
            
            # 补充整体误差信息
            errors["overall_mean_error"] = episode_data["bspline"]["overall_mean_error"]
            errors["max"] = np.max(np.abs(original_7d - reconstructed_7d.T))
            errors["min"] = np.min(np.abs(original_7d - reconstructed_7d.T))
            
            # 检查是否失败（max error > 0.1）
            if errors["max"] > MAX_ERROR_THRESHOLD:
                failed_episodes.append({
                    "episode_id": episode_idx,
                    "dataset_index": dataset_start_idx,
                    "mean_error": errors["overall_mean_error"],
                    "task": task_name
                })
                print(f"  ⚠ 压缩失败 (Max Error: {errors['max']:.6f} > {MAX_ERROR_THRESHOLD}). 进行可视化。")
            
                # 可视化
                save_path = VISUAL_OUTPUT_DIR / f"episode_{episode_idx}_comparison.png"
                visualize_episode(original_7d, reconstructed_7d, 
                                episode_idx, task_name,
                                num_knots_6d, num_forced_knots,
                                errors, save_path)
            
            print(f"  ✓ 重建完成 | Mean Error: {errors['overall_mean_error']:.6f}")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存失败的 episode 列表
    if failed_episodes:
        print(f"\n发现 {len(failed_episodes)} 个压缩失败的 episode，保存到 {COMPRESSION_FAIL_PATH}")
        with open(COMPRESSION_FAIL_PATH, 'w') as f:
            json.dump({
                "threshold": MAX_ERROR_THRESHOLD,
                "failed_count": len(failed_episodes),
                "episodes": failed_episodes
            }, f, indent=2)
        print(f"✓ 失败列表已保存")
    else:
        print("\n✓ 所有 episode 压缩成功！")
    
    print("\n" + "=" * 80)
    print(f"所有可视化已保存到: {VISUAL_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
