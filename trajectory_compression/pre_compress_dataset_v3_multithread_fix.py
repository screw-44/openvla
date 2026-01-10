import numpy as np
import pulp
import matplotlib.pyplot as plt
# 设置 Matplotlib 后端为 Agg，防止多进程绘图时报错
plt.switch_backend('Agg') 
import multiprocessing
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import BSpline, make_lsq_spline
import tqdm
from functools import partial

# ==========================================
# Configuration & Hardware Settings
# ==========================================
TOLERANCE = 0.01
RESULTS_JSON_PATH = Path("compression_results.json")
VISUAL_DIR = Path("test/visual_fig")
ERROR_THRESHOLD = 0.1  # 误差阈值，超过则需要重新拟合

# --- 并行配置 ---
TOTAL_CORES = 128
NUM_WORKERS = 16  # 并行进程数
THREADS_PER_SOLVER = 8  # 每个进程使用的线程数 (16 * 8 = 128)

# ==========================================
# JSON 数据管理函数
# ==========================================
def load_results_from_json(json_path: Path) -> Dict:
    """加载已保存的压缩结果"""
    if not json_path.exists():
        return {
            "metadata": {
                "tolerance": TOLERANCE,
                "degree": 3,
                "dataset": "HuggingFaceVLA/libero"
            },
            "episodes": {}
        }
    
    with open(json_path, 'r') as f:
        return json.load(f)


def save_results_to_json(results: Dict, json_path: Path):
    """保存压缩结果到JSON文件"""
    # 先写入临时文件再重命名，防止写入中断导致文件损坏
    temp_path = json_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(results, f, indent=2)
    temp_path.replace(json_path)
    # print(f"  ✓ 结果已保存") # 减少日志刷屏


def is_episode_computed(episode_idx: int, results: Dict) -> bool:
    """
    检查指定episode是否已正确计算
    返回 False 的情况：
    1. 未计算过
    2. forced_knots 数量等于 num_knots（说明求解失败）
    3. overall_mean_error > 0.5（精度不足）
    """
    ep_str = str(episode_idx)
    episodes = results.get("episodes", {})
    
    # 情况1：未计算过
    if ep_str not in episodes:
        return False
    
    ep_data = episodes[ep_str]
    bspline = ep_data.get("bspline", {})
    
    # 情况2：forced_knots 数量等于 num_knots（求解失败）
    forced_knots = bspline.get("forced_knots", [])
    num_knots = bspline.get("num_knots", 0)
    if len(forced_knots) == num_knots:
        print(f"  [重新计算] Episode {episode_idx}: forced_knots数量({len(forced_knots)}) == num_knots({num_knots})，求解失败")
        return False
    
    # 情况3：overall_mean_error > 0.5（精度不足）
    mean_error = bspline.get("overall_mean_error", 0.0)
    if mean_error > 0.5:
        print(f"  [重新计算] Episode {episode_idx}: mean_error={mean_error:.6f} > 0.5，精度不足")
        return False
    
    # TODO: 情况4, suboptimal 状态。.. 暂时不处理

    return True


# ==========================================
# 工具函数 (保持不变)
# ==========================================
def extract_forced_knots(time_points, gripper_data):
    diff = np.diff(gripper_data)
    change_indices = np.where(diff != 0)[0] + 1
    forced_knot_times = []
    for idx in change_indices:
        if 0 < idx < len(time_points) - 1:
            forced_knot_times.append(time_points[idx])
    return forced_knot_times


def convert_to_bspline_coeffs(time_points, data_1d, internal_knots, degree=3):
    t0, t_end = time_points[0], time_points[-1]
    sorted_knots = sorted([k for k in internal_knots if t0 < k < t_end])
    full_knot_vector = np.concatenate([
        np.repeat(t0, degree + 1),
        sorted_knots,
        np.repeat(t_end, degree + 1)
    ])
    try:
        bspline = make_lsq_spline(time_points, data_1d, full_knot_vector, k=degree)
        return bspline.c, full_knot_vector, bspline
    except Exception as e:
        return None, None, None


def reconstruct_bspline_trajectory(control_points, knot_vector, time_points, degree=3):
    bspline = BSpline(knot_vector, control_points, degree, extrapolate=False)
    return bspline(time_points)

# ==========================================
# 2. 求解器 (修改：接受 threads 参数 & 600s timeout & 次优解)
# ==========================================
def solve_6d_with_forced_knots(time_points, data_6d, forced_knot_times, degree=3, tol_ratio=0.03, time_limit=600, solver_threads=1):
    num_dims = data_6d.shape[0]
    # 下采样候选集以提高速度，如果追求极致精度改为 1
    candidate_knots = time_points[1:-1:3] 

    prob = pulp.LpProblem("RobotArm_6D", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", candidate_knots, cat="Binary")

    # 强制 knot 约束
    forced_count = 0
    for f_t in forced_knot_times:
        best_j = None
        min_dist = 1e-5
        for j in candidate_knots:
            dist = abs(j - f_t)
            if dist < min_dist:
                min_dist = dist
                best_j = j
        if best_j is not None:
            prob += y[best_j] == 1
            forced_count += 1
            
    alpha = {}
    beta = {}
    BigMs = []
    epsilons = []

    for d in range(num_dims):
        d_range = np.max(data_6d[d]) - np.min(data_6d[d])
        if d_range < 1e-6: d_range = 1.0
        eps = d_range * tol_ratio
        epsilons.append(eps)
        BigMs.append(d_range * 1.0) # NOTE: M的range改成1就行，有一些fail就是这里的原因（计算复杂度要求太多了）
        alpha[d] = pulp.LpVariable.dicts(f"alpha_{d}", range(degree + 1), lowBound=None)
        beta[d] = pulp.LpVariable.dicts(f"beta_{d}", candidate_knots, lowBound=None)

    # prob += pulp.lpSum([y[j] for j in candidate_knots])
    # NOTE: 对knots数量进行约束，加速搜索速度
    total_knots = pulp.lpSum([y[j] for j in candidate_knots])
    prob += total_knots         # 1. 设为目标函数 (最小化)
    prob += total_knots >= 5    # 2. 强约束：下界 (直接提升 Best Possible 到 5)
    prob += total_knots <= 70   # 3. 强约束：上界 (限制搜索空间)

    for d in range(num_dims):
        M_d = BigMs[d]
        eps_d = epsilons[d]
        for j in candidate_knots:
            prob += beta[d][j] <= M_d * y[j]
            prob += beta[d][j] >= -M_d * y[j]
        
        # 预计算幂基矩阵以加速构建
        for i, t_val in enumerate(time_points):
            P_val = data_6d[d][i]
            poly_part = pulp.lpSum([alpha[d][k] * (t_val**k) for k in range(degree + 1)])
            # 只生成非零项
            knot_terms = [beta[d][j] * ((t_val - j) ** degree) for j in candidate_knots if t_val > j]
            S_t = poly_part + pulp.lpSum(knot_terms)
            prob += S_t - P_val <= eps_d
            prob += P_val - S_t <= eps_d

    # 【关键修改】使用指定的线程数 & 时间限制
    # msg=False 关闭求解器日志，防止多进程时控制台混乱
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=solver_threads)
    prob.solve(solver)

    # 状态与目标值检查：接受可行的次优解
    status_str = pulp.LpStatus[prob.status]
    objective_val = pulp.value(prob.objective)

    is_suboptimal = False
    if status_str == "Optimal":
        pass
    elif objective_val is not None:
        # 求解器在时限内找到了可行解，但未证明全局最优
        is_suboptimal = True
    else:
        # 彻底失败：无可行解
        return [], None, False

    active_knots = [j for j in candidate_knots if pulp.value(y[j]) > 0.5]
    
    # 只需要 active knots，不需要在这里重建曲线，节省时间
    # 曲线重建在 b-spline 转换步骤做
    return active_knots, is_suboptimal 

# ==========================================
# 3. 数据集加载 (保持不变)
# ==========================================
def load_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        'HuggingFaceVLA/libero',
        root=Path("/inspire/hdd/project/robot-decision/public/datasets/")/'HuggingFaceVLA/libero',
        delta_timestamps={"abs_aff":[]}
    )
    return dataset

def load_episode_by_index(dataset, dataset_index: int):
    item = dataset[dataset_index]
    abs_aff = np.asarray(item['abs_aff'])
    abs_aff_absolute = abs_aff.copy()
    abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
    episode_idx = int(item['episode_index'])
    frame_idx = int(item['frame_index'])
    task_name = item['task'] if 'task' in item else "Unknown"
    return abs_aff_absolute, episode_idx, frame_idx, task_name

# ==========================================
# 4. 可视化 (完整版，对标v3)
# ==========================================
def visualize_episode_safe(data_7d, tpb_curves_7d, bspline_curves_7d, knots, ep_idx, task_name, x_axis, save_path):
    """线程安全的可视化函数（2行8列对比TPB和B-spline）"""
    try:
        fig, axes = plt.subplots(2, 8, figsize=(32, 10))
        axes = axes.flatten()
        
        dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
        dim_order = [0, 1, 2, 6, 3, 4, 5]  # x,y,z,gripper,yaw,pitch,roll
        
        # 左侧：TPB方法
        for plot_idx, dim_idx in enumerate(dim_order):
            ax = axes[plot_idx]
            ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, label='Ground Truth', zorder=1)
            line_color = 'red' if dim_idx < 6 else 'green'
            ax.plot(x_axis, tpb_curves_7d[dim_idx], color=line_color, linewidth=2, label='TPB Fit', zorder=2, alpha=0.8)
            if knots:
                for k in knots:
                    ax.axvline(x=k, color='blue', linestyle='--', alpha=0.3, linewidth=1, zorder=0)
            error = np.abs(tpb_curves_7d[dim_idx] - data_7d[dim_idx])
            ax.set_title(f"[TPB] {dim_names[dim_idx]}\nErr: {np.mean(error):.5f} | Max: {np.max(error):.5f}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Index", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)
            if dim_idx == 6:
                y_min, y_max = data_7d[dim_idx].min(), data_7d[dim_idx].max()
                y_range = max(y_max - y_min, 0.5)
                ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
        
        # 右侧：B-spline方法
        for plot_idx, dim_idx in enumerate(dim_order):
            ax = axes[plot_idx + 8]
            ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, label='Ground Truth', zorder=1)
            line_color = 'darkblue' if dim_idx < 6 else 'darkgreen'
            ax.plot(x_axis, bspline_curves_7d[dim_idx], color=line_color, linewidth=2, label='B-spline Fit', zorder=2, alpha=0.8)
            if knots:
                for k in knots:
                    ax.axvline(x=k, color='orange', linestyle='--', alpha=0.3, linewidth=1, zorder=0)
            error = np.abs(bspline_curves_7d[dim_idx] - data_7d[dim_idx])
            ax.set_title(f"[B-spline] {dim_names[dim_idx]}\nErr: {np.mean(error):.5f} | Max: {np.max(error):.5f}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Index", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)
            if dim_idx == 6:
                y_min, y_max = data_7d[dim_idx].min(), data_7d[dim_idx].max()
                y_range = max(y_max - y_min, 0.5)
                ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
        
        # 统计信息
        ax_stats_left = axes[7]
        ax_stats_left.axis('off')
        tpb_errors = np.abs(tpb_curves_7d - data_7d)
        stats_text_left = f"TPB Method\n\nKnots: {len(knots)}\n\nErrors:\n  Mean: {np.mean(tpb_errors):.6f}\n  Max: {np.max(tpb_errors):.6f}\n  Std: {np.std(tpb_errors):.6f}"
        ax_stats_left.text(0.1, 0.5, stats_text_left, fontsize=9, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        ax_stats_right = axes[15]
        ax_stats_right.axis('off')
        bspline_errors = np.abs(bspline_curves_7d - data_7d)
        stats_text_right = f"B-spline Method\n\nControl Points: ~{len(knots)+4}\n\nErrors:\n  Mean: {np.mean(bspline_errors):.6f}\n  Max: {np.max(bspline_errors):.6f}\n  Std: {np.std(bspline_errors):.6f}\n\n✓ Local Control"
        ax_stats_right.text(0.1, 0.5, stats_text_right, fontsize=9, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        fig.suptitle(f'Compression Comparison - Episode {ep_idx}\nTask: {task_name[:80]}...', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"绘图失败 (EP {ep_idx}): {e}")

# ==========================================
# 5. Worker 处理逻辑 (多进程入口)
# ==========================================

# 全局变量，用于子进程共享 Dataset，避免 Pickle 巨大对象
_worker_dataset = None

def init_worker():
    """子进程初始化函数"""
    global _worker_dataset
    # 在 Linux 下，fork 机制会复制父进程内存，这一步可能不需要重新 load
    # 但为了保险，如果 _worker_dataset 为空则 load
    if _worker_dataset is None:
        print(f"[Worker {multiprocessing.current_process().name}] Loading Dataset...")
        _worker_dataset = load_dataset()

def process_episode_task(args):
    """
    子进程执行的单一任务
    Args: (dataset_index, tolerance)
    """
    dataset_index, tolerance = args
    global _worker_dataset
    
    try:
        # 1. 加载数据
        data_7d, episode_idx, frame_idx, task_name = load_episode_by_index(_worker_dataset, dataset_index)
        
        # 2. 准备数据
        T_points = data_7d.shape[0]
        x_axis = np.arange(T_points)
        data_7d = data_7d.T
        gripper_traj = data_7d[6]
        data_6d = data_7d[:6]
        
        forced_knots = extract_forced_knots(x_axis, gripper_traj)
        
        # 3. MILP 求解 (600s timeout * 2，接受次优解)
        knots, is_suboptimal = solve_6d_with_forced_knots(
            x_axis, data_6d, forced_knots, 
            degree=3, tol_ratio=tolerance, time_limit=600 * 2, # NOTE： 把time limit首先拉小，快速处理调现在能处理调的东西，然后对于那种点数特别多，然后还是suboptimial的东西，可以慢慢跑。
            solver_threads=THREADS_PER_SOLVER
        )
        
        if knots is None:
            return None  # 求解失败

        # 4. B-spline 转换
        bspline_control_points = []
        bspline_knot_vectors = []
        bspline_curves_6d = np.zeros_like(data_6d)
        
        for d in range(6):
            ctrl_pts, knot_vec, _ = convert_to_bspline_coeffs(x_axis, data_6d[d], knots, degree=3)
            if ctrl_pts is None: return None
            bspline_control_points.append(ctrl_pts.tolist())
            bspline_knot_vectors.append(knot_vec.tolist())
            bspline_curves_6d[d] = reconstruct_bspline_trajectory(ctrl_pts, knot_vec, x_axis, degree=3)

        # Gripper 处理 (0阶B-spline)
        gripper_ctrl_pts, gripper_knot_vec, _ = convert_to_bspline_coeffs(x_axis, gripper_traj, forced_knots, degree=0)
        if gripper_ctrl_pts is None:
            bspline_control_points.append(gripper_traj.tolist())
            bspline_knot_vectors.append([])
            gripper_bspline = gripper_traj
        else:
            bspline_control_points.append(gripper_ctrl_pts.tolist())
            bspline_knot_vectors.append(gripper_knot_vec.tolist())
            gripper_bspline = reconstruct_bspline_trajectory(gripper_ctrl_pts, gripper_knot_vec, x_axis, degree=0)

        # 5. 组装结果
        bspline_curves_7d = np.vstack([bspline_curves_6d, gripper_bspline])
        overall_mean = float(np.mean(np.abs(bspline_curves_7d - data_7d)))
        
        # TPB 曲线（用于可视化对比）
        gripper_recon = gripper_traj.copy()
        tpb_curves_7d = np.vstack([bspline_curves_6d, gripper_recon])  # 近似用B-spline代替
        
        # 可视化
        vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
        visualize_episode_safe(data_7d, tpb_curves_7d, bspline_curves_7d, knots, episode_idx, task_name, x_axis, vis_path)

        result_dict = {
            "episode_index": episode_idx,
            "task_name": task_name,
            "trajectory_length": T_points,
            "status": "suboptimal" if is_suboptimal else "optimal",
            "bspline": {
                "control_points": bspline_control_points,
                "knot_vectors": bspline_knot_vectors,
                "internal_knots": [float(k) for k in knots],
                "forced_knots": [float(k) for k in forced_knots],
                "num_knots": len(knots),
                "errors_per_dim": [{
                    "mean": float(np.mean(np.abs(bspline_curves_7d[d] - data_7d[d]))),
                    "max": float(np.max(np.abs(bspline_curves_7d[d] - data_7d[d]))),
                    "std": float(np.std(np.abs(bspline_curves_7d[d] - data_7d[d])))
                } for d in range(7)],
                "overall_mean_error": overall_mean
            },
            "visualization_path": str(vis_path)
        }
        return result_dict

    except Exception as e:
        # 返回错误信息但不中断进程
        return {"error": str(e), "episode_idx": locals().get('episode_idx', -1)}

# ==========================================
# 6. 主程序
# ==========================================
def main():
    print("="*80)
    print(f"并行 B-Spline 压缩 | 进程数: {NUM_WORKERS} | 线程/进程: {THREADS_PER_SOLVER} | 总核心占用: {NUM_WORKERS * THREADS_PER_SOLVER}")
    print("="*80)
    
    # 1. 主进程加载数据集 (用于扫描索引)
    print("正在扫描数据集结构...")
    main_dataset = load_dataset()
    total_samples = len(main_dataset)
    
    # 2. 扫描所有 Episode 的起始索引
    # 为了避免在多进程中处理复杂的 next_idx 逻辑，我们先生成一个清晰的任务列表
    task_indices = []
    current_idx = 0
    
    # 这一步是单线程的，但非常快
    pbar = tqdm.tqdm(total=total_samples, desc="Indexing")
    while current_idx < total_samples:
        try:
            item = main_dataset[current_idx]
            ep_idx = int(item['episode_index'])
            ep_len = len(item['abs_aff'])
            
            task_indices.append((current_idx, ep_idx))
            
            pbar.update(ep_len)
            current_idx += ep_len
        except Exception as e:
            print(f"索引扫描错误 at {current_idx}: {e}")
            current_idx += 1
    pbar.close()
    
    print(f"扫描完成，共发现 {len(task_indices)} 个 Episode")

    # 3. 过滤已完成的任务（使用统一的检查逻辑）
    results = load_results_from_json(RESULTS_JSON_PATH)
    
    tasks_to_run = []
    reprocess_count = 0
    for start_idx, ep_id in task_indices:
        if not is_episode_computed(ep_id, results):
            # 未完成或需要重新处理的任务
            tasks_to_run.append((start_idx, TOLERANCE))
            # 判断是否为重新处理
            if str(ep_id) in results.get("episodes", {}):
                reprocess_count += 1
    
    print(f"剩余待处理任务: {len(tasks_to_run)} (其中重新处理: {reprocess_count})")
    
    if not tasks_to_run:
        print("所有任务已完成。")
        return

    # 4. 启动多进程池
    # initializer=init_worker 会在每个子进程启动时调用，加载 dataset
    with multiprocessing.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        
        # 使用 imap_unordered 乱序处理，提高效率
        # chunksize 设置为 1 即可，因为每个任务耗时较长
        iterator = pool.imap_unordered(process_episode_task, tasks_to_run, chunksize=1)
        
        success_count = 0
        fail_count = 0
        
        # 进度条
        pbar_run = tqdm.tqdm(iterator, total=len(tasks_to_run), desc="Processing")
        
        for res in pbar_run:
            if res is None:
                fail_count += 1
                continue
                
            if "error" in res:
                fail_count += 1
                continue
            
            # 保存结果 (主进程串行写入，无锁问题)，覆盖旧数据
            ep_id = str(res['episode_index'])
            results["episodes"][ep_id] = res
            success_count += 1
            
            # 定期保存 (每完成 1 个)
            if success_count % 1 == 0:
                save_results_to_json(results, RESULTS_JSON_PATH)
                pbar_run.set_postfix({"Saved": success_count, "Fail": fail_count, "Saved episode": ep_id, "Mean Error:": res["bspline"]["overall_mean_error"]})
        
        pbar_run.close()

    # 最后保存一次
    save_results_to_json(results, RESULTS_JSON_PATH)
    print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    # 设置启动方法，Linux 默认 fork (快)，Windows/Mac 必须 spawn
    try:
        multiprocessing.set_start_method('fork') 
    except:
        pass
    main()