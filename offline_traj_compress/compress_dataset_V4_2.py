import numpy as np
import pulp
import matplotlib.pyplot as plt
import multiprocessing
import json
import os  # 新增
import glob # 新增
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import BSpline, make_lsq_spline

# ==========================================
# Configuration & Environment Detection
# ==========================================
TOLERANCE = 0.01
# 最终合并的大文件路径 (用于读取已有的历史数据)
RESULTS_JSON_PATH = Path("compression_results_v2.json") 
# 临时结果存放目录 (Slurm模式下每个episode存一个文件，避免写冲突)
PARTIAL_RESULTS_DIR = Path("results_v2_partial") 
VISUAL_DIR = Path("visual_fig_v2")
EP_MAP_PATH = Path("epsidoe_2_dataset_index.json")

TIME_LIMIT_SECONDS = 600

# 如果想跑全量，把这个设为 None 或 []
# PROCESS_INDEX = [67] 
PROCESS_INDEX = None # [31] # 129, 130, 131, 132] # [128] 

def get_env_info():
    """
    自适应获取运行环境信息
    返回: (rank, world_size, num_cpus)
    """
    # 1. 尝试从 Slurm 环境变量获取
    # SLURM_PROCID: 当前进程的序号 (0, 1, 2...)
    # SLURM_NTASKS: 总任务数
    # SLURM_CPUS_PER_TASK: 每个任务分配的核数
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    # 获取 CPU 核数限制
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        num_cpus = int(slurm_cpus)
    else:
        # 普通模式下，留一点余量，或者用全部
        num_cpus = max(1, multiprocessing.cpu_count() - 1)
        
    return rank, world_size, num_cpus

# 初始化环境信息
RANK, WORLD_SIZE, CPU_THREADS = get_env_info()

if RANK == 0:
    print(f"Running Mode: Rank {RANK}/{WORLD_SIZE}, Threads per task: {CPU_THREADS}")
    # 只有 Rank 0 负责创建目录
    PARTIAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# JSON 数据管理函数 (修改版)
# ==========================================
def load_results_from_json(json_path: Path) -> Dict:
    """加载历史大文件结果，避免重复计算"""
    if not json_path.exists():
        return {"episodes": {}}
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return {"episodes": {}}

def is_episode_computed(episode_idx: int, main_results: Dict) -> bool:
    """
    检查是否已计算：
    1. 检查主 JSON 文件 (historical)
    2. 检查分片结果目录 (current run)
    """
    ep_str = str(episode_idx)
    
    # 1. 检查主文件
    if ep_str in main_results.get("episodes", {}):
        ep_data = main_results["episodes"][ep_str]
        if ep_data.get("status") != "failed":
            return True

    # 2. 检查分片文件 (防止并行运行时其他进程已经算完了)
    partial_path = PARTIAL_RESULTS_DIR / f"ep_{episode_idx}.json"
    if partial_path.exists():
        # 简单检查文件是否完整
        try:
            with open(partial_path, 'r') as f:
                json.load(f)
            return True
        except:
            return False # 文件损坏，重新算
            
    return False

def save_single_episode_result(episode_res: Dict):
    """
    【核心修改】
    不修改大 JSON，而是保存单个小 JSON 文件。
    文件名例如: results_v2_partial/ep_67.json
    """
    ep_idx = episode_res["episode_index"]
    save_path = PARTIAL_RESULTS_DIR / f"ep_{ep_idx}.json"
    
    # 为了线程安全，先写临时文件再重命名 (Atomic write)
    temp_path = save_path.with_suffix(".tmp")
    with open(temp_path, 'w') as f:
        json.dump(episode_res, f, indent=2)
    os.rename(temp_path, save_path)
    print(f"  Process {RANK}: Saved partial result to {save_path}")

# ==========================================
# 1. 工具函数 (保持不变)
# ==========================================
def extract_forced_knots(time_points, gripper_data) -> List[int]:
    diff = np.diff(gripper_data)
    change_indices = np.where(diff != 0)[0] + 1
    forced_knot_times = []
    max_idx = len(time_points) - 1
    for idx in change_indices:
        if 0 < idx < max_idx:
            forced_knot_times.append(int(time_points[idx]))
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
        print(f"  [Warning] B-spline转换失败: {e}")
        return None, None, None

def reconstruct_bspline_trajectory(control_points, knot_vector, time_points, degree=3):
    bspline = BSpline(knot_vector, control_points, degree, extrapolate=False)
    return bspline(time_points)

# ==========================================
# 注意：上面的 solve_6d 函数我进行了简化演示
# 请把你的 原版 solve_6d 函数复制过来，
# 唯一需要改的地方是 solver = ... 那一行
# 改为 threads=num_threads
# ==========================================

# 为了让你直接能用，我在这里把 原版 solve_6d 加上 threads 修改版写全：

def solve_6d_with_forced_knots_full(time_points, data_6d, forced_knot_times,
                               degree=3, tol_ratio=0.03, time_limit=600, 
                               use_gurobi=False, num_threads=1):
    T = len(time_points)
    num_dims = data_6d.shape[0]
    check_step = 1
    candidates = list(range(1, T - 1))
    forced_set = set(int(t) for t in forced_knot_times if 1 <= int(t) <= T - 2)

    t0, t_end = float(time_points[0]), float(time_points[-1])
    internal = np.arange(1, T - 1, dtype=float)
    U = np.concatenate([np.repeat(t0, degree + 1), internal, np.repeat(t_end, degree + 1)])
    n_basis = len(U) - degree - 1
    indices_to_check = list(range(0, T, check_step))
    if (T - 1) not in indices_to_check: indices_to_check.append(T - 1)

    # 预计算 B
    B_check = np.zeros((len(indices_to_check), n_basis), dtype=float)
    eye = np.eye(n_basis, dtype=float)
    for j in range(n_basis):
        bj = BSpline(U, eye[j], degree, extrapolate=False)
        B_check[:, j] = bj(time_points)[indices_to_check]
    row_nz_check = [np.where(B_check[i] > 1e-12)[0].tolist() for i in range(len(indices_to_check))]

    # Step 1: L1
    prob_lp = pulp.LpProblem("BSpline_L1", pulp.LpMinimize)
    c_lp = {}
    for d in range(num_dims):
        for j in range(n_basis):
            c_lp[(d, j)] = pulp.LpVariable(f"c_{d}_{j}", cat='Continuous')
    y_lp = {}
    for k in candidates:
        y_lp[k] = pulp.LpVariable(f"y_{k}", 0, 1, cat='Continuous')
    for k in forced_set: prob_lp += y_lp[k] == 1
    
    g_lp = {}
    M_relax = 200.0
    for d in range(num_dims):
        for k in candidates:
            i = k + 3
            g_lp[(d, k)] = pulp.LpVariable(f"g_{d}_{k}", cat='Continuous')
            delta4 = (c_lp[(d, i)] - 4*c_lp[(d, i-1)] + 6*c_lp[(d, i-2)] - 4*c_lp[(d, i-3)] + c_lp[(d, i-4)])
            prob_lp += g_lp[(d, k)] == delta4
            prob_lp += g_lp[(d, k)] <= M_relax * y_lp[k]
            prob_lp += g_lp[(d, k)] >= -M_relax * y_lp[k]
    
    for d in range(num_dims):
        d_range = float(np.max(data_6d[d]) - np.min(data_6d[d]))
        if d_range < 1e-6: d_range = 1.0
        eps = d_range * float(tol_ratio)
        for idx_in_check, real_t_idx in enumerate(indices_to_check):
            P = float(data_6d[d][real_t_idx])
            idxs = row_nz_check[idx_in_check]
            fit_expr = pulp.lpSum([float(B_check[idx_in_check, j]) * c_lp[(d, j)] for j in idxs])
            prob_lp += fit_expr - P <= eps
            prob_lp += P - fit_expr <= eps
            
    prob_lp += pulp.lpSum([y_lp[k] for k in candidates])

    # Solver Step 1
    if use_gurobi:
        try:
            solver_lp = pulp.GUROBI(msg=False, threads=num_threads, timeLimit=time_limit)
        except:
            solver_lp = pulp.PULP_CBC_CMD(msg=False, threads=num_threads, timeLimit=time_limit)
    else:
        solver_lp = pulp.PULP_CBC_CMD(msg=False, threads=num_threads, timeLimit=time_limit)
    
    prob_lp.solve(solver_lp)

    # Step 2: MILP
    prob_milp = pulp.LpProblem("BSpline_MILP", pulp.LpMinimize)
    c_milp = {}
    for d in range(num_dims):
        for j in range(n_basis):
            c_milp[(d, j)] = pulp.LpVariable(f"c_{d}_{j}", cat='Continuous')
    y_milp = {}
    for k in candidates:
        y_milp[k] = pulp.LpVariable(f"y_{k}", cat='Binary')
    for k in forced_set: prob_milp += y_milp[k] == 1
    
    g_milp = {}
    M_milp = 200.0
    for d in range(num_dims):
        for k in candidates:
            i = k + 3
            g_milp[(d, k)] = pulp.LpVariable(f"g_{d}_{k}", cat='Continuous')
            delta4 = (c_milp[(d, i)] - 4*c_milp[(d, i-1)] + 6*c_milp[(d, i-2)] - 4*c_milp[(d, i-3)] + c_milp[(d, i-4)])
            prob_milp += g_milp[(d, k)] == delta4
            prob_milp += g_milp[(d, k)] <= M_milp * y_milp[k]
            prob_milp += g_milp[(d, k)] >= -M_milp * y_milp[k]

    for d in range(num_dims):
        d_range = float(np.max(data_6d[d]) - np.min(data_6d[d]))
        if d_range < 1e-6: d_range = 1.0
        eps = d_range * float(tol_ratio)
        for idx_in_check, real_t_idx in enumerate(indices_to_check):
            P = float(data_6d[d][real_t_idx])
            idxs = row_nz_check[idx_in_check]
            fit_expr = pulp.lpSum([float(B_check[idx_in_check, j]) * c_milp[(d, j)] for j in idxs])
            prob_milp += fit_expr - P <= eps
            prob_milp += P - fit_expr <= eps

    prob_milp += pulp.lpSum([y_milp[k] for k in candidates])
    
    # Solver Step 2
    if use_gurobi:
        try:
            solver_milp = pulp.GUROBI(msg=True, threads=num_threads, timeLimit=time_limit)
        except:
            solver_milp = pulp.PULP_CBC_CMD(msg=True, threads=num_threads, timeLimit=time_limit)
    else:
        solver_milp = pulp.PULP_CBC_CMD(msg=True, threads=num_threads, timeLimit=time_limit)
        
    prob_milp.solve(solver_milp)
    
    if prob_milp.status in (pulp.LpStatusNotSolved, pulp.LpStatusUndefined):
        return [], None, False

    active_knots = [int(k) for k in candidates if pulp.value(y_milp[k]) > 0.5]
    fitted_curves = np.zeros_like(data_6d, dtype=float)
    for d in range(num_dims):
        ctrl_pts = [pulp.value(c_milp[(d, j)]) for j in range(n_basis)]
        spl = BSpline(U, ctrl_pts, degree, extrapolate=False)
        fitted_curves[d] = spl(time_points)
        
    return active_knots, fitted_curves, prob_milp.status


# ==========================================
# 3. 数据加载与处理 (保持不变)
# ==========================================
def load_dataset():
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    use_default_root = (torch.cuda.is_available() and torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name(0))
    kwargs = dict(delta_timestamps={"abs_aff": []})
    if not use_default_root:
        kwargs["root"] = Path("/inspire/hdd/project/robot-decision/public/datasets/") / "HuggingFaceVLA/libero"
    dataset = LeRobotDataset("HuggingFaceVLA/libero", **kwargs)
    return dataset

def load_episode_by_index(dataset, dataset_index):
    item = dataset[dataset_index]
    abs_aff = np.asarray(item['abs_aff'])
    abs_aff_absolute = abs_aff.copy()
    abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
    return abs_aff_absolute, int(item['episode_index']), int(item['frame_index']), item.get('task', "Unknown")

def load_or_build_episode_index_map(dataset):
    if EP_MAP_PATH.exists():
        with open(EP_MAP_PATH, "r") as f:
            data = json.load(f)
        return {int(k): int(v) for k, v in data.items()}
    mapping = {}
    idx = 0
    total = len(dataset)
    while idx < total:
        item = dataset[idx]
        frame_idx = int(item["frame_index"])
        if frame_idx != 0:
            idx += 1
            continue
        mapping[int(item["episode_index"])] = idx
        idx += len(item["abs_aff"])
    with open(EP_MAP_PATH, "w") as f: json.dump(mapping, f, indent=2)
    return mapping

# ==========================================
# 4. 可视化 (2x4 布局，仅显示 B-spline)
# ==========================================
def visualize_episode(data_7d: np.ndarray,
                      bspline_curves_7d: np.ndarray, knots: List, 
                      episode_idx: int, task_name: str, x_axis: np.ndarray, 
                      save_path: Path):
    # 创建 2 行 4 列的子图，只显示 B-spline 结果
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    dim_order = [0, 1, 2, 6, 3, 4, 5]  # 7个维度

    for plot_idx, dim_idx in enumerate(dim_order):
        ax = axes[plot_idx]
        ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, label='Original')
        ax.plot(x_axis, bspline_curves_7d[dim_idx], color='darkblue' if dim_idx < 6 else 'darkgreen', 
                linewidth=2, label='B-spline')
        if knots:
            for k in knots:
                ax.axvline(x=k, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f"[B-spline] {dim_names[dim_idx]}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # 在最后一个子图显示统计信息
    ax_stats = axes[-1]
    ax_stats.axis('off')
    
    # 计算误差统计
    errors = np.abs(bspline_curves_7d - data_7d)
    error_mean = np.mean(errors)
    error_max = np.max(errors)
    error_std = np.std(errors)
    
    # 计算控制点数量 (knots + 边界控制点)
    # 对于 degree=3 的 B-spline: n_control_points = n_knots + degree + 1
    n_control_points = len(knots) + 3 + 1  # knots + degree + 1
    
    # 显示统计信息
    stats_text = f"""
Statistics Summary
{'='*30}

Control Points: {n_control_points}
Internal Knots: {len(knots)}

Error Metrics (L1):
  Mean:   {error_mean:.6f}
  Max:    {error_max:.6f}
  Std:    {error_std:.6f}

Trajectory Length: {len(x_axis)}
Compression Ratio: {len(x_axis) / n_control_points:.2f}x
"""
    
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f"Episode {episode_idx} - B-spline Compression (Knots: {len(knots)})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 5. 处理函数 (修改: 使用多线程参数)
# ==========================================
def process_single_episode(dataset, dataset_index: int, main_results: Dict):
    data_7d, episode_idx, frame_idx, task_name = load_episode_by_index(dataset, dataset_index)
    x_axis = np.arange(data_7d.shape[0])
    data_7d = data_7d.T
    
    # 使用修改后的 Solver, 传入 CPU_THREADS
    knots, curves_6d, status = solve_6d_with_forced_knots_full(
        x_axis, data_7d[:6], extract_forced_knots(x_axis, data_7d[6]),
        degree=3, tol_ratio=TOLERANCE, time_limit=TIME_LIMIT_SECONDS, 
        use_gurobi=False, num_threads=CPU_THREADS
    )
    
    if curves_6d is None: return None
    
    # ... B-spline 转换逻辑 (保持不变) ...
    bspline_control_points = []
    bspline_curves_7d = np.zeros_like(data_7d)
    
    # 6D Arm
    for d in range(6):
        c, _, _ = convert_to_bspline_coeffs(x_axis, data_7d[d], knots, 3)
        bspline_control_points.append(c.tolist())
        bspline_curves_7d[d] = reconstruct_bspline_trajectory(
            c, np.concatenate([np.repeat(x_axis[0], 4), sorted(knots), np.repeat(x_axis[-1], 4)]), x_axis, 3)
            
    # Gripper
    gc, _, _ = convert_to_bspline_coeffs(x_axis, data_7d[6], knots, 0)
    bspline_control_points.append([int(round(v)) for v in gc])
    bspline_curves_7d[6] = reconstruct_bspline_trajectory(
        gc, np.concatenate([np.repeat(x_axis[0], 1), sorted(knots), np.repeat(x_axis[-1], 1)]), x_axis, 0)
    
    # Save Vis
    vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
    visualize_episode(data_7d, bspline_curves_7d, knots, episode_idx, task_name, x_axis, vis_path)
    
    # Return Dict
    t0, t_end = int(x_axis[0]), int(x_axis[-1])
    full_knots = [t0]*4 + sorted([int(k) for k in knots]) + [t_end]*4
    
    return {
        "episode_index": episode_idx,
        "task_name": task_name,
        "status":  pulp.LpStatus.get(status),
        "bspline": {
            "knots_vector": full_knots,
            "control_points": bspline_control_points,
            "overall_mean_error": float(np.mean(np.abs(bspline_curves_7d - data_7d)))
        },
        "visualization_path": str(vis_path)
    }

# ==========================================
# 6. 主程序 (Slurm 并行化核心逻辑)
# ==========================================
def main():
    print("="*80)
    print(f"MILP Compression | Rank {RANK}/{WORLD_SIZE} | CPUs: {CPU_THREADS}")
    print("="*80)
    
    dataset = load_dataset()
    ep_index_map = load_or_build_episode_index_map(dataset)
    
    # 加载已有的主结果 (只读)
    main_results = load_results_from_json(RESULTS_JSON_PATH)
    
    # 1. 确定要处理的任务列表
    all_episodes = sorted(list(ep_index_map.keys()))
    
    # 如果指定了 PROCESS_INDEX，则只处理指定的
    if PROCESS_INDEX is not None and len(PROCESS_INDEX) > 0:
        target_episodes = [ep for ep in all_episodes if ep in PROCESS_INDEX]
    else:
        target_episodes = all_episodes
        
    # 2. Slurm 任务切分 (Strided Slicing)
    # 例如: Rank 0 处理 [0, 100, 200...], Rank 1 处理 [1, 101, 201...]
    my_episodes = target_episodes[RANK::WORLD_SIZE]
    
    print(f"  [P{RANK}] Total episodes: {len(target_episodes)}. My workload: {len(my_episodes)}")
    
    processed_count = 0
    for episode_idx in my_episodes:
        dataset_idx = ep_index_map[episode_idx]
        
        # 检查是否已完成
        if is_episode_computed(episode_idx, main_results):
            # print(f"  [P{RANK}] Skip {episode_idx}")
            continue

        try:
            print(f"  [P{RANK}] Processing Ep {episode_idx}...")
            res = process_single_episode(dataset, dataset_idx, main_results)
            if res:
                # 【重要】保存为独立的小文件
                save_single_episode_result(res)
                processed_count += 1
        except Exception as e:
            print(f"  [P{RANK}] Error {episode_idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[P{RANK}] Done. Processed: {processed_count}")

    # ======================================
    # 可选：合并脚本 (仅在所有任务结束后，手动运行或由Rank0运行)
    # ======================================
    # 这里我们只打印一条提示，建议单独写一个 merge.py 或者在作业脚本最后跑一次
    # if RANK == 0:
    #    merge_partial_results() 
    
def merge_partial_results_tool():
    """
    这是一个辅助工具函数，用于最后合并所有 json
    可以在所有 Slurm 任务跑完后，单独调用一次
    """
    print("Merging partial results...")
    final_data = load_results_from_json(RESULTS_JSON_PATH)
    if "episodes" not in final_data: final_data["episodes"] = {}
    
    partial_files = glob.glob(str(PARTIAL_RESULTS_DIR / "*.json"))
    for p_file in partial_files:
        try:
            with open(p_file, 'r') as f:
                data = json.load(f)
                ep_idx = str(data["episode_index"])
                final_data["episodes"][ep_idx] = data
        except:
            print(f"Skipping bad file: {p_file}")
            
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"Merged {len(partial_files)} files into {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    # 如果作为脚本参数传入 --merge，则执行合并
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_partial_results_tool()
    else:
        main()