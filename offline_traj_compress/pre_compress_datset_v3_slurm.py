import numpy as np
import pulp
import matplotlib.pyplot as plt
import multiprocessing
import json
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import BSpline, make_lsq_spline

# ==========================================
# Configuration
# ==========================================
TOLERANCE = 0.01
RESULTS_JSON_PATH = Path("compression_results.json") # 代码会自动追加 _part_{rank}.json
VISUAL_DIR = Path("test/visual_fig")
EP_MAP_PATH = Path("epsidoe_2_dataset_index.json")

# ==========================================
# 0. HPC 环境感知 (新增)
# ==========================================
def get_slurm_context():
    """
    自动检测 Slurm 环境，返回 (当前进程ID, 总进程数, 分配的CPU核数)
    """
    # 1. 获取 Rank (我是第几个)
    rank = os.getenv('SLURM_PROCID')
    if rank is None:
        rank = 0
    else:
        rank = int(rank)

    # 2. 获取 World Size (总共有几个)
    world_size = os.getenv('SLURM_NTASKS')
    if world_size is None:
        world_size = 1
    else:
        world_size = int(world_size)

    # 3. 获取分配给当前任务的 CPU 核数 (用于 CBC Solver)
    cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    if cpus_per_task is None:
        # 单机模式下，默认保守一点，或者使用 CPU 核心的一半
        cpus_per_task = max(1, multiprocessing.cpu_count() // 2)
    else:
        cpus_per_task = int(cpus_per_task)

    return rank, world_size, cpus_per_task

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
    
    # 增加容错，防止读取空文件报错
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[Warn] 文件 {json_path} 损坏或为空，重新开始。")
        return {
            "metadata": {
                "tolerance": TOLERANCE,
                "degree": 3,
                "dataset": "HuggingFaceVLA/libero"
            },
            "episodes": {}
        }


def save_results_to_json(results: Dict, json_path: Path):
    """保存压缩结果到JSON文件"""
    # 确保父目录存在
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    # print(f"  ✓ 结果已保存到: {json_path}") # 减少日志刷屏


def is_episode_computed(episode_idx: int, results: Dict) -> bool:
    """
    检查指定episode是否已正确计算
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
        # print(f"  [重新计算] Episode {episode_idx}: 求解失败标记")
        return False
    
    # 情况3：overall_mean_error > 0.5（精度不足）
    mean_error = bspline.get("overall_mean_error", 0.0)
    if mean_error > 0.5:
        # print(f"  [重新计算] Episode {episode_idx}: 精度不足")
        return False
    
    return True


# ==========================================
# 1. 工具函数
# ==========================================
def extract_forced_knots(time_points, gripper_data):
    """返回 gripper 跳变位置对应的 knot 时间"""
    diff = np.diff(gripper_data)
    change_indices = np.where(diff != 0)[0] + 1

    forced_knot_times = []
    for idx in change_indices:
        if 0 < idx < len(time_points) - 1:
            forced_knot_times.append(time_points[idx])
    return forced_knot_times


def convert_to_bspline_coeffs(time_points: np.ndarray, data_1d: np.ndarray, 
                               internal_knots: List[float], degree: int = 3) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[BSpline]]:
    """将MILP算出的Knots转换为标准B-spline的控制点"""
    t0, t_end = time_points[0], time_points[-1]
    sorted_knots = sorted([k for k in internal_knots if t0 < k < t_end])
    full_knot_vector = np.concatenate([
        np.repeat(t0, degree + 1),
        sorted_knots,
        np.repeat(t_end, degree + 1)
    ])
    
    try:
        bspline = make_lsq_spline(time_points, data_1d, full_knot_vector, k=degree)
        control_points = bspline.c
        return control_points, full_knot_vector, bspline
    except Exception as e:
        print(f"  [Warning] B-spline转换失败: {e}")
        return None, None, None


def reconstruct_bspline_trajectory(control_points: np.ndarray, knot_vector: np.ndarray, 
                                    time_points: np.ndarray, degree: int = 3) -> np.ndarray:
    """从B-spline控制点重建完整轨迹"""
    bspline = BSpline(knot_vector, control_points, degree, extrapolate=False)
    return bspline(time_points)

# ==========================================
# 2. 求解器 (修改：接受 threads 参数)
# ==========================================
def solve_6d_with_forced_knots(time_points, data_6d, forced_knot_times, degree=3, tol_ratio=0.03, time_limit=600, threads=1):
    num_dims = data_6d.shape[0]
    candidate_knots = time_points[1:-1:1] 

    prob = pulp.LpProblem("RobotArm_6D_Constrained", pulp.LpMinimize)
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
    
    # 只有 Rank 0 或 debug 时才打印这个，避免刷屏
    # print(f"  [Info] 已添加 {forced_count} 个强制 Knot 约束")

    alpha = {}
    beta = {}
    BigMs = []
    epsilons = []

    for d in range(num_dims):
        d_range = np.max(data_6d[d]) - np.min(data_6d[d])
        if d_range < 1e-6:
            d_range = 1.0

        eps = d_range * tol_ratio
        epsilons.append(eps)
        BigMs.append(d_range * 1.0) 
        alpha[d] = pulp.LpVariable.dicts(f"alpha_{d}", range(degree + 1), lowBound=None)
        beta[d] = pulp.LpVariable.dicts(f"beta_{d}", candidate_knots, lowBound=None)

    total_knots = pulp.lpSum([y[j] for j in candidate_knots])
    prob += total_knots         
    prob += total_knots >= 5    
    prob += total_knots <= 70   

    for d in range(num_dims):
        M_d = BigMs[d]
        eps_d = epsilons[d]

        for j in candidate_knots:
            prob += beta[d][j] <= M_d * y[j]
            prob += beta[d][j] >= -M_d * y[j]

        for i, t_val in enumerate(time_points):
            P_val = data_6d[d][i]
            poly_part = pulp.lpSum([alpha[d][k] * (t_val**k) for k in range(degree + 1)])
            knot_terms = [beta[d][j] * ((t_val - j) ** degree) for j in candidate_knots if t_val > j]
            S_t = poly_part + pulp.lpSum(knot_terms)
            prob += S_t - P_val <= eps_d
            prob += P_val - S_t <= eps_d

    # [修改点] 移除 multiprocessing.cpu_count(), 使用传入的 threads
    print(f"  正在求解... Solver: CBC (Threads: {threads}) | TimeLimit: {time_limit}s")
    
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=threads) # msg=False 减少日志
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]
    objective_val = pulp.value(prob.objective)

    is_suboptimal = False
    if status_str == "Optimal":
        pass
    elif objective_val is not None:
        print(f"  [Info] 时间耗尽，返回次优解 (Knots: {int(objective_val)})")
        is_suboptimal = True
    else:
        print(f"  [Error] 求解失败。Status: {status_str}")
        return [], None, False

    active_knots = [j for j in candidate_knots if pulp.value(y[j]) > 0.5]

    fitted_curves = np.zeros_like(data_6d)
    for d in range(num_dims):
        a_vals = [pulp.value(alpha[d][k]) for k in range(degree + 1)]
        b_vals = {j: pulp.value(beta[d][j]) for j in candidate_knots}
        for i, t_val in enumerate(time_points):
            val = sum([a_vals[k] * (t_val**k) for k in range(degree + 1)])
            for j in candidate_knots:
                if t_val > j:
                    val += b_vals[j] * ((t_val - j) ** degree)
            fitted_curves[d, i] = val

    return active_knots, fitted_curves, is_suboptimal

# ==========================================
# 3. 数据加载与映射 (修改：Rank 0 写入权限)
# ==========================================
def load_dataset():
    """加载完整数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # 假设 Dataset 是只读的，多进程读取没问题
    dataset = LeRobotDataset(
        'HuggingFaceVLA/libero',
        root=Path("/inspire/hdd/project/robot-decision/public/datasets/")/'HuggingFaceVLA/libero',
        delta_timestamps={"abs_aff":[]}
    )
    return dataset


def load_episode_by_index(dataset, dataset_index: int) -> Tuple[np.ndarray, int, int, str]:
    item = dataset[dataset_index]
    abs_aff = np.asarray(item['abs_aff'])
    abs_aff_absolute = abs_aff.copy()
    abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
    
    episode_idx = int(item['episode_index'])
    frame_idx = int(item['frame_index'])
    task_name = item['task'] if 'task' in item else "Unknown"
    return abs_aff_absolute, episode_idx, frame_idx, task_name


def load_or_build_episode_index_map(dataset, rank=0) -> Dict[int, int]:
    """加载或构建映射。注意：只有 Rank 0 允许写入文件"""
    if EP_MAP_PATH.exists():
        # print(f"[Rank {rank}] 发现映射文件，正在加载...")
        try:
            with open(EP_MAP_PATH, "r") as f:
                data = json.load(f)
            return {int(k): int(v) for k, v in data.items()}
        except:
            print(f"[Rank {rank}] 映射文件读取失败，尝试重新构建...")

    print(f"[Rank {rank}] 开始遍历数据集构建映射...")
    mapping = {}
    idx = 0
    total = len(dataset)
    while idx < total:
        item = dataset[idx]
        ep_idx = int(item["episode_index"])
        frame_idx = int(item["frame_index"])
        if frame_idx != 0:
            idx += 1
            continue

        mapping[ep_idx] = idx
        ep_len = len(item["abs_aff"])
        idx += ep_len 

    # [修改点] 只有 Rank 0 保存文件，防止并发写入损坏
    if rank == 0:
        with open(EP_MAP_PATH, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"[Rank {rank}] 映射已保存到 {EP_MAP_PATH}")
    
    return mapping


# ==========================================
# 4. 可视化函数 (无重大修改，仅调整路径逻辑)
# ==========================================
def visualize_episode(data_7d, tpb_curves_7d, bspline_curves_7d, knots, 
                      episode_idx, task_name, x_axis, save_path):
    # 代码保持原样，省略以节省篇幅，功能不变
    # ... (Keep existing visualization code)
    # 为保证代码完整性，这里简略写，实际运行请保留原来的 visualize_episode 代码
    pass 
    # (实际上你的代码里这里已经写好了，我就不重复占位了，用你原来的即可)
    # 为了防止报错，我把你的可视化函数核心部分再粘贴回来：
    fig, axes = plt.subplots(2, 8, figsize=(32, 10))
    axes = axes.flatten()
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    dim_order = [0, 1, 2, 6, 3, 4, 5] 
    
    for plot_idx, dim_idx in enumerate(dim_order):
        ax = axes[plot_idx]
        ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, label='GT')
        line_color = 'red' if dim_idx < 6 else 'green'
        ax.plot(x_axis, tpb_curves_7d[dim_idx], color=line_color, linewidth=2, label='TPB')
        if knots:
            for k in knots: ax.axvline(x=k, color='blue', linestyle='--', alpha=0.3)
        ax.set_title(f"[TPB] {dim_names[dim_idx]}")
    
    for plot_idx, dim_idx in enumerate(dim_order):
        ax = axes[plot_idx + 8]
        ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, label='GT')
        line_color = 'darkblue' if dim_idx < 6 else 'darkgreen'
        ax.plot(x_axis, bspline_curves_7d[dim_idx], color=line_color, linewidth=2, label='BSpline')
        if knots:
            for k in knots: ax.axvline(x=k, color='orange', linestyle='--', alpha=0.3)
        ax.set_title(f"[BSpline] {dim_names[dim_idx]}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight') # 降低dpi稍微快点
    plt.close(fig)


# ==========================================
# 5. Episode 处理函数 (修改：接受 solver_threads)
# ==========================================
def process_single_episode(dataset, dataset_index: int, results: Dict, solver_threads: int = 1) -> Optional[Dict]:
    """处理单个episode"""
    data_7d, episode_idx, frame_idx, task_name = load_episode_by_index(dataset, dataset_index)
    assert frame_idx == 0
    
    # print(f"Processing Ep {episode_idx} | Len: {len(data_7d)}") # 减少日志
    
    T_points = data_7d.shape[0]
    x_axis = np.arange(T_points)
    data_7d = data_7d.T
    gripper_traj = data_7d[6]
    data_6d = data_7d[:6]
    
    forced_knots = extract_forced_knots(x_axis, gripper_traj)
    
    # [修改点] 传入 solver_threads
    knots, curves_6d, is_suboptimal = solve_6d_with_forced_knots(
        time_points=x_axis,
        data_6d=data_6d,
        forced_knot_times=forced_knots,
        degree=3,
        tol_ratio=TOLERANCE,
        time_limit=600 * 2,
        threads=solver_threads 
    )
    
    if curves_6d is None:
        print(f"  ✗ Episode {episode_idx} 求解失败")
        return None
    
    # ... (B-spline 转换部分保持不变) ...
    bspline_control_points = []
    bspline_knot_vectors = []
    bspline_curves_6d = np.zeros_like(data_6d)
    
    for d in range(6):
        ctrl_pts, knot_vec, bspline_obj = convert_to_bspline_coeffs(
            x_axis, data_6d[d], knots, degree=3
        )
        if ctrl_pts is None: return None
        bspline_control_points.append(ctrl_pts.tolist())
        bspline_knot_vectors.append(knot_vec.tolist())
        bspline_curves_6d[d] = reconstruct_bspline_trajectory(ctrl_pts, knot_vec, x_axis, degree=3)
    
    gripper_ctrl_pts, gripper_knot_vec, _ = convert_to_bspline_coeffs(
        x_axis, gripper_traj, forced_knots, degree=0
    )
    if gripper_ctrl_pts is not None:
        bspline_control_points.append(gripper_ctrl_pts.tolist())
        bspline_knot_vectors.append(gripper_knot_vec.tolist())
        gripper_bspline = reconstruct_bspline_trajectory(gripper_ctrl_pts, gripper_knot_vec, x_axis, degree=0)
    else:
        bspline_control_points.append(gripper_traj.tolist())
        bspline_knot_vectors.append([])
        gripper_bspline = gripper_traj.copy()
    
    bspline_curves_7d = np.vstack([bspline_curves_6d, gripper_bspline])
    
    # 简单的误差计算
    bspline_overall_mean = float(np.mean(np.abs(bspline_curves_7d - data_7d)))
    
    # 可视化 (需要补全参数)
    tpb_curves_7d = np.vstack([curves_6d, gripper_traj])
    vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
    # 注意：为了运行速度，大规模测试时可以注释掉 visualize_episode
    visualize_episode(data_7d, tpb_curves_7d, bspline_curves_7d, knots, 
                      episode_idx, task_name, x_axis, vis_path)
    
    episode_result = {
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
            "overall_mean_error": bspline_overall_mean
        },
        "visualization_path": str(vis_path)
    }
    
    print(f"  ✓ Episode {episode_idx} 完成 | Err: {bspline_overall_mean:.6f}")
    return episode_result


# ==========================================
# 6. 主程序 (核心修改：任务分发与环境感知)
# ==========================================
def main():
    # 1. 获取 HPC 环境上下文
    rank, world_size, cpus_per_task = get_slurm_context()

    print("="*80)
    print(f"HPC Job Started: Rank {rank}/{world_size} | Threads per solver: {cpus_per_task}")
    print("="*80)
    
    # 2. 定义【独立】的输出文件路径
    # 例如: compression_results_part_0.json
    my_json_path = RESULTS_JSON_PATH.parent / f"{RESULTS_JSON_PATH.stem}_part_{rank}.json"
    print(f"[Rank {rank}] 结果将保存到: {my_json_path}")

    # 3. 加载数据集
    print(f"[Rank {rank}] 正在加载数据集...")
    dataset = load_dataset()
    
    # 4. 加载映射 (内部已处理并发安全，只有rank 0写)
    ep_index_map = load_or_build_episode_index_map(dataset, rank=rank)
    
    # 获取全局任务列表并排序（确保顺序一致）
    all_episode_ids = sorted(list(ep_index_map.keys()))
    
    # ==========================================
    # 关键：动态任务切片 (Stride Slicing)
    # ==========================================
    # 进程 0: [0, 4, 8...]
    # 进程 1: [1, 5, 9...]
    my_episode_ids = all_episode_ids[rank::world_size]
    
    print(f"[Rank {rank}] 全局任务数: {len(all_episode_ids)} | 本地分配: {len(my_episode_ids)}")
    
    # 5. 加载【自己的】历史进度 (断点续传)
    results = load_results_from_json(my_json_path)
    
    processed_count = 0
    skipped_count = 0

    for episode_idx in my_episode_ids:
        dataset_idx = ep_index_map[episode_idx]

        # 检查是否已计算 (检查自己的文件)
        if is_episode_computed(episode_idx, results):
            skipped_count += 1
            continue

        try:
            # 传入 cpus_per_task
            episode_result = process_single_episode(
                dataset, dataset_idx, results, 
                solver_threads=cpus_per_task
            )

            if episode_result is not None:
                results["episodes"][str(episode_idx)] = episode_result
                # 每处理一个就保存一次，或者每 N 个保存一次
                save_results_to_json(results, my_json_path)
                processed_count += 1
        
        except Exception as e:
            print(f"\n[Rank {rank}] ✗ Error Episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()

    print("="*80)
    print(f"[Rank {rank}] 完成。处理: {processed_count}, 跳过: {skipped_count}")
    print("="*80)


if __name__ == "__main__":
    main()