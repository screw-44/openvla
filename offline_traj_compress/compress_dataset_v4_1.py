    # 67, 91, 128, 179, 223, 235, 247, 267, 316, 360, 378, 386, 
    # 387, 388, 394, 402, 416, 431, 447, 457, 459, 464, 468, 491, 
    # 492, 495, 508, 514, 522, 532, 567, 587, 613, 620, 626, 629, 635, 
    # 637, 641, 649, 653, 655, 658, 677, 679, 682, 689, 698, 706, 711, 
    # 712, 718, 720, 727, 743, 745, 747, 760, 773, 774, 781, 788, 789, 
    # 796, 811, 812, 815, 822, 823, 826, 829, 842, 843, 847, 852, 853, 
    # 858, 865, 867, 871, 880, 883, 891, 893, 898, 899, 902, 903, 906, 
    # 913, 922, 924, 933, 940, 952, 955, 959, 962, 967, 979, 984, 987, 
    # 988, 995, 1001, 1006, 1007, 1015, 1021, 1028, 1031, 1047, 1050, 
    # 1051, 1063, 1065, 1066, 1068, 1094, 1096, 1097, 1098, 1100, 1107, 
    # 1108, 1111, 1121, 1122, 1124, 1131, 1137, 1139, 1153, 1157, 1171, 
    # 1172, 1175, 1186, 1191, 1192, 1193, 1196, 1200, 1204, 1205, 1210, 
    # 1217, 1224, 1249, 1251, 1259, 1264, 1272, 1274, 1275, 1278, 1288, 
    # 1292, 1299, 1306, 1312, 1313, 1320, 1343, 1347, 1354, 1370, 1386, 
    # 1388, 1390, 1395, 1396, 1397, 1400, 1401, 1402, 1404, 1414, 1425, 
    # 1441, 1446, 1449, 1450, 1451, 1460, 1462, 1464, 1468, 1471, 1481, 
    # 1524, 1535, 1550, 1553, 1560, 1562, 1565, 1570, 1574, 1577, 1602, 
    # 1610, 1615, 1616, 1617, 1622, 1630, 1646, 1653, 1659, 1660, 1662, 
    # 1667, 1669, 1671, 1672, 1677, 1680, 1682, 1685] # 仅仅需要处理的index

#  "internal_knots": [
        #   11,
        #   19,
        #   31,
        #   45,
        #   54,
        #   64,
        #   72,
        #   84,
        #   93,
        #   113,
        #   118,
        #   121,
        #   132,
        #   146,
        #   152,
        #   163,
        #   172,
        #   188,
        #   193,
        #   209,
        #   215,
        #   229,
        #   234,
        #   246,
        #   257,
        #   265
        # ],
import numpy as np
import pulp
import matplotlib.pyplot as plt
import multiprocessing
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import BSpline, make_lsq_spline

# ==========================================
# Configuration
# ==========================================
TOLERANCE = 0.01
RESULTS_JSON_PATH = Path("compression_results_v2_no_time_limit.json")  # Modified V2
VISUAL_DIR = Path("visual_fig_v2") # Modified dir to separate from v1
EP_MAP_PATH = Path("epsidoe_2_dataset_index.json")

TIME_LIMIT_SECONDS = 999999  # 每个 episode 的求解时间限制，单位秒

# 保持原有的处理列表 / 忽略掉
PROCESS_INDEX = [0, 1, 2, 3] 

# ==========================================
# JSON 数据管理函数
# ==========================================
def load_results_from_json(json_path: Path) -> Dict:
    """加载已保存的压缩结果"""
    if not json_path.exists():
        return {
            "metadata": {
                "tolerance": TOLERANCE,
                "degree_arm": 3,
                "degree_gripper": 0,
                "dataset": "HuggingFaceVLA/libero",
                "version": "v2_forced_knots_prior"
            },
            "episodes": {}
        }
    
    with open(json_path, 'r') as f:
        return json.load(f)


def save_results_to_json(results: Dict, json_path: Path):
    """保存压缩结果到JSON文件"""
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ 结果已保存到: {json_path}")


def is_episode_computed(episode_idx: int, results: Dict) -> bool:
    """
    检查指定episode是否已正确计算
    """
    ep_str = str(episode_idx)
    episodes = results.get("episodes", {})
    
    if ep_str not in episodes:
        return False
    
    ep_data = episodes[ep_str]
    bspline = ep_data.get("bspline", {})
    
    # 检查求解状态
    if ep_data.get("status") == "failed":
        return False

    # 简单检查精度
    mean_error = bspline.get("overall_mean_error", 100.0)
    if mean_error > 0.5:
        print(f"  [重新计算] Episode {episode_idx}: mean_error={mean_error:.6f} > 0.5")
        return False
    
    return True


# ==========================================
# 1. 工具函数
# ==========================================
def extract_forced_knots(time_points, gripper_data) -> List[int]:
    """
    返回 gripper 跳变位置对应的 knot 时间 (Int)
    """
    diff = np.diff(gripper_data)
    # 找到发生变化的索引 (对应时间点)
    change_indices = np.where(diff != 0)[0] + 1
    
    forced_knot_times = []
    max_idx = len(time_points) - 1
    for idx in change_indices:
        if 0 < idx < max_idx:
            # 强制转换为 int
            forced_knot_times.append(int(time_points[idx]))
    return forced_knot_times


def convert_to_bspline_coeffs(time_points: np.ndarray, data_1d: np.ndarray, 
                               internal_knots: List[float], degree: int = 3) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[BSpline]]:
    """
    转换计算控制点
    """
    t0, t_end = time_points[0], time_points[-1]
    
    # 排序并过滤内部knots
    sorted_knots = sorted([k for k in internal_knots if t0 < k < t_end])
    
    # 构建 full knot vector
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
        print(f"  [Warning] B-spline转换失败 (Deg={degree}): {e}")
        return None, None, None


def reconstruct_bspline_trajectory(control_points: np.ndarray, knot_vector: np.ndarray, 
                                    time_points: np.ndarray, degree: int = 3) -> np.ndarray:
    bspline = BSpline(knot_vector, control_points, degree, extrapolate=False)
    return bspline(time_points)

# ==========================================
# 2. 求解器 (核心修改部分)
# ==========================================
def solve_6d_with_forced_knots(time_points, data_6d, forced_knot_times,
                               degree=3, tol_ratio=0.03, time_limit=600, 
                               use_gurobi=False):
    """
    使用 PuLP 前端建模，支持 CBC 和 Gurobi 求解器
    
    Args:
        use_gurobi: 如果为 True 且 Gurobi 可用，则使用 Gurobi；否则使用 CBC
    """
    T = len(time_points)
    num_dims = data_6d.shape[0]

    # --- 性能优化配置 ---
    # 1. 降采样步长：减少约束数量
    check_step = 1
    # if T > 200: check_step = 2
    # if T > 500: check_step = 4
    # if T > 1000: check_step = 5

    candidates = list(range(1, T - 1))
    forced_set = set(int(t) for t in forced_knot_times if 1 <= int(t) <= T - 2)

    # 1. 构造最大均匀 Knot
    t0, t_end = float(time_points[0]), float(time_points[-1])
    internal = np.arange(1, T - 1, dtype=float)
    U = np.concatenate([np.repeat(t0, degree + 1), internal, np.repeat(t_end, degree + 1)])
    n_basis = len(U) - degree - 1

    # 2. 预计算 B 矩阵 (只计算 check_step 覆盖的点)
    indices_to_check = list(range(0, T, check_step))
    if (T - 1) not in indices_to_check:
        indices_to_check.append(T - 1)

    from scipy.interpolate import BSpline
    # B_check 用于约束检查
    B_check = np.zeros((len(indices_to_check), n_basis), dtype=float)
    eye = np.eye(n_basis, dtype=float)
    for j in range(n_basis):
        bj = BSpline(U, eye[j], degree, extrapolate=False)
        vals = bj(time_points)
        B_check[:, j] = vals[indices_to_check]

    row_nz_check = [np.where(B_check[i] > 1e-12)[0].tolist() for i in range(len(indices_to_check))]

    # ==========================================
    # 阶段一：快速 L1 松弛预求解 (连续变量)
    # ==========================================
    print(f"  [Step 1] 运行 L1 松弛预求解 (连续松弛)...")
    
    prob_lp = pulp.LpProblem("BSpline_L1_Relaxed", pulp.LpMinimize)
    
    # 变量定义 - 控制点
    c_lp = {}
    for d in range(num_dims):
        for j in range(n_basis):
            c_lp[(d, j)] = pulp.LpVariable(f"c_{d}_{j}", lowBound=None, cat='Continuous')
    
    # 变量定义 - y (连续松弛 [0,1])
    y_lp = {}
    for k in candidates:
        y_lp[k] = pulp.LpVariable(f"y_{k}", lowBound=0.0, upBound=1.0, cat='Continuous')
    
    # 强制 Knot 约束
    for k in forced_set:
        prob_lp += y_lp[k] == 1, f"forced_knot_{k}"
    
    # Delta^4 约束 (Big-M)
    g_lp = {}
    M_relax = 200.0
    for d in range(num_dims):
        for k in candidates:
            i = k + 3
            g_lp[(d, k)] = pulp.LpVariable(f"g_{d}_{k}", lowBound=None, cat='Continuous')
            
            # Delta^4 定义
            delta4_expr = (c_lp[(d, i)] - 4.0 * c_lp[(d, i-1)] + 6.0 * c_lp[(d, i-2)] 
                          - 4.0 * c_lp[(d, i-3)] + c_lp[(d, i-4)])
            prob_lp += g_lp[(d, k)] == delta4_expr, f"delta4_def_{d}_{k}"
            
            # Big-M 约束: |g| <= M * y
            prob_lp += g_lp[(d, k)] <= M_relax * y_lp[k], f"bigM_upper_{d}_{k}"
            prob_lp += g_lp[(d, k)] >= -M_relax * y_lp[k], f"bigM_lower_{d}_{k}"
    
    # 拟合误差约束
    for d in range(num_dims):
        d_range = float(np.max(data_6d[d]) - np.min(data_6d[d]))
        if d_range < 1e-6: d_range = 1.0
        eps = d_range * float(tol_ratio)
        
        for idx_in_check, real_t_idx in enumerate(indices_to_check):
            P = float(data_6d[d][real_t_idx])
            idxs = row_nz_check[idx_in_check]
            
            fit_expr = pulp.lpSum([float(B_check[idx_in_check, j]) * c_lp[(d, j)] for j in idxs])
            prob_lp += fit_expr - P <= eps, f"fit_upper_{d}_{idx_in_check}"
            prob_lp += P - fit_expr <= eps, f"fit_lower_{d}_{idx_in_check}"
    
    # 目标函数：最小化 sum(y)
    prob_lp += pulp.lpSum([y_lp[k] for k in candidates]), "minimize_knots"
    
    # 选择求解器
    cpu_cores = multiprocessing.cpu_count()
    if use_gurobi:
        try:
            solver_lp = pulp.GUROBI(msg=False, threads=cpu_cores, timeLimit=time_limit)
        except:
            print("  [Warning] Gurobi 不可用，使用 CBC")
            solver_lp = pulp.PULP_CBC_CMD(msg=False, threads=cpu_cores, timeLimit=time_limit)
    else:
        solver_lp = pulp.PULP_CBC_CMD(msg=False, threads=cpu_cores, timeLimit=time_limit)
    
    prob_lp.solve(solver_lp)
    
    # 收集松弛解
    warm_start_values = {}
    if prob_lp.status == pulp.LpStatusOptimal:
        knot_guess_count = 0
        for k in candidates:
            val = pulp.value(y_lp[k])
            if val > 0.01:
                warm_start_values[k] = 1
                knot_guess_count += 1
            else:
                warm_start_values[k] = 0
        print(f"  -> L1 预估 Knot 数: {knot_guess_count}")
    else:
        print("  [Warning] L1 预求解失败，将直接运行 MILP")

    # ==========================================
    # 阶段二：MILP 求解 (二值变量)
    # ==========================================
    print(f"  [Step 2] 运行 MILP 精细求解...")
    
    prob_milp = pulp.LpProblem("BSpline_MILP", pulp.LpMinimize)
    
    # 变量定义 - 控制点
    c_milp = {}
    for d in range(num_dims):
        for j in range(n_basis):
            c_milp[(d, j)] = pulp.LpVariable(f"c_{d}_{j}", lowBound=None, cat='Continuous')
    
    # 变量定义 - y (二值变量)
    y_milp = {}
    for k in candidates:
        # 如果有 warm start，用作初始猜测（PuLP/CBC 不完全支持，但不影响）
        y_milp[k] = pulp.LpVariable(f"y_{k}", cat='Binary')
    
    # 强制 Knot 约束
    for k in forced_set:
        prob_milp += y_milp[k] == 1, f"forced_knot_{k}"
    
    # Delta^4 约束 (Big-M)
    g_milp = {}
    M_milp = 200.0
    for d in range(num_dims):
        for k in candidates:
            i = k + 3
            g_milp[(d, k)] = pulp.LpVariable(f"g_{d}_{k}", lowBound=None, cat='Continuous')
            
            # Delta^4 定义
            delta4_expr = (c_milp[(d, i)] - 4.0 * c_milp[(d, i-1)] + 6.0 * c_milp[(d, i-2)] 
                          - 4.0 * c_milp[(d, i-3)] + c_milp[(d, i-4)])
            prob_milp += g_milp[(d, k)] == delta4_expr, f"delta4_def_{d}_{k}"
            
            # Big-M 约束
            prob_milp += g_milp[(d, k)] <= M_milp * y_milp[k], f"bigM_upper_{d}_{k}"
            prob_milp += g_milp[(d, k)] >= -M_milp * y_milp[k], f"bigM_lower_{d}_{k}"
    
    # 拟合误差约束
    for d in range(num_dims):
        d_range = float(np.max(data_6d[d]) - np.min(data_6d[d]))
        if d_range < 1e-6: d_range = 1.0
        eps = d_range * float(tol_ratio)
        
        for idx_in_check, real_t_idx in enumerate(indices_to_check):
            P = float(data_6d[d][real_t_idx])
            idxs = row_nz_check[idx_in_check]
            
            fit_expr = pulp.lpSum([float(B_check[idx_in_check, j]) * c_milp[(d, j)] for j in idxs])
            prob_milp += fit_expr - P <= eps, f"fit_upper_{d}_{idx_in_check}"
            prob_milp += P - fit_expr <= eps, f"fit_lower_{d}_{idx_in_check}"
    
    # 目标函数
    prob_milp += pulp.lpSum([y_milp[k] for k in candidates]), "minimize_knots"
    
    # 选择求解器
    if use_gurobi:
        try:
            solver_milp = pulp.GUROBI(msg=True, threads=cpu_cores, timeLimit=time_limit)
        except:
            print("  [Warning] Gurobi 不可用，使用 CBC")
            solver_milp = pulp.PULP_CBC_CMD(msg=True, threads=cpu_cores, timeLimit=time_limit)
    else:
        solver_milp = pulp.PULP_CBC_CMD(msg=True, threads=cpu_cores, timeLimit=time_limit)
    
    prob_milp.solve(solver_milp)
    
    # ==========================================
    # 结果提取
    # ==========================================
    is_suboptimal = False
    status = prob_milp.status
    
    if status == pulp.LpStatusOptimal:
        status_str = "Optimal"
    elif status in (pulp.LpStatusNotSolved, pulp.LpStatusUndefined):
        print(f"  [Error] 求解失败。Status={pulp.LpStatus[status]}")
        return [], None, False
    else:
        status_str = pulp.LpStatus[status]
        is_suboptimal = True
    
    # 提取激活的 knots
    active_knots = [int(k) for k in candidates if pulp.value(y_milp[k]) > 0.5]
    
    # 重建曲线
    fitted_curves = np.zeros_like(data_6d, dtype=float)
    for d in range(num_dims):
        ctrl_pts = [pulp.value(c_milp[(d, j)]) for j in range(n_basis)]
        spl = BSpline(U, ctrl_pts, degree, extrapolate=False)
        fitted_curves[d] = spl(time_points)
    
    print(f"  -> 状态: {status_str}, Knot 数: {len(active_knots)}")
    return active_knots, fitted_curves, is_suboptimal
# ==========================================
# 3. 数据加载与处理
# ==========================================
def load_dataset():
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # 条件：只有 1 张 GPU 且名字里包含 4090
    use_default_root = (
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "4090" in torch.cuda.get_device_name(0)
    )

    kwargs = dict(delta_timestamps={"abs_aff": []})

    if not use_default_root:
        kwargs["root"] = Path("/inspire/hdd/project/robot-decision/public/datasets/") / "HuggingFaceVLA/libero"

    dataset = LeRobotDataset("HuggingFaceVLA/libero", **kwargs)
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

def load_or_build_episode_index_map(dataset) -> Dict[int, int]:
    if EP_MAP_PATH.exists():
        with open(EP_MAP_PATH, "r") as f:
            data = json.load(f)
        return {int(k): int(v) for k, v in data.items()}

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
        idx += len(item["abs_aff"])

    with open(EP_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
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
# 5. Episode 处理函数 (修改结果结构)
# ==========================================
def process_single_episode(dataset, dataset_index: int, results: Dict) -> Optional[Dict]:
    data_7d, episode_idx, frame_idx, task_name = load_episode_by_index(dataset, dataset_index)
    
    T_points = data_7d.shape[0]
    x_axis = np.arange(T_points)
    data_7d = data_7d.T
    
    gripper_traj = data_7d[6]
    data_6d = data_7d[:6]
    
    # 1. 提取 Gripper 强制 Knots (int list)
    forced_knots = extract_forced_knots(x_axis, gripper_traj)
    print(f"  强制 Knot (来自 Gripper): {forced_knots}")
    
    # 2. 求解 6D 轨迹 (传入强制列表)
    # use_gurobi=True 使用 Gurobi (如果可用)，False 使用 CBC
    knots, curves_6d, is_suboptimal = solve_6d_with_forced_knots(
        time_points=x_axis,
        data_6d=data_6d,
        forced_knot_times=forced_knots,
        degree=3,
        tol_ratio=TOLERANCE,
        time_limit=TIME_LIMIT_SECONDS,
        use_gurobi=False  # 默认使用 CBC，改为 True 可使用 Gurobi
    )
    
    if curves_6d is None:
        return None
    
    # 3. 统一转换为 B-spline
    #    注意：这里所有维度都使用相同的 internal_knots (即 MILP 输出的 knots)
    #    Knots 已经包含了 forced_knots
    print(f"  [Info] 转换为B-spline表示 (所有维度共享 Internal Knots)...")
    
    bspline_control_points = []
    bspline_curves_7d = np.zeros_like(data_7d)
    
    # 处理前 6 维 (Degree 3)
    for d in range(6):
        ctrl_pts, _, _ = convert_to_bspline_coeffs(
            time_points=x_axis,
            data_1d=data_6d[d],
            internal_knots=knots,
            degree=3
        )
        if ctrl_pts is None: return None
        bspline_control_points.append(ctrl_pts.tolist()) # list of floats
        bspline_curves_7d[d] = reconstruct_bspline_trajectory(
            ctrl_pts, 
            np.concatenate([np.repeat(x_axis[0], 4), sorted(knots), np.repeat(x_axis[-1], 4)]), 
            x_axis, 3
        )

    # 处理 Gripper (Dim 6, Degree 0)
    # [Requirement] Gripper 也要使用这套 Knots，保存为 int
    # Degree 0 的 knot vector 构造方式略有不同 (k=0)，但内部断点依然可以是 knots
    gripper_ctrl_pts, _, _ = convert_to_bspline_coeffs(
        time_points=x_axis,
        data_1d=gripper_traj,
        internal_knots=knots, # 使用完全相同的 Knots
        degree=0
    )
    if gripper_ctrl_pts is None: return None
    
    # 转换 Gripper control points 为 int (0 或 1，或者具体数值)
    # 通常 gripper 是 float 0.0 ~ 1.0，但这里要求保存为 int? 
    # 如果 data 本身是 0/1，则 int。如果是连续值，保持 float 比较好。
    # 用户要求: "knot还有gripper的数值用int来保存即可" -> 假设 gripper data 是离散状态
    gripper_cp_list = [int(round(v)) for v in gripper_ctrl_pts] 
    bspline_control_points.append(gripper_cp_list)

    bspline_curves_7d[6] = reconstruct_bspline_trajectory(
        gripper_ctrl_pts, # 重建时用 float 没关系
        np.concatenate([np.repeat(x_axis[0], 1), sorted(knots), np.repeat(x_axis[-1], 1)]), 
        x_axis, 0
    )
    
    # 计算误差
    bspline_overall_mean = float(np.mean(np.abs(bspline_curves_7d - data_7d)))
    
    # 构造完整的 knot vector (包含前后重复端点)
    # 对于 degree=3: [t0, t0, t0, t0, internal_knots..., t_end, t_end, t_end, t_end]
    t0, t_end = int(x_axis[0]), int(x_axis[-1])
    full_knots_vector = (
        [t0] * 4 +  # degree + 1 = 4 个起始点
        sorted([int(k) for k in knots]) +  # 内部 knots
        [t_end] * 4  # degree + 1 = 4 个结束点
    )
    
    # 可视化
    vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
    visualize_episode(data_7d, bspline_curves_7d, knots, 
                      episode_idx, task_name, x_axis, vis_path)
    
    # 4. 构造结果 (V2 格式)
    episode_result = {
        "episode_index": episode_idx,
        "task_name": task_name,
        "trajectory_length": T_points,
        "status": "suboptimal" if is_suboptimal else "optimal",
        "bspline": {
            # Shared Metadata
            "degree_arm": 3,
            "degree_gripper": 0,
            
            # 完整的 knot vector (可直接用于 B-spline 解码)
            "knots_vector": full_knots_vector, 
            
            # Control Points (7 Dims)
            "control_points": bspline_control_points,
            
            "overall_mean_error": bspline_overall_mean
        },
        "visualization_path": str(vis_path)
    }
    
    print(f"  ✓ Episode {episode_idx} 完成 | Knots: {len(knots)} | Err: {bspline_overall_mean:.6f}")
    return episode_result

# ==========================================
# 6. 主程序
# ==========================================
def main():
    print("="*80)
    print("MILP B-Spline Compression V2 (Forced Knots Priority)")
    print("="*80)
    
    dataset = load_dataset()
    results = load_results_from_json(RESULTS_JSON_PATH)
    ep_index_map = load_or_build_episode_index_map(dataset)
    
    processed_count = 0
    
    for episode_idx in sorted(ep_index_map.keys()):
        if PROCESS_INDEX is not None and episode_idx not in PROCESS_INDEX:
            continue

        dataset_idx = ep_index_map[episode_idx]
        
        if is_episode_computed(episode_idx, results):
            print(f"Skip {episode_idx}")
            continue

        try:
            res = process_single_episode(dataset, dataset_idx, results)
            if res:
                results["episodes"][str(episode_idx)] = res
                save_results_to_json(results, RESULTS_JSON_PATH)
                processed_count += 1
        except Exception as e:
            print(f"Error {episode_idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Processed: {processed_count}")

if __name__ == "__main__":
    main()