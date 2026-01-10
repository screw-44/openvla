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
RESULTS_JSON_PATH = Path("compression_results.json")
VISUAL_DIR = Path("test/visual_fig")
EP_MAP_PATH = Path("epsidoe_2_dataset_index.json")
PROCESS_INDEX = [
    67, 91, 128, 179, 223, 235, 247, 267, 316, 360, 378, 386, 
    387, 388, 394, 402, 416, 431, 447, 457, 459, 464, 468, 491, 
    492, 495, 508, 514, 522, 532, 567, 587, 613, 620, 626, 629, 635, 
    637, 641, 649, 653, 655, 658, 677, 679, 682, 689, 698, 706, 711, 
    712, 718, 720, 727, 743, 745, 747, 760, 773, 774, 781, 788, 789, 
    796, 811, 812, 815, 822, 823, 826, 829, 842, 843, 847, 852, 853, 
    858, 865, 867, 871, 880, 883, 891, 893, 898, 899, 902, 903, 906, 
    913, 922, 924, 933, 940, 952, 955, 959, 962, 967, 979, 984, 987, 
    988, 995, 1001, 1006, 1007, 1015, 1021, 1028, 1031, 1047, 1050, 
    1051, 1063, 1065, 1066, 1068, 1094, 1096, 1097, 1098, 1100, 1107, 
    1108, 1111, 1121, 1122, 1124, 1131, 1137, 1139, 1153, 1157, 1171, 
    1172, 1175, 1186, 1191, 1192, 1193, 1196, 1200, 1204, 1205, 1210, 
    1217, 1224, 1249, 1251, 1259, 1264, 1272, 1274, 1275, 1278, 1288, 
    1292, 1299, 1306, 1312, 1313, 1320, 1343, 1347, 1354, 1370, 1386, 
    1388, 1390, 1395, 1396, 1397, 1400, 1401, 1402, 1404, 1414, 1425, 
    1441, 1446, 1449, 1450, 1451, 1460, 1462, 1464, 1468, 1471, 1481, 
    1524, 1535, 1550, 1553, 1560, 1562, 1565, 1570, 1574, 1577, 1602, 
    1610, 1615, 1616, 1617, 1622, 1630, 1646, 1653, 1659, 1660, 1662, 
    1667, 1669, 1671, 1672, 1677, 1680, 1682, 1685] # 仅仅需要处理的index

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
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ 结果已保存到: {json_path}")


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
# 1. 工具函数: 提取 Gripper 的强制 Knots
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
    """
    将MILP算出的Knots转换为标准B-spline的控制点
    
    Args:
        time_points: 时间轴
        data_1d: 单个维度的数据
        internal_knots: MILP算出的内部knots（不包含首尾）
        degree: B-spline阶数
        
    Returns:
        control_points: B-spline控制点数组
        full_knot_vector: 完整的knot向量（包含首尾填充）
        bspline: BSpline对象
    """
    t0, t_end = time_points[0], time_points[-1]
    
    # 排序并过滤内部knots
    sorted_knots = sorted([k for k in internal_knots if t0 < k < t_end])
    
    # 构建clamped knot vector: [t0]*(degree+1) + internal + [t_end]*(degree+1)
    full_knot_vector = np.concatenate([
        np.repeat(t0, degree + 1),
        sorted_knots,
        np.repeat(t_end, degree + 1)
    ])
    
    try:
        # 使用最小二乘法求解控制点
        bspline = make_lsq_spline(time_points, data_1d, full_knot_vector, k=degree)
        control_points = bspline.c
        return control_points, full_knot_vector, bspline
    except Exception as e:
        print(f"  [Warning] B-spline转换失败: {e}")
        return None, None, None


def reconstruct_bspline_trajectory(control_points: np.ndarray, knot_vector: np.ndarray, 
                                    time_points: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    从B-spline控制点重建完整轨迹
    
    Args:
        control_points: B-spline控制点
        knot_vector: 完整的knot向量
        time_points: 采样时间点
        degree: B-spline阶数
        
    Returns:
        trajectory: 重建的轨迹
    """
    bspline = BSpline(knot_vector, control_points, degree, extrapolate=False)
    return bspline(time_points)

# ==========================================
# 2. 求解器: 6维拟合 + 强制 Knot 约束
# ==========================================
def solve_6d_with_forced_knots(time_points, data_6d, forced_knot_times, degree=3, tol_ratio=0.03, time_limit=600):
    num_dims = data_6d.shape[0]
    candidate_knots = time_points[1:-1:1] # 考虑到所有的可能的位置，如果要加速可以调整候选knot之间的距离。

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
    print(f"  [Info] 已添加 {forced_count} 个强制 Knot 约束 (来自 Gripper)")

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
        BigMs.append(d_range * 1.0) # NOTE： M=1（实际上可以更小，但是TPB不熟悉，不调整了
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

        for i, t_val in enumerate(time_points):
            P_val = data_6d[d][i]
            poly_part = pulp.lpSum([alpha[d][k] * (t_val**k) for k in range(degree + 1)])
            knot_terms = [beta[d][j] * ((t_val - j) ** degree) for j in candidate_knots if t_val > j]
            S_t = poly_part + pulp.lpSum(knot_terms)
            prob += S_t - P_val <= eps_d
            prob += P_val - S_t <= eps_d

    cpu_cores = multiprocessing.cpu_count() 
    print(f"正在求解 6D 轨迹... Tol: {tol_ratio*100}% | Solver: CBC ({cpu_cores} core) | TimeLimit: {time_limit}s")
    
    # 使用 CBC，并设置时间限制与线程数, 设置knot最小间距是0.99 (已经关闭time limit)
    solver = pulp.PULP_CBC_CMD(msg=True, threads=cpu_cores)
    # solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, threads=cpu_cores)
    prob.solve(solver)

    # 状态与目标值检查：接受可行的次优解
    status_str = pulp.LpStatus[prob.status]
    objective_val = pulp.value(prob.objective)

    is_suboptimal = False
    if status_str == "Optimal":
        pass
    elif objective_val is not None:
        # 求解器在时限内找到了可行解，但未证明全局最优
        print(f"  [Info] 时间耗尽或早停。返回当前次优解 (Knots: {int(objective_val)}) | Status: {status_str}")
        is_suboptimal = True
    else:
        print(f"  [Error] 求解失败或未找到可行解。Status: {status_str}")
        return [], None, False

    knot_count = pulp.value(prob.objective)
    print(f"  -> 状态: {status_str}, 总 Knot 数: {int(knot_count)} (包含强制Knot)")

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
# 3. 从 LeRobotDataset 加载数据
# ==========================================
def load_dataset():
    """加载完整数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(
        'HuggingFaceVLA/libero',
        root=Path("/inspire/hdd/project/robot-decision/public/datasets/")/'HuggingFaceVLA/libero',
        delta_timestamps={"abs_aff":[]}
    )
    return dataset


def load_episode_by_index(dataset, dataset_index: int) -> Tuple[np.ndarray, int, int, str]:
    """
    从数据集指定索引位置加载一个完整episode
    
    Returns:
        abs_aff_absolute: (T, 7) 绝对坐标轨迹
        episode_idx: episode索引
        frame_idx: 帧索引（应该为0）
        task_name: 任务名称
    """
    item = dataset[dataset_index]
    abs_aff = np.asarray(item['abs_aff'])
    
    # 转换为绝对坐标
    abs_aff_absolute = abs_aff.copy()
    abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
    
    episode_idx = int(item['episode_index'])
    frame_idx = int(item['frame_index'])
    task_name = item['task'] if 'task' in item else "Unknown"
    
    return abs_aff_absolute, episode_idx, frame_idx, task_name


def load_or_build_episode_index_map(dataset) -> Dict[int, int]:
    """加载或构建 episode_index -> dataset_index 的映射"""
    if EP_MAP_PATH.exists():
        print(f"发现映射文件: {EP_MAP_PATH}, 正在加载...")
        with open(EP_MAP_PATH, "r") as f:
            data = json.load(f)
        # keys/values 转为 int
        return {int(k): int(v) for k, v in data.items()}

    print(f"未发现映射文件，将遍历数据集构建 {EP_MAP_PATH} ...")
    mapping = {}
    idx = 0
    total = len(dataset)
    while idx < total:
        item = dataset[idx]
        ep_idx = int(item["episode_index"])
        frame_idx = int(item["frame_index"])
        if frame_idx != 0:
            # 如果不是episode起始，向下一个样本推进，尽量自愈
            idx += 1
            continue

        mapping[ep_idx] = idx
        ep_len = len(item["abs_aff"])
        idx += ep_len  # 直接跳到下一个episode起始

    with open(EP_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"映射已保存到 {EP_MAP_PATH} (共 {len(mapping)} 条)")
    return mapping


# ==========================================
# 4. 可视化函数 (2行8列，对比TPB和B-spline)
# ==========================================
def visualize_episode(data_7d: np.ndarray, tpb_curves_7d: np.ndarray, 
                      bspline_curves_7d: np.ndarray, knots: List, 
                      episode_idx: int, task_name: str, x_axis: np.ndarray, 
                      save_path: Path):
    """
    可视化单个episode的拟合结果，对比TPB和B-spline两种方法
    布局: 2行8列
      - 左侧4列: TPB方法 (X, Y, Z, Gripper, Yaw, Pitch, Roll, 统计)
      - 右侧4列: B-spline方法 (X, Y, Z, Gripper, Yaw, Pitch, Roll, 统计)
    """
    fig, axes = plt.subplots(2, 8, figsize=(32, 10))
    axes = axes.flatten()
    
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    dim_order = [0, 1, 2, 6, 3, 4, 5]  # x,y,z,gripper, yaw,pitch,roll
    
    # 左侧：TPB方法
    for plot_idx, dim_idx in enumerate(dim_order):
        ax = axes[plot_idx]
        
        # 原始数据
        ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, 
                   label='Ground Truth', zorder=1)
        
        # TPB拟合曲线
        line_color = 'red' if dim_idx < 6 else 'green'
        ax.plot(x_axis, tpb_curves_7d[dim_idx], color=line_color, linewidth=2,  
                label='TPB Fit', zorder=2, alpha=0.8)
        
        # Knots
        if knots:
            for k in knots:
                ax.axvline(x=k, color='blue', linestyle='--', alpha=0.3, linewidth=1, zorder=0)
        
        # 误差计算
        error = np.abs(tpb_curves_7d[dim_idx] - data_7d[dim_idx])
        mean_err = np.mean(error)
        max_err = np.max(error)
        
        # 标题和标签
        ax.set_title(f"[TPB] {dim_names[dim_idx]}\nErr: {mean_err:.5f} | Max: {max_err:.5f}", 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Time Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)
        
        # Gripper特殊处理
        if dim_idx == 6:
            y_min, y_max = data_7d[dim_idx].min(), data_7d[dim_idx].max()
            y_range = max(y_max - y_min, 0.5)
            ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
    
    # 右侧：B-spline方法
    for plot_idx, dim_idx in enumerate(dim_order):
        ax = axes[plot_idx + 8]  # 右侧偏移8个位置
        
        # 原始数据
        ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4, 
                   label='Ground Truth', zorder=1)
        
        # B-spline拟合曲线
        line_color = 'darkblue' if dim_idx < 6 else 'darkgreen'
        ax.plot(x_axis, bspline_curves_7d[dim_idx], color=line_color, linewidth=2, 
                label='B-spline Fit', zorder=2, alpha=0.8)
        
        # Knots
        if knots:
            for k in knots:
                ax.axvline(x=k, color='orange', linestyle='--', alpha=0.3, linewidth=1, zorder=0)
        
        # 误差计算
        error = np.abs(bspline_curves_7d[dim_idx] - data_7d[dim_idx])
        mean_err = np.mean(error)
        max_err = np.max(error)
        
        # 标题和标签
        ax.set_title(f"[B-spline] {dim_names[dim_idx]}\nErr: {mean_err:.5f} | Max: {max_err:.5f}", 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Time Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)
        
        # Gripper特殊处理
        if dim_idx == 6:
            y_min, y_max = data_7d[dim_idx].min(), data_7d[dim_idx].max()
            y_range = max(y_max - y_min, 0.5)
            ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
    
    # 左下角统计信息 (TPB)
    ax_stats_left = axes[7]
    ax_stats_left.axis('off')
    tpb_errors = np.abs(tpb_curves_7d - data_7d)
    stats_text_left = f"""TPB Method
    
Knots: {len(knots)}
    
Errors:
  Mean: {np.mean(tpb_errors):.6f}
  Max: {np.max(tpb_errors):.6f}
  Std: {np.std(tpb_errors):.6f}
"""
    ax_stats_left.text(0.1, 0.5, stats_text_left, fontsize=9, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='lightcoral', alpha=0.3))
    
    # 右下角统计信息 (B-spline)
    ax_stats_right = axes[15]
    ax_stats_right.axis('off')
    bspline_errors = np.abs(bspline_curves_7d - data_7d)
    stats_text_right = f"""B-spline Method
    
Control Points: ~{len(knots)+4}
    
Errors:
  Mean: {np.mean(bspline_errors):.6f}
  Max: {np.max(bspline_errors):.6f}
  Std: {np.std(bspline_errors):.6f}

✓ Local Control
"""
    ax_stats_right.text(0.1, 0.5, stats_text_right, fontsize=9, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', 
                        facecolor='lightblue', alpha=0.3))
    
    # 总标题
    fig.suptitle(f'Compression Comparison - Episode {episode_idx}\nTask: {task_name[:80]}...', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 可视化已保存: {save_path}")


# ==========================================
# 5. Episode 处理函数
# ==========================================
def process_single_episode(dataset, dataset_index: int, results: Dict) -> Optional[Dict]:
    """
    处理单个episode: 加载->压缩->可视化->保存
    
    Returns:
        episode_result: 包含knots, errors等信息的字典，失败返回None
    """
    # 加载数据
    data_7d, episode_idx, frame_idx, task_name = load_episode_by_index(dataset, dataset_index)
    
    # 断言验证
    assert frame_idx == 0, f"Episode {episode_idx}: 期望 frame_index=0, 实际 {frame_idx}"
    
    print(f"\n{'='*70}")
    print(f"处理 Episode {episode_idx} | Task: {task_name} | Length: {len(data_7d)}")
    print(f"{'='*70}")
    
    # 创建时间轴, 采用绝对的时间index
    T_points = data_7d.shape[0]
    x_axis = np.arange(T_points)
    
    # 转置为 (7, T)
    data_7d = data_7d.T
    
    # 分离gripper和6D数据
    gripper_traj = data_7d[6]
    data_6d = data_7d[:6]
    
    # 提取强制knots
    forced_knots = extract_forced_knots(x_axis, gripper_traj)
    print(f"  强制 Knot (来自 Gripper): {len(forced_knots)} 个")
    
    # 求解6D轨迹
    knots, curves_6d, is_suboptimal = solve_6d_with_forced_knots(
        time_points=x_axis,
        data_6d=data_6d,
        forced_knot_times=forced_knots,
        degree=3,
        tol_ratio=TOLERANCE,
        time_limit=600  # 每个episode最多20分钟,
    )
    
    if curves_6d is None:
        print(f"  ✗ Episode {episode_idx} 求解失败")
        return None
    
    # 第二步：将MILP找到的knots转换为B-spline控制点
    print(f"  [Info] 转换为B-spline表示...")
    bspline_control_points = []
    bspline_knot_vectors = []
    bspline_curves_6d = np.zeros_like(data_6d)
    
    for d in range(6):
        ctrl_pts, knot_vec, bspline_obj = convert_to_bspline_coeffs(
            time_points=x_axis,
            data_1d=data_6d[d],
            internal_knots=knots,
            degree=3
        )
        
        if ctrl_pts is None:
            print(f"  ✗ Dimension {d} B-spline转换失败")
            return None
        
        bspline_control_points.append(ctrl_pts.tolist())
        bspline_knot_vectors.append(knot_vec.tolist())
        bspline_curves_6d[d] = reconstruct_bspline_trajectory(ctrl_pts, knot_vec, x_axis, degree=3)
    
    # Gripper使用0阶B-spline（分段常数）
    gripper_ctrl_pts, gripper_knot_vec, _ = convert_to_bspline_coeffs(
        time_points=x_axis,
        data_1d=gripper_traj,
        internal_knots=forced_knots,
        degree=0
    )
    
    if gripper_ctrl_pts is not None:
        bspline_control_points.append(gripper_ctrl_pts.tolist())
        bspline_knot_vectors.append(gripper_knot_vec.tolist())
        gripper_bspline = reconstruct_bspline_trajectory(gripper_ctrl_pts, gripper_knot_vec, x_axis, degree=0)
    else:
        print(f"  [Warning] Gripper B-spline转换失败，使用原始数据")
        bspline_control_points.append(gripper_traj.tolist())
        bspline_knot_vectors.append([])
        gripper_bspline = gripper_traj.copy()
    
    # 组合完整的7D B-spline轨迹
    bspline_curves_7d = np.vstack([bspline_curves_6d, gripper_bspline])
    
    # 计算B-spline误差
    bspline_errors_per_dim = []
    for d in range(7):
        error = np.abs(bspline_curves_7d[d] - data_7d[d])
        bspline_errors_per_dim.append({
            "mean": float(np.mean(error)),
            "max": float(np.max(error)),
            "std": float(np.std(error))
        })
    
    bspline_overall_mean = float(np.mean(np.abs(bspline_curves_7d - data_7d)))
    
    print(f"  ✓ B-spline转换完成 | 误差: {bspline_overall_mean:.6f}")
    
    # TPB轨迹（用于可视化对比）
    gripper_recon = gripper_traj.copy()
    tpb_curves_7d = np.vstack([curves_6d, gripper_recon])
    
    # 可视化对比
    vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
    visualize_episode(data_7d, tpb_curves_7d, bspline_curves_7d, knots, 
                      episode_idx, task_name, x_axis, vis_path)
    
    # 构造结果字典（只保存B-spline数据）
    episode_result = {
        "episode_index": episode_idx,
        "task_name": task_name,
        "trajectory_length": T_points,
        "status": "suboptimal" if is_suboptimal else "optimal",
        "bspline": {
            "control_points": bspline_control_points,  # 7个维度的控制点
            "knot_vectors": bspline_knot_vectors,      # 7个维度的knot向量
            "internal_knots": [float(k) for k in knots],  # 前6维共享的内部knots
            "forced_knots": [float(k) for k in forced_knots],  # gripper的强制knots
            "num_knots": len(knots),
            "errors_per_dim": bspline_errors_per_dim,
            "overall_mean_error": bspline_overall_mean
        },
        "visualization_path": str(vis_path)
    }
    
    print(f"  ✓ Episode {episode_idx} 处理完成 | B-spline误差: {bspline_overall_mean:.6f}")
    
    return episode_result


# ==========================================
# 6. 主程序 - 遍历全数据集
# ==========================================
def main():
    """主函数：遍历数据集所有episode并增量计算"""
    print("="*80)
    print("MILP B-Spline 压缩 - 全数据集处理")
    print("="*80)
    print(f"配置: Tolerance={TOLERANCE}, Solver=CBC")
    print(f"结果保存: {RESULTS_JSON_PATH}")
    print(f"可视化目录: {VISUAL_DIR}")
    print("="*80)
    
    # 加载数据集
    print("\n正在加载数据集...")
    dataset = load_dataset()
    total_samples = len(dataset)
    print(f"数据集加载完成，共 {total_samples} 个样本")
    
    # 加载已有结果
    results = load_results_from_json(RESULTS_JSON_PATH)
    initial_completed = len(results["episodes"])
    print(f"已完成 {initial_completed} 个 episode")
    
    # 加载或构建 episode -> dataset_index 映射
    ep_index_map = load_or_build_episode_index_map(dataset)
    all_episode_ids = sorted(ep_index_map.keys())

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for episode_idx in all_episode_ids:
        # 处理哪些index
        if episode_idx not in PROCESS_INDEX:
            continue

        dataset_idx = ep_index_map[episode_idx]

        # 检查是否已计算
        if is_episode_computed(episode_idx, results):
            print(f"\n[跳过] Episode {episode_idx} 已计算")
            skipped_count += 1
            continue

        try:
            episode_result = process_single_episode(dataset, dataset_idx, results)

            if episode_result is not None:
                results["episodes"][str(episode_idx)] = episode_result
                save_results_to_json(results, RESULTS_JSON_PATH)
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"\n✗ 处理 episode {episode_idx} (dataset_idx={dataset_idx}) 时出错: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # 总结
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"总样本数: {total_samples}")
    print(f"映射中的 episode 数: {len(all_episode_ids)}")
    print(f"本次处理: {processed_count} 个")
    print(f"跳过(已完成): {skipped_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"累计完成: {len(results['episodes'])} 个 episode")
    print(f"结果文件: {RESULTS_JSON_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()