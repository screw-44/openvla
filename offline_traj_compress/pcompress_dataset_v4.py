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
RESULTS_JSON_PATH = Path("compression_results_v2.json")  # Modified V2
VISUAL_DIR = Path("visual_fig_v2") # Modified dir to separate from v1
EP_MAP_PATH = Path("epsidoe_2_dataset_index.json")

# 保持原有的处理列表
PROCESS_INDEX = [1,] 

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
                               degree=3, tol_ratio=0.03, time_limit=600):
    """
    B-spline(固定最大基) + Δ^4 稀疏(用y选择) + Gurobi Indicator 的MILP。
    返回:
      active_knots: 选择的时间knot (int list, 1..T-2)
      fitted_curves: (6, T) 由该MILP在最大基下拟合出的曲线(用于你现有可视化)
      is_suboptimal: 是否为次优/时间到解
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("需要 gurobipy 才能使用该求解器版本。请确认已安装并有license。") from e

    T = len(time_points)
    num_dims = data_6d.shape[0]
    assert degree == 3, "当前实现默认 cubic (degree=3)，若要其他degree需要同步改Δ阶数与基构造。"

    # 只允许内部knot: 1..T-2（与你原candidate一致）
    candidates = list(range(1, T - 1))  # 1..T-2
    forced_set = set(int(t) for t in forced_knot_times if 1 <= int(t) <= T - 2)

    # ================
    # 1) 构造最大 open-uniform knot vector U_max
    # U = [0,0,0,0, 1,2,...,T-2, T-1,T-1,T-1,T-1]
    # n_basis = len(U) - p - 1 = (T+6) - 3 - 1 = T+2
    # ================
    t0 = float(time_points[0])
    t_end = float(time_points[-1])
    internal = np.arange(1, T - 1, dtype=float)  # 1..T-2
    U = np.concatenate([np.repeat(t0, degree + 1), internal, np.repeat(t_end, degree + 1)])
    n_basis = len(U) - degree - 1  # T+2

    # ================
    # 2) 预计算 B-spline 设计矩阵 B (T x n_basis)
    # 每行最多 4 个非零（cubic），后面用稀疏索引构线性表达式
    # ================
    from scipy.interpolate import BSpline
    B = np.zeros((T, n_basis), dtype=float)

    # 构造每个基函数的单位控制点并评估
    # （对T~几百/几千仍可接受；而且只做一次/episode）
    eye = np.eye(n_basis, dtype=float)
    for j in range(n_basis):
        bj = BSpline(U, eye[j], degree, extrapolate=False)
        B[:, j] = bj(time_points)

    # 行稀疏索引（阈值过滤掉数值0）
    row_nz = [np.where(B[i] > 1e-12)[0].tolist() for i in range(T)]

    # ================
    # 3) 建模
    #   变量: c[d,j] 控制点(连续), y[k] 是否允许在k处有“拐点/跳变”(二进制), g[d,k]=Δ^4(c) (连续)
    #   约束:
    #     - y[k]=1 for forced
    #     - g[d,k] = Δ^4 c 在控制点索引 i=k+3 处 (与k=1..T-2对齐)
    #     - Indicator: y[k]==0 -> g[d,k]==0
    #     - 拟合: | sum_j B[i,j]*c[d,j] - P | <= eps_d
    #   目标: min sum_k y[k]
    # ================
    m = gp.Model("BSpline_Delta4_Sparse")
    m.Params.OutputFlag = 1
    m.Params.Threads = multiprocessing.cpu_count()
    m.Params.TimeLimit = time_limit

    # 控制点变量
    c = m.addVars(num_dims, n_basis, lb=-GRB.INFINITY, name="c")

    # y变量（只对内部knot时刻k）
    y = m.addVars(candidates, vtype=GRB.BINARY, name="y")

    # 强制knot
    if forced_set:
        for k in forced_set:
            m.addConstr(y[k] == 1, name=f"force_y_{k}")
    print(f"  [Info] 已将 {len(forced_set)} 个 Gripper 变化点锁定为强制 Knots (Indicator版本)")

    # Δ^4 变量与约束：g[d,k] = c[d,i] -4c[d,i-1] +6c[d,i-2] -4c[d,i-3] + c[d,i-4]
    # 这里 i = k+3，使得 k=1 -> i=4（用到c0..c4），k=T-2 -> i=T+1（合法，n_basis=T+2）
    g = m.addVars(num_dims, candidates, lb=-GRB.INFINITY, name="g")
    for d in range(num_dims):
        for k in candidates:
            i = k + 3
            m.addConstr(
                g[d, k] ==
                c[d, i] - 4.0 * c[d, i - 1] + 6.0 * c[d, i - 2] - 4.0 * c[d, i - 3] + c[d, i - 4],
                name=f"delta4_{d}_{k}"
            )
            # Indicator: y[k]==0 -> g[d,k]==0
            m.addGenConstrIndicator(y[k], 0, g[d, k], GRB.EQUAL, 0.0, name=f"ind_{d}_{k}")

    # 拟合误差约束
    epsilons = []
    for d in range(num_dims):
        d_range = float(np.max(data_6d[d]) - np.min(data_6d[d]))
        if d_range < 1e-6:
            d_range = 1.0
        eps = d_range * float(tol_ratio)
        epsilons.append(eps)

        for i in range(T):
            P = float(data_6d[d][i])
            idxs = row_nz[i]  # 至多4个
            expr = gp.quicksum(float(B[i, j]) * c[d, j] for j in idxs)
            m.addConstr(expr - P <= eps, name=f"fit_pos_{d}_{i}")
            m.addConstr(P - expr <= eps, name=f"fit_neg_{d}_{i}")

    # 目标：最小化选择的knot数
    m.setObjective(gp.quicksum(y[k] for k in candidates), GRB.MINIMIZE)

    print(f"正在求解 6D 轨迹... Tol: {tol_ratio*100}% | Solver: Gurobi (Indicator) | T={T} | n_basis={n_basis}")
    m.optimize()

    # ================
    # 4) 读取结果
    # ================
    is_suboptimal = False
    if m.Status == GRB.OPTIMAL:
        status_str = "Optimal"
    elif m.Status in (GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        status_str = "Suboptimal/TimeLimit"
        is_suboptimal = True
    elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        print(f"  [Error] 不可行/无界。Status={m.Status}")
        return [], None, False
    else:
        print(f"  [Error] 求解失败。Status={m.Status}")
        return [], None, False

    obj_val = m.ObjVal if m.SolCount > 0 else None
    if obj_val is None:
        print(f"  [Error] 没有可用解。Status={m.Status}")
        return [], None, False

    print(f"  -> 状态: {status_str}, 总 Knot 数: {int(round(obj_val))}")

    active_knots = [int(k) for k in candidates if y[k].X > 0.5]

    # 用最大基下的控制点重建 fitted_curves（用于你现有可视化/对比）
    fitted_curves = np.zeros_like(data_6d, dtype=float)
    # 取出每个维度的控制点向量
    for d in range(num_dims):
        cvec = np.array([c[d, j].X for j in range(n_basis)], dtype=float)
        fitted_curves[d] = B @ cvec

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
# 4. 可视化 (保持不变，仅修改保存路径)
# ==========================================
def visualize_episode(data_7d: np.ndarray, tpb_curves_7d: np.ndarray, 
                      bspline_curves_7d: np.ndarray, knots: List, 
                      episode_idx: int, task_name: str, x_axis: np.ndarray, 
                      save_path: Path):
    # 代码内容与原版一致，为了节省篇幅，这里略去具体绘图代码
    # 逻辑完全复用上一版即可
    fig, axes = plt.subplots(2, 8, figsize=(32, 10))
    axes = axes.flatten()
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    dim_order = [0, 1, 2, 6, 3, 4, 5]

    for plot_idx, dim_idx in enumerate(dim_order):
        ax = axes[plot_idx] # TPB
        ax.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4)
        ax.plot(x_axis, tpb_curves_7d[dim_idx], color='red' if dim_idx < 6 else 'green')
        if knots:
            for k in knots:
                ax.axvline(x=k, color='blue', linestyle='--', alpha=0.3)
        ax.set_title(f"[TPB] {dim_names[dim_idx]}")

        ax2 = axes[plot_idx + 8] # B-spline
        ax2.scatter(x_axis, data_7d[dim_idx], s=12, color='gray', alpha=0.4)
        ax2.plot(x_axis, bspline_curves_7d[dim_idx], color='darkblue' if dim_idx < 6 else 'darkgreen')
        if knots:
            for k in knots:
                ax2.axvline(x=k, color='orange', linestyle='--', alpha=0.3)
        ax2.set_title(f"[B-spline] {dim_names[dim_idx]}")

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
    knots, curves_6d, is_suboptimal = solve_6d_with_forced_knots(
        time_points=x_axis,
        data_6d=data_6d,
        forced_knot_times=forced_knots,
        degree=3,
        tol_ratio=TOLERANCE
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
    
    # 可视化
    vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
    tpb_curves_7d = np.vstack([curves_6d, bspline_curves_7d[6]]) # TPB 暂用 B-spline 的 gripper
    visualize_episode(data_7d, tpb_curves_7d, bspline_curves_7d, knots, 
                      episode_idx, task_name, x_axis, vis_path)
    
    # 4. 构造结果 (V2 格式)
    # [Requirement] knot_vector 只保存一份 (internal_knots)
    episode_result = {
        "episode_index": episode_idx,
        "task_name": task_name,
        "trajectory_length": T_points,
        "status": "suboptimal" if is_suboptimal else "optimal",
        "bspline": {
            # Shared Metadata
            "degree_arm": 3,
            "degree_gripper": 0,
            "internal_knots": [int(k) for k in knots], # [Requirement] Int List, Shared
            
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
        if episode_idx not in PROCESS_INDEX:
            continue

        dataset_idx = ep_index_map[episode_idx]
        
        # if is_episode_computed(episode_idx, results):
        #     print(f"Skip {episode_idx}")
        #     continue

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