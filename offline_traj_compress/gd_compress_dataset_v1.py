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
RESULTS_JSON_PATH = Path("gd_compression_results_v1.json")  # Modified V2
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
def solve_6d_with_forced_knots(
    time_points,
    data_6d,
    forced_knot_times,
    degree=3,
    tol_ratio=0.03,
    time_limit=600,
):
    """
    R1: 6D group trend filtering (order=3 cubic) via sparse 4th-difference.
    Solve:
        min_X 0.5||X - Y||_F^2 + lam * sum_t ||(D4 X)_[:,t]||_2
    where D4 is 4th finite difference along time.

    Then infer knots from non-zero group norms of D4X.
    Forced knots: do not shrink corresponding D4 positions.

    Returns:
        active_knots: List[int] in [1, T-2]
        fitted_curves: np.ndarray shape (6, T)
        is_suboptimal: bool
    """
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    assert degree == 3, "R1这里按 cubic(=3) 做：四阶差分稀疏。"
    start_time = time.time()

    # ----------------------------
    # basic setup
    # ----------------------------
    Y_np = np.asarray(data_6d, dtype=np.float32)  # (6, T)
    D, T = Y_np.shape
    if T < 6:
        # 太短没法做四阶差分，直接返回原数据
        knots = sorted(set(int(t) for t in forced_knot_times if 1 <= int(t) <= T - 2))
        return knots, Y_np, False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = torch.from_numpy(Y_np).to(device)  # (6, T)

    # tolerance (L∞ per-dim)
    ranges = np.max(Y_np, axis=1) - np.min(Y_np, axis=1)
    ranges[ranges < 1e-6] = 1.0
    eps = torch.from_numpy((ranges * float(tol_ratio)).astype(np.float32)).to(device)  # (6,)

    forced_set = set(int(t) for t in forced_knot_times if 1 <= int(t) <= T - 2)

    # D4 index r maps to "knot time" r+2 (center of 5-point stencil)
    # r ranges [0, T-5] => knot time in [2, T-3]
    forced_r = set()
    for k in forced_set:
        r = k - 2
        if 0 <= r <= T - 5:
            forced_r.add(r)

    # ----------------------------
    # build D4 and D4^T via depthwise conv1d
    # ----------------------------
    # kernel for 4th finite difference: [1, -4, 6, -4, 1]
    k = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], device=device, dtype=torch.float32)
    w = k.view(1, 1, 5).repeat(D, 1, 1)  # (D,1,5) depthwise

    def d4(x_DT):
        # x_DT: (D, T) -> (D, T-4)
        x = x_DT.unsqueeze(0)  # (1,D,T)
        out = F.conv1d(x, w, padding=0, groups=D)  # (1,D,T-4)
        return out.squeeze(0)

    def d4T(v_DTm4):
        # v_DTm4: (D, T-4) -> (D, T)  using padding=4 gives length T
        v = v_DTm4.unsqueeze(0)  # (1,D,T-4)
        out = F.conv1d(v, w, padding=4, groups=D)  # (1,D,T)
        return out.squeeze(0)

    # Linear operator A(x)= x + rho * D4^T D4 x
    def A(x_DT, rho):
        return x_DT + rho * d4T(d4(x_DT))

    # ----------------------------
    # Conjugate Gradient for X-update
    # ----------------------------
    @torch.no_grad()
    def cg_solve(rhs_DT, rho, x0_DT=None, max_iter=60, tol=1e-5):
        # Solve A(x)=rhs. x,r,p shapes (D,T)
        x = rhs_DT.clone() if x0_DT is None else x0_DT.clone()
        r = rhs_DT - A(x, rho)
        p = r.clone()
        rs_old = torch.sum(r * r)
        if rs_old.item() < 1e-20:
            return x

        for _ in range(max_iter):
            Ap = A(p, rho)
            denom = torch.sum(p * Ap) + 1e-12
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)
            if rs_new.item() <= tol * tol * (rhs_DT.numel()):
                break
            p = r + (rs_new / (rs_old + 1e-12)) * p
            rs_old = rs_new
        return x

    # ----------------------------
    # ADMM solve for a given lambda
    # ----------------------------
    @torch.no_grad()
    def solve_for_lambda(lam, rho=1.0, admm_iters=200, cg_iters=60):
        # Variables:
        # X: (D,T)
        # Z,U: (D,T-4)
        X = Y.clone()
        Z = torch.zeros((D, T - 4), device=device, dtype=torch.float32)
        U = torch.zeros((D, T - 4), device=device, dtype=torch.float32)

        forced_mask = torch.zeros((1, T - 4), device=device, dtype=torch.bool)
        if forced_r:
            rr = torch.tensor(sorted(list(forced_r)), device=device, dtype=torch.long)
            forced_mask[:, rr] = True  # (1, T-4)

        # ADMM loop
        for it in range(admm_iters):
            if (time.time() - start_time) > time_limit:
                break

            # X-update: (I + rho D4^T D4) X = Y + rho D4^T(Z - U)
            rhs = Y + rho * d4T(Z - U)
            X = cg_solve(rhs, rho=rho, x0_DT=X, max_iter=cg_iters, tol=1e-5)

            # Z-update: group shrinkage on V = D4X + U across dims (group over D)
            V = d4(X) + U  # (D, T-4)
            norms = torch.sqrt(torch.sum(V * V, dim=0, keepdim=True) + 1e-12)  # (1, T-4)
            shrink = torch.clamp(1.0 - (lam / (rho * norms)), min=0.0)  # (1, T-4)

            # do NOT shrink forced positions
            if forced_r:
                shrink = torch.where(forced_mask, torch.ones_like(shrink), shrink)

            Z = V * shrink  # broadcast over D

            # U-update
            U = U + d4(X) - Z

            # small early stop (optional)
            if (it + 1) % 25 == 0:
                prim = torch.norm(d4(X) - Z) / (torch.norm(Z) + 1e-12)
                if prim.item() < 1e-3:
                    break

        return X

    # ----------------------------
    # choose lambda by binary search to satisfy L∞ tolerance
    # (largest lambda s.t. max_abs_error_per_dim <= eps)
    # ----------------------------
    @torch.no_grad()
    def max_abs_err(X):
        E = torch.abs(X - Y)  # (D,T)
        return torch.max(E, dim=1).values  # (D,)

    # quick bounds
    lam_lo = 0.0
    lam_hi = 1.0

    # find upper bound where error exceeds eps
    # (error increases with lambda; if not, keep increasing)
    X_hi = solve_for_lambda(lam_hi, rho=1.0, admm_iters=120, cg_iters=40)
    err_hi = max_abs_err(X_hi)

    while torch.all(err_hi <= eps) and lam_hi < 1e6 and (time.time() - start_time) < time_limit * 0.5:
        lam_hi *= 2.0
        X_hi = solve_for_lambda(lam_hi, rho=1.0, admm_iters=120, cg_iters=40)
        err_hi = max_abs_err(X_hi)

    # If even huge lambda still fits tolerance, accept it (minimal knots)
    if torch.all(err_hi <= eps):
        X_best = X_hi
        lam_best = lam_hi
    else:
        # binary search
        X_best = None
        lam_best = lam_lo
        for _ in range(18):  # enough
            if (time.time() - start_time) > time_limit:
                break
            lam_mid = 0.5 * (lam_lo + lam_hi)
            X_mid = solve_for_lambda(lam_mid, rho=1.0, admm_iters=200, cg_iters=60)
            err_mid = max_abs_err(X_mid)

            if torch.all(err_mid <= eps):
                lam_best = lam_mid
                X_best = X_mid
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

        # if never feasible (very strict eps), fall back to lam=0
        if X_best is None:
            X_best = solve_for_lambda(0.0, rho=1.0, admm_iters=80, cg_iters=30)
            lam_best = 0.0

    # ----------------------------
    # infer knots: Top-K (budget) instead of tiny threshold
    # ----------------------------
    with torch.no_grad():
        D4 = d4(X_best)  # (D, T-4)
        gn = torch.sqrt(torch.sum(D4 * D4, dim=0) + 1e-12)  # (T-4,)
        gn_cpu = gn.detach().cpu().numpy()

        # r -> knot time = r+2 in [2, T-3]
        knot_times = np.arange(2, T - 2, dtype=int)  # length T-4

        # ---- 你想要的“点数越少越好”在这里调 ----
        MAX_KNOTS = 20            # 总 knot 上限（含 forced）
        MIN_GAP = 1               # (可选) knot 之间最小间隔，=0 表示不限制

        forced_times = sorted([k for k in forced_set if 1 <= k <= T - 2])
        budget = max(0, MAX_KNOTS - len(forced_times))

        # 给 forced 打无穷大分数，保证一定选中
        scores = gn_cpu.copy()
        for k in forced_times:
            r = k - 2
            if 0 <= r < len(scores):
                scores[r] = 1e30

        # 从大到小排序候选
        order = np.argsort(-scores)

        selected = []
        chosen_times = set(forced_times)

        def ok_with_gap(t):
            if MIN_GAP <= 0:
                return True
            for tt in chosen_times:
                if abs(t - tt) < MIN_GAP:
                    return False
            return True

        # 先塞 forced
        for t in forced_times:
            selected.append(t)

        # 再选 top-K（带最小间隔）
        for idx in order:
            t = int(knot_times[idx])
            if t in chosen_times:
                continue
            if budget <= 0:
                break
            if ok_with_gap(t):
                selected.append(t)
                chosen_times.add(t)
                budget -= 1

        active_knots = sorted(set(selected))

    fitted_curves = X_best.detach().cpu().numpy().astype(np.float32)

    # not "suboptimal" in MILP sense; here mark suboptimal if time limit hit hard
    is_suboptimal = (time.time() - start_time) > time_limit * 0.98

    print(f"  [R1] lambda={lam_best:.4g} | knots={len(active_knots)} | "
          f"max_err={torch.max(max_abs_err(X_best)).item():.6f} | device={device}")

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
def visualize_episode(data_7d: np.ndarray, bspline_curves_7d: np.ndarray, knots: List, 
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
    
    # 构造完整的 knot vector (包含前后重复端点)
    # 对于 degree=3: [t0, t0, t0, t0, internal_knots..., t_end, t_end, t_end, t_end]
    t0, t_end = int(x_axis[0]), int(x_axis[-1])
    full_knots_vector_deg3 = (
        [t0] * 4 +  # degree + 1 = 4 个起始点
        sorted([int(k) for k in knots]) +  # 内部 knots
        [t_end] * 4  # degree + 1 = 4 个结束点
    )
    
    # 对于 gripper (degree=0): [t0, internal_knots..., t_end]
    full_knots_vector_deg0 = (
        [t0] +  # degree + 1 = 1 个起始点
        sorted([int(k) for k in knots]) +  # 内部 knots (与上面相同)
        [t_end]  # degree + 1 = 1 个结束点
    )
    
    # 可视化
    vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
    # tpb_curves_7d = np.vstack([curves_6d, bspline_curves_7d[6]]) # TPB 暂用 B-spline 的 gripper
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
            "knots_vector_deg3": full_knots_vector_deg3,  # 用于前6维 (arm)
            "knots_vector_deg0": full_knots_vector_deg0,  # 用于第7维 (gripper)
            
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