import numpy as np
import pulp
import matplotlib.pyplot as plt

# ==========================================
import multiprocessing

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

# ==========================================
# 2. 求解器: 6维拟合 + 强制 Knot 约束
# ==========================================
def solve_6d_with_forced_knots(time_points, data_6d, forced_knot_times, degree=3, tol_ratio=0.03, time_limit=60):
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
        BigMs.append(d_range * 100.0)

        alpha[d] = pulp.LpVariable.dicts(f"alpha_{d}", range(degree + 1), lowBound=None)
        beta[d] = pulp.LpVariable.dicts(f"beta_{d}", candidate_knots, lowBound=None)

    prob += pulp.lpSum([y[j] for j in candidate_knots])

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
    print(f"正在求解 6D 轨迹... Tol: {tol_ratio*100}% | Solver: CBC ({cpu_cores} core)")
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=cpu_cores)
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]
    if status_str != "Optimal":
        print(f"Warning: {status_str}")
        return [], None

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

    return active_knots, fitted_curves

# ==========================================
# 2. 从 LeRobotDataset 加载 abs_aff 数据
# ==========================================
def load_abs_aff_from_dataset(episode_index=0):
    """从 LeRobotDataset 加载指定 episode 的 abs_aff 数据"""
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(
        'HuggingFaceVLA/libero',
        root=Path("/inspire/hdd/project/robot-decision/public/datasets/")/'HuggingFaceVLA/libero',
        delta_timestamps={"abs_aff":[]}  # 获取完整的 abs_aff 序列
    )

    # 获取该 episode 完整的 abs_aff 数据
    item = dataset[episode_index]
    abs_aff = np.asarray(item['abs_aff'])  # [T, 7]
    
    # 转换为绝对坐标 (前6维累加，gripper保持不变)
    abs_aff_absolute = abs_aff.copy()
    abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
    
    episode_idx = int(item['episode_index'])
    frame_idx = int(item['frame_index'])
    task_name = item['task'] if 'task' in item else "Unknown"
    
    print(f"\n加载数据成功:")
    print(f"  Episode: {episode_idx}, Frame: {frame_idx}")
    print(f"  Task: {task_name}")
    print(f"  Shape: {abs_aff_absolute.shape}")
    print(f"  前3个点:\n{abs_aff_absolute[:3]}")
    
    return abs_aff_absolute, episode_idx, task_name

# ==========================================
# 3. 主程序 - 从 LeRobotDataset 加载并压缩
# ==========================================

# 加载真实数据
EPISODE_INDEX = 30000  # 可以修改为其他 episode
data_7d, episode_idx, task_name = load_abs_aff_from_dataset(EPISODE_INDEX)

# 创建时间轴 (归一化到 [0, 10] 方便MILP求解)
T_points = data_7d.shape[0]
x_axis = np.linspace(0, 10, T_points)

# 转置为 (7, T) 以适配求解器
data_7d = data_7d.T  # 从 (T, 7) -> (7, T)

# gripper 数据 (index=6) 与 6D 连续轨迹分离
gripper_traj = data_7d[6]
data_6d = data_7d[:6]

# 提取 gripper 跳变，生成强制 knots
forced_knots = extract_forced_knots(x_axis, gripper_traj)
print(f"强制 Knot (来自 gripper 跳变): {forced_knots}")

# 对比三种 Tolerance
tol_ratios = [0.02, 0.01, 0.001]  # 如需更多精度可增加列表
scenarios = ["0.02", "0.01", 0.001]

results = []
for ratio in tol_ratios:
    knots, curves_6d = solve_6d_with_forced_knots(
        time_points=x_axis,
        data_6d=data_6d,
        forced_knot_times=forced_knots,
        degree=3,
        tol_ratio=ratio,
        time_limit=60,
    )
    # gripper 复原：使用原始分段常值信号（已知强制 knot）
    if curves_6d is None:
        full_curves = None
    else:
        gripper_recon = gripper_traj.copy()
        full_curves = np.vstack([curves_6d, gripper_recon])
    results.append((knots, full_curves))

# ==========================================
# 4. 绘图与保存
# ==========================================
fig, axes = plt.subplots(7, 3, figsize=(16, 16), sharex=True)
dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]

for col_idx, (ratio, (knots, curves)) in enumerate(zip(tol_ratios, results)):
    
    # 如果求解失败 curves 为 None
    if curves is None: 
        curves = np.zeros_like(data_7d)
    
    scenario_name = scenarios[col_idx]
    
    for row_idx in range(7):
        ax = axes[row_idx, col_idx]
        
        # 1. 原始数据
        ax.scatter(x_axis, data_7d[row_idx], s=12, color='gray', alpha=0.5, label='Data', zorder=1)
        
        # 2. 拟合曲线
        line_color = 'red' if row_idx < 6 else 'green'  # gripper用绿色
        line_width = 2 if row_idx < 6 else 2.5
        ax.plot(x_axis, curves[row_idx], color=line_color, linewidth=line_width, 
                label='Fit', zorder=2, alpha=0.8)
        
        # 3. Knots (控制点位置)
        if knots:
            for k in knots:
                ax.axvline(x=k, color='blue', linestyle='--', alpha=0.4, linewidth=1, zorder=0)
        # gripper 强制 knot 也可显示（与 knots 一致，这里不重复）
        
        # 4. 计算误差
        if curves is not None:
            error = np.abs(curves[row_idx] - data_7d[row_idx])
            mean_err = np.mean(error)
            max_err = np.max(error)
            ax.text(0.98, 0.02, f'E_mean={mean_err:.4f}\nE_max={max_err:.4f}', 
                   transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
        # 标注
        if row_idx == 0:
            knot_info = f"Knots: {len(knots)}" if knots else "Failed"
            ax.set_title(f"{scenario_name} (Tol={ratio*100}%)\n{knot_info}", 
                         fontsize=11, fontweight='bold')
        
        if col_idx == 0:
            ax.set_ylabel(dim_names[row_idx], fontsize=10, fontweight='bold')
            
        # 为gripper设置固定Y轴
        if row_idx == 6:
            y_min, y_max = data_7d[row_idx].min(), data_7d[row_idx].max()
            y_range = max(y_max - y_min, 0.5)
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        ax.grid(True, alpha=0.2)

# 总标题
fig.suptitle(f'MILP B-Spline Compression (6D+Forced Knots) - Episode {episode_idx}\nTask: {task_name}', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# 保存图片
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"MILP_Compression_Ep{episode_idx}_forced_knots.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n图片已保存为: {filename}")

# plt.show()  # 注释掉以便在服务器上运行