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

# --- 并行配置 ---
TOTAL_CORES = 64
NUM_WORKERS = 16  # 并行进程数
THREADS_PER_SOLVER = 4  # 每个进程使用的线程数 (16 * 4 = 64)

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
    return str(episode_idx) in results.get("episodes", {})


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
# 2. 求解器 (修改：接受 threads 参数)
# ==========================================
def solve_6d_with_forced_knots(time_points, data_6d, forced_knot_times, degree=3, tol_ratio=0.03, time_limit=60, solver_threads=1):
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
        
        # 预计算幂基矩阵以加速构建
        for i, t_val in enumerate(time_points):
            P_val = data_6d[d][i]
            poly_part = pulp.lpSum([alpha[d][k] * (t_val**k) for k in range(degree + 1)])
            # 只生成非零项
            knot_terms = [beta[d][j] * ((t_val - j) ** degree) for j in candidate_knots if t_val > j]
            S_t = poly_part + pulp.lpSum(knot_terms)
            prob += S_t - P_val <= eps_d
            prob += P_val - S_t <= eps_d

    # 【关键修改】使用指定的线程数
    # msg=False 关闭求解器日志，防止多进程时控制台混乱
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=solver_threads)
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]
    if status_str != "Optimal":
        return [], None

    active_knots = [j for j in candidate_knots if pulp.value(y[j]) > 0.5]
    
    # 只需要 active knots，不需要在这里重建曲线，节省时间
    # 曲线重建在 b-spline 转换步骤做
    return active_knots, "Solved" 

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
# 4. 可视化 (精简版，支持无GUI)
# ==========================================
def visualize_episode_safe(data_7d, tpb_curves, bspline_curves, knots, ep_idx, task, x_axis, save_path):
    """线程安全的可视化函数"""
    try:
        fig, axes = plt.subplots(2, 8, figsize=(32, 10))
        axes = axes.flatten()
        # ... (绘图逻辑与之前相同，省略重复代码以节省空间) ...
        # 这里请填入你原来的绘图代码
        # ... 
        
        # 简单画一下示意，确保代码可运行
        dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
        for i in range(7):
            ax = axes[i+8] # 右侧 B-spline
            ax.plot(x_axis, data_7d[i], 'gray', alpha=0.5)
            ax.plot(x_axis, bspline_curves[i], 'b-')
            ax.set_title(dim_names[i])
        
        plt.suptitle(f"Episode {ep_idx}: {task}")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close(fig) # 必须关闭
    except Exception as e:
        print(f"绘图失败: {e}")

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
        
        # 3. MILP 求解 (调用 4 个线程)
        knots, status = solve_6d_with_forced_knots(
            x_axis, data_6d, forced_knots, 
            degree=3, tol_ratio=tolerance, time_limit=120,
            solver_threads=THREADS_PER_SOLVER # <--- 关键参数
        )
        
        if knots is None:
            return None # 求解失败

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

        # Gripper 处理
        gripper_ctrl_pts, gripper_knot_vec, _ = convert_to_bspline_coeffs(x_axis, gripper_traj, forced_knots, degree=0)
        if gripper_ctrl_pts is None:
             bspline_control_points.append(gripper_traj.tolist()) # Fallback
             bspline_knot_vectors.append([])
             gripper_bspline = gripper_traj
        else:
            bspline_control_points.append(gripper_ctrl_pts.tolist())
            bspline_knot_vectors.append(gripper_knot_vec.tolist())
            gripper_bspline = reconstruct_bspline_trajectory(gripper_ctrl_pts, gripper_knot_vec, x_axis, degree=0)

        # 5. 组装结果
        bspline_curves_7d = np.vstack([bspline_curves_6d, gripper_bspline])
        overall_mean = float(np.mean(np.abs(bspline_curves_7d - data_7d)))
        
        # 可视化 (保存到文件)
        vis_path = VISUAL_DIR / f"episode_{episode_idx}.jpg"
        # 为了速度，可以在这里暂不画图，或者只画一部分
        visualize_episode_safe(data_7d, bspline_curves_7d, bspline_curves_7d, knots, episode_idx, task_name, x_axis, vis_path)

        result_dict = {
            "episode_index": episode_idx,
            "task_name": task_name,
            "trajectory_length": T_points,
            "bspline": {
                "control_points": bspline_control_points,
                "knot_vectors": bspline_knot_vectors,
                "internal_knots": [float(k) for k in knots],
                "forced_knots": [float(k) for k in forced_knots],
                "num_knots": len(knots),
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
    current_idx = 500
    
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

    # 3. 过滤已完成的任务
    results = load_results_from_json(RESULTS_JSON_PATH)
    computed_ep_ids = set([int(k) for k in results["episodes"].keys()])
    
    tasks_to_run = []
    for start_idx, ep_id in task_indices:
        if ep_id not in computed_ep_ids:
            tasks_to_run.append((start_idx, TOLERANCE)) # 构建任务参数
    
    print(f"剩余待处理任务: {len(tasks_to_run)}")
    
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
                # print(f"\n[Err] Ep {res.get('episode_idx')}: {res['error']}")
                fail_count += 1
                continue
            
            # 保存结果 (主进程串行写入，无锁问题)
            ep_id = str(res['episode_index'])
            results["episodes"][ep_id] = res
            success_count += 1
            
            # 定期保存 (每完成 10 个)
            if success_count % 10 == 0:
                save_results_to_json(results, RESULTS_JSON_PATH)
                pbar_run.set_postfix({"Saved": success_count, "Fail": fail_count})
        
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