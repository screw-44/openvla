"""
检查压缩结果质量的脚本

功能：
1. 扫描 compression_results.json，找出需要重新计算的 episode
   - 情况1：未计算过
   - 情况2：forced_knots 数量等于 num_knots（求解失败）
   - 情况3：overall_mean_error > 0.5（精度不足）
   - 情况4：is_suboptimal 为 True（次优解）

2. 验证模式（可选）：从 JSON 重建 B-spline 曲线，计算实际误差
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from scipy.interpolate import BSpline
from collections import defaultdict
import tqdm

# ==========================================
# Configuration
# ==========================================
RESULTS_JSON_PATH = Path("compression_results.json")
EPISODE_INDEX_PATH = Path("epsidoe_2_dataset_index.json")  # 注意拼写错误，保持和原文件一致
DATASET_ROOT = Path("/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA/libero")

# 是否启用验证模式（重建 B-spline 并计算误差）
ENABLE_VALIDATION = True
TOP_N_ERRORS = 300  # 打印前 N 个最大误差的 episode


# ==========================================
# 检查函数
# ==========================================
def check_episode_status(episode_idx: int, ep_data: Dict) -> Tuple[bool, str]:
    """
    检查单个 episode 的状态
    
    Returns:
        (is_valid, reason) - is_valid=False 表示需要重新计算，reason 说明原因
    """
    bspline = ep_data.get("bspline", {})
    
    # 情况2：forced_knots 数量等于 num_knots（求解失败）
    forced_knots = bspline.get("forced_knots", [])
    num_knots = bspline.get("num_knots", 0)
    if len(forced_knots) == num_knots:
        return False, f"求解失败: forced_knots({len(forced_knots)}) == num_knots({num_knots})"
    
    # 情况3：overall_mean_error > 0.5（精度不足）
    mean_error = bspline.get("overall_mean_error", 0.0)
    if mean_error > 0.5:
        return False, f"精度不足: mean_error={mean_error:.6f} > 0.5"
    
    # 情况4：次优解 (检查顶层的 status 字段 或 bspline 下的 is_suboptimal 字段)
    status = ep_data.get("status", "")
    is_suboptimal = bspline.get("is_suboptimal", False)
    if status == "suboptimal" or is_suboptimal:
        return False, f"次优解: status={status}, is_suboptimal={is_suboptimal}"
    
    return True, "正常"


def load_compression_results() -> Dict:
    """加载压缩结果 JSON"""
    if not RESULTS_JSON_PATH.exists():
        print(f"❌ 错误: 文件不存在 {RESULTS_JSON_PATH}")
        return {}
    
    with open(RESULTS_JSON_PATH, 'r') as f:
        return json.load(f)


def scan_all_episodes(results: Dict) -> Tuple[Set[int], Dict[str, List[int]]]:
    """
    扫描所有 episode，分类统计需要重新计算的情况
    
    Returns:
        (all_recompute_episodes, category_episodes)
        - all_recompute_episodes: 所有需要重新计算的 episode 集合
        - category_episodes: 按原因分类的 episode 列表
    """
    episodes = results.get("episodes", {})
    
    # 分类统计
    categories = {
        "未计算": [],
        "求解失败": [],
        "精度不足": [],
        "次优解": []
    }
    
    all_recompute = set()
    
    # 获取所有已计算的 episode
    computed_episodes = {int(ep_id) for ep_id in episodes.keys()}
    
    # 检查已存在的 episode
    for ep_str, ep_data in episodes.items():
        ep_idx = int(ep_str)
        is_valid, reason = check_episode_status(ep_idx, ep_data)
        
        if not is_valid:
            all_recompute.add(ep_idx)
            
            # 分类
            if "求解失败" in reason:
                categories["求解失败"].append(ep_idx)
            elif "精度不足" in reason:
                categories["精度不足"].append(ep_idx)
            elif "次优解" in reason:
                categories["次优解"].append(ep_idx)
    
    # 检查未计算的 episode（需要从 episode_2_dataset_index.json 获取完整列表）
    if EPISODE_INDEX_PATH.exists():
        with open(EPISODE_INDEX_PATH, 'r') as f:
            episode_index_map = json.load(f)
        
        all_episodes = {int(ep_id) for ep_id in episode_index_map.keys()}
        missing_episodes = all_episodes - computed_episodes
        
        if missing_episodes:
            categories["未计算"] = sorted(list(missing_episodes))
            all_recompute.update(missing_episodes)
    
    return all_recompute, categories


def print_statistics(all_recompute: Set[int], categories: Dict[str, List[int]]):
    """打印统计信息"""
    print("\n" + "="*80)
    print("压缩结果检查报告")
    print("="*80)
    
    # 分类打印
    for category, episodes in categories.items():
        episodes_sorted = sorted(episodes)
        print(f"\n【{category}】 共 {len(episodes_sorted)} 个:")
        if len(episodes_sorted) > 0:
            # 每行打印 10 个
            for i in range(0, len(episodes_sorted), 10):
                chunk = episodes_sorted[i:i+10]
                print(f"  {chunk}")
        else:
            print(f"  (无)")
    
    # 汇总统计
    print("\n" + "-"*80)
    print(f"【汇总】 需要重新计算的 Episode (共 {len(all_recompute)} 个):")
    all_sorted = sorted(list(all_recompute))
    if len(all_sorted) > 0:
        print(all_sorted)
    else:
        print(f"  (所有 Episode 均已正确计算)")
    
    print("="*80 + "\n")


# ==========================================
# 验证模式：重建 B-spline 并计算误差
# ==========================================
def load_dataset():
    """加载数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        'HuggingFaceVLA/libero',
        root=DATASET_ROOT,
        delta_timestamps={"abs_aff": []}
    )
    return dataset


def load_episode_data(dataset, dataset_index: int) -> Tuple[np.ndarray, int]:
    """
    加载单个 episode 的轨迹数据
    
    Returns:
        (data_7d, episode_idx) - 7维轨迹数据 (T, 7) 和 episode 索引
    """
    item = dataset[dataset_index]
    abs_aff = np.asarray(item['abs_aff'])
    abs_aff_absolute = abs_aff.copy()
    abs_aff_absolute[:, :-1] = np.cumsum(abs_aff_absolute[:, :-1], axis=0)
    episode_idx = int(item['episode_index'])
    return abs_aff_absolute, episode_idx


def reconstruct_bspline(control_points_list: List, knot_vectors_list: List, time_points: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    从控制点和节点向量重建 B-spline 曲线
    
    Args:
        control_points_list: 控制点列表 (7个维度，每个维度一个数组)
        knot_vectors_list: 节点向量列表 (7个维度，每个维度一个knot vector)
        time_points: 时间点
        degree: B-spline 阶数（前6维用3，gripper用0）
        
    Returns:
        curves_7d: 重建的 7 维曲线 (7, T)
    """
    curves = []
    for dim_idx, (control_points, knot_vector) in enumerate(zip(control_points_list, knot_vectors_list)):
        # Gripper (dim 6) 使用 degree=0
        if dim_idx == 6:
            if len(knot_vector) == 0:
                # 如果没有 knot vector，直接使用 control points（可能是原始数据）
                curves.append(np.array(control_points))
            else:
                bspline = BSpline(knot_vector, control_points, k=0, extrapolate=False)
                curve = bspline(time_points)
                curves.append(curve)
        else:
            # 前6维使用 degree=3
            bspline = BSpline(knot_vector, control_points, k=3, extrapolate=False)
            curve = bspline(time_points)
            curves.append(curve)
    
    return np.array(curves)  # shape: (7, T)


def validate_episode(ep_idx: int, ep_data: Dict, dataset, episode_index_map: Dict) -> Tuple[float, float, np.ndarray]:
    """
    验证单个 episode 的实际误差
    
    Args:
        ep_idx: episode 索引
        ep_data: 压缩结果数据
        dataset: 数据集对象
        episode_index_map: episode 到 dataset_index 的映射
        
    Returns:
        (mean_error, max_error, errors_per_dim) - 平均误差、最大误差、各维度误差
    """
    # 1. 从 JSON 读取 B-spline 参数
    bspline_data = ep_data.get("bspline", {})
    control_points_list = bspline_data.get("control_points", [])
    # 注意：字段名是 knot_vectors（复数）而不是 knot_vector
    knot_vectors = bspline_data.get("knot_vectors", [])
    
    if not control_points_list or not knot_vectors:
        return np.inf, np.inf, np.array([np.inf] * 7)
    
    # 2. 加载原始轨迹
    ep_str = str(ep_idx)
    if ep_str not in episode_index_map:
        return np.inf, np.inf, np.array([np.inf] * 7)
    
    dataset_index = episode_index_map[ep_str]
    data_7d, _ = load_episode_data(dataset, dataset_index)  # shape: (T, 7)
    data_7d = data_7d.T  # shape: (7, T)
    
    # 3. 重建 B-spline 曲线
    T_points = data_7d.shape[1]
    time_points = np.arange(T_points)
    
    try:
        bspline_curves_7d = reconstruct_bspline(control_points_list, knot_vectors, time_points)
    except Exception as e:
        print(f"  ❌ Episode {ep_idx} 重建失败: {e}")
        return np.inf, np.inf, np.array([np.inf] * 7)
    
    # 4. 计算误差
    errors = np.abs(bspline_curves_7d - data_7d)  # shape: (7, T)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    errors_per_dim = np.max(errors, axis=1)  # 每个维度的最大误差
    
    return mean_error, max_error, errors_per_dim


def validation_mode(results: Dict):
    """
    验证模式：重建所有 B-spline 并计算实际误差
    """
    print("\n" + "="*80)
    print("验证模式：重建 B-spline 并计算误差")
    print("="*80)
    
    # 1. 加载数据集和索引映射
    print("正在加载数据集...")
    dataset = load_dataset()
    
    if not EPISODE_INDEX_PATH.exists():
        print(f"❌ 错误: 索引文件不存在 {EPISODE_INDEX_PATH}")
        return
    
    with open(EPISODE_INDEX_PATH, 'r') as f:
        episode_index_map = json.load(f)
    
    # 2. 遍历所有已计算的 episode
    episodes = results.get("episodes", {})
    error_records = []
    
    print(f"正在验证 {len(episodes)} 个 episode...")
    print(f"索引映射示例: {list(episode_index_map.items())[:3]}")
    
    failed_count = 0
    
    for ep_str, ep_data in tqdm.tqdm(episodes.items(), desc="验证中"):
        ep_idx = int(ep_str)
        
        mean_error, max_error, errors_per_dim = validate_episode(
            ep_idx, ep_data, dataset, episode_index_map
        )
        
        if np.isinf(mean_error):
            failed_count += 1
        
        # 记录误差
        # 检查 is_suboptimal: 优先从顶层 status 字段读取，其次从 bspline 读取
        status = ep_data.get("status", "Not Found")
    
        error_records.append({
            "episode_idx": ep_idx,
            "mean_error": mean_error,
            "max_error": max_error,
            "errors_per_dim": errors_per_dim,
            "json_mean_error": ep_data.get("bspline", {}).get("overall_mean_error", 0.0),
            "num_knots": ep_data.get("bspline", {}).get("num_knots", 0),
            "is_suboptimal": status
        })
    
    print(f"\n验证完成: 失败 {failed_count}/{len(episodes)} 个episode")
    
    # 3. 按误差排序
    print("\n正在统计误差分布...")
    
    # 过滤掉无效结果
    valid_records = [r for r in error_records if not np.isinf(r["mean_error"])]
    
    # 按 mean_error 排序
    sorted_by_mean = sorted(valid_records, key=lambda x: x["mean_error"], reverse=True)
    
    # 按 max_error 排序
    sorted_by_max = sorted(valid_records, key=lambda x: x["max_error"], reverse=True)
    
    # 4. 打印统计信息
    print("\n" + "-"*80)
    print(f"【误差统计】")
    print(f"  总计: {len(valid_records)} 个有效 episode")
    print(f"  平均误差: mean={np.mean([r['mean_error'] for r in valid_records]):.6f}, "
          f"std={np.std([r['mean_error'] for r in valid_records]):.6f}")
    print(f"  最大误差: mean={np.mean([r['max_error'] for r in valid_records]):.6f}, "
          f"std={np.std([r['max_error'] for r in valid_records]):.6f}")
    
    # 5. 打印前 N 个最大 mean_error
    print("\n" + "-"*80)
    print(f"【按 Mean Error 排序】前 {TOP_N_ERRORS} 个:")
    print(f"{'Rank':<6} {'Episode':<10} {'Mean Error':<15} {'Max Error':<15} {'Knots':<8} {'Suboptimal':<12} {'JSON Error':<15}")
    print("-"*80)
    
    for i, record in enumerate(sorted_by_mean[:TOP_N_ERRORS], 1):
        print(f"{i:<6} {record['episode_idx']:<10} {record['mean_error']:<15.6f} "
              f"{record['max_error']:<15.6f} {record['num_knots']:<8} "
              f"{str(record['is_suboptimal']):<12} {record['json_mean_error']:<15.6f}")
    
    # 6. 打印前 N 个最大 max_error
    print("\n" + "-"*80)
    print(f"【按 Max Error 排序】前 {TOP_N_ERRORS} 个:")
    print(f"{'Rank':<6} {'Episode':<10} {'Mean Error':<15} {'Max Error':<15} {'Knots':<8} {'Suboptimal':<12} {'JSON Error':<15}")
    print("-"*80)
    
    for i, record in enumerate(sorted_by_max[:TOP_N_ERRORS], 1):
        print(f"{i:<6} {record['episode_idx']:<10} {record['mean_error']:<15.6f} "
              f"{record['max_error']:<15.6f} {record['num_knots']:<8} "
              f"{str(record['is_suboptimal']):<12} {record['json_mean_error']:<15.6f}")
    
    # 7. 维度误差分析
    print("\n" + "-"*80)
    print(f"【各维度最大误差统计】")
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    
    all_dim_errors = np.array([r["errors_per_dim"] for r in valid_records if len(r["errors_per_dim"]) == 7])
    
    if len(all_dim_errors) > 0:
        print(f"{'Dimension':<12} {'Mean':<15} {'Max':<15} {'Std':<15}")
        print("-"*80)
        for i, dim_name in enumerate(dim_names):
            dim_errors = all_dim_errors[:, i]
            print(f"{dim_name:<12} {np.mean(dim_errors):<15.6f} {np.max(dim_errors):<15.6f} {np.std(dim_errors):<15.6f}")
    
    print("="*80 + "\n")


# ==========================================
# 主函数
# ==========================================
def main():
    # 1. 加载压缩结果
    print("正在加载压缩结果...")
    results = load_compression_results()
    
    if not results:
        print("❌ 无法加载压缩结果，退出")
        return
    
    # 2. 扫描并分类统计
    print("正在扫描 episode 状态...")
    all_recompute, categories = scan_all_episodes(results)
    
    # 3. 打印统计信息
    print_statistics(all_recompute, categories)
    
    # 4. 验证模式（可选）
    if ENABLE_VALIDATION:
        validation_mode(results)


if __name__ == "__main__":
    main()


# ================================================================================
# 压缩结果检查报告
# ================================================================================

# 【未计算】 共 0 个:
#   (无)

# 【求解失败】 共 4 个:
#   [128, 822, 906, 1615]

# 【精度不足】 共 43 个:
#   [67, 91, 179, 223, 235, 267, 316, 360, 378, 387]
#   [416, 431, 491, 711, 826, 842, 852, 880, 924, 955]
#   [984, 1001, 1021, 1047, 1051, 1094, 1096, 1097, 1107, 1137]
#   [1299, 1312, 1343, 1347, 1396, 1441, 1446, 1450, 1562, 1630]
#   [1659, 1671, 1685]

# 【次优解】 共 173 个:
#   [247, 386, 388, 394, 402, 447, 457, 459, 464, 468]
#   [492, 495, 508, 514, 522, 532, 567, 587, 613, 620]
#   [626, 629, 635, 637, 641, 649, 653, 655, 658, 677]
#   [679, 682, 689, 698, 706, 712, 718, 720, 727, 743]
#   [745, 747, 760, 773, 774, 781, 788, 789, 796, 811]
#   [812, 815, 823, 829, 843, 847, 853, 858, 865, 867]
#   [871, 883, 891, 893, 898, 899, 902, 903, 913, 922]
#   [933, 940, 952, 959, 962, 967, 979, 987, 988, 995]
#   [1006, 1007, 1015, 1028, 1031, 1050, 1063, 1065, 1066, 1068]
#   [1098, 1100, 1108, 1111, 1121, 1122, 1124, 1131, 1139, 1153]
#   [1157, 1171, 1172, 1175, 1186, 1191, 1192, 1193, 1196, 1200]
#   [1204, 1205, 1210, 1217, 1224, 1249, 1251, 1259, 1264, 1272]
#   [1274, 1275, 1278, 1288, 1292, 1306, 1313, 1320, 1354, 1370]
#   [1386, 1388, 1390, 1395, 1397, 1400, 1401, 1402, 1404, 1414]
#   [1425, 1449, 1451, 1460, 1462, 1464, 1468, 1471, 1481, 1524]
#   [1535, 1550, 1553, 1560, 1565, 1570, 1574, 1577, 1602, 1610]
#   [1616, 1617, 1622, 1646, 1653, 1660, 1662, 1667, 1669, 1672]
#   [1677, 1680, 1682]

# --------------------------------------------------------------------------------
# 【汇总】 需要重新计算的 Episode (共 47 个):
#   [67, 91, 128, 179, 223, 235, 267, 316, 360, 378]
#   [387, 416, 431, 491, 711, 822, 826, 842, 852, 880]
#   [906, 924, 955, 984, 1001, 1021, 1047, 1051, 1094, 1096]
#   [1097, 1107, 1137, 1299, 1312, 1343, 1347, 1396, 1441, 1446]
#   [1450, 1562, 1615, 1630, 1659, 1671, 1685]
# ================================================================================
# --------------------------------------------------------------------------------
# 【按 Max Error 排序】前 100 个:
# Rank   Episode    Mean Error      Max Error       Knots    Suboptimal   JSON Error     
# --------------------------------------------------------------------------------
# 1      316        2.297507        17.821464       0        False        2.297507       
# 2      378        1.532878        12.484756       5        True         1.532878       
# 3      179        1.290440        12.075339       5        True         1.290440       
# 4      67         1.619133        10.399843       5        False        1.619133       
# 5      387        0.870922        10.335525       5        True         0.870922       
# 6      1441       1.121766        9.549488        5        True         1.121766       
# 7      852        0.786741        9.379718        6        True         0.786741       
# 8      360        1.011390        8.747005        5        True         1.011390       
# 9      1659       0.532501        8.433726        5        True         0.532501       
# 10     1096       0.659551        8.359747        6        True         0.659551       
# 11     416        0.832405        8.166654        5        True         0.832405       
# 12     91         0.869344        8.069135        15       True         0.869344       
# 13     711        0.928792        7.788973        5        True         0.928792       
# 14     826        0.753580        7.604012        5        True         0.753580       
# 15     1562       0.659524        7.586589        5        True         0.659524       
# 16     1100       0.447061        7.505343        6        True         0.447061       
# 17     128        0.979382        7.452811        4        True         0.979382       
# 18     1097       0.814367        7.286036        5        True         0.814367       
# 19     1685       0.775749        7.246369        5        True         0.775749       
# 20     1292       0.367312        6.990377        20       True         0.367312       
# 21     235        0.858411        6.663799        5        True         0.858411       
# 22     811        0.475156        6.635759        5        True         0.475156       
# 23     1275       0.475131        6.569781        14       True         0.475131       
# 24     267        0.673836        6.525260        8        True         0.673836       
# 25     1615       0.957550        6.306546        2        True         0.957550       
# 26     1343       0.583534        6.152509        3        True         0.583534       
# 27     1622       0.449457        6.099414        4        True         0.449457       
# 28     491        0.592750        6.041832        5        True         0.592750       
# 29     223        0.635047        6.021118        7        True         0.635047       
# 30     1653       0.476605        5.968143        6        True         0.476605       
# 31     880        0.560185        5.964155        5        True         0.560185       
# 32     1312       0.557615        5.956221        3        True         0.557615       
# 33     984        0.559766        5.954119        4        True         0.559766       
# 34     1672       0.443285        5.843091        7        True         0.443285       
# 35     1347       0.686106        5.835185        3        True         0.686106       
# 36     906        0.606452        5.818208        2        True         0.606452       
# 37     620        0.374370        5.765478        5        True         0.374370       
# 38     1299       0.662070        5.734002        5        True         0.662070       
# 39     1396       0.641513        5.733591        4        True         0.641513       
# 40     1001       0.625662        5.654321        5        True         0.625662       
# 41     718        0.485201        5.632817        9        True         0.485201       
# 42     988        0.455158        5.606190        6        True         0.455158       
# 43     1021       0.625603        5.568002        4        True         0.625603       
# 44     1446       0.555762        5.514436        5        True         0.555762       
# 45     1610       0.368713        5.456118        5        True         0.368713       
# 46     1404       0.294367        5.417013        9        True         0.294367       
# 47     924        0.504851        5.392135        7        True         0.504851       
# 48     842        0.660972        5.327123        5        True         0.660972       
# 49     1449       0.381756        5.292816        11       True         0.381756       
# 50     1450       0.659663        5.232750        6        True         0.659663       
# 51     822        0.519113        5.190780        2        True         0.519113       
# 52     987        0.323402        5.163323        10       True         0.323402       
# 53     1051       0.535026        5.098215        5        True         0.535026       
# 54     1068       0.328831        5.086962        13       True         0.328831       
# 55     1390       0.465743        5.028237        5        True         0.465743       
# 56     1210       0.400380        4.960629        14       True         0.400380       
# 57     431        0.544880        4.957790        5        True         0.544880       
# 58     1481       0.382832        4.903587        15       True         0.382832       
# 59     913        0.405397        4.874551        14       True         0.405397       
# 60     1570       0.325924        4.698635        4        True         0.325924       
# 61     1047       0.678648        4.693585        5        True         0.678648       
# 62     1107       0.515431        4.688076        5        True         0.515431       
# 63     1370       0.385651        4.638890        8        True         0.385651       
# 64     1121       0.290610        4.466669        16       True         0.290610       
# 65     1646       0.288428        4.444200        7        True         0.288428       
# 66     1137       0.529097        4.416285        5        True         0.529097       
# 67     1414       0.353613        4.387353        11       True         0.353613       
# 68     955        0.518329        4.382893        5        True         0.518329       
# 69     1402       0.366289        4.208298        14       True         0.366289       
# 70     1669       0.355682        4.196655        4        True         0.355682       
# 71     1065       0.322106        4.172070        4        True         0.322106       
# 72     1175       0.322091        4.090971        4        True         0.322091       
# 73     247        0.357406        4.074426        17       True         0.357406       
# 74     1574       0.242985        4.011255        8        True         0.242985       
# 75     1662       0.482989        4.008951        5        True         0.482989       
# 76     1462       0.398835        3.997556        12       True         0.398835       
# 77     1249       0.359276        3.974803        6        True         0.359276       
# 78     522        0.470798        3.952058        5        True         0.470798       
# 79     1007       0.342157        3.908159        11       True         0.342157       
# 80     1094       0.510234        3.905600        5        True         0.510234       
# 81     1028       0.371203        3.897681        5        True         0.371203       
# 82     979        0.233844        3.874264        7        True         0.233844       
# 83     1259       0.364460        3.861059        8        True         0.364460       
# 84     677        0.439164        3.845614        5        True         0.439164       
# 85     1066       0.161665        3.798188        15       True         0.161665       
# 86     1139       0.262026        3.788690        20       True         0.262026       
# 87     1157       0.297197        3.746177        8        True         0.297197       
# 88     1186       0.496264        3.729156        5        True         0.496264       
# 89     1192       0.264755        3.711876        11       True         0.264755       
# 90     899        0.323462        3.678737        6        True         0.323462       
# 91     1400       0.337364        3.657486        7        True         0.337364       
# 92     1630       0.656840        3.656702        5        True         0.656840       
# 93     893        0.165144        3.650339        19       True         0.165143       
# 94     796        0.287710        3.647327        5        True         0.287710       
# 95     1205       0.270747        3.630237        11       True         0.270747       
# 96     891        0.368307        3.627009        4        True         0.368307       
# 97     1251       0.261519        3.587720        6        True         0.261519       
# 98     1677       0.231099        3.583343        12       True         0.231099       
# 99     679        0.417958        3.576257        7        True         0.417958       
# 100    402        0.253853        3.535027        8        True         0.253853       

# --------------------------------------------------------------------------------
# 【各维度最大误差统计】
# Dimension    Mean            Max             Std            
# --------------------------------------------------------------------------------
# X            0.470458        8.199321        0.887247       
# Y            0.396698        17.821464       0.874673       
# Z            0.602717        12.484756       1.323836       
# Yaw          0.042879        1.044632        0.074122       
# Pitch        0.080121        1.756921        0.146026       
# Roll         0.067299        3.398925        0.143918       
# Gripper      0.009282        1.983051        0.134708       
# ================================================================================