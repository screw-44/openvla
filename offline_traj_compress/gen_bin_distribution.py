"""
从 compression_results.json 中提取 control points 的 bin distribution
保存为 control_point_bin_distribution.npy，供 tokenizer 使用
仅处理 6 个维度：x, y, z, yaw, pitch, roll

使用 K-Means 聚类找到最优的 bin 中心，最小化重构误差
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans


def generate_control_point_bin_distribution(
    json_path: str = "compression_results.json",
    output_path: str = "control_point_bin_distribution.npy",
    n_bins: int = 512,
    n_dims: int = 6
):
    """
    从 compression_results.json 中提取所有 control points 的 bin distribution
    
    Args:
        json_path: compression_results.json 的路径
        output_path: 输出的 npy 文件路径
        n_bins: bin 数量（默认 512）
        n_dims: 维度数（默认 6，不包括 gripper）
    """
    json_path = Path(json_path)
    output_path = Path(output_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"❌ 文件不存在: {json_path}")
    
    print(f"📖 正在加载 {json_path}...")
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    episodes = results.get("episodes", {})
    print(f"✓ 加载了 {len(episodes)} 个 episodes")
    
    # Step 1: 收集前 n_dims 个维度的 control point 数据
    print(f"\n📊 正在收集 {n_dims} 个维度的 control points 数据...")
    all_data = [[] for _ in range(n_dims)]
    
    for ep_idx_str, ep_data in tqdm(episodes.items(), desc="收集中"):
        bspline = ep_data.get("bspline", {})
        control_points = bspline.get("control_points", [])
        
        if not control_points:
            continue
        
        for dim in range(min(n_dims, len(control_points))):
            all_data[dim].extend(control_points[dim])
    
    print(f"✓ 收集完成")
    
    # Step 2: 使用 K-Means 找到最优的 bin 中心
    print(f"\n🎯 使用 K-Means 计算最优的 {n_bins} 个 bin 中心...")
    edges = np.zeros((n_dims, n_bins + 1), dtype=np.float32)  # 保持兼容性，存储排序后的簇心 + 边界
    bin_centers_list = []  # 存储每个维度的簇心
    
    dim_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    
    for dim in range(n_dims):
        if len(all_data[dim]) > 0:
            data_array = np.array(all_data[dim], dtype=np.float32).reshape(-1, 1)
            
            # K-Means 聚类
            kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10, verbose=0)
            kmeans.fit(data_array)
            
            # 得到簇心并排序
            cluster_centers = np.sort(kmeans.cluster_centers_.flatten()).astype(np.float32)
            bin_centers_list.append(cluster_centers)
            
            # 构造边界：两个相邻簇心的中点
            # 前边界和后边界设置为簇心外侧
            min_val = cluster_centers[0]
            max_val = cluster_centers[-1]
            
            edges[dim, 0] = min_val - (cluster_centers[1] - cluster_centers[0]) / 2 if n_bins > 1 else min_val - 1
            edges[dim, -1] = max_val + (cluster_centers[-1] - cluster_centers[-2]) / 2 if n_bins > 1 else max_val + 1
            
            # 中间边界是相邻簇心的中点
            for i in range(1, n_bins):
                edges[dim, i] = (cluster_centers[i-1] + cluster_centers[i]) / 2.0
    
    # Step 3: 计算离散化误差
    print(f"\n📊 计算离散化误差...")
    print(f"{'维度':<8} {'数据点数':<12} {'范围':<35} {'MaxErr':<12} {'MeanErr':<12} {'StdErr':<12} {'MSE':<12}")
    print("-" * 110)
    
    quantization_errors = []
    for dim in range(n_dims):
        if len(all_data[dim]) > 0:
            data_array = np.array(all_data[dim], dtype=np.float32)
            cluster_centers = bin_centers_list[dim]
            
            # 找到每个数据点最近的簇心
            distances = np.abs(data_array[:, np.newaxis] - cluster_centers[np.newaxis, :])
            nearest_idx = np.argmin(distances, axis=1)
            quantized_values = cluster_centers[nearest_idx]
            
            # 计算误差
            errors = np.abs(data_array - quantized_values)
            max_err = np.max(errors)
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            mse = np.mean(errors ** 2)
            
            quantization_errors.append({
                'dim': dim_names[dim],
                'max': max_err,
                'mean': mean_err,
                'std': std_err,
                'mse': mse
            })
            
            range_str = f"[{data_array.min():10.4f}, {data_array.max():10.4f}]"
            n_points = len(all_data[dim])
            print(f"{dim_names[dim]:<8} {n_points:<12} {range_str:<35} {max_err:<12.6f} {mean_err:<12.6f} {std_err:<12.6f} {mse:<12.6f}")
    
    print(f"\n💡 K-Means 方法说明:")
    print(f"    - 每个维度独立使用 K-Means 聚类，得到 {n_bins} 个最优的簇心")
    print(f"    - 簇心作为量化的目标值，最小化重构均方误差 (MSE)")
    print(f"    - 相比分位数方法，可显著降低 max error 和 mean error")
    
    # Step 4: 保存 edges
    print(f"\n💾 保存到 {output_path}...")
    np.save(output_path, edges)
    print(f"✓ 完成！Edges 形状: {edges.shape}")
    
    return edges


if __name__ == "__main__":
    asset_json = Path(__file__).parent.parent / "assets" / "compression_results_v2.json"
    output_npy = Path(__file__).parent / "control_point_bin_distribution.npy"
    
    print(f"📁 JSON 路径: {asset_json}")
    print(f"📁 输出路径: {output_npy}\n")
    
    edges = generate_control_point_bin_distribution(
        json_path=str(asset_json),
        output_path=str(output_npy),
        n_bins=512,
        n_dims=6
    )
    
    print(f"\n✅ 完成！")


# 📊 计算离散化误差...
# 维度       数据点数         范围                                  非空bins     MaxErr       MeanErr      StdErr      
# ----------------------------------------------------------------------------------------------------
# x        41046        [  -50.1885,    52.8019]            512        14.345612    0.065102     0.665614    
# y        41046        [  -69.3363,    72.5234]            512        22.512955    0.084292     0.899412    
# z        41046        [  -98.6969,    39.2960]            512        20.918144    0.082390     0.824966    
# yaw      41046        [  -11.8946,    10.8398]            512        1.921101     0.012413     0.087433    
# pitch    41046        [  -13.9759,    17.5974]            512        3.791125     0.017898     0.164113    
# roll     41046        [  -24.8003,    24.5268]            512        4.152792     0.027434     0.176028    

# 🎯 使用 K-Means 计算最优的 512 个 bin 中心...

# 📊 计算离散化误差...
# 维度       数据点数         范围                                  MaxErr       MeanErr      StdErr       MSE         
# --------------------------------------------------------------------------------------------------------------
# x        41046        [  -50.1885,    52.8019]            0.248451     0.020642     0.014989     0.000651    
# y        41046        [  -69.3363,    72.5234]            0.350449     0.030767     0.021994     0.001430    
# z        41046        [  -98.6969,    39.2960]            0.361328     0.024287     0.019071     0.000954    
# yaw      41046        [  -11.8946,    10.8398]            0.062275     0.004020     0.003700     0.000030    
# pitch    41046        [  -13.9759,    17.5974]            0.079699     0.005720     0.004743     0.000055    
# roll     41046        [  -24.8003,    24.5268]            0.125874     0.010570     0.009616     0.000204    

# 💡 K-Means 方法说明:
#     - 每个维度独立使用 K-Means 聚类，得到 512 个最优的簇心
#     - 簇心作为量化的目标值，最小化重构均方误差 (MSE)
#     - 相比分位数方法，可显著降低 max error 和 mean error