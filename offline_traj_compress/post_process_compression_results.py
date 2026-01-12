"""
后处理脚本：修正 compression_results.json
1. 合并重复的 knot_vectors（6个维度相同）
2. 扩展 gripper control points 使其与其他维度长度一致
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

# 测试数据（episode 721）
TEST_DATA = {
    "721": {
        "episode_index": 721,
        "task_name": "put the bowl on the stove",
        "trajectory_length": 102,
        "bspline": {
            "control_points": [
                [0.11292475328308214, -0.2833614996061926, 7.447671906511874, 12.356048121326204, 12.06760457723716, 11.550427509623336, 9.313778446243974, 8.955251580052522, 9.025895911275343, 8.499949057366278, 8.736652533227671, 6.483466542115199, 1.774562750210055, 0.966966289255685, 1.6401776656142386, -0.6605823465858455, -0.6493822823232394, -0.6029064391683474, -0.6376325453443524, -0.6222879244348991],
                [0.12518503128627828, -0.807661210356401, 2.303104964370253, 3.138940331298607, 2.832433329858978, 2.829742730030092, 1.5527617038671238, 1.9469322261565998, 1.5431130547436958, 2.2381228466475642, 3.4375095340601995, 6.00443336640837, 11.443572232362898, 16.670174676625727, 19.31658334789332, 18.867161891321555, 18.812891978146048, 18.632355886244724, 18.75252491450917, 18.701221730486512],
                [0.05904046997451281, -0.2625783016372673, 0.5144838722379644, -0.7914696728423971, -5.668181637574171, -10.96302071447276, -16.292908198706698, -19.114401692897438, -18.92126697722447, -16.910679764884915, -11.176214914627558, -7.136777341991432, -5.864620695780417, -6.01437948950865, -9.656889270348104, -14.925725071438306, -16.451492148398756, -15.741745015577381, -15.172735588593453, -13.9963693133169],
                [-0.0050160725148838394, 0.048640977356163996, -0.2649001829110734, -0.2593925363830008, -0.24624753234235447, -0.2750997394762413, -0.20676078755410518, -0.41399846458718575, -0.7398410126510848, -1.0908717108959922, -1.1526004655712812, -1.0948976593050022, -1.181785773848413, -1.2302287636261753, -1.219103743469978, -1.0240547630849253, -1.0965530348735906, -1.0812984685759608, -1.1341175387590048, -1.1146815331608217],
                [0.008271551932795359, 0.023575034536022714, 0.02152122877313086, 0.08356926123798983, -0.0013945384123134185, -0.05730456115201746, -0.2819190006621692, -0.2657084736321945, -0.16059099604793817, -0.2160425734829739, -0.03512281235443722, 0.0003130660379489705, -0.0034444120285584824, -0.029194562654405766, 0.48316602778040757, 0.9979564270121721, 1.2162243926969782, 1.196872396122288, 1.1543226610053041, 1.2436163615196734],
                [0.005445333352511787, -0.02161705494532379, 0.0006682579104062303, -0.03639237145114296, 0.1415414035835977, 0.20477510234105584, 0.1565743962419917, 0.21112902624475038, 0.0461794377081243, -0.08596122418922084, -0.4079130416643006, -0.34297805541205406, -0.38055700621218724, -0.3583755175867595, -0.37758828742380557, -0.2270158858005327, -0.1951756849078393, -0.528848388300902, -0.5873684885845178, -0.7093585065063424],
                [-1.0, 1.0, -1.0]
            ],
            "knot_vectors": [
                [0, 0, 0, 0, 13, 19, 25, 28, 37, 40, 49, 55, 58, 64, 73, 79, 82, 88, 94, 97, 101, 101, 101, 101],
                [0, 0, 0, 0, 13, 19, 25, 28, 37, 40, 49, 55, 58, 64, 73, 79, 82, 88, 94, 97, 101, 101, 101, 101],
                [0, 0, 0, 0, 13, 19, 25, 28, 37, 40, 49, 55, 58, 64, 73, 79, 82, 88, 94, 97, 101, 101, 101, 101],
                [0, 0, 0, 0, 13, 19, 25, 28, 37, 40, 49, 55, 58, 64, 73, 79, 82, 88, 94, 97, 101, 101, 101, 101],
                [0, 0, 0, 0, 13, 19, 25, 28, 37, 40, 49, 55, 58, 64, 73, 79, 82, 88, 94, 97, 101, 101, 101, 101],
                [0, 0, 0, 0, 13, 19, 25, 28, 37, 40, 49, 55, 58, 64, 73, 79, 82, 88, 94, 97, 101, 101, 101, 101],
                [0, 40, 93, 101]
            ],
            "internal_knots": [13.0, 19.0, 25.0, 28.0, 37.0, 40.0, 49.0, 55.0, 58.0, 64.0, 73.0, 79.0, 82.0, 88.0, 94.0, 97.0],
            "forced_knots": [40.0, 93.0],
            "num_knots": 16,
            "overall_mean_error": 0.018200077387126576
        },
        "visualization_path": "test/visual_fig/episode_721.jpg"
    }
}


def expand_gripper_control_points(gripper_cp: List[float], shared_knot_vector: List[float], 
                                   gripper_knot_vector: List[float], degree: int = 3) -> List[float]:
    """
    扩展gripper control points使其与其他维度长度一致
    
    策略：gripper是分段常数函数，根据gripper_knot_vector确定状态变化
    B-spline控制点数量 = len(knot_vector) - degree - 1
    
    Args:
        gripper_cp: 原始gripper control points
        shared_knot_vector: 共享的完整knot vector（包含首尾重复）
        gripper_knot_vector: gripper维度的knot vector（用于确定状态变化点）
        degree: B-spline阶数（默认3）
        
    Returns:
        扩展后的gripper control points（长度与其他维度一致）
    """
    # 计算需要的控制点数量
    n_control_points = len(shared_knot_vector) - degree - 1
    
    # 从shared_knot_vector提取内部knots（去掉首尾重复）
    internal_knots = shared_knot_vector[degree+1 : -(degree+1)]
    
    # 特殊情况：如果gripper只有1个control point，说明整个trajectory保持不变
    if len(gripper_cp) == 1:
        return [gripper_cp[0]] * n_control_points
    
    # 从gripper_knot_vector提取状态变化点（去掉首尾）
    # gripper使用0阶B-spline，gripper_knot_vector的内部值就是状态变化点
    if len(gripper_knot_vector) > 2:
        gripper_transition_knots = gripper_knot_vector[1:-1]
    else:
        # 没有内部转换点，保持第一个值
        return [gripper_cp[0]] * n_control_points
    
    # 构建gripper的分段常数映射
    expanded_gripper = []
    
    # 对每个控制点位置，确定gripper值
    for i in range(n_control_points):
        if i < degree:
            # 前几个控制点：在起始区域
            knot_pos = shared_knot_vector[degree]  # 起始时间
        elif i >= len(internal_knots) + degree:
            # 后几个控制点：在结束区域
            knot_pos = shared_knot_vector[-(degree+1)]  # 结束时间
        else:
            # 中间控制点：对应内部knots
            knot_pos = internal_knots[i - degree]
        
        # 根据knot_pos在gripper_transition_knots中的位置确定gripper值
        # gripper_cp[i]对应区间 [gripper_transition_knots[i-1], gripper_transition_knots[i])
        idx = 0
        for transition_knot in gripper_transition_knots:
            if knot_pos >= transition_knot:
                idx += 1
            else:
                break
        
        # 确保idx不超出范围
        idx = min(idx, len(gripper_cp) - 1)
        gripper_val = gripper_cp[idx]
        
        expanded_gripper.append(gripper_val)
    
    return expanded_gripper


def post_process_episode(episode_data: Dict) -> Dict:
    """
    后处理单个episode数据
    1. 合并重复的knot_vectors
    2. 扩展gripper control points
    """
    bspline = episode_data["bspline"]
    
    # 获取数据
    control_points = bspline["control_points"]
    knot_vectors = bspline["knot_vectors"]
    forced_knots = bspline["forced_knots"]
    
    # 验证前6个knot_vectors是否相同
    shared_knot_vector = knot_vectors[0]
    for i in range(1, 6):
        assert knot_vectors[i] == shared_knot_vector, f"Knot vector {i} 与第0个不一致"
    
    # 扩展gripper control points
    gripper_cp_original = control_points[6]
    gripper_knot_vector = knot_vectors[6]  # 使用gripper维度的knot_vector
    gripper_cp_expanded = expand_gripper_control_points(
        gripper_cp_original, shared_knot_vector, gripper_knot_vector, degree=3
    )
    
    # 验证长度
    expected_len = len(control_points[0])
    assert len(gripper_cp_expanded) == expected_len, \
        f"扩展后gripper长度({len(gripper_cp_expanded)})与预期({expected_len})不一致"

    # 过滤一下gripper_cp，只保留-1或1，取更接近的
    gripper_cp_expanded = [1 if val >= 0 else -1   for val in gripper_cp_expanded]
    
    # 构建新的数据结构
    new_bspline = {
        "control_points": control_points[:6] + [gripper_cp_expanded],  # 更新gripper
        "knot_vector": shared_knot_vector,  # 共享的knot vector
        "num_knots": bspline["num_knots"],
        "overall_mean_error": bspline.get("overall_mean_error", 0.0)
    }
    
    # 保留errors_per_dim（如果存在）
    if "errors_per_dim" in bspline:
        new_bspline["errors_per_dim"] = bspline["errors_per_dim"]
    
    # 构建新的episode数据
    new_episode_data = {
        "episode_index": episode_data["episode_index"],
        "task_name": episode_data["task_name"],
        "trajectory_length": episode_data["trajectory_length"],
        "bspline": new_bspline,
        "visualization_path": episode_data.get("visualization_path", "")
    }
    
    # 保留status字段（如果存在）
    if "status" in episode_data:
        new_episode_data["status"] = episode_data["status"]
    
    return new_episode_data


def visualize_comparison(original_data: Dict, processed_data: Dict, save_path: str = "/tmp/post_process_test.png"):
    """
    可视化对比原始数据和处理后的数据
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    dim_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll", "Gripper"]
    
    orig_bspline = original_data["bspline"]
    proc_bspline = processed_data["bspline"]
    
    orig_cp = orig_bspline["control_points"]
    proc_cp = proc_bspline["control_points"]
    
    for i in range(7):
        ax = axes[i]
        
        # 原始control points
        orig_vals = orig_cp[i]
        proc_vals = proc_cp[i]
        
        # 绘制
        x_orig = np.arange(len(orig_vals))
        x_proc = np.arange(len(proc_vals))
        
        ax.plot(x_orig, orig_vals, 'o-', label='Original', markersize=6, linewidth=2, alpha=0.7)
        ax.plot(x_proc, proc_vals, 's--', label='Processed', markersize=5, linewidth=1.5, alpha=0.7)
        
        ax.set_title(f"{dim_names[i]}\nOrig: {len(orig_vals)} | Proc: {len(proc_vals)}", 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Control Point Index")
        ax.set_ylabel("Value")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # 最后一个子图显示统计信息
    ax_stats = axes[7]
    ax_stats.axis('off')
    
    stats_text = f"""后处理统计
    
原始数据:
  - Knot vectors: 7 个 (重复)
  - Gripper CP: {len(orig_cp[6])} 个
  
处理后:
  - Shared knot vector: 1 个
  - Gripper knot vector: 1 个
  - Gripper CP: {len(proc_cp[6])} 个 ✓
  
节省空间: ~85%
    """
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', 
                  facecolor='lightgreen', alpha=0.3))
    
    episode_idx = original_data["episode_index"]
    task_name = original_data["task_name"]
    fig.suptitle(f"Post-Process Comparison - Episode {episode_idx}\n{task_name}", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 可视化已保存: {save_path}")


def test_with_sample_data():
    """使用示例数据测试后处理逻辑"""
    print("="*80)
    print("测试模式：使用 Episode 721 示例数据")
    print("="*80)
    
    # 获取测试数据
    original_episode = TEST_DATA["721"]
    
    print(f"\n原始数据:")
    print(f"  Episode: {original_episode['episode_index']}")
    print(f"  Task: {original_episode['task_name']}")
    print(f"  Trajectory length: {original_episode['trajectory_length']}")
    
    orig_bspline = original_episode["bspline"]
    print(f"  Control points: {len(orig_bspline['control_points'])} 个维度")
    for i, cp in enumerate(orig_bspline['control_points']):
        print(f"    Dim {i}: {len(cp)} 个控制点")
    print(f"  Knot vectors: {len(orig_bspline['knot_vectors'])} 个")
    print(f"  Forced knots: {len(orig_bspline['forced_knots'])} 个")
    
    # 后处理
    print("\n" + "-"*80)
    print("正在后处理...")
    processed_episode = post_process_episode(original_episode)
    
    print(f"\n处理后数据:")
    proc_bspline = processed_episode["bspline"]
    print(f"  Control points: {len(proc_bspline['control_points'])} 个维度")
    for i, cp in enumerate(proc_bspline['control_points']):
        print(f"    Dim {i}: {len(cp)} 个控制点")
    print(f"  Shared knot vector: {len(proc_bspline['shared_knot_vector'])} 个元素")
    print(f"  Gripper knot vector: {len(proc_bspline['gripper_knot_vector'])} 个元素")
    
    # 验证
    print("\n" + "-"*80)
    print("验证:")
    
    # 1. 长度一致性
    cp_lengths = [len(cp) for cp in proc_bspline['control_points']]
    all_same = len(set(cp_lengths)) == 1
    print(f"  ✓ 所有维度control points长度一致: {all_same} (长度={cp_lengths[0]})")
    
    # 2. Gripper值的合理性
    gripper_cp = proc_bspline['control_points'][6]
    unique_vals = set(gripper_cp)
    print(f"  ✓ Gripper唯一值: {unique_vals} (原始: {set(orig_bspline['control_points'][6])})")
    
    # 3. 打印扩展后的gripper详情
    print(f"\n  扩展后的Gripper control points:")
    print(f"    {gripper_cp}")
    
    # 可视化对比
    print("\n" + "-"*80)
    print("生成可视化对比...")
    visualize_comparison(original_episode, processed_episode)
    
    # 保存处理后的数据用于检查
    test_output = {"episodes": {"721": processed_episode}}
    test_output_path = Path("/tmp/test_processed_episode_721.json")
    with open(test_output_path, 'w') as f:
        json.dump(test_output, f, indent=2)
    print(f"  ✓ 测试结果已保存: {test_output_path}")
    
    print("\n" + "="*80)
    print("✅ 测试完成！请检查可视化图片和JSON文件")
    print("="*80)
    
    return processed_episode


def process_full_json(input_path: str, output_path: str):
    """
    处理完整的 compression_results.json 文件
    """
    print("="*80)
    print(f"处理完整JSON文件: {input_path}")
    print("="*80)
    
    # 加载原始数据
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return
    
    print(f"正在加载 {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    episodes = data.get("episodes", {})
    compression_statics = data.get("compression_statics", {})

    print("data keys:", data.keys())
    
    print(f"  原始数据: {len(episodes)} 个 episodes")
    
    # 处理每个episode
    processed_episodes = {}
    failed_count = 0
    
    for ep_key, ep_data in episodes.items():
        try:
            processed_ep = post_process_episode(ep_data)
            processed_episodes[ep_key] = processed_ep
            
            if len(processed_episodes) % 100 == 0:
                print(f"  已处理: {len(processed_episodes)}/{len(episodes)}")
        except Exception as e:
            print(f"  ✗ Episode {ep_key} 处理失败: {e}")
            failed_count += 1
            # 保留原始数据
            processed_episodes[ep_key] = ep_data
    
    # 构建输出数据
    output_data = {
        "metadata": metadata,
        "episodes": processed_episodes,
        "compression_statics": compression_statics
    }
    
    # 保存
    output_path = Path(output_path)
    print(f"\n正在保存到 {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✅ 处理完成！")
    print(f"  成功: {len(processed_episodes) - failed_count}/{len(episodes)}")
    print(f"  失败: {failed_count}")
    print(f"  输出: {output_path}")
    print("="*80)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # 处理完整JSON
        input_file = sys.argv[2] if len(sys.argv) > 2 else "compression_results.json"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "compression_results_processed.json"
        process_full_json(input_file, output_file)
    else:
        # 测试模式
        print("提示: 运行完整处理请使用: python post_process_compression_results.py --full [input.json] [output.json]")
        print()
        test_with_sample_data()


if __name__ == "__main__":
    main()
