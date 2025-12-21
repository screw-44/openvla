"""

实现用b-spline对整段轨迹进行压缩，目前实现三种情况 
1. 不压缩，直接插值biningg成定长的点
2. b-spline最小二乘法压缩成定长的点
3. b-spline压缩成不定长的点，使用padding来进行计算

"""
import numpy as np
from scipy.interpolate import BSpline, splrep, splev, make_lsq_spline
from scipy.linalg import lstsq # Compute least-squares solution to equation Ax = b.
from typing import Tuple
from abc import ABC, abstractmethod

TRAJECTORY_COMPRESSION_REGISTRY = {}

def register_trajectory_compression(name: str):
    def decorator(cls):
        TRAJECTORY_COMPRESSION_REGISTRY[name] = cls
        return cls
    return decorator


class BaseTrajectoryCompression(ABC):
    """
        压缩轨迹的基类方法，子类需要实现具体的压缩逻辑。
        输入为一个trajectory，应该是points_num * dim的numpy数组，输出为压缩后的trajectory。
    """
    @abstractmethod
    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        pass

@register_trajectory_compression("action_chunk")
class ActionChunk(BaseTrajectoryCompression):
    """ 回退到普通vla模式(默认ac长度为1,测试一下openvla的原始设定) """
    def __init__(self, action_chunk_size=1):
        self.action_chunk_size = action_chunk_size
        self.exp_type = "action_chunk"

    def __call__(self, trajectory):
        # 确保 trajectory 是 numpy 数组
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        # 如果输入是一维（单个action），转换为二维 [1, n_dims]
        if trajectory.ndim == 1: trajectory = trajectory.reshape(1, -1)
        
        return trajectory[:self.action_chunk_size]  # 截断到指定大小

@register_trajectory_compression("bining")
class BiningTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 50):
        self.exp_type = "bining"
        self.target_length = target_length

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        original_length = trajectory.shape[0]
        
        # 如果原始长度不足目标长度，先填充（与 action_chunk 保持一致）
        if original_length < self.target_length:
            # 重复最后一个点来填充
            last_point = trajectory[-1:]
            num_padding = self.target_length - original_length
            padding = np.repeat(last_point, num_padding, axis=0)
            trajectory = np.vstack([trajectory, padding])
            return trajectory
        
        # 如果长度足够，均匀采样
        indices = np.linspace(0, original_length - 1, num=self.target_length)
        compressed_trajectory = np.array([trajectory[int(idx)] for idx in indices])
        return compressed_trajectory

# 拟合方法，加权最小二乘, s这个参数代表是否平滑拟合
# scipy.interpolate.splrep(x, y, s=0, k=3) 名称：Spline representation, 输出：三元组 (t(knots), c(control), k)
# scipy.interpolate.splev(x_new, tck)  名称：Spline evaluation （这两个只是插值）
# scipy.interpolate.make_lsq_spline(x, y, t, k) 名称：Least-squares B-spline representation（真正的拟合函数）
@register_trajectory_compression("uniform_bspline")
class UniformBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 20, degree: int = 3):
        """
        固定点数的B样条压缩：用固定数量的内部节点来拟合B样条，通过最小二乘法优化控制点。
        采用时间参数化，控制点均匀分布在轨迹上。（模型不需要预测x轴）
        Args:
            target_length: 内部节点数量（控制点数量 = target_length + degree + 1）
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.target_length = target_length
        self.degree = degree
        self.exp_type = "uniform_bspline"

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        使用固定数量的内部节点进行B样条最小二乘拟合。
        
        工作流程：
        1. 在整条轨迹上均匀分布 target_length 个内部节点
        2. 使用 make_lsq_spline 拟合B样条（自动优化控制点的y值）
        3. 返回优化得到的控制点（不在原始曲线上）
        
        Args:
            trajectory: [points_num, dim] 的numpy数组
        
        Returns:
            control_points: [n_control_points, dim] 的numpy数组，n = target_length + degree + 1
            interior_knots: [target_length] 内部节点位置
        """

        original_length = trajectory.shape[0]
        dim = trajectory.shape[1] if len(trajectory.shape) > 1 else 1

        # B样条约束：数据点数量必须 >= 控制点数量
        # 控制点数量 = n_interior_knots + degree + 1
        n_control_points = self.target_length + self.degree + 1
        
        # 如果原始长度不足控制点数量，先填充
        if original_length < n_control_points:
            # 重复最后一个点来填充
            last_point = trajectory[-1:]
            num_padding = n_control_points - original_length
            padding = np.repeat(last_point, num_padding, axis=0)
            trajectory = np.vstack([trajectory, padding])
            original_length = trajectory.shape[0]  # 更新长度
        
        # 时间参数化
        x_data = np.arange(original_length, dtype=float)
        x_min, x_max = 0.0, float(original_length - 1)
        
        # 在轨迹上均匀分布内部节点（排除端点）
        # 因为是时间上均匀分布，interior_knots为均匀分布的时间点。
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]
        
        # 构造完整的knot vector
        knot_vector = np.concatenate([
            np.repeat(x_min, self.degree + 1),      # 起始重复节点
            interior_knots,                          # 内部节点
            np.repeat(x_max, self.degree + 1)       # 结束重复节点
        ])
        
        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []
        
        for d in range(dim):
            # 使用最小二乘法拟合B样条
            # make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(x_data, trajectory[:, d], knot_vector, k=self.degree)
            all_control_points.append(bspline.c)
        
        # 将控制点组合成 [n_control_points, dim] 的数组
        control_points = np.column_stack(all_control_points)

        # print("Uniform B-spline compression done.")
        # print("control points:", control_points.shape)
        # # 可视化的时候，左右还需要重复一次，但是可以不预测interior_knots了
        # print("interior knots:", interior_knots.shape) 
        return control_points
    
    def get_visualization_points(self, compressed_output: np.ndarray, original_trajectory: np.ndarray, num_points: int = 100):
        """
        获取用于可视化的B-spline重构点和对应的时间坐标
        
        Args:
            compressed_output: __call__() 返回的控制点 [n_control_points, dim]
            original_trajectory: 原始轨迹 [T, dim]，用于确定时间范围
            num_points: 重构的点数（默认100，用于平滑可视化）
        
        Returns:
            x_coords: 时间坐标 [num_points]
            trajectory_points: 重构的轨迹点 [num_points, dim]
        """
        control_points = compressed_output
        original_length = len(original_trajectory)
        dim = original_trajectory.shape[1]
        
        # 时间参数化
        x_min, x_max = 0.0, float(original_length - 1)
        
        # 重构 interior_knots（与 __call__ 中相同）
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]
        
        # 构造完整的 knot vector
        knot_vector = np.concatenate([
            np.repeat(x_min, self.degree + 1),
            interior_knots,
            np.repeat(x_max, self.degree + 1)
        ])
        
        # 生成均匀的评估点
        x_coords = np.linspace(x_min, x_max, num_points)
        
        # 对每个维度重构 B-spline 并评估
        trajectory_points = np.zeros((num_points, dim))
        for d in range(dim):
            bspline = BSpline(knot_vector, control_points[:, d], self.degree)
            trajectory_points[:, d] = bspline(x_coords)
        
        return x_coords, trajectory_points


# ==============================================================================
# 这里开始是fixed freq的压缩方法（实验区域
# ==============================================================================

@register_trajectory_compression("fix_freq_bining")
class FixFreqBiningTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 60):
        self.target_length = target_length
        self.exp_type = "fix_freq"
    
    def __call__(self, full_trajectory: np.ndarray, frame_percentage: float = 0.0) -> np.ndarray:
        """
        固定频率的bining压缩，支持从指定百分比位置开始截断。
        
        Args:
            full_trajectory: [points_num, dim] 的numpy数组
            frame_percentage: 起始位置百分比 [0.0, 1.0]，0.0 表示从开头开始
        
        Returns:
            压缩后的轨迹 [variable_length, dim]（变长，配合EOS token）
        """
        original_length = full_trajectory.shape[0]
        
        # NOTE: 12.17 均匀采样（如果原始长度不足target_length，采样到实际长度）
        actual_length = min(original_length, self.target_length)
        indices = np.linspace(0, original_length - 1, num=actual_length)
        sampled_trajectory = np.array([full_trajectory[int(idx)] for idx in indices])
        
        # NOTE: 12.17 根据 frame_percentage 截断，不再padding（使用EOS token表示结束）
        start_idx = int(actual_length * frame_percentage)
        start_idx = min(start_idx, actual_length - 1)  # 确保不超出范围
        compressed_trajectory = sampled_trajectory[start_idx:]
        
        return compressed_trajectory

# 拟合方法，加权最小二乘, s这个参数代表是否平滑拟合
# scipy.interpolate.splrep(x, y, s=0, k=3) 名称：Spline representation, 输出：三元组 (t(knots), c(control), k)
# scipy.interpolate.splev(x_new, tck)  名称：Spline evaluation （这两个只是插值）
# scipy.interpolate.make_lsq_spline(x, y, t, k) 名称：Least-squares B-spline representation（真正的拟合函数）
@register_trajectory_compression("fix_freq_uniform_bspline")
class FixFreqUniformBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 30, degree: int = 3):
        """
        固定点数和频率的B样条压缩：用固定数量的内部节点来拟合B样条，通过最小二乘法优化控制点。
        采用时间参数化，控制点均匀分布在轨迹上。（模型不需要预测x轴）
        Args:
            target_length: 内部节点数量（控制点数量 = target_length + degree + 1）
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.exp_type = "fix_freq"
        self.target_length = target_length
        self.degree = degree

    def __call__(self, full_trajectory: np.ndarray, frame_percentage: float = 0.0) -> np.ndarray:
        """
        使用固定数量的内部节点进行B样条最小二乘拟合，支持从指定百分比位置开始截断。
        
        工作流程：
        1. 在整条轨迹上均匀分布 target_length 个内部节点
        2. 使用 make_lsq_spline 拟合B样条（自动优化控制点的y值）
        3. 根据 frame_percentage 截断控制点
        4. 返回截断后的控制点
        
        Args:
            trajectory: [points_num, dim] 的numpy数组
            frame_percentage: 起始位置百分比 [0.0, 1.0]，0.0 表示从开头开始
        
        Returns:
            control_points: 截断后的控制点 [n_control_points, dim]
        """

        original_length = full_trajectory.shape[0]
        dim = full_trajectory.shape[1] if len(full_trajectory.shape) > 1 else 1

        # B样条约束：数据点数量必须 >= 控制点数量
        # 控制点数量 = n_interior_knots + degree + 1
        n_control_points = self.target_length + self.degree + 1
        
        # 如果原始长度不足控制点数量，先填充
        if original_length < n_control_points:
            # 重复最后一个点来填充
            last_point = full_trajectory[-1:]
            num_padding = n_control_points - original_length
            padding = np.repeat(last_point, num_padding, axis=0)
            full_trajectory = np.vstack([full_trajectory, padding])
            original_length = full_trajectory.shape[0]  # 更新长度
        
        # 时间参数化
        x_data = np.arange(original_length, dtype=float)
        x_min, x_max = 0.0, float(original_length - 1)
        
        # 在轨迹上均匀分布内部节点（排除端点）
        # 因为是时间上均匀分布，interior_knots为均匀分布的时间点。
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]
        
        # 构造完整的knot vector
        knot_vector = np.concatenate([
            np.repeat(x_min, self.degree + 1),      # 起始重复节点
            interior_knots,                          # 内部节点
            np.repeat(x_max, self.degree + 1)       # 结束重复节点
        ])
        
        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []
        
        for d in range(dim):
            # 使用最小二乘法拟合B样条
            # make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(x_data, full_trajectory[:, d], knot_vector, k=self.degree)
            all_control_points.append(bspline.c)
        
        # 将控制点组合成 [n_control_points, dim] 的数组
        control_points = np.column_stack(all_control_points)
        
        # NOTE: 12.17 根据 frame_percentage 截断控制点，不再padding（使用EOS token表示结束）
        start_idx = int(n_control_points * frame_percentage)
        start_idx = min(start_idx, n_control_points - 1)  # 确保不超出范围
        control_points = control_points[start_idx:]
        
        return control_points
    
    def get_visualization_points(self, compressed_output: np.ndarray, original_trajectory: np.ndarray, num_points: int = 100):
        """
        获取用于可视化的B-spline重构点和对应的时间坐标
        
        Args:
            compressed_output: __call__() 返回的控制点 [n_control_points, dim]
            original_trajectory: 原始轨迹 [T, dim]，用于确定时间范围
            num_points: 重构的点数（默认100，用于平滑可视化）
        
        Returns:
            x_coords: 时间坐标 [num_points]
            trajectory_points: 重构的轨迹点 [num_points, dim]
        """
        control_points = compressed_output
        original_length = len(original_trajectory)
        dim = original_trajectory.shape[1]
        
        # 时间参数化
        x_min, x_max = 0.0, float(original_length - 1)
        
        # 重构 interior_knots（与 __call__ 中相同）
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]
        
        # 构造完整的 knot vector
        knot_vector = np.concatenate([
            np.repeat(x_min, self.degree + 1),
            interior_knots,
            np.repeat(x_max, self.degree + 1)
        ])
        
        # 生成均匀的评估点
        x_coords = np.linspace(x_min, x_max, num_points)
        
        # 对每个维度重构 B-spline 并评估
        trajectory_points = np.zeros((num_points, dim))
        for d in range(dim):
            bspline = BSpline(knot_vector, control_points[:, d], self.degree)
            trajectory_points[:, d] = bspline(x_coords)
        
        return x_coords, trajectory_points


# ==============================================================================
# 这里开始是积分轨迹（positional）的压缩方法（实验区域
# ==============================================================================


@register_trajectory_compression("positional_bining")
class PositionalBiningTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 50):
        self.target_length = target_length
        self.exp_type = "positional"

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        # 对轨迹进行累积和操作（将差分转换为绝对位置）
        trajectory = np.cumsum(trajectory, axis=0)
        
        original_length = trajectory.shape[0]
        
        # 如果原始长度不足目标长度，先填充（与 action_chunk 保持一致）
        if original_length < self.target_length:
            # 重复最后一个点来填充
            last_point = trajectory[-1:]
            num_padding = self.target_length - original_length
            padding = np.repeat(last_point, num_padding, axis=0)
            trajectory = np.vstack([trajectory, padding])
            return trajectory
        
        # 如果长度足够，均匀采样
        indices = np.linspace(0, original_length - 1, num=self.target_length)
        compressed_trajectory = np.array([trajectory[int(idx)] for idx in indices])
        return compressed_trajectory

# 拟合方法，加权最小二乘, s这个参数代表是否平滑拟合
# scipy.interpolate.splrep(x, y, s=0, k=3) 名称：Spline representation, 输出：三元组 (t(knots), c(control), k)
# scipy.interpolate.splev(x_new, tck)  名称：Spline evaluation （这两个只是插值）
# scipy.interpolate.make_lsq_spline(x, y, t, k) 名称：Least-squares B-spline representation（真正的拟合函数）
@register_trajectory_compression("positional_uniform_bspline")
class PositionalUniformBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 20, degree: int = 3):
        """
        固定点数的B样条压缩：用固定数量的内部节点来拟合B样条，通过最小二乘法优化控制点。
        采用时间参数化，控制点均匀分布在轨迹上。（模型不需要预测x轴）
        Args:
            target_length: 内部节点数量（控制点数量 = target_length + degree + 1）
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.target_length = target_length
        self.degree = degree
        self.exp_type = "positional"

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        使用固定数量的内部节点进行B样条最小二乘拟合。
        
        工作流程：
        1. 在整条轨迹上均匀分布 target_length 个内部节点
        2. 使用 make_lsq_spline 拟合B样条（自动优化控制点的y值）
        3. 返回优化得到的控制点（不在原始曲线上）
        
        Args:
            trajectory: [points_num, dim] 的numpy数组
        
        Returns:
            control_points: [n_control_points, dim] 的numpy数组，n = target_length + degree + 1
            interior_knots: [target_length] 内部节点位置
        """
        # 对轨迹进行累积和操作（将差分转换为绝对位置）
        trajectory = np.cumsum(trajectory, axis=0)

        original_length = trajectory.shape[0]
        dim = trajectory.shape[1] if len(trajectory.shape) > 1 else 1

        # B样条约束：数据点数量必须 >= 控制点数量
        # 控制点数量 = n_interior_knots + degree + 1
        n_control_points = self.target_length + self.degree + 1
        
        # 如果原始长度不足控制点数量，先填充
        if original_length < n_control_points:
            # 重复最后一个点来填充
            last_point = trajectory[-1:]
            num_padding = n_control_points - original_length
            padding = np.repeat(last_point, num_padding, axis=0)
            trajectory = np.vstack([trajectory, padding])
            original_length = trajectory.shape[0]  # 更新长度
        
        # 时间参数化
        x_data = np.arange(original_length, dtype=float)
        x_min, x_max = 0.0, float(original_length - 1)
        
        # 在轨迹上均匀分布内部节点（排除端点）
        # 因为是时间上均匀分布，interior_knots为均匀分布的时间点。
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]
        
        # 构造完整的knot vector
        knot_vector = np.concatenate([
            np.repeat(x_min, self.degree + 1),      # 起始重复节点
            interior_knots,                          # 内部节点
            np.repeat(x_max, self.degree + 1)       # 结束重复节点
        ])
        
        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []
        
        for d in range(dim):
            # 使用最小二乘法拟合B样条
            # make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(x_data, trajectory[:, d], knot_vector, k=self.degree)
            all_control_points.append(bspline.c)
        
        # 将控制点组合成 [n_control_points, dim] 的数组
        control_points = np.column_stack(all_control_points)

        # print("Uniform B-spline compression done.")
        # print("control points:", control_points.shape)
        # # 可视化的时候，左右还需要重复一次，但是可以不预测interior_knots了
        # print("interior knots:", interior_knots.shape) 
        return control_points
    
    def get_visualization_points(self, compressed_output: np.ndarray, original_trajectory: np.ndarray, num_points: int = 100):
        """
        获取用于可视化的B-spline重构点和对应的时间坐标
        
        Args:
            compressed_output: __call__() 返回的控制点 [n_control_points, dim]
            original_trajectory: 原始轨迹 [T, dim]，用于确定时间范围
            num_points: 重构的点数（默认100，用于平滑可视化）
        
        Returns:
            x_coords: 时间坐标 [num_points]
            trajectory_points: 重构的轨迹点 [num_points, dim]
        """
        control_points = compressed_output
        original_length = len(original_trajectory)
        dim = original_trajectory.shape[1]
        
        # 时间参数化
        x_min, x_max = 0.0, float(original_length - 1)
        
        # 重构 interior_knots（与 __call__ 中相同）
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]
        
        # 构造完整的 knot vector
        knot_vector = np.concatenate([
            np.repeat(x_min, self.degree + 1),
            interior_knots,
            np.repeat(x_max, self.degree + 1)
        ])
        
        # 生成均匀的评估点
        x_coords = np.linspace(x_min, x_max, num_points)
        
        # 对每个维度重构 B-spline 并评估
        trajectory_points = np.zeros((num_points, dim))
        for d in range(dim):
            bspline = BSpline(knot_vector, control_points[:, d], self.degree)
            trajectory_points[:, d] = bspline(x_coords)
        
        return x_coords, trajectory_points



















# Adapative 还没有准备好，先在普通方法上弄好再弄


@register_trajectory_compression("adaptive_bspline")
class AdaptiveBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, max_length: int = 50, max_iteration: int = 100, max_error: float = 0.01, degree: int = 3,
                 improvement_epsilon: float = 1e-4, min_knot_spacing: float = 0.02):
        """
        自适应B样条压缩：动态增加内部节点直到满足误差要求。
        使用*弦长参数化*，放弃之间的时间参数化（时间参数化的效果较差，之前beast相当于是时间参数化）。
        模型需要预测x轴的位置。

        关键概念：
        - 内部节点（interior knots）: 决定B样条的灵活性，更多节点 = 更灵活的曲线
        - 控制点（control points）: 通过最小二乘法优化得到，数量 = n_interior_knots + degree + 1
        - 控制点不在原始曲线上，它们的位置是为了最小化拟合误差而优化的
        
        Args:
            max_length: 最大内部节点数量
            max_error: 允许的最大重建误差
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.max_iteration = max_iteration
        self.max_length = max_length
        self.max_error = max_error
        self.degree = degree
        self.improvement_epsilon = improvement_epsilon
        self.min_knot_spacing = min_knot_spacing

    def fit_bspline_scipy(self, points: np.ndarray, n_control: int) -> np.ndarray:
        """
        使用scipy拟合B-spline
        
        Parameters:
        -----------
        points : np.ndarray 数据点 (N, D)
        n_control : int 期望的控制点数量
            
        Returns:
        --------
        knots : np.ndarray ： 结点向量
        control_points : np.ndarray ： 控制点
        """
        # 参数化（累积弦长）
        n_points = len(points)
        assert n_control >= self.degree + 1 and n_points >= 2, "控制点数量必须至少为阶次+1，且数据点数量必须至少为2, 数据点数量: {}".format(n_points)
        
        # 参数化u，映射到[0,1]
        # 把第 1…N 个数据点按“走过的弧长”均匀映射到 [0,1]，像给每个点一个时间戳。代码用的是弦长参数化（chord-length）：
        diffs = np.diff(points, axis=0) # 算数组中相邻元素的差值
        chord_lengths = np.linalg.norm(diffs, axis=1) # 算出矩阵的 2 范数（欧几里得范数）
        cumulative = np.concatenate([[0], np.cumsum(chord_lengths)])
        u = cumulative / cumulative[-1]
        
        # 确定内部结点数量,B-spline的定义
        n_interior_knots = max(0, n_control - self.degree - 1)
        
        # 使用scipy的make_interp_spline
        if n_interior_knots <= 0:
            # 最简单情况：只有边界结点
            t = np.concatenate([
                [0] * (self.degree + 1),
                [1] * (self.degree + 1)
            ])
        else:
            # 均匀分布内部结点。结点的数量比控制点多，因为包括边界结点。
            interior = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
            t = np.concatenate([
                [0] * (self.degree + 1),
                interior,
                [1] * (self.degree + 1)
            ])
        
        # 构建基函数矩阵并最小二乘拟合
        n_control_actual = len(t) - self.degree - 1
        
        # 构建基函数矩阵
        N = np.zeros((n_points, n_control_actual))
        for i in range(n_control_actual):
            # 单位基函数
            c = np.zeros(n_control_actual)
            c[i] = 1.0 # 第 i 个基函数为 1，其余为 0
            bspl = BSpline(t, c, self.degree) 
            N[:, i] = bspl(u) # 评估一列 N[:, i] = bspl(u)
        
        # 最小二乘拟合
        control_points, _, _, _ = lstsq(N, points)
        
        return t, control_points

    def __call__(self, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        自适应选择内部节点位置并优化控制点以满足最大误差要求。
        
        工作流程：
        1. 从少量内部节点开始
        2. 使用make_lsq_spline拟合B样条（自动计算最优控制点）
        3. 找到重建误差最大的位置，在那里添加新的内部节点
        4. 重复步骤2-3，直到误差满足要求或达到最大节点数
        
        创新：
        1. chord-length参数化，效果更好
        2. 二分法插入控制点和knots，避免过于密集

        Args:
            trajectory: [points_num, dim] 的numpy数组
        
        Returns:
            control_with_knots: [n_control_points, dim+1] 拟合得到的控制点, 以及对应的节点位置  

        """
        self.original_curve = trajectory.copy()
        n_points, dim = trajectory.shape
        # 弦长参数化到u域
        diffs = np.diff(trajectory, axis=0)
        chord_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(chord_lengths)])
        total = cumulative[-1] if cumulative[-1] > 0 else 1.0
        u_data = cumulative / total

        # 从无内部结点开始，仅端点重复
        interior_u = []
        best_control = None
        best_knots = None
        best_max_err = float('inf')  # Track best error
        prev_max_err = None

        for it in range(self.max_iteration):
            # 构造完整knot向量
            interior_arr = np.array(interior_u, dtype=float) if len(interior_u) else np.array([])
            knots = np.concatenate([
                np.zeros(self.degree + 1),
                interior_arr,
                np.ones(self.degree + 1)
            ])

            # 最小二乘拟合并重建
            reconstructed = np.zeros_like(trajectory)
            all_c = []
            for d in range(dim):
                bs = make_lsq_spline(u_data, trajectory[:, d], knots, k=self.degree)
                all_c.append(bs.c)
                reconstructed[:, d] = bs(u_data)

            control_points = np.column_stack(all_c)
            errors = np.linalg.norm(trajectory - reconstructed, axis=1)
            max_err = float(np.max(errors))
            
            print(f"[Adaptive] Iteration {it}: interior_knots={len(interior_u)}, control_pts={len(control_points)}, max_err={max_err:.6f}")

            # 保留最佳结果
            if max_err < best_max_err:
                best_control = control_points.copy()
                best_knots = knots.copy()
                best_max_err = max_err
                print(f"[Adaptive]   -> New best! Saving state with error={max_err:.6f}")

            # 停止条件：误差阈值
            if max_err <= self.max_error:
                print(f"[Adaptive] Stopped: max_err ({max_err:.6f}) <= threshold ({self.max_error})")
                break

            # 改进不足（至少添加过3个结点以后）
            # if len(interior_u) >= 3 and prev_max_err is not None and (prev_max_err - max_err) < self.improvement_epsilon:
            #     print(f"[Adaptive] Stopped: improvement ({prev_max_err - max_err:.6f}) < epsilon ({self.improvement_epsilon})")
            #     break
            # prev_max_err = max_err

            # 达到最大内部结点数
            if len(interior_u) >= self.max_length:
                print(f"[Adaptive] Stopped: reached max_length ({self.max_length})")
                break

            # 新策略：在包含最大误差点的两个已有节点之间插入新节点（二分法）
            worst_idx = int(np.argmax(errors))
            u_worst = float(u_data[worst_idx])
            
            # 构建完整的节点列表（包含边界0和1）
            all_knots = np.concatenate([[0.0], interior_u, [1.0]])
            
            # 找到包含u_worst的区间
            # u_worst在哪两个节点之间
            interval_idx = np.searchsorted(all_knots, u_worst) - 1
            interval_idx = max(0, min(interval_idx, len(all_knots) - 2))
            
            u_left = all_knots[interval_idx]
            u_right = all_knots[interval_idx + 1]
            
            # 在区间中点插入新节点
            u_new = (u_left + u_right) / 2.0
            
            # 检查是否满足最小间距要求
            if len(interior_u) == 0:
                # 第一个内部节点，直接插入
                interior_u.append(u_new)
                interior_u.sort()
                print(f"[Adaptive]   -> Inserted knot at u={u_new:.4f} (midpoint of [{u_left:.4f}, {u_right:.4f}])")
            else:
                # 检查与已有节点的最小距离
                min_dist = np.min(np.abs(np.array(interior_u) - u_new))
                if min_dist >= self.min_knot_spacing:
                    interior_u.append(u_new)
                    interior_u.sort()
                    print(f"[Adaptive]   -> Inserted knot at u={u_new:.4f} (midpoint of [{u_left:.4f}, {u_right:.4f}])")
                else:
                    # 如果中点不满足间距要求，尝试寻找其他可插入的区间
                    print(f"[Adaptive]   -> Midpoint u={u_new:.4f} too close to existing knots (min_dist={min_dist:.4f})")
                    
                    # 按区间大小排序，尝试在最大的可用区间中插入
                    inserted = False
                    for i in range(len(all_knots) - 1):
                        u_l = all_knots[i]
                        u_r = all_knots[i + 1]
                        u_mid = (u_l + u_r) / 2.0
                        
                        # 跳过边界节点
                        if u_mid <= 0.0 + 1e-9 or u_mid >= 1.0 - 1e-9:
                            continue
                        
                        # 检查间距
                        if len(interior_u) == 0 or np.min(np.abs(np.array(interior_u) - u_mid)) >= self.min_knot_spacing:
                            interior_u.append(u_mid)
                            interior_u.sort()
                            print(f"[Adaptive]   -> Inserted knot at u={u_mid:.4f} (alternative interval [{u_l:.4f}, {u_r:.4f}])")
                            inserted = True
                            break
                    
                    if not inserted:
                        print(f"[Adaptive]   -> No valid interval found for insertion, stopping.")
                        break

        # 最终返回：将interior_knots保存在控制点矩阵中
        if best_control is None or best_knots is None:
            # 回退到最后一次拟合
            best_control = control_points
            best_knots = knots
        
        # 从best_knots中提取interior_knots
        # best_knots结构: [0, 0, 0, 0, interior_knots..., 1, 1, 1, 1] (degree=3)
        interior_knots = best_knots[self.degree + 1: -(self.degree + 1)]
        
        # 创建扩展矩阵：[n_control_points, dim+1]
        # 前dim列是控制点坐标，最后一列存储interior_knots（用0填充到相同长度）
        n_control = len(best_control)
        n_interior = len(interior_knots)
        
        # 最后一列：前n_interior个位置存储interior_knots，其余用0填充
        last_column = np.zeros(n_control)
        last_column[:n_interior] = interior_knots
        
        # 合并：[n_control_points, dim+1]
        control_with_knots = np.column_stack([best_control, last_column])
        
        return control_with_knots
    
    def get_visualization_points(self, compressed_output: np.ndarray, original_trajectory: np.ndarray, num_points: int = 100):
        """
        获取用于可视化的B-spline重构点和对应的时间坐标
        
        对于 adaptive_bspline，需要将弦长参数化的 u 值映射回原始时间索引
        
        Args:
            compressed_output: __call__() 返回的 [n_control_points, dim+1]
                前 dim 列是控制点，最后一列是 interior_knots (u 值 [0,1])
            original_trajectory: 原始轨迹 [T, dim]，用于弦长参数化
            num_points: 重构的点数（默认100，用于平滑可视化）
        
        Returns:
            x_coords: 时间坐标 [num_points]（映射回原始时间索引）
            trajectory_points: 重构的轨迹点 [num_points, dim]
        """
        # 分离控制点和 interior_knots
        control_points = compressed_output[:, :-1]  # [n_control, dim]
        interior_knots_u = compressed_output[:, -1]  # [n_control]
        
        # 过滤出非零的 interior_knots（因为有0填充）
        interior_knots_u = interior_knots_u[interior_knots_u > 0]
        
        dim = control_points.shape[1]
        original_length = len(original_trajectory)
        
        # 构造完整的 knot vector（u 域 [0,1]）
        knot_vector = np.concatenate([
            np.zeros(self.degree + 1),
            interior_knots_u,
            np.ones(self.degree + 1)
        ])
        
        # 在 u 域生成均匀的评估点
        u_eval = np.linspace(0, 1, num_points)
        
        # 对每个维度重构 B-spline 并评估
        trajectory_points = np.zeros((num_points, dim))
        for d in range(dim):
            bspline = BSpline(knot_vector, control_points[:, d], self.degree)
            trajectory_points[:, d] = bspline(u_eval)
        
        # 将 u 值映射回原始时间索引
        # 使用弦长参数化的反向映射
        diffs = np.diff(original_trajectory, axis=0)
        chord_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(chord_lengths)])
        total_length = cumulative[-1] if cumulative[-1] > 0 else 1.0
        u_original = cumulative / total_length
        
        # 通过插值将 u_eval 映射回时间索引
        x_coords = np.interp(u_eval, u_original, np.arange(original_length))
        
        return x_coords, trajectory_points


# ============================================================================
# Main Function for Trajectory Compression Visualization
# ============================================================================

if __name__ == "__main__":
    """
    可视化和对比不同轨迹压缩方法的效果
    
    功能：
    1. 从 Libero 数据集中加载机器人动作轨迹
    2. 使用4种压缩方法进行压缩：action_chunk, bining, uniform_bspline, adaptive_bspline
    3. 可视化对比压缩前后的轨迹、压缩率、重构误差等
    4. 保存可视化结果到指定目录
    
    使用方式：
        python trajectory_compression.py --task_ids 0 1 2 --num_episodes 5 --output_dir ./compression_results
    """
    import argparse
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys
    import logging
    
    # 添加项目路径以便导入模块
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from prismatic.vla.tokenizer import DummyTokenizer
    from prismatic.vla.dataset import MyLeRobotDataset
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # ========================================================================
    # 解析命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(description="轨迹压缩方法对比可视化")
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="要加载的 Libero 任务 ID 列表 (默认: 0 1 2)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="每个任务加载的 episode 数量 (默认: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trajectory_compression_results",
        help="保存可视化结果的目录 (默认: ./trajectory_compression_results)"
    )
    parser.add_argument(
        "--compression_methods",
        type=str,
        nargs="+",
        default=["action_chunk", "bining", "uniform_bspline", "adaptive_bspline"],
        choices=["action_chunk", "bining", "uniform_bspline", "adaptive_bspline"],
        help="要测试的压缩方法 (默认: 全部)"
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=20,
        help="目标压缩长度 (默认: 20)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="HuggingFaceVLA/libero",
        help="数据集 repo ID (默认: HuggingFaceVLA/libero)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/inspire/hdd/project/robot-decision/public/datasets/",
        help="数据集根目录 (默认: /inspire/hdd/project/robot-decision/public/datasets/)"
    )
    parser.add_argument(
        "--visualization_mode",
        type=str,
        default="single_episode",
        choices=["single_episode", "multi_episode_per_frame"],
        help="可视化模式: single_episode (单个episode多个frame) 或 multi_episode_per_frame (多个episode的同一frame)"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # ========================================================================
    # 辅助函数：可视化单个轨迹的压缩对比
    # ========================================================================
    def visualize_trajectory_comparison(
        original_trajectory: np.ndarray,
        compressed_trajectories: dict,
        episode_idx: int,
        frame_idx: int,
        output_path: Path,
        compressors: dict
    ):
        """
        可视化原始轨迹与各压缩方法的对比
        
        Args:
            original_trajectory: 原始轨迹 [T, D]
            compressed_trajectories: {method_name: compressed_trajectory} 字典
            episode_idx: episode 索引
            frame_idx: frame 索引
            output_path: 输出图片路径
            compressors: {method_name: compressor_instance} 压缩器字典
        """
        n_methods = len(compressed_trajectories)
        n_dims = original_trajectory.shape[1]
        
        # 创建图形布局 - 调整为4行
        # 第1行: 3D轨迹对比 (1个原始 + 4个压缩方法)
        # 第2行: X, Y, Z 维度
        # 第3行: Yaw, Pitch, Roll 维度
        # 第4行: Gripper + 统计信息
        fig = plt.figure(figsize=(28, 20))
        
        # 颜色方案 - 降低原始轨迹的透明度
        colors = {
            'original': 'blue',
            'action_chunk': 'red',
            'bining': 'green',
            'uniform_bspline': 'orange',
            'adaptive_bspline': 'purple'
        }
        
        # 原始轨迹的透明度
        original_alpha = 0.3
        
        # ====================================================================
        # 第一行：3D/2D 轨迹对比
        # ====================================================================
        if n_dims >= 3:
            # 3D 轨迹可视化
            ax_3d = fig.add_subplot(3, n_methods + 1, 1, projection='3d')
            ax_3d.plot(
                original_trajectory[:, 0],
                original_trajectory[:, 1],
                original_trajectory[:, 2],
                'o-', color=colors['original'], label='Original', 
                markersize=3, alpha=original_alpha, linewidth=1.5
            )
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title('Original Trajectory', fontweight='bold')
            ax_3d.legend()
            ax_3d.grid(True, alpha=0.3)
            
            # 各压缩方法的 3D 轨迹
            for idx, (method_name, compressed) in enumerate(compressed_trajectories.items()):
                ax = fig.add_subplot(3, n_methods + 1, idx + 2, projection='3d')
                
                # 原始轨迹（浅色）
                ax.plot(
                    original_trajectory[:, 0],
                    original_trajectory[:, 1],
                    original_trajectory[:, 2],
                    'o-', color='gray', alpha=0.2, markersize=2, linewidth=1
                )
                
                # 压缩后的轨迹
                if compressed.shape[1] >= 3:
                    if method_name in ['uniform_bspline', 'adaptive_bspline']:
                        # B-spline方法：分别显示control points和decoded曲线
                        # 1. 显示control points（离散点）
                        ax.scatter(
                            compressed[:, 0],
                            compressed[:, 1],
                            compressed[:, 2],
                            color=colors.get(method_name, 'black'),
                            s=50, alpha=0.9, marker='o', label=f'{method_name} (control pts)',
                            edgecolors='black', linewidths=1
                        )
                        
                        # 2. 获取decoded曲线并显示（连续线，无点）
                        compressor = compressors[method_name]
                        x_coords, decoded_trajectory = compressor.get_visualization_points(
                            compressed, original_trajectory, num_points=100
                        )
                        if decoded_trajectory.shape[1] >= 3:
                            ax.plot(
                                decoded_trajectory[:, 0],
                                decoded_trajectory[:, 1],
                                decoded_trajectory[:, 2],
                                '-', color=colors.get(method_name, 'black'),
                                linewidth=2, alpha=0.7, label=f'{method_name} (curve)'
                            )
                    else:
                        # 其他方法：正常显示点和线
                        ax.plot(
                            compressed[:, 0],
                            compressed[:, 1],
                            compressed[:, 2],
                            'o-', color=colors.get(method_name, 'black'),
                            label=method_name, markersize=4, alpha=0.8, linewidth=2
                        )
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'{method_name}', fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # ====================================================================
        # ====================================================================
        # 第二、三、四行：各维度时间序列对比
        # 第二行: X, Y, Z (3个维度)
        # 第三行: Yaw, Pitch, Roll (3个维度)
        # 第四行: Gripper (1个维度) + 统计信息
        # ====================================================================
        dim_names = ['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Gripper']
        n_plot_dims = min(n_dims, 7)  # 显示全部7个维度
        
        # 准备各压缩方法的x坐标和数据
        compressed_coords = {}
        control_points_data = {}  # 存储B-spline的control points用于单独绘制
        
        for method_name, compressed in compressed_trajectories.items():
            if method_name == 'action_chunk':
                # action_chunk: 当前时间往后的连续点
                x_coords = np.arange(len(compressed))
                data_points = compressed
                compressed_coords[method_name] = (x_coords, data_points)
                
            elif method_name == 'bining':
                # bining: 均匀采样的索引
                x_coords = np.linspace(0, len(original_trajectory) - 1, len(compressed))
                data_points = compressed
                compressed_coords[method_name] = (x_coords, data_points)
                
            elif method_name == 'uniform_bspline':
                # uniform_bspline: 需要重构B-spline
                compressor = compressors[method_name]
                x_coords, data_points = compressor.get_visualization_points(
                    compressed, original_trajectory, num_points=100
                )
                compressed_coords[method_name] = (x_coords, data_points)
                # 保存control points（用于单独绘制）
                # control points的x坐标通过均匀分布在轨迹长度上估算
                control_x = np.linspace(0, len(original_trajectory) - 1, len(compressed))
                control_points_data[method_name] = (control_x, compressed)
                
            elif method_name == 'adaptive_bspline':
                # adaptive_bspline: 需要映射u值回时间索引
                compressor = compressors[method_name]
                x_coords, data_points = compressor.get_visualization_points(
                    compressed, original_trajectory, num_points=100
                )
                compressed_coords[method_name] = (x_coords, data_points)
                # 保存control points（从compressed中提取，去掉最后一列的knots信息）
                control_points = compressed[:, :-1] if compressed.shape[1] > n_dims else compressed
                control_x = np.linspace(0, len(original_trajectory) - 1, len(control_points))
                control_points_data[method_name] = (control_x, control_points)
                
            else:
                # 默认情况
                x_coords = np.arange(len(compressed))
                data_points = compressed
                compressed_coords[method_name] = (x_coords, data_points)
        
        # 绘制各个维度
        for dim_idx in range(n_plot_dims):
            # 确定subplot位置
            if dim_idx < 3:
                # 第二行: X, Y, Z
                row = 2
                col = dim_idx + 1
            elif dim_idx < 6:
                # 第三行: Yaw, Pitch, Roll
                row = 3
                col = dim_idx - 3 + 1
            else:
                # 第四行: Gripper (左边第一个位置)
                row = 4
                col = 1
            
            ax = fig.add_subplot(4, n_methods + 1, (row - 1) * (n_methods + 1) + col)
            
            # 原始轨迹
            original_x = np.arange(len(original_trajectory))
            ax.plot(
                original_x,
                original_trajectory[:, dim_idx],
                'o-', color=colors['original'], label='Original',
                markersize=3, alpha=original_alpha, linewidth=1.5
            )
            
            # 各压缩方法
            for method_name, (x_coords, data_points) in compressed_coords.items():
                if data_points.shape[1] > dim_idx:
                    if method_name in ['uniform_bspline', 'adaptive_bspline']:
                        # B-spline方法：先画曲线（无点），再画control points（离散点）
                        # 1. 画decoded曲线（连续线，无点）
                        ax.plot(
                            x_coords,
                            data_points[:, dim_idx],
                            '-', color=colors.get(method_name, 'black'),
                            label=f'{method_name} (curve)', linewidth=1.5, alpha=0.7
                        )
                        # 2. 画control points（离散点）
                        if method_name in control_points_data:
                            control_x, control_pts = control_points_data[method_name]
                            if control_pts.shape[1] > dim_idx:
                                ax.scatter(
                                    control_x,
                                    control_pts[:, dim_idx],
                                    color=colors.get(method_name, 'black'),
                                    s=40, alpha=0.9, marker='o',
                                    edgecolors='black', linewidths=1,
                                    label=f'{method_name} (ctrl pts)', zorder=5
                                )
                    else:
                        # 其他方法：正常显示点和线
                        ax.plot(
                            x_coords,
                            data_points[:, dim_idx],
                            'o-', color=colors.get(method_name, 'black'),
                            label=method_name, markersize=3, alpha=0.7, linewidth=1.5
                        )
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{dim_names[dim_idx]} Value')
            ax.set_title(f'{dim_names[dim_idx]} Dimension', fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        
        # ====================================================================
        # 第四行右侧：显示统计信息文本
        # ====================================================================
        
        # 计算统计信息
        method_names = list(compressed_trajectories.keys())
        
        # 1. 压缩率
        compression_ratios = {}
        for method_name, compressed in compressed_trajectories.items():
            compression_ratios[method_name] = len(compressed) / len(original_trajectory)
        
        # 2. 重构误差（RMSE）
        reconstruction_errors = {}
        for method_name, compressed in compressed_trajectories.items():
            # 使用decoded曲线计算误差（对B-spline方法）
            if method_name in ['uniform_bspline', 'adaptive_bspline']:
                # 使用decoded的曲线数据
                _, decoded_trajectory = compressed_coords[method_name]
                # 线性插值到原始长度
                from scipy.interpolate import interp1d
                t_decoded = np.linspace(0, 1, len(decoded_trajectory))
                t_original = np.linspace(0, 1, len(original_trajectory))
                
                interpolated = np.zeros_like(original_trajectory)
                for dim_idx in range(min(n_dims, decoded_trajectory.shape[1])):
                    f = interp1d(t_decoded, decoded_trajectory[:, dim_idx], kind='linear', fill_value='extrapolate')
                    interpolated[:, dim_idx] = f(t_original)
                
                error = np.sqrt(np.mean((original_trajectory[:, :decoded_trajectory.shape[1]] - interpolated[:, :decoded_trajectory.shape[1]]) ** 2))
            else:
                # 对其他方法
                if len(compressed) != len(original_trajectory):
                    from scipy.interpolate import interp1d
                    t_compressed = np.linspace(0, 1, len(compressed))
                    t_original = np.linspace(0, 1, len(original_trajectory))
                    
                    interpolated = np.zeros_like(original_trajectory)
                    for dim_idx in range(min(n_dims, compressed.shape[1])):
                        f = interp1d(t_compressed, compressed[:, dim_idx], kind='linear', fill_value='extrapolate')
                        interpolated[:, dim_idx] = f(t_original)
                    
                    error = np.sqrt(np.mean((original_trajectory[:, :compressed.shape[1]] - interpolated[:, :compressed.shape[1]]) ** 2))
                else:
                    error = np.sqrt(np.mean((original_trajectory[:, :compressed.shape[1]] - compressed) ** 2))
            
            reconstruction_errors[method_name] = error
        
        # 3. 数据点数量
        point_counts = {method_name: len(compressed) for method_name, compressed in compressed_trajectories.items()}
        
        # 在第四行创建右侧的统计信息区域（占据第2到第5列，共4列）
        ax_stats = plt.subplot2grid((4, n_methods + 1), (3, 2), colspan=n_methods - 1, fig=fig)
        ax_stats.axis('off')
        
        # 构建统计信息文本
        stats_text = f"Original Trajectory Length: {len(original_trajectory)} points\n\n"
        stats_text += "Compression Statistics:\n"
        stats_text += "-" * 100 + "\n"
        stats_text += f"{'Method':<20} {'Points':<15} {'Compression Ratio':<20} {'RMSE':<15}\n"
        stats_text += "-" * 100 + "\n"
        
        for method_name in method_names:
            points = point_counts[method_name]
            ratio = compression_ratios[method_name]
            error = reconstruction_errors[method_name]
            stats_text += f"{method_name:<20} {points:<15} {ratio:<20.4f} {error:<15.6f}\n"
        
        stats_text += "-" * 100 + "\n"
        
        # 显示文本
        ax_stats.text(
            0.5, 0.5, stats_text,
            ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        # 添加总标题
        fig.suptitle(
            f'Trajectory Compression Comparison\n'
            f'Episode {episode_idx} | Frame {frame_idx} | Original Length: {len(original_trajectory)}',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存可视化结果: {output_path}")
    
    # ========================================================================
    # 主流程：加载数据并进行压缩对比
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("开始轨迹压缩对比实验")
    logger.info("=" * 80)
    
    # 创建 dummy tokenizer
    logger.info("创建 DummyTokenizer...")
    dummy_tokenizer = DummyTokenizer()
    
    # 创建压缩器字典
    logger.info(f"初始化压缩方法: {args.compression_methods}")
    compressors = {}
    for method_name in args.compression_methods:
        if method_name == "action_chunk":
            compressors[method_name] = TRAJECTORY_COMPRESSION_REGISTRY["action_chunk"](
                action_chunk_size=args.target_length
            )
        elif method_name == "bining":
            compressors[method_name] = TRAJECTORY_COMPRESSION_REGISTRY["bining"](
                target_length=args.target_length
            )
        elif method_name == "uniform_bspline":
            compressors[method_name] = TRAJECTORY_COMPRESSION_REGISTRY["uniform_bspline"](
                target_length=args.target_length, degree=3
            )
        # elif method_name == "adaptive_bspline":
        #     compressors[method_name] = TRAJECTORY_COMPRESSION_REGISTRY["adaptive_bspline"](
        #         max_length=args.target_length, max_error=0.01, degree=3
        #     )
    
    # 加载数据集（使用 bining 作为默认压缩器，实际压缩在后面手动进行）
    logger.info(f"加载数据集: {args.repo_id}")
    logger.info(f"任务 IDs: {args.task_ids}")
    
    try:
        dataset = MyLeRobotDataset(
            repo_id=args.repo_id,
            tokenizer=dummy_tokenizer,
            trajectory_compression=TRAJECTORY_COMPRESSION_REGISTRY["bining"](target_length=50),
            real_root=Path(args.data_root),
            task_ids=args.task_ids,
            train_val_split=(0.9, 0.1)
        )
        logger.info(f"数据集加载成功！总样本数: {len(dataset)}")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 遍历数据集并进行压缩对比
    logger.info("=" * 80)
    logger.info(f"可视化模式: {args.visualization_mode}")
    logger.info(f"开始处理 {min(args.num_episodes, len(dataset))} 个样本...")
    logger.info("=" * 80)
    
    if args.visualization_mode == "single_episode":
        # 单 episode 多 frame 模式（原有逻辑）
        for sample_idx in range(min(args.num_episodes, len(dataset))):
            try:
                # 获取样本
                item = dataset.dataset.__getitem__(sample_idx)
                
                # 提取原始轨迹
                episode_id = item['episode_index'].item()
                frame_id = item['frame_index'].item()
                
                # 获取原始轨迹（从当前帧到 episode 结束）
                episode_from_id = dataset.dataset.meta.episodes['dataset_from_index'][episode_id]
                episode_to_id = dataset.dataset.meta.episodes['dataset_to_index'][episode_id]
                original_trajectory = np.array(
                    dataset.dataset.hf_dataset['action'][episode_from_id + frame_id:episode_to_id]
                )
                
                logger.info(f"\n样本 {sample_idx + 1}/{min(args.num_episodes, len(dataset))}")
                logger.info(f"  Episode ID: {episode_id} | Frame ID: {frame_id}")
                logger.info(f"  原始轨迹形状: {original_trajectory.shape}")
                
                # 如果轨迹太短，跳过
                if len(original_trajectory) < 5:
                    logger.warning(f"  轨迹太短 ({len(original_trajectory)} 点)，跳过")
                    continue
                
                # 使用各压缩方法进行压缩
                compressed_trajectories = {}
                for method_name, compressor in compressors.items():
                    compressed = compressor(original_trajectory.copy())
                    compressed_trajectories[method_name] = compressed
                    logger.info(f"  {method_name:20s}: {len(original_trajectory):4d} -> {len(compressed):4d} 点 "
                              f"(压缩率: {len(compressed)/len(original_trajectory):.2f})")
                
                # 可视化对比
                output_path = output_dir / f"sample_{sample_idx:03d}_ep{episode_id}_frame{frame_id}.png"
                visualize_trajectory_comparison(
                    original_trajectory,
                    compressed_trajectories,
                    episode_id,
                    frame_id,
                    output_path,
                    compressors
                )
                
            except Exception as e:
                logger.error(f"处理样本 {sample_idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    else:  # multi_episode_per_frame 模式
        # 多 episode 单 frame 模式：按 frame 分组
        from collections import defaultdict
        
        # 收集所有样本并按 (episode_id, frame_id) 分组
        samples_by_frame = defaultdict(list)
        
        logger.info("收集样本数据...")
        for sample_idx in range(min(args.num_episodes, len(dataset))):
            try:
                item = dataset.dataset.__getitem__(sample_idx)
                episode_id = item['episode_index'].item()
                frame_id = item['frame_index'].item()
                
                # 获取原始轨迹
                episode_from_id = dataset.dataset.meta.episodes['dataset_from_index'][episode_id]
                episode_to_id = dataset.dataset.meta.episodes['dataset_to_index'][episode_id]
                original_trajectory = np.array(
                    dataset.dataset.hf_dataset['action'][episode_from_id + frame_id:episode_to_id]
                )
                
                # 跳过太短的轨迹
                if len(original_trajectory) < 5:
                    continue
                
                # 按 frame_id 分组（不同 episode 的同一 frame）
                samples_by_frame[frame_id].append({
                    'episode_id': episode_id,
                    'frame_id': frame_id,
                    'original_trajectory': original_trajectory,
                    'sample_idx': sample_idx
                })
                
            except Exception as e:
                logger.error(f"收集样本 {sample_idx} 时出错: {e}")
                continue
        
        logger.info(f"共收集到 {len(samples_by_frame)} 个不同的 frame")
        
        # 为每个 frame 创建一张包含多个 episode 的图
        for frame_id in sorted(samples_by_frame.keys()):
            samples = samples_by_frame[frame_id]
            logger.info(f"\n处理 Frame {frame_id}: {len(samples)} 个 episodes")
            
            # 为这个 frame 的所有 episodes 创建一张大图
            # 每个 episode 一行，每行包含该 episode 在此 frame 的压缩对比
            fig = plt.figure(figsize=(28, 8 * len(samples)))
            
            for row_idx, sample_data in enumerate(samples):
                episode_id = sample_data['episode_id']
                original_trajectory = sample_data['original_trajectory']
                
                # 压缩轨迹
                compressed_trajectories = {}
                for method_name, compressor in compressors.items():
                    compressed = compressor(original_trajectory.copy())
                    compressed_trajectories[method_name] = compressed
                
                # 在子图中绘制（使用 subplot 的行索引）
                # 每个 episode 占一行，每行显示: 3D图 + 7个维度图
                base_row = row_idx * 8  # 每个 episode 8 行空间
                
                # 添加 episode 标题
                ax_title = fig.add_subplot(len(samples), 1, row_idx + 1)
                ax_title.text(0.5, 0.5, f'Episode {episode_id} | Frame {frame_id} | Length: {len(original_trajectory)}',
                             ha='center', va='center', fontsize=14, fontweight='bold')
                ax_title.axis('off')
            
            # 保存
            output_path = output_dir / f"frame_{frame_id:03d}_multi_episodes.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"保存可视化结果: {output_path}")
    
    logger.info("=" * 80)
    logger.info("轨迹压缩对比实验完成！")
    logger.info(f"结果保存在: {output_dir}")
    logger.info("=" * 80)

