"""

实现用b-spline对整段轨迹进行压缩，目前实现三种情况
1. 不压缩，直接插值biningg成定长的点
2. b-spline最小二乘法压缩成定长的点
3. b-spline压缩成不定长的点，使用padding来进行计算

"""

import numpy as np
from scipy.interpolate import BSpline, splrep, splev, make_lsq_spline
from scipy.linalg import lstsq  # Compute least-squares solution to equation Ax = b.
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
    """回退到普通vla模式(默认ac长度为1,测试一下openvla的原始设定)"""

    def __init__(self, action_chunk_size=1):
        self.action_chunk_size = action_chunk_size
        self.exp_type = "action_chunk"

    def __call__(self, trajectory):
        # 确保 trajectory 是 numpy 数组
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)

        # 如果输入是一维（单个action），转换为二维 [1, n_dims]
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, -1)

        return trajectory[: self.action_chunk_size]  # 截断到指定大小


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
        knot_vector = np.concatenate(
            [
                np.repeat(x_min, self.degree + 1),  # 起始重复节点
                interior_knots,  # 内部节点
                np.repeat(x_max, self.degree + 1),  # 结束重复节点
            ]
        )

        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []

        for d in range(dim):
            # 使用最小二乘法拟合B样条
            # make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(
                x_data, trajectory[:, d], knot_vector, k=self.degree
            )
            all_control_points.append(bspline.c)

        # 将控制点组合成 [n_control_points, dim] 的数组
        control_points = np.column_stack(all_control_points)

        # print("Uniform B-spline compression done.")
        # print("control points:", control_points.shape)
        # # 可视化的时候，左右还需要重复一次，但是可以不预测interior_knots了
        # print("interior knots:", interior_knots.shape)
        return control_points

    def get_visualization_points(
        self,
        compressed_output: np.ndarray,
        original_trajectory: np.ndarray,
        num_points: int = 100,
    ):
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
        knot_vector = np.concatenate(
            [
                np.repeat(x_min, self.degree + 1),
                interior_knots,
                np.repeat(x_max, self.degree + 1),
            ]
        )

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


@register_trajectory_compression("aff_bining")
class AffBiningTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 60):
        self.target_length = target_length
        self.exp_type = "aff"

    def __call__(self, aff_trajectory: np.ndarray) -> np.ndarray:
        """
        固定频率的bining压缩，支持从指定百分比位置开始截断。

        Args:
            aff_trajectory: [points_num, dim] 的numpy数组
        Returns:
            压缩后的轨迹 [target_length, dim]（定长，配合EOS token）
        """
        original_length = aff_trajectory.shape[0]
        # 如果原始长度不足目标长度，直接返回原始轨迹
        if original_length < self.target_length:
            return aff_trajectory
        # 均匀采样 target_length 个点
        indices = np.linspace(0, original_length - 1, num=self.target_length)
        compressed_trajectory = np.array(
            [aff_trajectory[int(round(idx))] for idx in indices]
        )

        # print("compressed trajectory:", compressed_trajectory)
        return compressed_trajectory


# 拟合方法，加权最小二乘, s这个参数代表是否平滑拟合
# scipy.interpolate.splrep(x, y, s=0, k=3) 名称：Spline representation, 输出：三元组 (t(knots), c(control), k)
# scipy.interpolate.splev(x_new, tck)  名称：Spline evaluation （这两个只是插值）
# scipy.interpolate.make_lsq_spline(x, y, t, k) 名称：Least-squares B-spline representation（真正的拟合函数）
@register_trajectory_compression("aff_uniform_bspline")
class AffUniformBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 30, degree: int = 3):
        """
        固定点数和频率的B样条压缩：用固定数量的内部节点来拟合B样条，通过最小二乘法优化控制点。
        采用时间参数化，控制点均匀分布在轨迹上。（模型不需要预测x轴）
        Args:
            target_length: 内部节点数量（控制点数量 = target_length + degree + 1）
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.exp_type = "aff"
        self.target_length = target_length
        self.degree = degree

    def __call__(self, aff_trajectory: np.ndarray) -> np.ndarray:
        """
        使用固定数量的内部节点进行B样条最小二乘拟合。

        工作流程：
        1. 在整条轨迹上均匀分布 target_length 个内部节点
        2. 使用 make_lsq_spline 拟合B样条（自动优化控制点的y值）
        3. 返回优化得到的控制点（不在原始曲线上）

        Args:
            aff_trajectory: [points_num, dim] 的numpy数组

        Returns:
            control_points: [n_control_points, dim] 的numpy数组
        """

        original_length = aff_trajectory.shape[0]

        dim = aff_trajectory.shape[1] if len(aff_trajectory.shape) > 1 else 1
        # B样条约束：数据点数量必须 >= 控制点数量
        # 控制点数量 = n_interior_knots + degree + 1
        n_control_points = self.target_length + self.degree + 1
        # 如果原始长度不足控制点数量，直接返回原始轨迹
        if original_length < n_control_points:
            return aff_trajectory
        
        # 时间参数化
        x_data = np.arange(original_length, dtype=float)
        x_min, x_max = 0.0, float(original_length - 1)

        # 正常情况：使用self.target_length
        n_interior_knots = self.target_length
        interior_knots = np.linspace(x_min, x_max, self.target_length + 2)[1:-1]

        # 构造完整的knot vector
        knot_vector = np.concatenate(
            [
                np.repeat(x_min, self.degree + 1),  # 起始重复节点
                interior_knots,  # 内部节点
                np.repeat(x_max, self.degree + 1),  # 结束重复节点
            ]
        )


        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []

        for d in range(dim):
            # 使用最小二乘法拟合B样条
            # make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(
                x_data, aff_trajectory[:, d], knot_vector, k=self.degree
            )
            all_control_points.append(bspline.c)

        # 将控制点组合成 [n_control_points, dim] 的数组
        control_points = np.column_stack(all_control_points)

        return control_points

    def get_visualization_points(
        self,
        compressed_output: np.ndarray,
        original_trajectory: np.ndarray,
        num_points: int = 100,
    ):
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
        knot_vector = np.concatenate(
            [
                np.repeat(x_min, self.degree + 1),
                interior_knots,
                np.repeat(x_max, self.degree + 1),
            ]
        )

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
        knot_vector = np.concatenate(
            [
                np.repeat(x_min, self.degree + 1),  # 起始重复节点
                interior_knots,  # 内部节点
                np.repeat(x_max, self.degree + 1),  # 结束重复节点
            ]
        )

        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []

        for d in range(dim):
            # 使用最小二乘法拟合B样条
            # make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(
                x_data, trajectory[:, d], knot_vector, k=self.degree
            )
            all_control_points.append(bspline.c)

        # 将控制点组合成 [n_control_points, dim] 的数组
        control_points = np.column_stack(all_control_points)

        # print("Uniform B-spline compression done.")
        # print("control points:", control_points.shape)
        # # 可视化的时候，左右还需要重复一次，但是可以不预测interior_knots了
        # print("interior knots:", interior_knots.shape)
        return control_points

    def get_visualization_points(
        self,
        compressed_output: np.ndarray,
        original_trajectory: np.ndarray,
        num_points: int = 100,
    ):
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
        knot_vector = np.concatenate(
            [
                np.repeat(x_min, self.degree + 1),
                interior_knots,
                np.repeat(x_max, self.degree + 1),
            ]
        )

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
    def __init__(
        self,
        max_length: int = 50,
        max_iteration: int = 100,
        max_error: float = 0.01,
        degree: int = 3,
        improvement_epsilon: float = 1e-4,
        min_knot_spacing: float = 0.02,
    ):
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
        assert (
            n_control >= self.degree + 1 and n_points >= 2
        ), "控制点数量必须至少为阶次+1，且数据点数量必须至少为2, 数据点数量: {}".format(
            n_points
        )

        # 参数化u，映射到[0,1]
        # 把第 1…N 个数据点按“走过的弧长”均匀映射到 [0,1]，像给每个点一个时间戳。代码用的是弦长参数化（chord-length）：
        diffs = np.diff(points, axis=0)  # 算数组中相邻元素的差值
        chord_lengths = np.linalg.norm(
            diffs, axis=1
        )  # 算出矩阵的 2 范数（欧几里得范数）
        cumulative = np.concatenate([[0], np.cumsum(chord_lengths)])
        u = cumulative / cumulative[-1]

        # 确定内部结点数量,B-spline的定义
        n_interior_knots = max(0, n_control - self.degree - 1)

        # 使用scipy的make_interp_spline
        if n_interior_knots <= 0:
            # 最简单情况：只有边界结点
            t = np.concatenate([[0] * (self.degree + 1), [1] * (self.degree + 1)])
        else:
            # 均匀分布内部结点。结点的数量比控制点多，因为包括边界结点。
            interior = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
            t = np.concatenate(
                [[0] * (self.degree + 1), interior, [1] * (self.degree + 1)]
            )

        # 构建基函数矩阵并最小二乘拟合
        n_control_actual = len(t) - self.degree - 1

        # 构建基函数矩阵
        N = np.zeros((n_points, n_control_actual))
        for i in range(n_control_actual):
            # 单位基函数
            c = np.zeros(n_control_actual)
            c[i] = 1.0  # 第 i 个基函数为 1，其余为 0
            bspl = BSpline(t, c, self.degree)
            N[:, i] = bspl(u)  # 评估一列 N[:, i] = bspl(u)

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
        best_max_err = float("inf")  # Track best error
        prev_max_err = None

        for it in range(self.max_iteration):
            # 构造完整knot向量
            interior_arr = (
                np.array(interior_u, dtype=float) if len(interior_u) else np.array([])
            )
            knots = np.concatenate(
                [np.zeros(self.degree + 1), interior_arr, np.ones(self.degree + 1)]
            )

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

            print(
                f"[Adaptive] Iteration {it}: interior_knots={len(interior_u)}, control_pts={len(control_points)}, max_err={max_err:.6f}"
            )

            # 保留最佳结果
            if max_err < best_max_err:
                best_control = control_points.copy()
                best_knots = knots.copy()
                best_max_err = max_err
                print(
                    f"[Adaptive]   -> New best! Saving state with error={max_err:.6f}"
                )

            # 停止条件：误差阈值
            if max_err <= self.max_error:
                print(
                    f"[Adaptive] Stopped: max_err ({max_err:.6f}) <= threshold ({self.max_error})"
                )
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
                print(
                    f"[Adaptive]   -> Inserted knot at u={u_new:.4f} (midpoint of [{u_left:.4f}, {u_right:.4f}])"
                )
            else:
                # 检查与已有节点的最小距离
                min_dist = np.min(np.abs(np.array(interior_u) - u_new))
                if min_dist >= self.min_knot_spacing:
                    interior_u.append(u_new)
                    interior_u.sort()
                    print(
                        f"[Adaptive]   -> Inserted knot at u={u_new:.4f} (midpoint of [{u_left:.4f}, {u_right:.4f}])"
                    )
                else:
                    # 如果中点不满足间距要求，尝试寻找其他可插入的区间
                    print(
                        f"[Adaptive]   -> Midpoint u={u_new:.4f} too close to existing knots (min_dist={min_dist:.4f})"
                    )

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
                        if (
                            len(interior_u) == 0
                            or np.min(np.abs(np.array(interior_u) - u_mid))
                            >= self.min_knot_spacing
                        ):
                            interior_u.append(u_mid)
                            interior_u.sort()
                            print(
                                f"[Adaptive]   -> Inserted knot at u={u_mid:.4f} (alternative interval [{u_l:.4f}, {u_r:.4f}])"
                            )
                            inserted = True
                            break

                    if not inserted:
                        print(
                            f"[Adaptive]   -> No valid interval found for insertion, stopping."
                        )
                        break

        # 最终返回：将interior_knots保存在控制点矩阵中
        if best_control is None or best_knots is None:
            # 回退到最后一次拟合
            best_control = control_points
            best_knots = knots

        # 从best_knots中提取interior_knots
        # best_knots结构: [0, 0, 0, 0, interior_knots..., 1, 1, 1, 1] (degree=3)
        interior_knots = best_knots[self.degree + 1 : -(self.degree + 1)]

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

    def get_visualization_points(
        self,
        compressed_output: np.ndarray,
        original_trajectory: np.ndarray,
        num_points: int = 100,
    ):
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
        knot_vector = np.concatenate(
            [np.zeros(self.degree + 1), interior_knots_u, np.ones(self.degree + 1)]
        )

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


# 可视化删除了，烦了。以后单独处理好了
