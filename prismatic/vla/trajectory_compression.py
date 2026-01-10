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
    def __call__(self, trajectory: np.ndarray, **kwargs) -> np.ndarray:
        pass


@register_trajectory_compression("action_chunk")
class ActionChunk(BaseTrajectoryCompression):
    """回退到普通vla模式(默认ac长度为1,测试一下openvla的原始设定)"""

    def __init__(self, action_chunk_size=1):
        self.action_chunk_size = action_chunk_size
        self.exp_type = "action_chunk"

    def __call__(self, trajectory, **kwargs):
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

    def __call__(self, trajectory: np.ndarray, **kwargs) -> np.ndarray:
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

    def __call__(self, trajectory: np.ndarray, **kwargs) -> np.ndarray:
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

    def __call__(self, aff_trajectory: np.ndarray, **kwargs) -> np.ndarray:
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

        print("compressed trajectory:", compressed_trajectory)
        return compressed_trajectory


# 拟合方法，加权最小二乘, s这个参数代表是否平滑拟合
# scipy.interpolate.splrep(x, y, s=0, k=3) 名称：Spline representation, 输出：三元组 (t(knots), c(control), k)
# scipy.interpolate.splev(x_new, tck)  名称：Spline evaluation （这两个只是插值）
# scipy.interpolate.make_lsq_spline(x, y, t, k) 名称：Least-squares B-spline representation（真正的拟合函数）
# NOTE： 这个实现忘记输出时间了，无法进行复原，故不进行测试。
@register_trajectory_compression("aff_uniform_bspline")
class AffUniformBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 100, degree: int = 3):
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

    def __call__(self, aff_trajectory: np.ndarray, **kwargs) -> np.ndarray:
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

    def decode_to_trajectory(
        self, control_points: np.ndarray, num_points: int = None, original_length: int = None
    ) -> np.ndarray:
        """
        从控制点解码回aff_trajectory。其实不知道解码到时候是多少个点，所以这个是一个问题（规定需要移动的距离是多少？)

        在压缩阶段（只返回 control points），如果没有保存 original_length（原始轨迹长度），那么在 decode（重建）时就无法100%还原出和原始轨迹完全对齐的 knot vector，导致重建轨迹和原始轨迹的采样点不一一对应，decode 结果会有偏差。

        结论：

        仅仅保存 control points（控制点）是不够的，decode 时还需要 original_length（或等价的时间参数化信息），否则无法实现无损重建。
        如果想实现无损 decode，建议在压缩时一并保存 original_length 或相关参数，decode 时用同样的 knot vector 还原。


        Args:
            control_points: [n_control_points, dim] 的numpy数组，由 __call__() 返回的控制点
            num_points: 解码后的轨迹点数（可选）。如果为None，使用original_length
            original_length: 原始轨迹长度（可选）。用于重建knot vector。如果为None且num_points也为None，使用默认值100

        Returns:
            aff_trajectory: [num_points, dim] 的numpy数组，解码后的轨迹
        """
        dim = control_points.shape[1]
        
        # 确定输出点数
        if num_points is None and original_length is None:
            num_points = 100  # 默认值
        elif num_points is None:
            num_points = original_length
        
        # 确定用于重建knot vector的长度参数
        if original_length is None:
            original_length = num_points
        
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
        aff_trajectory = np.zeros((num_points, dim))
        for d in range(dim):
            bspline = BSpline(knot_vector, control_points[:, d], self.degree)
            aff_trajectory[:, d] = bspline(x_coords)
        
        return aff_trajectory

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
# 这里开始是在整个轨迹上进行b-spline的压缩（只进行一次），然后去找现在frame index后面的点进行预测的方式。
# Fixed-Knot B-spline Compression
# ==============================================================================
@register_trajectory_compression("abs_aff_uniform_bspline")
class AbsAffUniformBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 30, degree: int = 3):
        """
        固定点数和频率的B样条压缩：用固定数量的内部节点来拟合整条轨迹的B样条，通过最小二乘法优化控制点。
        采用时间参数化，控制点均匀分布在政整条轨迹上。（模型不需要预测时间轴）
        Args:
            target_length: 内部节点数量（控制点数量 = target_length + degree + 1）
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.exp_type = "abs_aff" # 重要，决定了dataset load那种数据
        self.target_length = target_length # 代表的是控制点的总数量，其中包含内部结点和端点重复
        self.degree = degree

        self.episode_cache = {}  # 存储每个episode的B样条数据

    def compress_to_control_point(self, aff_trajectory: np.ndarray):
        """
        这里采用last mile的切分方式，从夹爪变化点左右两边来slice出来需要的东西
        """
        gripper_traj = aff_trajectory[:, 6] # 最后一个gripper的维度
        diff_gripper = np.diff(gripper_traj)
        gripper_knot_vector = np.where(diff_gripper != 0)[0] + 1 # gripper所对应的变化点，存储变化后一刻的点（+1）（解码时候注意处理）

        process_trajectory = aff_trajectory[:, :6] # 只处理前六维的x, y, z, yaw, pitch, row
        # TODO：首先这里可以换成曲率均一化的方法，而不是时间上均一化
        sampled_knot_vector = np.linspace(
            0, 
            process_trajectory.shape[0]-1, # 到结尾的index
            self.target_length - len(gripper_knot_vector) - self.degree + 1 
        )[1:-1] # 去掉头尾，避免和端点重复. 最后长度是+1-2=-1
        # print("gripper knot vector num:", len(gripper_knot_vector), ", processed_knot_vector:", len(processed_knot_vector))
        internal_knot_vector = np.sort(np.concatenate([gripper_knot_vector, sampled_knot_vector]))

        knot_vector = np.concatenate([
            np.repeat(0.0, self.degree+1),
            internal_knot_vector,
            np.repeat(aff_trajectory.shape[0]-1, self.degree+1)
        ]).astype(int)
        assert len(knot_vector) == self.target_length + self.degree + 1
        # print("knot_vector length:", len(knot_vector))

        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []
        x_data = np.arange(aff_trajectory.shape[0], dtype=float) # 时间参数化
        # print("knot_vector and process_trajectoy length:", len(knot_vector), " ", len(process_trajectory))
        for d in range(process_trajectory.shape[1]):
            # 使用最小二乘法拟合B样条,make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(
                x_data, process_trajectory[:, d], knot_vector, k=self.degree
            )
            all_control_points.append(bspline.c)    
            # print("contorl point length:", len(bspline.c))
            # print("bspline.k:", bspline.k)  
            # print("internal knot:", internal_knot_vector)        
            # exit()
        # ga_conv_kernel = np.ones(self.degree) / self.degree
        # greville_abscissae_time_indices = np.convolve(knot_vector[1:-1], ga_conv_kernel, mode='valid')
        # print("greville_absciasse:", greville_abscissae_time_indices)
        # print("gripper slice:", gripper_knot_vector)
        # exit()
        # 这里的slice是我未来是clamped start，open end方式来采用的slice设计.让其变成N+K的长度和c统一长度。[0, 1, xxx, n, n, n, n]
        sliced_knot_vector = knot_vector[self.degree:-1] # 不能把第一个knot完全删除调, 所以变成这样
        all_control_points.append(gripper_traj[sliced_knot_vector]) # 加上gripper的控制点
        all_control_points.append(sliced_knot_vector) # 最后加上internal knot的参数

        # 将控制点组合成 [n_control_points, dim+1] 的数组，这里长度不包含前后重复的端点，就是target length
        control_points = np.column_stack(all_control_points)
        assert len(control_points) == self.target_length 
        return control_points


    def __call__(self, aff_trajectory: np.ndarray, **kwargs) -> np.ndarray:
        """
        使用固定数量的内部节点进行B样条最小二乘拟合，然后slice到后续的实际的trajectory
        """
        frame_index, episode_index = kwargs['frame_index'], kwargs['episode_index']
        
        # NOTE：对于xyz,yaw,pitch,row进行类加，在绝对位置进行学习
        # print("before aff:", aff_trajectory[:5])
        aff_trajectory[:, :-1] = np.cumsum(aff_trajectory[:, :-1], axis=0)
        # print("aff:", aff_trajectory[:5])
        if episode_index in self.episode_cache.keys():
            full_traj = self.episode_cache[episode_index]
        else:
            full_traj = self.compress_to_control_point(aff_trajectory=aff_trajectory)
            self.episode_cache[episode_index] = full_traj
        
        idx = np.searchsorted(full_traj[:,-1], frame_index)

        predict_values = full_traj[idx:, :]
        predict_values[:, -1] = predict_values[:, -1] - frame_index.item()  # NOTE: 时间参数化调整为从0开始
        # print("predict values:", predict_values)
        return predict_values

    def decode_to_action(
        self, control_points: np.ndarray, current_eef_pose: np.ndarray
    ) -> np.ndarray:
        """从控制点解码回aff_trajectory。 不管用A还是S，都是当前pose到预测的这些点上去移动，设置移动的速度。"""
        assert control_points.shape[0] >= 3, "控制点数量不足degree + 1，无法解码轨迹。"

        current_eef_pose = np.append(current_eef_pose, 0)  # 添加时间维度，当前时间为0  

        internal_knot_vector = control_points[:, -1] # 最后一个dim的knot vector
        knot_vector = np.concatenate([
            np.repeat(current_eef_pose[-1], self.degree + 1),  # 起始重复节点, 因为起始的slice方式让其丢失开始的点？ TODO，没太理解
            internal_knot_vector,
            np.repeat(internal_knot_vector[-1], 1)  # 结束重复节点
        ])
        extended_cp = np.vstack([current_eef_pose[:6], control_points[:, :6]]) # 前六维是aff的
        bspline = BSpline(knot_vector, extended_cp, self.degree) # 前六维是aff的

        # 目前实现中action没有作用
        return None, bspline



# ==============================================================================
# 这里开始是在整个轨迹上进行b-spline的压缩（只进行一次），然后去找现在frame index后面的点进行预测的方式。
# Fixed-Knot B-spline Compression
# ==============================================================================
@register_trajectory_compression("abs_aff_adap_bspline")
class AbsAffAdapBSplineTrajectoryCompression(BaseTrajectoryCompression):
    def __init__(self, target_length: int = 30, degree: int = 3):
        """
        固定点数和频率的B样条压缩：用固定数量的内部节点来拟合整条轨迹的B样条，通过最小二乘法优化控制点。
        采用时间参数化，控制点均匀分布在政整条轨迹上。（模型不需要预测时间轴）
        Args:
            target_length: 内部节点数量（控制点数量 = target_length + degree + 1）
            degree: B样条的阶数（默认3 = 三次样条）
        """
        self.exp_type = "abs_aff" # 重要，决定了dataset load那种数据
        self.target_length = target_length # 代表的是控制点的总数量，其中包含内部结点和端点重复
        self.degree = degree

        self.episode_cache = {}  # 存储每个episode的B样条数据

    def compress_to_control_point(self, aff_trajectory: np.ndarray):
        """
        这里采用last mile的切分方式，从夹爪变化点左右两边来slice出来需要的东西
        """
        gripper_traj = aff_trajectory[:, 6] # 最后一个gripper的维度
        diff_gripper = np.diff(gripper_traj)
        gripper_knot_vector = np.where(diff_gripper != 0)[0] + 1 # gripper所对应的变化点，存储变化后一刻的点（+1）（解码时候注意处理）

        process_trajectory = aff_trajectory[:, :6] # 只处理前六维的x, y, z, yaw, pitch, row
        # TODO：首先这里可以换成曲率均一化的方法，而不是时间上均一化
        sampled_knot_vector = np.linspace(
            0, 
            process_trajectory.shape[0]-1, # 到结尾的index
            self.target_length - len(gripper_knot_vector) - self.degree + 1 
        )[1:-1] # 去掉头尾，避免和端点重复. 最后长度是+1-2=-1
        # print("gripper knot vector num:", len(gripper_knot_vector), ", processed_knot_vector:", len(processed_knot_vector))
        internal_knot_vector = np.sort(np.concatenate([gripper_knot_vector, sampled_knot_vector]))

        knot_vector = np.concatenate([
            np.repeat(0.0, self.degree+1),
            internal_knot_vector,
            np.repeat(aff_trajectory.shape[0]-1, self.degree+1)
        ]).astype(int)
        assert len(knot_vector) == self.target_length + self.degree + 1
        # print("knot_vector length:", len(knot_vector))

        # 对每个维度分别进行B样条最小二乘拟合
        all_control_points = []
        x_data = np.arange(aff_trajectory.shape[0], dtype=float) # 时间参数化
        # print("knot_vector and process_trajectoy length:", len(knot_vector), " ", len(process_trajectory))
        for d in range(process_trajectory.shape[1]):
            # 使用最小二乘法拟合B样条,make_lsq_spline 会自动计算最优的控制点
            bspline = make_lsq_spline(
                x_data, process_trajectory[:, d], knot_vector, k=self.degree
            )
            all_control_points.append(bspline.c)    
            # print("contorl point length:", len(bspline.c))
            # print("bspline.k:", bspline.k)  
            # print("internal knot:", internal_knot_vector)        
            # exit()
        # ga_conv_kernel = np.ones(self.degree) / self.degree
        # greville_abscissae_time_indices = np.convolve(knot_vector[1:-1], ga_conv_kernel, mode='valid')
        # print("greville_absciasse:", greville_abscissae_time_indices)
        # print("gripper slice:", gripper_knot_vector)
        # exit()
        # 这里的slice是我未来是clamped start，open end方式来采用的slice设计.让其变成N+K的长度和c统一长度。[0, 1, xxx, n, n, n, n]
        sliced_knot_vector = knot_vector[self.degree:-1] # 不能把第一个knot完全删除调, 所以变成这样
        all_control_points.append(gripper_traj[sliced_knot_vector]) # 加上gripper的控制点
        all_control_points.append(sliced_knot_vector) # 最后加上internal knot的参数

        # 将控制点组合成 [n_control_points, dim+1] 的数组，这里长度不包含前后重复的端点，就是target length
        control_points = np.column_stack(all_control_points)
        assert len(control_points) == self.target_length 
        return control_points


    def __call__(self, aff_trajectory: np.ndarray, **kwargs) -> np.ndarray:
        """
        使用固定数量的内部节点进行B样条最小二乘拟合，然后slice到后续的实际的trajectory
        """
        frame_index, episode_index = kwargs['frame_index'], kwargs['episode_index']
        
        # NOTE：对于xyz,yaw,pitch,row进行类加，在绝对位置进行学习
        # print("before aff:", aff_trajectory[:5])
        aff_trajectory[:, :-1] = np.cumsum(aff_trajectory[:, :-1], axis=0)
        # print("aff:", aff_trajectory[:5])
        if episode_index in self.episode_cache.keys():
            full_traj = self.episode_cache[episode_index]
        else:
            full_traj = self.compress_to_control_point(aff_trajectory=aff_trajectory)
            self.episode_cache[episode_index] = full_traj
        
        idx = np.searchsorted(full_traj[:,-1], frame_index)

        predict_values = full_traj[idx:, :]
        predict_values[:, -1] = predict_values[:, -1] - frame_index.item()  # NOTE: 时间参数化调整为从0开始
        # print("predict values:", predict_values)
        return predict_values

    def decode_to_action(
        self, control_points: np.ndarray, current_eef_pose: np.ndarray
    ) -> np.ndarray:
        """从控制点解码回aff_trajectory。 不管用A还是S，都是当前pose到预测的这些点上去移动，设置移动的速度。"""
        assert control_points.shape[0] >= 3, "控制点数量不足degree + 1，无法解码轨迹。"

        current_eef_pose = np.append(current_eef_pose, 0)  # 添加时间维度，当前时间为0  

        internal_knot_vector = control_points[:, -1] # 最后一个dim的knot vector
        knot_vector = np.concatenate([
            np.repeat(current_eef_pose[-1], self.degree + 1),  # 起始重复节点, 因为起始的slice方式让其丢失开始的点？ TODO，没太理解
            internal_knot_vector,
            np.repeat(internal_knot_vector[-1], 1)  # 结束重复节点
        ])
        extended_cp = np.vstack([current_eef_pose[:6], control_points[:, :6]]) # 前六维是aff的
        bspline = BSpline(knot_vector, extended_cp, self.degree) # 前六维是aff的

        # 目前实现中action没有作用
        return None, bspline





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

    def __call__(self, trajectory: np.ndarray, **kwargs) -> np.ndarray:
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

