"""

实现用b-spline对整段轨迹进行压缩，目前实现三种情况
1. 不压缩，直接插值biningg成定长的点
2. b-spline最小二乘法压缩成定长的点
3. b-spline压缩成不定长的点，使用padding来进行计算

"""

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.linalg import lstsq  # Compute least-squares solution to equation Ax = b.
from typing import Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import json
import copy

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
# offline 去用MILP拿到最佳knot，然后保存json对应的bspline拟合参数。
# ==============================================================================
@register_trajectory_compression("bspline_v3")
class BSplineTrajectoryCompressionV3(BaseTrajectoryCompression):
    def __init__(self):
        self.exp_type = "v3" # 代码设置不兼容了，这个不重要了
        self.degree = 3

        compression_json = Path(__file__).parent.parent.parent / "assets" / "compression_results_v2.json"
        with open(compression_json, "r") as f:
            offline_compression_results = json.load(f)
        
        self.episode_cache = offline_compression_results["episodes"]  # 存储每个episode的B样条数据
        print("self.episode_cache keys:", list(self.episode_cache.keys())[:10])

        self._cache_smoothed_trajectoy = None

    # cp代表compression points
    def get_cp_from_episode(self, episode_index: int):
        episode_index = str(int(episode_index)) # 本来是tensor，转成int再转成str
        if episode_index not in self.episode_cache.keys():
            raise ValueError(f"Episode index {episode_index} not found in compression results.")
        
        episode_data = copy.deepcopy(self.episode_cache[episode_index]) # 同一个episode可能会被多次调用，深拷贝避免修改原数据
        bspline_data = episode_data["bspline"]
        control_points = bspline_data["control_points"]
        # gripper这个dim进行补全
        control_points[-1].extend([-1,] * self.degree) # NOTE：补全 -1 (注意为占位符号)
        # print("control_points:", control_points)
        # print("controlpoints shape of row 0, and row -1:", len(control_points[0]), " ", len(control_points[-1]))
        control_points = np.array(control_points)
        knot_vector = np.array(bspline_data["knots_vector"]) 
        return control_points, knot_vector

    def compress_to_control_point(self):
        raise NotImplementedError("该方法不需要实现，直接从离线结果中获取控制点。")

    def __call__(self, frame_index, episode_index) -> np.ndarray:
        control_points, knot_vector = self.get_cp_from_episode(episode_index)
        # print("control_points:", control_points)
        # print("knot_vector:", knot_vector)

        full_traj = np.concatenate([control_points, [knot_vector[3:-1], ]], axis=0) # knot vector去掉前3个和后一个，这样实现长度一致（能contact）
        # 装一下
        full_traj = full_traj.T  # 转置成 [n_points, dim+1]
        # print("full traj", full_traj)
        # print("full_traj shape:", full_traj.shape)

        idx = np.searchsorted(full_traj[:,-1], frame_index)
        # NOTE: 我们设定需要拿到-3的点开始预测, 维持i后的轨迹一致
        slice_idx = max(0, idx - 3)

        predict_values = full_traj[slice_idx:, :]
        # NOTE: 对于如果前面的 - frame_index 变成负数的情况，需要进行调整
        predict_values[:, -1] = predict_values[:, -1] - frame_index.item()  # 时间参数化调整为从0开始
        # print("predict values:", predict_values)
        return predict_values
    
    # 从0到10的平滑过渡到曲线上
    def start_offset(self, current_eef_pos, target_pred_pos, decay_distance=10):
        delta = current_eef_pos[:6] - target_pred_pos[:6]
        t = np.arange(decay_distance)
        
        # 使用余弦函数生成 1 -> 0 的平滑曲线
        # cos(0) = 1, cos(pi) = -1
        # 变换公式: 0.5 * (1 + cos(pi * t / decay_distance))
        weights = 0.5 * (1 + np.cos(np.pi * t / decay_distance))
        
        correction_segment = np.outer(weights, delta)
        return correction_segment


    def decode_to_action(self, pred_points: np.ndarray) -> Tuple:
        """从控制点解码回aff_trajectory。 x, y, z, yaw, pitch, roll, ---,  knot_vector"""
        # assert pred_points.shape[0] >= self.degree + 1, "控制点数量不足degree + 1，无法解码轨迹。"

        control_points = pred_points[:, :6]  # 只处理前六维的x, y, z, yaw, pitch, roll
        knot_vector = pred_points[:, -1]  # 最后一个dim的knot vector
        knot_vector = np.concatenate([
            np.repeat(knot_vector[0], self.degree),
            knot_vector,
            np.repeat(knot_vector[-1], 1)
        ])

        # print("knot_vector after extend:", knot_vector)
        bspline = BSpline(knot_vector, control_points, k=self.degree)

        # print("control points shape:", control_points.shape)
        # print("control points:", control_points)

        # NOTE: gripper的bspline是0阶的step function, 注意padding的处理
        gripper_bspline = BSpline(knot_vector[self.degree:-self.degree], pred_points[:, 6][:-self.degree], k=0)

        return bspline, gripper_bspline


    def smooth_traj(
            self, pred_trajectory: np.ndarray, pred_gap: int, smooth_ratio: float = 1, reset: bool = False,
        ):
        """
            笛卡尔空间轨迹平滑

            传入参数：
            - pred_trajectory 是当前预测的控制点序列，从0时刻开始(笛卡尔空间）。
            - pred_gap：两次预测之间的时间间隔（帧数）。
            - smooth_ratio: 控制平滑的比例因子，范围在0到1之间。实际平滑的数量是 1 / smooth_ratio下的数据做平滑。
            - 是否reset：如果为True，表示重新开始平滑处理，一般在新episode开始时使用。
        """
        if self._cache_smoothed_trajectoy is None or reset:
            self._cache_smoothed_trajectoy = pred_trajectory
            return self._cache_smoothed_trajectoy
        
        if smooth_ratio == 1:
            self._cache_smoothed_trajectoy = pred_trajectory
            return self._cache_smoothed_trajectoy

        cached_traj = self._cache_smoothed_trajectoy[pred_gap:] # 去掉已经执行过的部分

        shared_length = min(len(pred_trajectory), len(cached_traj))
        print("shared_length for smoothing:", shared_length, ". pred_trajectory length:", len(pred_trajectory), 
              ", cached_traj length:", len(cached_traj))

        pred_trajectory[:shared_length] = (
            smooth_ratio * pred_trajectory[:shared_length] + (1 - smooth_ratio) * cached_traj[:shared_length] 
        ) 
        self._cache_smoothed_trajectoy = pred_trajectory
        return self._cache_smoothed_trajectoy
