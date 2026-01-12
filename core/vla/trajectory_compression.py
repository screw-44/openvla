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

        compression_json = Path(__file__).parent.parent.parent / "assets" / "compression_results_processed.json"
        with open(compression_json, "r") as f:
            offline_compression_results = json.load(f)
        
        self.episode_cache = offline_compression_results["episodes"]  # 存储每个episode的B样条数据

    # cp代表compression points
    def get_cp_from_episode(self, episode_index: int):
        episode_index = str(int(episode_index)) # 本来是tensor，转成int再转成str
        if episode_index not in self.episode_cache.keys():
            raise ValueError(f"Episode index {episode_index} not found in compression results.")
        
        episode_data = self.episode_cache[episode_index]
        bspline_data = episode_data["bspline"]
        control_points = np.array(bspline_data["control_points"])
        knot_vector = np.array(bspline_data["knot_vector"]) # 取第一个维度的knot vector
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
    
    # 为了服务decode_to_action, 还需要添加start_reanchor_expoential()方法, 起步的时候消除静态误差， decay_sigma控制指数衰减的快慢(时间)
    def start_reanchor_expoential(self, pred_points, current_eef_points, decay_sigma=0.5, degree=3):
        # 0. 重新组合成控制点和knot vector
        control_points = pred_points[:, :6]  # 只处理前六维的x, y, z, yaw, pitch, roll
        knot_vector = pred_points[:, -1]  # 最后一个dim的knot vector
        knot_vector = np.concatenate([
            np.repeat(knot_vector[0], degree),
            knot_vector,
            np.repeat(knot_vector[-1], 1)
        ])

        # 1. 计算 t=0 处的预测位置和误差
        temp_spline = BSpline(knot_vector, control_points, k=degree)
        pred_start = temp_spline(0.0)
        current_eef_pose = current_eef_points[:6] # 只取前六维
        delta = current_eef_pose - pred_start # 误差向量
        
        # 2. 找到 t=0 时刻相关的控制点索引
        # 在 B-spline 中，t=0 受 index 为 [idx_0 - degree] 到 [idx_0] 的控制点影响
        # searchsorted 找到 0 应该插入的位置，减 1 就是最后那个影响 t=0 的 knot 的索引
        # 注意：B-spline 定义域通常是 [knots[degree], knots[-degree-1]]
        
        # 我们不仅要平移 t=0，还要平移 t<0 的历史，所以我们要找到所有影响 t<=0 的控制点
        # 简单做法：找到 t=0 所在的 knot span 对应的最后一个控制点索引
        span_idx = np.searchsorted(knot_vector, 0.0, side='right') - 1
        
        # 3. 构建局部掩码 (Local Mask)
        N_cp = len(control_points)
        mask = np.zeros(N_cp)
        
        # --- A. 历史锁定区 (t <= 0) ---
        # 影响 t=0 及其之前的所有控制点，权重必须是 1
        # 这样保证 t=0 处的 位置+速度+加速度 整体平移，没有顿挫
        # 对于 p=3, 影响 t=0 的控制点是 span_idx, span_idx-1, span_idx-2, span_idx-3
        history_end_idx = span_idx 
        mask[:history_end_idx + 1] = 1.0 
        
        # --- B. 指数衰减区 (t > 0) ---
        # 从 history_end_idx 后面开始衰减
        # 我们需要知道每个控制点大致对应的时间，才能做指数衰减
        # Greville Abscissae (格雷维尔坐标) 是估算控制点对应时间的标准方法
        # t_i = sum(knots[i+1 : i+p+1]) / p
        
        for i in range(history_end_idx + 1, N_cp):
            # 计算该控制点对应的“物理时间”
            # 取该控制点支撑区间的 knot 平均值
            knot_sub = knot_vector[i+1 : i+degree+1]
            if len(knot_sub) == 0: break # 边界保护
            t_approx = np.mean(knot_sub)
            
            # 如果这个控制点的时间已经很大了，说明是后续操作，不用改
            if t_approx > 0:
                # 指数衰减公式: w = exp(-t / sigma)
                # t越大，w越接近0
                w = np.exp(-t_approx / decay_sigma)
                
                # 截断：如果权重已经很小了，直接置0，保证绝对安全
                if w < 0.01:
                    w = 0.0
                
                mask[i] = w
            else:
                # 理论上 t_approx > 0 的分支应该覆盖后面，这里以防万一
                mask[i] = 1.0

        # 4. 应用修正
        # correction: (N, 1) * (1, Dim) = (N, Dim)
        correction = mask[:, np.newaxis] * delta[np.newaxis, :]
        new_control_points = control_points + correction

        bpsline = BSpline(knot_vector, new_control_points, k=degree)
        return bpsline, knot_vector

    def decode_to_action(
        self, control_points: np.ndarray, current_eef_pose: np.ndarray
    ) -> Tuple:
        """从控制点解码回aff_trajectory。 不管用A还是S，都是当前pose到预测的这些点上去移动，设置移动的速度。"""
        assert control_points.shape[0] >= 3, "控制点数量不足degree + 1，无法解码轨迹。"
        # x, y, z, yaw, pitch, roll, ---,  knot_vector
        bspline, knot_vector = self.start_reanchor_expoential(control_points, current_eef_pose)

        # 2. 解码 Gripper 轨迹 (0阶/Step Function)
        # internal_knot_vector 已经是相对时间了 (例如: -3, -2, -1, 0, 5, 10...)
        internal_knot_vector = control_points[:, -1]
        gripper_cp = control_points[:, 6]
        # print("internal_knot_vector:", internal_knot_vector)
        print("knots for gripper:", knot_vector)
        print("gripper cp:", gripper_cp)
        
        # 确定评估时间点：从 0 (当前) 开始，直到轨迹结束
        # 向上取整以覆盖最后的时间段
        max_time = np.ceil(knot_vector[-1])
        t_eval = np.arange(0, int(max_time) + 1)

        gripper_cp = control_points[:, 6]
        knot_span_indices = np.searchsorted(knot_vector, t_eval, side='right')
        gripper_indices = knot_span_indices - (self.degree - 1)
        
        # 边界保护：防止 t_eval 比第一个 knot 还小 (变成 -1) 或 比最后一个还大
        # 对于 t < 第一个knot，通常保持第一个状态
        gripper_indices = np.clip(gripper_indices, 0, len(gripper_cp) - 1)
        print("gripper indices:", gripper_indices)
        
        gripper_traj = gripper_cp[gripper_indices]
        return bspline, gripper_traj

    # TODO：在decode_to_action之后， 还需要在笛卡尔空间进行平滑

