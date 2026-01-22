import json

compression_result = json.load(open("../assets/compression_results_v2.json", "r"))

print(compression_result["episodes"]["0"])
episode_0 = compression_result["episodes"]["0"]["bspline"]
knots_vector = episode_0["knots_vector"]
contorl_points = episode_0["control_points"]
print("knots_vector shape:", len(knots_vector), "knots_vector:", knots_vector)
print("control_points shape:", f"{len(contorl_points)}*{len(contorl_points[0])}, last{len(contorl_points[-1])}", "control_points:", contorl_points)

print("control point 0, 1, 2: \n", contorl_points[0], "\n", contorl_points[1], "\n", contorl_points[2])

episodes = compression_result["episodes"]
count = 0
for episode_id in episodes:
    episode = episodes[episode_id]["bspline"]
    knots_vector = episode["knots_vector"]
    min_knot, max_knot = min(knots_vector), max(knots_vector)
    if len(knots_vector) <= 8:
        print(f"Episode {episode_id} has less than 8 knots: {len(knots_vector)}")
        continue
    internal_knots = knots_vector[4:-4]
    min_internal_knot, max_internal_knot = min(internal_knots), max(internal_knots)
    if min_internal_knot < 4 or max_internal_knot > max_knot - 3:
        print(f"Episode {episode_id} has internal knots out of range: {min_internal_knot}, {max_internal_knot}. Full range: {min_knot} to {max_knot}")
        count += 1
    # print(f"Episode {episode_id}:")
    # print("  knots_vector shape:", len(knots_vector))
    # print("  control_points shape:", f"{len(contorl_points)}*{len(contorl_points[0])}, last{len(contorl_points[-1])}")
print("counts are:", count)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "HuggingFaceVLA/libero"
dataset = LeRobotDataset(repo_id=repo_id)
print("dataset meta info(fps/features)", dataset.meta.info)
print(dataset[0].keys())
print(dataset[0]["observation.images.image"].shape)


import shutil
from pathlib import Path
import numpy as np
import copy
import torch
from PIL import Image
from tqdm import tqdm

# 删除已存在的数据集目录
dataset_path = Path.home() / "Software/huggingface/lerobot/HxyFace/libero_cp"
if dataset_path.exists():
    shutil.rmtree(dataset_path)
    print(f"Deleted existing dataset at {dataset_path}")

cp_dataset = LeRobotDataset.create(
    repo_id="HxyFace/libero_cp",
    fps=10,
    features={
        'observation.images.image': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel'], 'fps': 10.0}, 
        'observation.images.image2': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel'], 'fps': 10.0}, 
        'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': ['state'], 'fps': 10.0}, 
        'action': {'dtype': 'float32', 'shape': (7,), 'names': ['actions'], 'fps': 10.0},  # 变成control point 
        'timestamp': {'dtype': 'float32', 'shape': (1,), 'names': None, 'fps': 10.0}, 
        'frame_index': {'dtype': 'int64', 'shape': (1,), 'names': None, 'fps': 10.0}, 
        'episode_index': {'dtype': 'int64', 'shape': (1,), 'names': None, 'fps': 10.0}, 
        'index': {'dtype': 'int64', 'shape': (1,), 'names': None, 'fps': 10.0}, # 注意这里要处理
        'task_index': {'dtype': 'int64', 'shape': (1,), 'names': None, 'fps': 10.0},
        'predict_status': {'dtype': 'int64', 'shape': (1,), 'names': None, 'fps': 10.0} # 这里放string，step的类别。（ 0，关键点 1，中间点 2）
    }
    )

def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] in (1,3,4):  # CHW
        x = x.permute(1,2,0)                  # HWC
    # float -> uint8
    if x.dtype != torch.uint8:
        x = (x.clamp(0,1) * 255).to(torch.uint8)
    return Image.fromarray(x.numpy())

重复点 = np.array([0,])
关键点 = np.array([1,])
中间点 = np.array([2,])

# 变成 [0, xxx, end * 3] 的长度，和contorl point对齐，最后end+1， end+2
index_offset = 0
for step_data in tqdm(dataset):
    episode_index = step_data["episode_index"].item()
    frame_index = step_data["frame_index"].item()
    episode_compression_data =  copy.deepcopy(compression_result["episodes"][str(step_data["episode_index"].item())])
    episode_knot_vector = episode_compression_data["bspline"]["knots_vector"]
    filtered_episode_knot_vector = episode_knot_vector[3:] # 去掉前3个重复的
    frame_knot_index = filtered_episode_knot_vector.index(frame_index) if frame_index in filtered_episode_knot_vector else -1
    # print("frame_knot_index at:", frame_knot_index)
    # print("filtered_episode_knot_vector:", filtered_episode_knot_vector)
    # print("frame_index at:", step_data["frame_index"].item())

    control_points = episode_compression_data["bspline"]["control_points"]
    # print("last control point length before trim:", len(control_points[-1]))
    control_points[-1].extend([-1.0] * 3) # 补齐control point的长度
    episode_contorl_points = np.array(control_points).T
    # print("episode_contorl_points shape:", episode_contorl_points.shape)
    # print("knot vector shape:", len(episode_knot_vector))

    # 处理一下图像
    step_data["observation.images.image"] = tensor_to_image(step_data['observation.images.image'])
    step_data["observation.images.image2"] = tensor_to_image(step_data['observation.images.image2'])


    step_data_frame_index = step_data["frame_index"].item()


    REMOVE_KEY = {'episode_index', 'timestamp', 'index', 'task_index', 'frame_index'}
    for key in REMOVE_KEY:
        step_data.pop(key, None)

    if step_data_frame_index not in episode_knot_vector:
        step_data["predict_status"] = 中间点
        # step_data["index"] = step_data["index"] + index_offset
        cp_dataset.add_frame(step_data)
    elif step_data_frame_index == max(episode_knot_vector):
        # 结尾1
        step_data["predict_status"] = 关键点
        step_data["action"] = np.array(episode_contorl_points[frame_knot_index], dtype=np.float32)
        cp_dataset.add_frame(copy.deepcopy(step_data))
        # print("Adding end control point at frame_index:", step_data)
        # 重复2
        step_data["predict_status"] = 重复点
        step_data["action"] = np.array(episode_contorl_points[frame_knot_index+1], dtype=np.float32)
        # step_data["frame_index"] = step_data["frame_index"] + 1
        # step_data["index"] = step_data["index"] + 1 + index_offset
        cp_dataset.add_frame(copy.deepcopy(step_data))
        # 重复3
        step_data["predict_status"] = 重复点
        step_data["action"] = np.array(episode_contorl_points[frame_knot_index+2], dtype=np.float32)
        # step_data["frame_index"] = step_data["frame_index"] + 2
        # step_data["index"] = step_data["index"] + 2 + index_offset
        cp_dataset.add_frame(step_data)

        # 一个episode结束了，可以保存这个episode了
        cp_dataset.save_episode()
        # index_offset += 2  # 因为多加了2个点
    else:
        # 除了结尾外的关键点
        step_data["predict_status"] = 关键点
        step_data["action"] = np.array(episode_contorl_points[frame_knot_index], dtype=np.float32)
        # step_data["index"] = step_data["index"] + index_offset

        # print("step_data:", step_data)
        cp_dataset.add_frame(step_data)



    # print("step_data", step_data)

cp_dataset.finalize()

print("dataset has been processed and finalized.")
# cp_dataset.finalize()