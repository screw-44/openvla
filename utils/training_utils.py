import re
from pathlib import Path
from typing import Optional, Tuple

from prismatic.overwatch.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

# === Utility Function: Extract Step and Epoch from Checkpoint Path ===
def extract_step_epoch_from_checkpoint(checkpoint_path: Optional[Path]) -> Tuple[int, int]:
    """
    从 checkpoint 文件路径中提取 step 和 epoch 信息
    
    预期文件名格式示例:
        - step-010000-epoch-01-loss=0.2341.pt
        - step-005000-epoch-00.pt
    Returns:
        (resume_step, resume_epoch) 元组，如果无法解析则返回 (0, 0)
    """
    if checkpoint_path is None:
        return 0, 0
    
    filename = checkpoint_path.name
    # 尝试匹配 step-XXXXXX-epoch-XX 格式
    step_match = re.search(r'step-(\d+)', filename)
    epoch_match = re.search(r'epoch-(\d+)', filename)
    
    resume_step = int(step_match.group(1)) if step_match else 0
    resume_epoch = int(epoch_match.group(1)) if epoch_match else 0
    return resume_step, resume_epoch

def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """
    在指定目录下查找最新的 checkpoint 文件
    
    Args:
        run_dir: 运行目录（包含 checkpoints 子目录）
        
    Returns:
        最新的 checkpoint 文件路径，如果没有找到则返回 None
    """
    checkpoint_dir = run_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        overwatch.info(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # 查找所有 .pt 文件
    checkpoint_files = list(checkpoint_dir.glob("*.safetensors"))
    
    if not checkpoint_files:
        overwatch.info(f"No checkpoint files found in: {checkpoint_dir}")
        return None
    
    # 提取每个文件的 step 数字，找到最大的
    checkpoints_with_steps = []
    for ckpt_path in checkpoint_files:
        step, _ = extract_step_epoch_from_checkpoint(ckpt_path)
        checkpoints_with_steps.append((step, ckpt_path))
    
    # 按 step 排序，取最大的
    checkpoints_with_steps.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = checkpoints_with_steps[0][1]
    
    overwatch.info(f"Found latest checkpoint: {latest_checkpoint.name} (step={checkpoints_with_steps[0][0]})")
    return latest_checkpoint
