#!/bin/bash
# LeRobot-Eval Integration for VLA Policy
#
# This script demonstrates the correct way to use lerobot-eval with a locally trained VLA policy.
#
# Problem Solved:
#   HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 
#   './runs/distilgpt2+b16+x7--1_test_action-chunk'
#
# Root Cause:
#   lerobot's PreTrainedConfig.from_pretrained() tries to download from HuggingFace Hub
#   when given a local path that doesn't have proper config.json format
#
# Solution:
#   1. Convert checkpoint to lerobot-compatible format with config.json + model.safetensors
#   2. Use the converted checkpoint path with --policy.path
#   3. lerobot will auto-discover policy type from config.json's "type" field
#
# Step 1: Convert checkpoint to lerobot format
echo "Step 1: Converting checkpoint to lerobot-compatible format..."
python scripts/convert_checkpoint_to_lerobot_format.py \
    --input_checkpoint "./runs/distilgpt2+b16+x7--1_test_action-chunk" \
    --output_dir "./lerobot_checkpoints/distilgpt2_vla"

if [ $? -ne 0 ]; then
    echo "❌ Checkpoint conversion failed!"
    exit 1
fi

echo ""
echo "Step 2: Running lerobot-eval with converted checkpoint..."
echo ""

# Step 2: Run lerobot-eval with the converted checkpoint
# Parameters explanation:
#   --policy.path: ✅ Use --policy.path (NOT --policy.id which doesn't exist)
#                     Points to converted checkpoint directory
#   --env.type:    Environment type (libero, pusht, etc.)
#   --env.task:    Specific task description for the environment
#   --eval.n_episodes: Number of episodes to evaluate
#   --eval.batch_size: Batch size for parallel evaluation
#   
# How lerobot loads the policy:
#   1. PreTrainedConfig.from_pretrained("./lerobot_checkpoints/distilgpt2_vla")
#      └─ Reads config.json
#   2. Extracts config.type = "vla"
#   3. factory.get_policy_class("vla") → returns VLAPolicy class
#   4. VLAPolicy.from_pretrained() → loads model.safetensors

# Run lerobot-eval with VLA policy registered
# Note: We need to import hf_wrapper before lerobot-eval to register VLAConfig
python -c "
import sys
import hf_wrapper  # Trigger @PreTrainedConfig.register_subclass('vla') decorator

# Now call lerobot-eval main with arguments
from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

# Set up arguments
sys.argv = [
    'lerobot-eval',
    '--policy.path=./lerobot_checkpoints/distilgpt2_vla',
    '--env.type=libero',
    '--env.task=libero_10',
    '--eval.n_episodes=10',
    '--eval.batch_size=1',
    '--policy.device=cuda',
    '--output_dir=./eval_results',
]

lerobot_eval_main()
"