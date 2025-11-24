# OpenVLA Processor Wrapper - Quick Start Guide

## 📁 What Was Implemented

A processor wrapper system that makes OpenVLA compatible with LeRobot evaluation framework, **without modifying any lerobot library code**. All code is in `Project/Aff/vla/`.

### File Structure

```
Project/Aff/vla/
├── prismatic/vla/processor_wrapper/    (NEW)
│   ├── __init__.py                     - Package exports
│   ├── policy_wrapper.py               - Main wrapper class
│   ├── processor_steps.py              - Individual processing steps
│   └── processor_factory.py            - Pipeline factory
│
└── vla-scripts/test/                   (NEW)
    ├── __init__.py
    ├── test_processor_steps.py         - Unit tests for steps
    ├── test_consistency.py             - Critical: tests output matches original
    ├── test_integration.py             - Integration tests
    ├── eval_with_wrapper.py            - Example evaluation script
    ├── run_tests.sh                    - Run all tests
    └── README.md                       - Test documentation
```

## 🚀 Quick Start

### 1. Run Tests

**IMPORTANT**: Before using the wrapper, verify it works correctly!

```bash
cd /inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/vla-scripts/test

# First, update checkpoint paths in test_consistency.py and test_integration.py
# Then run tests:

export HF_TOKEN=your_hf_token_here
chmod +x run_tests.sh
./run_tests.sh
```

Expected output:
```
✓ Processor Steps Tests PASSED
✓ Consistency Tests PASSED          <- Most important!
✓ Integration Tests PASSED
✓ ALL TESTS PASSED!
```

### 2. Use the Wrapper

#### Option A: Simple Demo (No Environment)

```bash
python eval_with_wrapper.py \
    --checkpoint /path/to/checkpoint.pt \
    --mode demo \
    --num-steps 10
```

#### Option B: Compare with Original

```bash
python eval_with_wrapper.py \
    --checkpoint /path/to/checkpoint.pt \
    --mode compare
```

#### Option C: Full LIBERO Evaluation

```bash
python eval_with_wrapper.py \
    --checkpoint /path/to/checkpoint.pt \
    --mode libero \
    --task-suite libero_spatial \
    --num-episodes 50
```

### 3. Use in Your Own Code

```python
from prismatic.vla.processor_wrapper import OpenVLAPolicyWrapper

# Load policy
policy = OpenVLAPolicyWrapper.from_pretrained(
    "/path/to/checkpoint.pt",
    device="cuda",
)

# Use in environment loop
for step in range(max_steps):
    observation = {
        'full_image': env.get_image(),  # numpy array [H,W,3]
        'task': 'pick up the object',
        'unnorm_key': 'libero_spatial',
    }
    
    action = policy.select_action(observation)
    env.step(action)
```

## 🔍 How It Works

### Data Flow

```
Original VLA Flow:
LeRobotDataset → get_trajectory_for_item → trajectory_compression → 
VlaTokenizer.tokenize_batch → model.forward → predict_action

Wrapped Flow (Same Logic, Different Structure):
Observation → TrajectoryRetrievalStep → TrajectoryCompressionStep → 
VLATokenizerStep → model.forward → VLAActionDecoderStep → Action
```

### Key Components

1. **OpenVLAPolicyWrapper** (`policy_wrapper.py`)
   - Main wrapper class
   - Provides `select_action()` for inference
   - Loads VLA using existing `load_vla()` function
   - No modification to VLA internals

2. **Processor Steps** (`processor_steps.py`)
   - `TrajectoryRetrievalProcessorStep`: Replicates `get_trajectory_for_item` logic
   - `TrajectoryCompressionProcessorStep`: Wraps trajectory compression
   - `VLATokenizerProcessorStep`: Wraps VLA tokenization
   - `VLAActionDecoderProcessorStep`: Decodes actions from tokens

3. **SimplePipeline** (`processor_factory.py`)
   - Chains steps together
   - Independent of lerobot's ProcessorPipeline
   - Lightweight custom implementation

## ✅ Testing Strategy

### Critical Test: `test_consistency.py`

This test ensures the wrapper doesn't introduce bugs:

```python
# Compares:
1. Action predictions (wrapped vs original)
2. Action tokenizer decoding
3. Model parameters
4. Normalization statistics

# If this passes, the wrapper is safe to use!
```

### Why This Matters

- Wrapper should be **transparent** - no numerical changes
- Same checkpoint → same predictions
- Any difference = bug in wrapper

## 📊 What's Preserved

✓ Exact same `load_vla()` loading mechanism  
✓ Exact same `get_trajectory_for_item` logic  
✓ Exact same trajectory compression methods  
✓ Exact same VlaTokenizer behavior  
✓ Exact same action prediction  
✓ No changes to lerobot library  

## 🔧 Configuration

The wrapper automatically loads configuration from checkpoint's `config.json`:

```json
{
  "vla": { ... },
  "dataset": {
    "repo_id": "HuggingFaceVLA/libero",
    "task_ids": [0, 1, 2],
    "trajectory_compression": "uniform_bspline",
    "exp_type": "positional"
  }
}
```

## 🐛 Troubleshooting

### Tests Fail with "Checkpoint not found"

Update checkpoint paths in test files:
- `test_consistency.py`: Line ~40
- `test_integration.py`: Multiple locations

### Import Errors

```bash
cd /path/to/Project/Aff/vla
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### HuggingFace Token Issues

```bash
export HF_TOKEN=your_token
```

Or create `.hf_token` file in project root.

### Actions Don't Match Original

This is a BUG! Check:
1. Are you using the same checkpoint?
2. Did you modify any processor step logic?
3. Run `test_consistency.py` to diagnose

## 🎯 Next Steps

After tests pass:

1. **Use for Evaluation**
   ```bash
   python eval_with_wrapper.py --checkpoint /path/to/ckpt --mode libero
   ```

2. **Integrate with lerobot_eval** (if available)
   - Wrapper provides `select_action()` interface
   - Compatible with standard evaluation scripts

3. **Compare with Training Results**
   - Verify evaluation results match training performance
   - Use same checkpoint for fair comparison

## 📝 Important Notes

### What This Wrapper Does NOT Do

❌ Modify lerobot library code  
❌ Change VLA training logic  
❌ Replace your training pipeline  
❌ Require retraining models  

### What This Wrapper DOES Do

✅ Provides LeRobot-compatible interface  
✅ Wraps existing VLA for evaluation  
✅ Maintains exact same predictions  
✅ Enables lerobot_eval compatibility  

## 🤝 Usage Tips

1. **Always run tests first**: Verify wrapper works correctly
2. **Check consistency**: Use `--mode compare` before evaluation
3. **Use same checkpoint**: For fair comparison with training
4. **Monitor memory**: Wrapper adds minimal overhead, but be aware

## 📚 Additional Resources

- Test README: `vla-scripts/test/README.md`
- Example evaluation: `vla-scripts/test/eval_with_wrapper.py`
- Original VLA code: `prismatic/models/vlas/openvla.py`

---

**Questions or Issues?**

If tests fail or wrapper behaves unexpectedly, check:
1. Checkpoint path is correct
2. HF_TOKEN is set
3. All dependencies are installed
4. Using correct Python environment

The wrapper is designed to be **zero-impact** - if something doesn't work,
it's likely a configuration issue, not a fundamental problem with the approach.
