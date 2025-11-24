"""
README for OpenVLA Processor Wrapper Tests

This directory contains tests for the OpenVLA processor wrapper implementation.

## Test Files

1. **test_processor_steps.py**
   - Unit tests for individual processor steps
   - Tests trajectory compression, action decoding, etc.
   - Run: `python test_processor_steps.py`

2. **test_consistency.py** (MOST IMPORTANT)
   - Verifies wrapped model produces identical outputs to original
   - Compares action predictions, tokenizer decoding, parameters
   - Run: `python test_consistency.py`
   - ⚠️ Update checkpoint_path before running!

3. **test_integration.py**
   - Integration tests for complete workflows
   - Tests policy initialization, action selection, multi-step rollouts
   - Run: `python test_integration.py`

## Setup

Before running tests, update checkpoint paths in:
- `test_consistency.py`: Line ~40
- `test_integration.py`: Lines where checkpoint_path is defined

Set your checkpoint path to a valid trained model, e.g.:
```python
checkpoint_path = Path(
    "/path/to/your/runs/YOUR_RUN_ID/checkpoints/step-010000-epoch-01.pt"
)
```

## Running Tests

### Run all tests:
```bash
cd /inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/vla-scripts/test
python -m pytest -v
```

### Run specific test file:
```bash
python test_consistency.py
```

### Run with HuggingFace token:
```bash
export HF_TOKEN=your_token_here
python test_consistency.py
```

## Expected Results

All tests should PASS with exact numerical matches (within tolerance).

If `test_consistency.py` passes, it means:
✓ Wrapped model produces identical predictions to original
✓ No bugs introduced by wrapper
✓ Safe to use for evaluation

## Troubleshooting

**Import errors:**
- Make sure you're in the correct directory
- Check that `prismatic` package is importable

**Checkpoint not found:**
- Update checkpoint paths in test files
- Ensure checkpoint file exists

**HuggingFace token issues:**
- Set HF_TOKEN environment variable
- Or pass token to test functions

## Next Steps

After tests pass:
1. Use wrapped policy for lerobot_eval
2. Run evaluation in LIBERO environment
3. Compare results with original training code
