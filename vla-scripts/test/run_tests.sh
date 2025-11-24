#!/bin/bash
# run_tests.sh
#
# Convenience script to run all tests for the OpenVLA processor wrapper.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo "OpenVLA Processor Wrapper Test Suite"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "test_consistency.py" ]; then
    echo -e "${RED}Error: Not in test directory!${NC}"
    echo "Please run from: Project/Aff/vla/vla-scripts/test/"
    exit 1
fi

# Print test environment info
echo -e "${BLUE}Test Environment Info${NC}"
echo "----------------------------------------"

# Check for checkpoint
DEFAULT_CKPT="../../runs/base+b32+x7--aff_representation_251117-action_chunk/checkpoints/latest-checkpoint.pt"
if [ -f "$DEFAULT_CKPT" ]; then
    echo -e "${GREEN}✓${NC} Default checkpoint found: $DEFAULT_CKPT"
else
    echo -e "${RED}✗${NC} Default checkpoint NOT found: $DEFAULT_CKPT"
    echo "  Tests will fail. Please check your runs/ directory."
    exit 1
fi

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠️${NC}  Warning: HF_TOKEN not set"
    echo "  Some model loading may use cached credentials"
else
    echo -e "${GREEN}✓${NC} HF_TOKEN is set"
fi

# Check HuggingFace cache
HF_CACHE="$HOME/.cache/huggingface"
if [ -d "$HF_CACHE" ]; then
    echo -e "${GREEN}✓${NC} HuggingFace cache: $HF_CACHE"
else
    echo -e "${YELLOW}⚠️${NC}  HuggingFace cache not found: $HF_CACHE"
fi

echo ""
echo "================================================"
echo "Running Tests"
echo "================================================"
echo ""

# Function to run a test file
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo -e "${YELLOW}Running: $test_name${NC}"
    echo "----------------------------------------"
    
    if python "$test_file"; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        echo ""
        return 1
    fi
}

# Track results
total_tests=0
passed_tests=0

# Run processor steps tests
if run_test "test_processor_steps.py" "Processor Steps Tests"; then
    ((passed_tests++))
fi
((total_tests++))

# Run consistency tests (most important!)
echo -e "${YELLOW}⚠️  Running critical consistency tests...${NC}"
echo "These tests verify wrapped model matches original implementation"
echo ""
if run_test "test_consistency.py" "Consistency Tests"; then
    ((passed_tests++))
fi
((total_tests++))

# Run integration tests
if run_test "test_integration.py" "Integration Tests"; then
    ((passed_tests++))
fi
((total_tests++))

# Print summary
echo "================================================"
echo "Test Summary"
echo "================================================"
echo -e "Passed: ${GREEN}$passed_tests${NC} / $total_tests"

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo ""
    echo "The processor wrapper is working correctly and produces"
    echo "identical outputs to the original VLA implementation."
    echo ""
    echo "You can now use it for evaluation with lerobot_eval!"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please fix failing tests before using the wrapper."
    exit 1
fi
