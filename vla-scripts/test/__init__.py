"""
Test suite for OpenVLA Processor Wrapper.

This module tests the processor wrapper implementation to ensure:
1. Each processor step works correctly in isolation
2. The complete pipeline works end-to-end
3. The wrapped model produces identical outputs to the original
"""

# Make test directory importable
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
