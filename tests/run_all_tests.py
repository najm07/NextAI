"""
Run all Track 1 tests.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tests.test_env_determinism import run_all_tests as run_env_tests
from tests.test_one_hot import run_all_tests as run_one_hot_tests
from tests.test_swap_invariance import run_all_tests as run_swap_tests
from tests.test_agent import run_all_tests as run_agent_tests


def main():
    """Run all test suites."""
    print("=" * 60)
    print("Track 1 (v0.1-A) Test Suite")
    print("=" * 60)
    
    try:
        run_env_tests()
        run_one_hot_tests()
        run_swap_tests()
        run_agent_tests()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

