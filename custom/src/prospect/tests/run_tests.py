"""
Test runner for PROSPECT
"""
import sys
import subprocess
from pathlib import Path


def run_all_tests():
    """Run all PROSPECT tests using pytest"""
    print("🧪 Running PROSPECT Tests...")
    print("=" * 60)
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Run pytest with verbose output
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        "-v",
        "--tb=short",
        "--color=yes",
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    return result.returncode


def run_custom_model_tests():
    """Run only custom model tests"""
    print("🧪 Running Custom Model Tests...")
    print("=" * 60)
    
    tests_dir = Path(__file__).parent
    test_file = tests_dir / "test_custom_model.py"
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_integration_tests():
    """Run integration tests"""
    print("🧪 Running Integration Tests...")
    print("=" * 60)
    
    tests_dir = Path(__file__).parent
    test_file = tests_dir / "test_runner_integration.py"
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_strategy_tests():
    """Run context strategy tests"""
    print("🧪 Running Context Strategy Tests...")
    print("=" * 60)
    
    tests_dir = Path(__file__).parent
    test_file = tests_dir / "test_context_strategies.py"
    
    if not test_file.exists():
        print(f"⚠️  Test file not found: {test_file}")
        return 0
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_quick_tests():
    """Run quick smoke tests only"""
    print("🧪 Running Quick Smoke Tests...")
    print("=" * 60)
    
    tests_dir = Path(__file__).parent
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        "-v",
        "-m", "not slow",  # Skip tests marked as slow
        "--tb=short",
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner with menu"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PROSPECT Test Runner")
    parser.add_argument(
        "--suite",
        choices=["all", "custom_model", "integration", "strategy", "quick"],
        default="all",
        help="Test suite to run",
    )
    
    args = parser.parse_args()
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed!")
        print("Install it with: pip install pytest")
        return 1
    
    print("\n" + "=" * 60)
    print("PROSPECT Test Runner")
    print("=" * 60 + "\n")
    
    # Run selected test suite
    if args.suite == "all":
        exit_code = run_all_tests()
    elif args.suite == "custom_model":
        exit_code = run_custom_model_tests()
    elif args.suite == "integration":
        exit_code = run_integration_tests()
    elif args.suite == "strategy":
        exit_code = run_strategy_tests()
    elif args.suite == "quick":
        exit_code = run_quick_tests()
    else:
        print(f"❌ Unknown test suite: {args.suite}")
        exit_code = 1
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print("=" * 60 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
