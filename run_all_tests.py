#!/usr/bin/env python
"""
Comprehensive test runner for karray that runs:
- All pytest tests
- All doctests
- With both dense and sparse backends
"""
import os
import sys
import subprocess
import doctest
from src.karray import source_code, settings


def run_doctests(verbose=False):
    """Run doctests for the karray package."""
    print("="*80)
    print(f"Running doctests with {settings.data_type} backend")
    print("="*80)

    # Ensure test data directory exists
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'data'), exist_ok=True)

    # Run doctests with flexible whitespace and ellipsis matching
    failure_count, test_count = doctest.testmod(
        source_code,
        verbose=verbose,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )

    return failure_count, test_count


def run_pytest(verbose=False, coverage=False):
    """Run pytest tests."""
    print("="*80)
    print(f"Running pytest tests with {settings.data_type} backend")
    print("="*80)

    cmd = ["python", "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src/karray", "--cov-report", "term"])

    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Run all tests with both backends."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    coverage = "--cov" in sys.argv

    failures = 0

    # Run tests with dense backend
    settings.data_type = 'dense'
    pytest_result = run_pytest(verbose, coverage)
    failures += pytest_result

    doctest_failures, _ = run_doctests(verbose)
    failures += doctest_failures

    # Run tests with sparse backend
    settings.data_type = 'sparse'
    pytest_result = run_pytest(verbose, coverage)
    failures += pytest_result

    doctest_failures, _ = run_doctests(verbose)
    failures += doctest_failures

    # Show summary
    print("="*80)
    if failures == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failures} test failures detected")
    print("="*80)

    return failures


if __name__ == "__main__":
    sys.exit(main())
