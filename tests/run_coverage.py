#!/usr/bin/env python3
"""
Coverage runner script for Infinite RL project.

This script provides convenient commands for running tests with coverage analysis.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_coverage():
    """Run tests with coverage analysis."""
    project_root = Path(__file__).parent.parent
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=infinite_rl",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80",
        "tests/",
    ]

    print("Running tests with coverage analysis...")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def update_badge():
    """Update coverage badge in README.md"""
    script_path = Path(__file__).parent / "update_coverage_badge.py"
    project_root = Path(__file__).parent.parent
    if script_path.exists():
        print("Updating coverage badge in README...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=project_root)
        return result.returncode
    else:
        print("Badge update script not found")
        return 1


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "run":
            sys.exit(run_coverage())
        elif command == "view":
            view_coverage()
        elif command == "badge":
            sys.exit(update_badge())
        elif command == "all":
            # Run coverage and update badge
            if run_coverage() == 0:
                update_badge()
            else:
                print("Tests failed, skipping badge update")
                sys.exit(1)
        else:
            print("Usage: python run_coverage.py [run|view|badge|all]")
            print("  run   - Run tests with coverage analysis")
            print("  view  - Open HTML coverage report in browser")
            print("  badge - Update coverage badge in README")
            print("  all   - Run tests and update badge")
            sys.exit(1)
    else:
        # Default: run coverage
        sys.exit(run_coverage())


if __name__ == "__main__":
    main()
