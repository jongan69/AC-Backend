#!/usr/bin/env python3
"""
Test Runner Script for Travel API
Provides various options for running the automated test suite.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path

def run_tests_with_options(test_file="test_api.py", verbose=False, coverage=False, 
                          html_report=False, parallel=False, markers=None, 
                          exit_first=False, show_local_vars=False):
    """
    Run tests with specified options.
    
    Args:
        test_file (str): Path to test file
        verbose (bool): Run with verbose output
        coverage (bool): Generate coverage report
        html_report (bool): Generate HTML coverage report
        parallel (bool): Run tests in parallel
        markers (str): Run only tests with specific markers
        exit_first (bool): Exit on first failure
        show_local_vars (bool): Show local variables on failures
    """
    
    # Build pytest command
    cmd = ["python3", "-m", "pytest", test_file]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=api", "--cov-report=term-missing"])
        if html_report:
            cmd.extend(["--cov-report=html:htmlcov"])
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    if markers:
        cmd.extend(["-m", markers])
    
    if exit_first:
        cmd.append("-x")
    
    if show_local_vars:
        cmd.append("-l")
    
    # Add some default options for better output
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def run_specific_test_categories():
    """Run specific test categories with predefined configurations."""
    
    categories = {
        "basic": {
            "description": "Basic functionality tests (health, root endpoints)",
            "markers": "basic",
            "verbose": True
        },
        "categorization": {
            "description": "Transaction categorization tests",
            "markers": "categorization",
            "verbose": True
        },
        "hotels": {
            "description": "Hotel search tests",
            "markers": "hotels",
            "verbose": True
        },
        "flights": {
            "description": "Flight search tests",
            "markers": "flights",
            "verbose": True
        },
        "trip_planning": {
            "description": "Trip planning tests",
            "markers": "trip_planning",
            "verbose": True
        },
        "airbnb": {
            "description": "Airbnb search tests",
            "markers": "airbnb",
            "verbose": True
        },
        "airports": {
            "description": "Airport search tests",
            "markers": "airports",
            "verbose": True
        },
        "predicthq": {
            "description": "PredictHQ events tests",
            "markers": "predicthq",
            "verbose": True
        },
        "functional": {
            "description": "Simple functional endpoints tests",
            "markers": "functional",
            "verbose": True
        },
        "itinerary": {
            "description": "Itinerary tests",
            "markers": "itinerary",
            "verbose": True
        },
        "performance": {
            "description": "Performance tests",
            "markers": "performance",
            "verbose": True
        },
        "integration": {
            "description": "Integration tests",
            "markers": "integration",
            "verbose": True
        }
    }
    
    print("Available test categories:")
    for key, config in categories.items():
        print(f"  {key}: {config['description']}")
    
    choice = input("\nEnter category to run (or 'all' for all tests): ").strip().lower()
    
    if choice == "all":
        return run_tests_with_options(verbose=True, coverage=True)
    elif choice in categories:
        config = categories[choice]
        print(f"\nRunning {config['description']}...")
        return run_tests_with_options(markers=config["markers"], verbose=config["verbose"])
    else:
        print("Invalid choice. Running all tests...")
        return run_tests_with_options(verbose=True)

def main():
    """Main function to handle command line arguments and run tests."""
    
    parser = argparse.ArgumentParser(description="Run Travel API automated tests")
    parser.add_argument("--test-file", default="test_api.py", 
                       help="Path to test file (default: test_api.py)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Run with verbose output")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--html-report", action="store_true",
                       help="Generate HTML coverage report")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel")
    parser.add_argument("-m", "--markers",
                       help="Run only tests with specific markers")
    parser.add_argument("-x", "--exit-first", action="store_true",
                       help="Exit on first failure")
    parser.add_argument("-l", "--show-local-vars", action="store_true",
                       help="Show local variables on failures")
    parser.add_argument("--categories", action="store_true",
                       help="Run specific test categories")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only (no external API calls)")
    
    args = parser.parse_args()
    
    # Check if test file exists
    if not os.path.exists(args.test_file):
        print(f"Error: Test file '{args.test_file}' not found")
        return 1
    
    # Check if required dependencies are installed
    try:
        import pytest
        import httpx
    except ImportError as e:
        print(f"Error: Missing required dependencies. Please install test requirements:")
        print("pip install -r test_requirements.txt")
        return 1
    
    print("Travel API Automated Test Suite")
    print("=" * 40)
    
    if args.categories:
        return run_specific_test_categories()
    
    # Build test options
    test_options = {
        "test_file": args.test_file,
        "verbose": args.verbose,
        "coverage": args.coverage,
        "html_report": args.html_report,
        "parallel": args.parallel,
        "markers": args.markers,
        "exit_first": args.exit_first,
        "show_local_vars": args.show_local_vars
    }
    
    # Add quick test marker if requested
    if args.quick:
        if test_options["markers"]:
            test_options["markers"] += " and quick"
        else:
            test_options["markers"] = "quick"
    
    # Run tests
    exit_code = run_tests_with_options(**test_options)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        if args.coverage and args.html_report:
            print("📊 HTML coverage report generated in 'htmlcov/' directory")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 