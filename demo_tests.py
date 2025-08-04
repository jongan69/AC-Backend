#!/usr/bin/env python3
"""
Demo script to show the automated test suite in action.
This script runs a subset of tests to demonstrate functionality.
"""

import subprocess
import sys
import os

def run_demo_tests():
    """Run a demonstration of the test suite."""
    
    print("🚀 Travel API Automated Test Suite Demo")
    print("=" * 50)
    
    # Test 1: Health check endpoint
    print("\n1. Testing Health Check Endpoint...")
    result1 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_health_check", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result1.returncode == 0:
        print("✅ Health check test passed!")
    else:
        print("❌ Health check test failed!")
        print(result1.stdout)
        print(result1.stderr)
    
    # Test 2: Root endpoint
    print("\n2. Testing Root Endpoint...")
    result2 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_root_endpoint", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("✅ Root endpoint test passed!")
    else:
        print("❌ Root endpoint test failed!")
        print(result2.stdout)
        print(result2.stderr)
    
    # Test 3: Transaction categorization (basic validation)
    print("\n3. Testing Transaction Categorization Validation...")
    result3 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_categorize_missing_transactions", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result3.returncode == 0:
        print("✅ Transaction validation test passed!")
    else:
        print("❌ Transaction validation test failed!")
        print(result3.stdout)
        print(result3.stderr)
    
    # Test 4: Hotel search validation
    print("\n4. Testing Hotel Search Validation...")
    result4 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_hotel_search_invalid_adults", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result4.returncode == 0:
        print("✅ Hotel validation test passed!")
    else:
        print("❌ Hotel validation test failed!")
        print(result4.stdout)
        print(result4.stderr)
    
    # Test 5: Flight search validation
    print("\n5. Testing Flight Search Validation...")
    result5 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_flight_search_missing_return_date", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result5.returncode == 0:
        print("✅ Flight validation test passed!")
    else:
        print("❌ Flight validation test failed!")
        print(result5.stdout)
        print(result5.stderr)
    
    # Test 6: Airport search functionality
    print("\n6. Testing Airport Search Functionality...")
    result6 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_nearest_airport_success", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result6.returncode == 0:
        print("✅ Airport search test passed!")
    else:
        print("❌ Airport search test failed!")
        print(result6.stdout)
        print(result6.stderr)
    
    # Test 7: PredictHQ events functionality
    print("\n7. Testing PredictHQ Events Functionality...")
    result7 = subprocess.run([
        "python3", "-m", "pytest", 
        "test_api.py::test_predicthq_events_success", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result7.returncode == 0:
        print("✅ PredictHQ events test passed!")
    else:
        print("❌ PredictHQ events test failed!")
        print(result7.stdout)
        print(result7.stderr)
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n📋 Test Suite Features Demonstrated:")
    print("   • Health check endpoint testing")
    print("   • Root endpoint testing") 
    print("   • Input validation testing")
    print("   • Error handling testing")
    print("   • Mock-based testing (external APIs)")
    
    print("\n📚 To run the full test suite:")
    print("   python3 run_tests.py -v")
    print("   python3 run_tests.py --coverage")
    print("   python3 run_tests.py --categories")
    
    print("\n📖 For more information, see TEST_README.md")

if __name__ == "__main__":
    run_demo_tests() 