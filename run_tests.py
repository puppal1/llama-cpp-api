#!/usr/bin/env python
import pytest
import sys

def main():
    """Run the test suite"""
    # Add any pytest arguments here
    args = [
        "llama_cpp_api_package/tests",  # Test directory
        "-v",                           # Verbose output
        "--tb=short",                   # Shorter traceback format
        "-s",                           # Show print statements
    ]
    
    # Run tests and exit with appropriate code
    exit_code = pytest.main(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 