#!/usr/bin/env python3
"""
Simple script to run and test vector_add_kernel.py without the LLM generation process.
"""

import os
import sys
import importlib.util
import numpy as np

def load_kernel_module(kernel_path):
    """
    Dynamically load the kernel module from the given path.
    """
    module_name = os.path.basename(kernel_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_vector_add(kernel_module):
    """
    Test the vector_add function from the loaded kernel module.
    """
    # Create test vectors
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    expected_output = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
    
    # Run the kernel
    result = kernel_module.vector_add(a, b)
    
    # Verify the result
    if not np.array_equal(result, expected_output):
        print(f"Error: Expected {expected_output}, but got {result}")
        return False
    
    print("Test passed: Vector addition was computed correctly.")
    return True

def main():
    # Path to the kernel module - adjust as needed for your system
    # Default path from your file
    kernel_module_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/run_manual_kernel.py"
    
    # Allow overriding the path via command line argument
    if len(sys.argv) > 1:
        kernel_module_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.isfile(kernel_module_path):
        print(f"Error: Kernel file not found at {kernel_module_path}")
        return
    
    try:
        # Load the kernel module
        print(f"Loading kernel from {kernel_module_path}")
        kernel_module = load_kernel_module(kernel_module_path)
        
        # Test the vector_add function
        test_vector_add(kernel_module)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()