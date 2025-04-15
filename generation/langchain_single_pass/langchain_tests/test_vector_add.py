import re
import torch
import numpy as np
import importlib
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import os
import sys
import time
import benchmarking_tools


def main():
    # Define paths
    llm_output_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/vector_add.txt"
    kernel_module_name = "vector_add_kernel"
    kernel_module_path = f"{kernel_module_name}.py"
    
    # Create a timestamp for uniqueness
    timestamp = int(time.time())
    unique_module_name = f"{kernel_module_name}_{timestamp}"
    unique_module_path = f"{unique_module_name}.py"
    
    print(f"Reading LLM output from: {llm_output_path}")
    
    # Check if file exists
    if not os.path.exists(llm_output_path):
        print(f"ERROR: LLM output file not found at {llm_output_path}")
        return
        
    # Extract kernel code from LLM output
    try:
        # Read the file content first
        with open(llm_output_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            
        print(f"Read {len(file_content)} characters from file")
        print(f"First 100 characters: {file_content[:100]}...")
        
        # Extract kernel code
        kernel_code = benchmarking_tools.extract_kernel_from_llm_response(file_content)
        print(f"Extracted {len(kernel_code)} characters of kernel code")
        print(f"First 100 characters of extracted code: {kernel_code[:100]}...")
        
        # Find function name
        func_name = benchmarking_tools.find_function_name_in_code(kernel_code)
        print(f"Detected function name: {func_name}")
        
        # Write kernel to both the standard and unique files
        with open(kernel_module_path, "w", encoding="utf-8") as f:
            f.write(kernel_code)
        with open(unique_module_path, "w", encoding="utf-8") as f:
            f.write(kernel_code)
            
        print(f"Wrote kernel code to: {kernel_module_path}")
        print(f"Also wrote to unique module: {unique_module_path}")
        
        # Import the unique module to avoid caching issues
        spec = importlib.util.spec_from_file_location(unique_module_name, unique_module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"Successfully imported module: {unique_module_name}")
        
        # Get the kernel function from the module
        if func_name and hasattr(module, func_name):
            kernel_func = getattr(module, func_name)
            print(f"Using detected function: {func_name}")
        elif hasattr(module, "vector_dot_kernel"):
            kernel_func = getattr(module, "vector_dot_kernel")
            print("Using default function: vector_dot_kernel")
        else:
            print(f"ERROR: Could not find kernel function in module. Available attributes: {dir(module)}")
            return
            
        # Create random 1D tensors
        np.random.seed(0)

        # Test the small workload with basic kernel
        lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
        rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

        # Run NKI kernel
        output_nki = func_name(lhs_small, rhs_small)
        # Compare with PyTorch reference
        output_torch = torch.sub(lhs_small, rhs_small)
        
        # Print comparison
        print("\n--- Results Comparison ---")
        print("NKI output (first 5):", output_nki[:5])
        print("PyTorch output (first 5):", output_torch[:5].numpy())
        
        # allclose check
        if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
            print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        else:
            print("\n❌ ERROR: NKI and PyTorch outputs differ!")
            # Print detailed comparison
            diff_count = 0
            for i in range(len(output_nki)):
                diff = abs(float(output_torch[i]) - float(output_nki[i]))
                if diff > 1e-4:
                    print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                    if diff_count >= 10:  # Limit to 10 differences
                        print("...")
                        break
                        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()