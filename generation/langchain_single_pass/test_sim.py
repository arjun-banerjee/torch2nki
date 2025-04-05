import torch_xla
from torch_xla.core import xla_model as xm
import os
import torch

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np


def test_torch_addition(device, nki_vector_add):
    """Test elementwise addition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((128,))
    rhs_small = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_vector_add,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.add(lhs_small, rhs_small)
        
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

def test_torch_subtraction(device, nki_vector_sub):
    """Test elementwise addition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((128,))
    rhs_small = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_vector_sub,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
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

def test_torch_multiplication(device, nki_vector_mul):
    """Test elementwise multiplication between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((128,))
    rhs_small = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_vector_mul,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.mul(lhs_small, rhs_small)
        
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


def test_torch_division(device, nki_vector_div):
    """Test elementwise division between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_vector_div: The NKI kernel function for elementwise division
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((128,))
    rhs_small = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_vector_div,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.div(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
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
        return 0

import numpy as np
import torch

def test_torch_absolute(device, nki_vector_abs):
    """Test elementwise absolute value between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_vector_abs: The NKI kernel function for elementwise absolute value
    
    Returns:
        int: 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    # Generate values in [-1, 1] so negatives are included.
    input_tensor = torch.rand((128,)) * 2 - 1
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_abs,
        np.array(input_tensor)
    )
    
    output_torch = torch.abs(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_exponential(device, nki_vector_exp):
    """Test elementwise exponential between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_exp: The NKI kernel function for exponential
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_exp,
        np.array(input_tensor)
    )
    
    output_torch = torch.exp(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_log(device, nki_vector_log):
    """Test elementwise natural logarithm between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_log: The NKI kernel function for logarithm
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Ensure positive input (avoid 0) by adding a small constant.
    input_tensor = torch.rand((128,)) + 0.1
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_log,
        np.array(input_tensor)
    )
    
    output_torch = torch.log(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_sqrt(device, nki_vector_sqrt):
    """Test elementwise square root between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_sqrt: The NKI kernel function for square root
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Use non-negative input
    input_tensor = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_sqrt,
        np.array(input_tensor)
    )
    
    output_torch = torch.sqrt(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_rsqrt(device, nki_vector_rsqrt):
    """Test elementwise reciprocal square root between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_rsqrt: The NKI kernel function for reciprocal square root
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Use nonzero positive input to avoid division by zero
    input_tensor = torch.rand((128,)) + 0.1
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_rsqrt,
        np.array(input_tensor)
    )
    
    output_torch = torch.rsqrt(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_power(device, nki_vector_power):
    """Test elementwise power (base ** exponent) between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_power: The NKI kernel function for power
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Base values; use a wider range.
    base = torch.rand((128,)) * 2
    # Exponent values in [0, 1] to avoid large numbers.
    exponent = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_power,
        np.array(base),
        np.array(exponent)
    )
    
    output_torch = torch.pow(base, exponent)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_sine(device, nki_vector_sine):
    """Test elementwise sine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_sine: The NKI kernel function for sine
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Generate input in the range [0, 2π]
    input_tensor = torch.rand((128,)) * 6.28318
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_sine,
        np.array(input_tensor)
    )
    
    output_torch = torch.sin(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_cosine(device, nki_vector_cosine):
    """Test elementwise cosine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_cosine: The NKI kernel function for cosine
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 6.28318
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_cosine,
        np.array(input_tensor)
    )
    
    output_torch = torch.cos(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_tangent(device, nki_vector_tangent):
    """Test elementwise tangent between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_tangent: The NKI kernel function for tangent
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Using an input range that avoids points where tan is undefined.
    input_tensor = torch.rand((128,)) * 1.0  # typically safe range
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_tangent,
        np.array(input_tensor)
    )
    
    output_torch = torch.tan(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_arcsine(device, nki_vector_arcsine):
    """Test elementwise arcsine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_arcsine: The NKI kernel function for arcsine
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Ensure input in [-1, 1]
    input_tensor = torch.rand((128,)) * 2 - 1
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_arcsine,
        np.array(input_tensor)
    )
    
    output_torch = torch.asin(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_arccosine(device, nki_vector_arccosine):
    """Test elementwise arccosine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_arccosine: The NKI kernel function for arccosine
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Ensure input in [-1, 1]
    input_tensor = torch.rand((128,)) * 2 - 1
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_arccosine,
        np.array(input_tensor)
    )
    
    output_torch = torch.acos(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_arctangent(device, nki_vector_arctangent):
    """Test elementwise arctangent between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_arctangent: The NKI kernel function for arctangent
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Input can be any real number; here we use a symmetric range.
    input_tensor = torch.rand((128,)) * 10 - 5
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_arctangent,
        np.array(input_tensor)
    )
    
    output_torch = torch.atan(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_hyperbolic_sine(device, nki_vector_sinh):
    """Test elementwise hyperbolic sine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_sinh: The NKI kernel function for hyperbolic sine
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # Input range chosen to include negatives.
    input_tensor = torch.rand((128,)) * 4 - 2
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_sinh,
        np.array(input_tensor)
    )
    
    output_torch = torch.sinh(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_hyperbolic_cosine(device, nki_vector_cosh):
    """Test elementwise hyperbolic cosine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_cosh: The NKI kernel function for hyperbolic cosine
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 4 - 2
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_cosh,
        np.array(input_tensor)
    )
    
    output_torch = torch.cosh(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_hyperbolic_tangent(device, nki_vector_tanh):
    """Test elementwise hyperbolic tangent between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_tanh: The NKI kernel function for hyperbolic tangent
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 4 - 2
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_tanh,
        np.array(input_tensor)
    )
    
    output_torch = torch.tanh(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_sigmoid(device, nki_vector_sigmoid):
    """Test elementwise sigmoid between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_sigmoid: The NKI kernel function for sigmoid
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 10 - 5  # a range with negatives and positives
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_sigmoid,
        np.array(input_tensor)
    )
    
    output_torch = torch.sigmoid(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_relu(device, nki_vector_relu):
    """Test elementwise ReLU between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on
        nki_vector_relu: The NKI kernel function for ReLU
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    # ReLU works with any input; using a range that includes negatives.
    input_tensor = torch.rand((128,)) * 2 - 1
    
    print("Running NKI kernel simulation...")
    output_nki = nki.simulate_kernel(
        nki_vector_relu,
        np.array(input_tensor)
    )
    
    output_torch = torch.relu(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0

def test_torch_threshold(device, nki_vector_threshold):
    """Test threshold operation between NKI and PyTorch implementations.
    
    For torch.threshold, each element in input_tensor is compared to a threshold.
    If the element is less than or equal to the threshold, it is replaced by a given value.
    
    Args:
        device: The device to run the test on
        nki_vector_threshold: The NKI kernel function for threshold
    
    Returns:
        int: 1 if outputs match, 0 otherwise
    """
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    # For this test, we use fixed scalar parameters.
    threshold_val = 0.5
    replacement_val = 0.0
    
    print("Running NKI kernel simulation...")
    # Pass the extra parameters as a two-element array.
    output_nki = nki.simulate_kernel(
        nki_vector_threshold,
        np.array(input_tensor),
        threshold_val,
        replacement_val
    )
    
    output_torch = torch.threshold(input_tensor, threshold_val, replacement_val)
    
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
                if diff_count >= 10:
                    print("...")
                    break
        return 0


# Helper function for comparing outputs.
def outputs_match(output_torch, output_nki):
    # Ensure both outputs are torch.Tensors.
    t_output = output_torch if isinstance(output_torch, torch.Tensor) else torch.tensor(output_torch)
    n_output = torch.tensor(output_nki)
    # For floating-point data, use allclose; otherwise, use exact equality.
    if t_output.dtype in [torch.float32, torch.float64]:
        return torch.allclose(t_output, n_output, atol=1e-4, rtol=1e-2)
    else:
        return torch.equal(t_output, n_output)

# Helper function for printing the first few elements.
def print_first_five(label, output):
    try:
        # If output is a scalar tensor.
        if isinstance(output, torch.Tensor) and output.dim() == 0:
            print(f"{label}:", output.item())
        else:
            # Try slicing (works for arrays and tensors with >0 elements)
            if isinstance(output, torch.Tensor):
                arr = output.detach().cpu().numpy()
            elif isinstance(output, np.ndarray):
                arr = output
            else:
                arr = output
            # Print first 5 elements if possible.
            print(f"{label} (first 5):", arr[:5] if hasattr(arr, '__getitem__') else arr)
    except Exception as e:
        print(f"{label}:", output)

# ---------------------------
# Test functions for single-input operations
# ---------------------------

def test_torch_softmax(device, nki_vector_softmax):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    dim = 0
    print("Running NKI kernel simulation for softmax...")
    output_nki = nki.simulate_kernel(
        nki_vector_softmax,
        np.array(input_tensor),
    )
    output_torch = torch.softmax(input_tensor, dim=dim)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_log_softmax(device, nki_vector_log_softmax):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    dim = 0
    print("Running NKI kernel simulation for log_softmax...")
    output_nki = nki.simulate_kernel(
        nki_vector_log_softmax,
        np.array(input_tensor),
    )
    output_torch = torch.log_softmax(input_tensor, dim=dim)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_max(device, nki_vector_max):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for max...")
    output_nki = nki.simulate_kernel(nki_vector_max, np.array(input_tensor))
    output_torch = torch.max(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_min(device, nki_vector_min):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for min...")
    output_nki = nki.simulate_kernel(nki_vector_min, np.array(input_tensor))
    output_torch = torch.min(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_sum(device, nki_vector_sum):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for sum...")
    output_nki = nki.simulate_kernel(nki_vector_sum, np.array(input_tensor))
    output_torch = torch.sum(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_mean(device, nki_vector_mean):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for mean...")
    output_nki = nki.simulate_kernel(nki_vector_mean, np.array(input_tensor))
    output_torch = torch.mean(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_var(device, nki_vector_var):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for var...")
    output_nki = nki.simulate_kernel(nki_vector_var, np.array(input_tensor))
    output_torch = torch.var(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_std(device, nki_vector_std):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for std...")
    output_nki = nki.simulate_kernel(nki_vector_std, np.array(input_tensor))
    output_torch = torch.std(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_norm(device, nki_vector_norm):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for norm...")
    output_nki = nki.simulate_kernel(nki_vector_norm, np.array(input_tensor))
    output_torch = torch.norm(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_cumsum(device, nki_vector_cumsum):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    dim = 0
    print("Running NKI kernel simulation for cumsum...")
    output_nki = nki.simulate_kernel(
        nki_vector_cumsum,
        np.array(input_tensor),
    )
    output_torch = torch.cumsum(input_tensor, dim=dim)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_cumprod(device, nki_vector_cumprod):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) + 0.1  # avoid zeros
    dim = 0
    print("Running NKI kernel simulation for cumprod...")
    output_nki = nki.simulate_kernel(
        nki_vector_cumprod,
        np.array(input_tensor),
    )
    output_torch = torch.cumprod(input_tensor, dim=dim)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_prod(device, nki_vector_prod):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) + 0.1  # avoid zero values
    print("Running NKI kernel simulation for prod...")
    output_nki = nki.simulate_kernel(nki_vector_prod, np.array(input_tensor))
    output_torch = torch.prod(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_round(device, nki_vector_round):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 10
    print("Running NKI kernel simulation for round...")
    output_nki = nki.simulate_kernel(nki_vector_round, np.array(input_tensor))
    output_torch = torch.round(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_floor(device, nki_vector_floor):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 10
    print("Running NKI kernel simulation for floor...")
    output_nki = nki.simulate_kernel(nki_vector_floor, np.array(input_tensor))
    output_torch = torch.floor(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_ceil(device, nki_vector_ceil):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 10
    print("Running NKI kernel simulation for ceil...")
    output_nki = nki.simulate_kernel(nki_vector_ceil, np.array(input_tensor))
    output_torch = torch.ceil(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_trunc(device, nki_vector_trunc):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 10 - 5  # include negatives
    print("Running NKI kernel simulation for trunc...")
    output_nki = nki.simulate_kernel(nki_vector_trunc, np.array(input_tensor))
    output_torch = torch.trunc(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_sign(device, nki_vector_sign):
    np.random.seed(0)
    input_tensor = torch.rand((128,)) * 2 - 1  # values in [-1, 1]
    print("Running NKI kernel simulation for sign...")
    output_nki = nki.simulate_kernel(nki_vector_sign, np.array(input_tensor))
    output_torch = torch.sign(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

# ---------------------------
# Test functions for multi-input or comparison operations
# ---------------------------

def test_torch_where(device, nki_vector_where):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    condition = input_tensor > 0.5
    x = input_tensor
    y = -input_tensor
    print("Running NKI kernel simulation for where...")
    output_nki = nki.simulate_kernel(
        nki_vector_where,
        np.array(condition),
        np.array(x),
        np.array(y)
    )
    output_torch = torch.where(condition, x, y)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_eq(device, nki_vector_eq):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    other_tensor = torch.full(input_tensor.shape, 0.5)
    print("Running NKI kernel simulation for eq...")
    output_nki = nki.simulate_kernel(
        nki_vector_eq,
        np.array(input_tensor),
        np.array(other_tensor)
    )
    output_torch = torch.eq(input_tensor, other_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_ne(device, nki_vector_ne):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    other_tensor = torch.full(input_tensor.shape, 0.5)
    print("Running NKI kernel simulation for ne...")
    output_nki = nki.simulate_kernel(
        nki_vector_ne,
        np.array(input_tensor),
        np.array(other_tensor)
    )
    output_torch = torch.ne(input_tensor, other_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_gt(device, nki_vector_gt):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    other_tensor = torch.full(input_tensor.shape, 0.5)
    print("Running NKI kernel simulation for gt...")
    output_nki = nki.simulate_kernel(
        nki_vector_gt,
        np.array(input_tensor),
        np.array(other_tensor)
    )
    output_torch = torch.gt(input_tensor, other_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_lt(device, nki_vector_lt):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    other_tensor = torch.full(input_tensor.shape, 0.5)
    print("Running NKI kernel simulation for lt...")
    output_nki = nki.simulate_kernel(
        nki_vector_lt,
        np.array(input_tensor),
        np.array(other_tensor)
    )
    output_torch = torch.lt(input_tensor, other_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_clamp(device, nki_vector_clamp):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    min_val = 0.3
    max_val = 0.7
    print("Running NKI kernel simulation for clamp...")
    output_nki = nki.simulate_kernel(
        nki_vector_clamp,
        np.array(input_tensor),
        min_val,
        max_val
    )
    output_torch = torch.clamp(input_tensor, min=min_val, max=max_val)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_sort(device, nki_vector_sort):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for sort...")
    output_nki = nki.simulate_kernel(nki_vector_sort, np.array(input_tensor))
    # Assume NKI returns only the sorted values.
    output_torch = torch.sort(input_tensor)[0]
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_topk(device, nki_vector_topk):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    k = 5
    print("Running NKI kernel simulation for topk...")
    output_nki = nki.simulate_kernel(
        nki_vector_topk,
        np.array(input_tensor),
        np.array([k])
    )
    # Compare only the values (not the indices).
    output_torch = torch.topk(input_tensor, k=k)[0]
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_kthvalue(device, nki_vector_kthvalue):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    k = 3
    print("Running NKI kernel simulation for kthvalue...")
    output_nki = nki.simulate_kernel(
        nki_vector_kthvalue,
        np.array(input_tensor),
        np.array([k])
    )
    output_torch = torch.kthvalue(input_tensor, k=k).values
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_median(device, nki_vector_median):
    np.random.seed(0)
    # Use an odd-length tensor so that median is unambiguous.
    input_tensor = torch.rand((129,))
    print("Running NKI kernel simulation for median...")
    output_nki = nki.simulate_kernel(nki_vector_median, np.array(input_tensor))
    output_torch = torch.median(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_mode(device, nki_vector_mode):
    np.random.seed(0)
    # Use integer values to get meaningful mode results.
    input_tensor = torch.randint(0, 5, (128,))
    print("Running NKI kernel simulation for mode...")
    output_nki = nki.simulate_kernel(nki_vector_mode, np.array(input_tensor))
    # Compare only the mode values (not indices).
    output_torch = torch.mode(input_tensor).values
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item() if output_torch.dim()==0 else output_torch[:5].numpy())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_percentile(device, nki_vector_percentile):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    q = 50  # 50th percentile (median)
    print("Running NKI kernel simulation for percentile...")
    output_nki = nki.simulate_kernel(
        nki_vector_percentile,
        np.array(input_tensor),
        np.array([q])
    )
    output_torch = torch.percentile(input_tensor, q)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_logsumexp(device, nki_vector_logsumexp):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    dim = 0
    print("Running NKI kernel simulation for logsumexp...")
    output_nki = nki.simulate_kernel(
        nki_vector_logsumexp,
        np.array(input_tensor),
    )
    output_torch = torch.logsumexp(input_tensor, dim=dim)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_amax(device, nki_vector_amax):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for amax...")
    output_nki = nki.simulate_kernel(nki_vector_amax, np.array(input_tensor))
    output_torch = torch.amax(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_amin(device, nki_vector_amin):
    np.random.seed(0)
    input_tensor = torch.rand((128,))
    print("Running NKI kernel simulation for amin...")
    output_nki = nki.simulate_kernel(nki_vector_amin, np.array(input_tensor))
    output_torch = torch.amin(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.item())
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_all(device, nki_vector_all):
    np.random.seed(0)
    # Create a boolean tensor.
    input_tensor = (torch.rand((128,)) > 0.3)
    print("Running NKI kernel simulation for all...")
    output_nki = nki.simulate_kernel(nki_vector_all, np.array(input_tensor))
    output_torch = torch.all(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", bool(output_torch))
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_any(device, nki_vector_any):
    np.random.seed(0)
    input_tensor = (torch.rand((128,)) > 0.7)
    print("Running NKI kernel simulation for any...")
    output_nki = nki.simulate_kernel(nki_vector_any, np.array(input_tensor))
    output_torch = torch.any(input_tensor)
    
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", bool(output_torch))
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_bincount(device, nki_vector_bincount):
    np.random.seed(0)
    input_tensor = torch.randint(0, 10, (128,))
    print("Running NKI kernel simulation for bincount...")
    output_nki = nki.simulate_kernel(nki_vector_bincount, np.array(input_tensor))
    output_torch = torch.bincount(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_unique(device, nki_vector_unique):
    np.random.seed(0)
    input_tensor = torch.randint(0, 10, (128,))
    print("Running NKI kernel simulation for unique...")
    output_nki = nki.simulate_kernel(nki_vector_unique, np.array(input_tensor))
    output_torch = torch.unique(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0

def test_torch_unique_consecutive(device, nki_vector_unique_consecutive):
    np.random.seed(0)
    # Create a tensor with consecutive duplicates.
    base = torch.randint(0, 5, (64,))
    input_tensor = torch.repeat_interleave(base, repeats=2)
    print("Running NKI kernel simulation for unique_consecutive...")
    output_nki = nki.simulate_kernel(nki_vector_unique_consecutive, np.array(input_tensor))
    output_torch = torch.unique_consecutive(input_tensor)
    
    print("\n--- Results Comparison ---")
    print_first_five("NKI output", output_nki)
    print_first_five("PyTorch output", output_torch)
    
    if outputs_match(output_torch, output_nki):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        return 0
    


def test_torch_inner(device, nki_inner):
    """Test inner product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_inner: The NKI kernel function for inner product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((10,))
    rhs_small = torch.rand((10,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_inner,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.inner(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        print(f"PyTorch={float(output_torch):.6f}, NKI={float(output_nki):.6f}, Diff={abs(float(output_torch) - float(output_nki)):.6f}")
        return 0

def test_torch_outer(device, nki_outer):
    """Test outer product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_outer: The NKI kernel function for outer product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((10,))
    rhs_small = torch.rand((12,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_outer,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.outer(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(output_nki.shape[0]):
            for j in range(output_nki.shape[1]):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                    if diff_count >= 10:  # Limit to 10 differences
                        print("...")
                        break
            if diff_count >= 10:
                break
        
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_dot(device, nki_dot):
    """Test dot product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_dot: The NKI kernel function for dot product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((10,))
    rhs_small = torch.rand((10,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_dot,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.dot(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        print(f"PyTorch={float(output_torch):.6f}, NKI={float(output_nki):.6f}, Diff={abs(float(output_torch) - float(output_nki)):.6f}")
        return 0

def test_torch_vdot(device, nki_vdot):
    """Test vdot product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_vdot: The NKI kernel function for vdot product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((10,))
    rhs_small = torch.rand((10,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_vdot,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.vdot(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        print(f"PyTorch={float(output_torch):.6f}, NKI={float(output_nki):.6f}, Diff={abs(float(output_torch) - float(output_nki)):.6f}")
        return 0

def test_torch_cross(device, nki_cross):
    """Test cross product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_cross: The NKI kernel function for cross product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((3,))
    rhs_small = torch.rand((3,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_cross,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.cross(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output:", output_nki)
    print("PyTorch output:", output_torch.numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        for i in range(len(output_nki)):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > 1e-4:
                print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
        return 0

def test_torch_matmul(device, nki_matmul):
    """Test matrix multiplication operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_matmul: The NKI kernel function for matrix multiplication
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((64, 128))
    rhs_small = torch.rand((128, 32))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_matmul,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.matmul(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(5, output_nki.shape[0])):
            for j in range(min(5, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                    if diff_count >= 10:  # Limit to 10 differences
                        print("...")
                        break
            if diff_count >= 10:
                break
        
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_mm(device, nki_mm):
    """Test matrix-matrix multiplication (mm) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_mm: The NKI kernel function for matrix-matrix multiplication
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((64, 128))
    rhs_small = torch.rand((128, 32))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_mm,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.mm(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(5, output_nki.shape[0])):
            for j in range(min(5, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                    if diff_count >= 10:  # Limit to 10 differences
                        print("...")
                        break
            if diff_count >= 10:
                break
        
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_mv(device, nki_mv):
    """Test matrix-vector multiplication (mv) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_mv: The NKI kernel function for matrix-vector multiplication
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((64, 128))
    rhs_small = torch.rand((128,))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_mv,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.mv(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
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
        return 0

def test_torch_bmm(device, nki_bmm):
    """Test batch matrix-matrix multiplication (bmm) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_bmm: The NKI kernel function for batch matrix-matrix multiplication
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((10, 64, 128))
    rhs_small = torch.rand((10, 128, 32))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_bmm,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.bmm(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first batch, 5x5):", output_nki[0, :5, :5])
    print("PyTorch output (first batch, 5x5):", output_torch[0, :5, :5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0, 0)
        for b in range(min(2, output_nki.shape[0])):
            for i in range(min(3, output_nki.shape[1])):
                for j in range(min(3, output_nki.shape[2])):
                    diff = abs(float(output_torch[b, i, j]) - float(output_nki[b, i, j]))
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_idx = (b, i, j)
                    if diff > 1e-4:
                        print(f"Element [{b},{i},{j}]: PyTorch={float(output_torch[b, i, j]):.6f}, NKI={float(output_nki[b, i, j]):.6f}, Diff={diff:.6f}")
                        diff_count += 1
                        if diff_count >= 10:  # Limit to 10 differences
                            print("...")
                            break
                if diff_count >= 10:
                    break
            if diff_count >= 10:
                break
        
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_hadamard(device, nki_hadamard):
    """Test Hadamard (element-wise) product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_hadamard: The NKI kernel function for Hadamard product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    lhs_small = torch.rand((64, 128))
    rhs_small = torch.rand((64, 128))
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_hadamard,
        np.array(lhs_small),
        np.array(rhs_small)
    )
        
    # Compare with PyTorch reference
    output_torch = torch.mul(lhs_small, rhs_small)
        
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].numpy())
        
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(5, output_nki.shape[0])):
            for j in range(min(5, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                    if diff_count >= 10:  # Limit to 10 differences
                        print("...")
                        break
            if diff_count >= 10:
                break
        
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_tensordot(device, nki_tensordot):
    """Test tensordot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_tensordot: The NKI kernel function for tensordot operation
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    a = torch.rand((4, 5, 6), dtype=torch.bfloat16, device=device)
    b = torch.rand((6, 7, 8), dtype=torch.bfloat16, device=device)
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_tensordot,
        a.to(torch.float32).numpy(),
        b.to(torch.float32).numpy(),
    )
    
    # Compare with PyTorch reference
    output_torch = torch.tensordot(a, b, dims=([2], [0]))
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].numpy())
    
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(3, output_nki.shape[0])):
            for j in range(min(3, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                if diff_count >= 10:  # Limit to 10 differences
                    print("...")
                    break
            if diff_count >= 10:
                break
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_einsum(device, nki_einsum):
    """Test einsum operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_einsum: The NKI kernel function for einsum operation
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    a = torch.rand((64, 128), dtype=torch.float32, device=device)
    b = torch.rand((128, 32), dtype=torch.float32, device=device)
    equation = "ij,jk->ik"
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_einsum,
        equation,
        a.to(torch.float32).numpy(),
        b.to(torch.float32).numpy(),
    )
    
    # Compare with PyTorch reference
    output_torch = torch.einsum(equation, a, b)
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first a: 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].to(torch.float32).numpy())
    
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(3, output_nki.shape[0])):
            for j in range(min(3, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                if diff_count >= 10:  # Limit to 10 differences
                    print("...")
                    break
            if diff_count >= 10:
                break
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0
def test_torch_kron(device, nki_kron):
    """Test Kronecker product operation between NKI and PyTorch implementations.
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_kron: The NKI kernel function for Kronecker product
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    a = torch.rand((3, 3), dtype=torch.bfloat16, device=device)
    b = torch.rand((3, 3), dtype=torch.bfloat16, device=device)
    print("Running NKI kernel simulation...")
    
    # Run NKI kernel using simulate_kernel with float32 inputs
    output_nki = nki.simulate_kernel(
        nki_kron,
        a.to(torch.float32).numpy(),
        b.to(torch.float32).numpy(),
    )
    
    # Compare with PyTorch reference - convert output_torch to float32 BEFORE comparison
    output_torch = torch.kron(a, b).to(torch.float32)
    
    # Convert NKI output to tensor for comparison
    output_nki_tensor = torch.tensor(output_nki, dtype=torch.float32, device=device)
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].cpu().numpy())
    
    # allclose check - both tensors are now float32
    if torch.allclose(output_torch, output_nki_tensor, atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(3, output_nki.shape[0])):
            for j in range(min(3, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j].cpu()) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j].cpu()):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                if diff_count >= 10:  # Limit to 10 differences
                    print("...")
                    break
                if diff_count >= 10:
                    break
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0
        
def test_torch_linalg_vecdot(device, nki_linalg_vecdot):
    """Test linalg_vecdot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_linalg_vecdot: The NKI kernel function for vector dot product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    a = torch.rand((5, 10), dtype=torch.bfloat16, device=device)
    b = torch.rand((5, 10), dtype=torch.bfloat16, device=device)
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_linalg_vecdot,
        a.to(torch.float32).numpy(),
        b.to(torch.float32).numpy(),
        dim=1
    )
    
    # Compare with PyTorch reference
    output_torch = torch.linalg.vecdot(a, b, dim=1)
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5):", output_nki[:5])
    print("PyTorch output (first 5):", output_torch[:5].numpy())
    
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = 0
        for i in range(min(5, output_nki.shape[0])):
            diff = abs(float(output_torch[i]) - float(output_nki[i]))
            if diff > max_diff:
                max_diff = diff
                max_diff_idx = i
            if diff > 1e-4:
                print(f"Element [{i}]: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                diff_count += 1
            if diff_count >= 10:  # Limit to 10 differences
                print("...")
                break
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0

def test_torch_linalg_multi_dot(device, nki_linalg_multi_dot):
    """Test linalg_multi_dot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_linalg_multi_dot: The NKI kernel function for multi-dot product
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    np.random.seed(0)
    A = torch.rand((10, 20), dtype=torch.bfloat16, device=device)
    B = torch.rand((20, 30), dtype=torch.bfloat16, device=device)
    C = torch.rand((30, 40), dtype=torch.bfloat16, device=device)
    matrices = [A, B, C]
    
    print("Running NKI kernel simulation...")
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_linalg_multi_dot,
        [np.array(m) for m in matrices]
    )
    
    # Compare with PyTorch reference
    output_torch = torch.linalg.multi_dot(matrices)
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print("NKI output (first 5x5):", output_nki[:5, :5])
    print("PyTorch output (first 5x5):", output_torch[:5, :5].numpy())
    
    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        return 1
    else:
        print("\n❌ ERROR: NKI and PyTorch outputs differ!")
        # Print detailed comparison
        diff_count = 0
        max_diff = 0
        max_diff_idx = (0, 0)
        for i in range(min(3, output_nki.shape[0])):
            for j in range(min(3, output_nki.shape[1])):
                diff = abs(float(output_torch[i, j]) - float(output_nki[i, j]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_idx = (i, j)
                if diff > 1e-4:
                    print(f"Element [{i},{j}]: PyTorch={float(output_torch[i, j]):.6f}, NKI={float(output_nki[i, j]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                if diff_count >= 10:  # Limit to 10 differences
                    print("...")
                    break
            if diff_count >= 10:
                break
        print(f"Maximum difference of {max_diff:.6f} at element {max_diff_idx}")
        return 0