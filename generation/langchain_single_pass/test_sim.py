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
        np.array([threshold_val, replacement_val])
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
