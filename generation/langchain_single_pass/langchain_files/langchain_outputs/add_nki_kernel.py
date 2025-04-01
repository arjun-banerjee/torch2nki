from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_add(a_tensor, b_tensor):
    """
    Performs element-wise addition of two tensors using NKI.
    
    Args:
        a_tensor: First input tensor
        b_tensor: Second input tensor
        
    Returns:
        Result tensor containing element-wise sum
    """
    # Verify input shapes match
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Input tensors must have the same shape")
    
    # Initialize result tensor in HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result
    nl.store(result, temp)
    
    return result