from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sub(a_tensor, b_tensor):
    """
    Perform element-wise subtraction of two tensors using NKI.
    
    Args:
        a_tensor: First input tensor
        b_tensor: Second input tensor
        
    Returns:
        Result tensor containing a - b
    """
    # Validate input shapes
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Input tensors must have the same shape")
        
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform subtraction
    temp = nl.subtract(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result