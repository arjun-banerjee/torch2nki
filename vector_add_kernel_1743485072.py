from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    """
    Adds two vectors element-wise using NKI.
    
    Args:
        a_tensor: First input vector tensor
        b_tensor: Second input vector tensor
    
    Returns:
        Result tensor containing element-wise sum
    """
    # Check that vectors have same length
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input vectors from HBM to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result