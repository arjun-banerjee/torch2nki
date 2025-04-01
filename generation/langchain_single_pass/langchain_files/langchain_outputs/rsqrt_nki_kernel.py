from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_rsqrt(a_tensor):
    """
    Calculates the reciprocal square root (1/sqrt(x)) of each element in the input tensor.
    
    Args:
        a_tensor: Input tensor
        
    Returns:
        A tensor containing the reciprocal square root of each element in the input tensor
    """
    # Initialize result tensor in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    input_tile = nl.load(a_tensor)
    
    # Compute rsqrt
    output_tile = nl.rsqrt(input_tile)
    
    # Store result back to HBM
    nl.store(result, value=output_tile)
    
    return result