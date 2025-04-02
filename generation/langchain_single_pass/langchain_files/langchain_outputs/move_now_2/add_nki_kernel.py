from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_add(a_tensor, b_tensor):
    """
    This kernel performs element-wise addition of two input tensors.

    Parameters:
    a_tensor (ndarray): First input tensor.
    b_tensor (ndarray): Second input tensor.

    Returns:
    result (ndarray): Output tensor containing the element-wise sum of a_tensor and b_tensor.
    """
    # Initialize the result tensor with the same shape and dtype as the input tensors
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensors from HBM
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition using NKI's add function
    dummy_result = nl.add(a_tile, b_tile)
    
    # Store the result back to HBM
    nl.store(result, dummy_result)

    return result