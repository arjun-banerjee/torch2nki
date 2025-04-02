from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_mul(a_tensor, b_tensor):
    """
    This kernel performs element-wise multiplication of two input tensors.
    
    Parameters:
    - a_tensor: Input tensor with shape (N, M)
    - b_tensor: Input tensor with shape (N, M)
    
    Returns:
    - result: A new tensor containing the element-wise products of a_tensor and b_tensor.
    """
    # Ensure both input tensors have the same shape
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Both input tensors must have the same shape")

    # Initialize the result array in HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load input tiles from HBM to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Perform element-wise multiplication
    dummy_result = nl.multiply(a_tile, b_tile)

    # Store the result back to HBM
    nl.store(result, dummy_result)

    return result