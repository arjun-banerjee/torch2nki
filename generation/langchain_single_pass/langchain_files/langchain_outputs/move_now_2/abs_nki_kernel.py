from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_abs(a_tensor):
    """
    Computes the absolute value of elements in the input tensor.

    Parameters:
    a_tensor (nl.ndarray): Input tensor containing values.

    Returns:
    nl.ndarray: Tensor containing the absolute values of the input tensor.
    """
    # Initialize a result array in HBM with the same shape and dtype as the input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data into on-chip memory
    input_tile = nl.load(a_tensor)
    
    # Calculate absolute value using NKI's built-in abs function
    abs_result_tile = nl.abs(input_tile)
    
    # Store the computed absolute values back to HBM
    nl.store(result, abs_result_tile)

    return result