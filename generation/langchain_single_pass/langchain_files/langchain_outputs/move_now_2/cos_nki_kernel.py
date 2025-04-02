from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_cos(a_tensor):
    """
    Calculate cosine of a_tensor using NKI cos function.

    Parameters:
    a_tensor (ndarray): Input tensor whose cosine values are to be calculated.

    Returns:
    ndarray: Cosine values of the input tensor.
    """
    # Initialize the result array in HBM with the same shape and dtype as a_tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load the input tensor from HBM into on-chip memory
    input_tile = nl.load(a_tensor)

    # Compute the cosine of each element in the input tile
    cos_output = nl.cos(input_tile)

    # Store the computed cosine values back to the result array in HBM
    nl.store(result, cos_output)

    # Return the resulting tensor containing cosine values
    return result