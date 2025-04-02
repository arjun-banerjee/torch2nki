from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_rsqrt(a_tensor):
    """
    Calculate the reciprocal square root (rsqrt) of the input tensor.

    Args:
    a_tensor (ndarray): Input tensor of float type.

    Returns:
    ndarray: Tensor containing the reciprocal square root of the input tensor.
    """

    # Initialize the result array with the same shape and dtype as the input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load the input tensor from HBM to SBUF
    input_tile = nl.load(a_tensor)

    # Compute the reciprocal square root using the NKI built-in function
    rsqrt_output = nl.rsqrt(input_tile)

    # Store the result back into HBM
    nl.store(result, rsqrt_output)

    return result