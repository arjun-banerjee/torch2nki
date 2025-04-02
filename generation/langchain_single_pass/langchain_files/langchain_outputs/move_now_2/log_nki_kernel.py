from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_log(a_tensor):
    """
    Compute the natural logarithm of the input tensor element-wise.

    :param a_tensor: Input tensor from HBM.
    :return: Result tensor containing the natural logarithm of each element in a_tensor.
    """
    # Initialize the result array with the same shape and dtype as the input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load input tensor into an SBUF tile
    input_tile = nl.load(a_tensor)

    # Compute the natural logarithm using the NKI API
    log_output = nl.log(input_tile)

    # Store the result back into the HBM
    nl.store(result, log_output)

    # Return the result
    return result