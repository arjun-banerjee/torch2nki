from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tan(a_tensor):
    """
    Calculate the tangent of a tensor using NKI's built-in tan function.

    Parameters:
    a_tensor : nl.ndarray
        A tensor containing the input values in radians.

    Returns:
    nl.ndarray
        A tensor containing the tangent values of the input tensor.
    """
    # Initialize the result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load input tensor from HBM into SBUF
    input_tile = nl.load(a_tensor)

    # Compute the tangent using the built-in NKI tan function
    tan_output = nl.tan(input_tile)

    # Store the result back to HBM
    nl.store(result, tan_output)

    return result