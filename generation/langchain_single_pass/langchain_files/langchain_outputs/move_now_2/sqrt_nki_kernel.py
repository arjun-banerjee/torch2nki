from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sqrt(a_tensor):
    """
    Computes the square root of elements in the input tensor using NKI.

    Parameters:
    a_tensor (ndarray): Input tensor containing numbers for which square root is to be calculated.

    Returns:
    result (ndarray): Tensor containing the square roots of the input tensor elements.
    """
    # Initialize the result array in HBM with the same shape and dtype as the input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load the input tensor into a tile from HBM
    input_tile = nl.load(a_tensor)

    # Compute the square root using NKI's built-in sqrt function
    sqrt_output = nl.sqrt(input_tile)

    # Store the result back to HBM
    nl.store(result, sqrt_output)

    # Return the result tensor
    return result