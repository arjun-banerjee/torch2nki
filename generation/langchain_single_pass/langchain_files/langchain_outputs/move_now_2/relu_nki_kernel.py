from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_relu(a_tensor):
    """
    This kernel implements the Rectified Linear Unit (ReLU) function using NKI.
    The ReLU operation returns 0 if the input is less than 0, and the input itself otherwise.

    Parameters:
    a_tensor (ndarray): Input tensor containing the values to apply ReLU.

    Returns:
    ndarray: Output tensor after applying ReLU.
    """
    # Initialize the result array with the same shape and dtype as input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load input tensor into an SBUF tile
    input_tile = nl.load(a_tensor)

    # Apply ReLU function
    relu_output = nl.relu(input_tile)

    # Store the result back to HBM
    nl.store(result, relu_output)

    return result