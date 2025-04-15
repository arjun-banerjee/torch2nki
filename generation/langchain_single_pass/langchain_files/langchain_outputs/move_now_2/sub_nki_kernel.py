from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sub(a_tensor, b_tensor):
    """
    Performs element-wise subtraction of two tensors.

    Parameters:
    a_tensor: The first input tensor (on HBM).
    b_tensor: The second input tensor (on HBM).

    Returns:
    result: A tensor containing the result of a_tensor - b_tensor (on HBM).
    """
    # Ensure both tensors have compatible shapes
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Input tensors must have the same shape")

    # Initialize the result array in the shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load both tensors into SBUF
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Perform element-wise subtraction using NKI's subtract function
    dummy_result = nl.subtract(a_tile, b_tile)

    # Store the result back into HBM
    nl.store(result, dummy_result)

    return result