from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_div(a_tensor, b_tensor):
    """
    Element-wise division of two tensors.

    Args:
    a_tensor (ndarray): Tensor of numerators, shape (N, M).
    b_tensor (ndarray): Tensor of denominators, shape (N, M).

    Returns:
    ndarray: Result tensor of the division operation, shape (N, M).
    """
    # Ensure both tensors have the same shape
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Input tensors must have the same shape.")
        
    # Initialize a result tensor with shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load input tensors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Perform element-wise division using the NKI divide function
    # Note: We use a mask to prevent division by zero by setting 
    # the denominator to a small positive value where it's zero.
    mask = (b_tile != 0)  # Create a mask where b_tile is not zero
    dummy_result = nl.divide(a_tile, b_tile, mask=mask)  # Perform division with masking

    # Store the result back to HBM
    nl.store(result, dummy_result)

    return result