from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    """
    Adds two tensors element-wise using NKI, supporting broadcasting.
    
    :param v1_tensor: Input tensor for the first vector (1D or broadcastable).
    :param v2_tensor: Input tensor for the second vector (1D or broadcastable).
    :return: Result tensor containing the sum of the two tensors.
    """
    
    # Validate input tensors
    if v1_tensor.ndim < 1 or v2_tensor.ndim < 1:
        raise ValueError("Input tensors must be at least 1-dimensional")

    # Determine the result shape based on broadcasting rules
    result_shape = nl.broadcast_shape(v1_tensor.shape, v2_tensor.shape)
    result = nl.zeros(result_shape, dtype=nl.promote_types(v1_tensor.dtype, v2_tensor.dtype))

    # Load the input tensors
    v1_tile = nl.load(v1_tensor)
    v2_tile = nl.load(v2_tensor)

    # Perform element-wise addition using broadcasting
    nl.store(result, nl.add(v1_tile, v2_tile))

    return result