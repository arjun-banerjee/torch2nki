from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    """
    Adds two vectors element-wise.
    
    :param v1_tensor: Input tensor for the first vector
    :param v2_tensor: Input tensor for the second vector
    :return: A tensor representing the sum of the two vectors
    """
    # Ensure both tensors are 1D
    if v1_tensor.shape[0] != v2_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")

    # Initialize the result tensor
    result = nl.ndarray(v1_tensor.shape, dtype=v1_tensor.dtype, buffer=nl.shared_hbm)

    # Compute the element-wise addition
    v1_elements = nl.load(v1_tensor)
    v2_elements = nl.load(v2_tensor)
    
    # Use a dummy variable to hold the addition result
    temp_result = nl.add(v1_elements, v2_elements)

    # Store the result in the result tensor
    nl.store(result, temp_result)

    return result