from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    """
    Adds two vectors element-wise using the NKI API.

    :param v1_tensor: Tensor representing the first vector (shape: [N])
    :param v2_tensor: Tensor representing the second vector (shape: [N])
    :return: Tensor representing the sum of the two vectors (shape: [N])
    """
    # Ensure both tensors are 1D
    if v1_tensor.shape[0] != v2_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize output tensor
    result_tensor = nl.zeros(v1_tensor.shape, dtype=v1_tensor.dtype)

    # Get the number of elements in the vector
    num_elements = v1_tensor.shape[0]

    # Use affine_range to loop through the elements
    for i in nl.affine_range(num_elements):
        # Load elements from each vector
        v1_value = nl.load(v1_tensor[i])
        v2_value = nl.load(v2_tensor[i])

        # Perform the addition
        result_value = nl.add(v1_value, v2_value)

        # Store the result back to the output tensor
        nl.store(result_tensor[i], result_value)

    return result_tensor