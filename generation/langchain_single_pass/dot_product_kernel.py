from neuronxcc import nki
import neuronxcc.nki.language as nl

def nki_dot_product(a_tensor, b_tensor):
    """
    Computes the dot product of two 1D tensors using NKI APIs.
    
    :param a_tensor: A 1D tensor containing the first vector.
    :param b_tensor: A 1D tensor containing the second vector.
    :return: A scalar representing the dot product of the two vectors.
    """
    # Ensure both tensors are 1D
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize a scalar to hold the sum result
    sum_result = nl.zeros((), dtype=nl.float32, buffer=nl.psum)

    # Load tensor values
    a_values = nl.load(a_tensor)
    b_values = nl.load(b_tensor)
    
    # Element-wise multiplication and accumulation
    sum_result += nl.sum(nl.multiply(a_values, b_values))

    return sum_result