from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    # Ensure both tensors are 1D
    if v1_tensor.shape[0] != v2_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize the result tensor to match the input shapes
    result_tensor = nl.zeros(v1_tensor.shape, dtype=v1_tensor.dtype)  # Adjusted line to ensure proper shape and rank

    # Process the vector addition
    for i in nl.affine_range(v1_tensor.shape[0]):
        a_value = nl.load(v1_tensor[i])
        b_value = nl.load(v2_tensor[i])
        result_tensor[i] = nl.add(a_value, b_value)

    return result_tensor