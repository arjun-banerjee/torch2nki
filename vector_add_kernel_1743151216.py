from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    # Ensure both tensors are 1D
    if v1_tensor.shape[0] != v2_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize the result tensor to be a 1D tensor matching the length of input tensors
    result_tensor = nl.zeros((v1_tensor.shape[0],), dtype=v1_tensor.dtype)  # Correctly initialize as 1D tensor

    # Process the vector addition
    for i in nl.affine_range(v1_tensor.shape[0]):
        v1_value = nl.load(v1_tensor[i])
        v2_value = nl.load(v2_tensor[i])
        result_tensor[i] = nl.add(v1_value, v2_value)

    return result_tensor