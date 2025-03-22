import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    Vector addition kernel that performs element-wise addition of two input tensors.
    
    Parameters:
    v1: Input tensor (first vector)
    v2: Input tensor (second vector)
    
    Returns:
    result: Output tensor containing the sum of v1 and v2
    """
    # Validate that the input vectors have the same shape
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    # Ensure both tensors have a valid shape for NKI
    if len(v1.shape) < 1 or len(v2.shape) < 1:
        raise ValueError("Input vectors must have at least one dimension")
    
    # Create an output tensor of the same shape, filled with zeros
    # If the shape is 1D, make it 2D by adding an additional dimension.
    if len(v1.shape) == 1:
        result_shape = (v1.shape[0], 1)
    else:
        result_shape = v1.shape
    
    result = nl.zeros(result_shape, dtype=v1.dtype)

    # Calculate the number of elements in the tensor
    num_elements = nl.prod(v1.shape)  # Get the total number of elements
    
    # Using multi-dimensional indexing to properly iterate over the tensor
    for idx in nl.arange(num_elements):
        # Compute the multi-dimensional index from the flat index
        multi_idx = nl.unravel_index(idx, v1.shape)
        
        # Load the elements from the input tensors
        a = nl.load(v1[multi_idx])
        b = nl.load(v2[multi_idx])

        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor
        nl.store(result[multi_idx], c)

    return result