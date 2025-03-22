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
    
    # Create an output tensor of the same shape with zeros
    result = nl.zeros((*v1.shape, 1), dtype=v1.dtype)  # Correctly create a tensor with an additional dimension

    # Check if the input tensors are 1D or 2D and handle accordingly
    if len(v1.shape) == 1:  # 1D case
        for i in nl.arange(v1.shape[0])[:, None]:  # Ensure i is reshaped to 2D
            a = nl.load(v1[i, :])  # Load the element from the first tensor
            b = nl.load(v2[i, :])  # Load the element from the second tensor
            c = nl.add(a, b)  # Perform element-wise addition
            nl.store(result[i, :], c)  # Store the result

    else:  # 2D case
        for i in nl.arange(v1.shape[0])[:, None]:  # Ensure i is reshaped to 2D
            for j in nl.arange(v1.shape[1])[:, None]:  # Ensure j is reshaped to 2D
                # Load the elements from the input tensors
                a = nl.load(v1[i, j])  # Use slicing to maintain dimensionality
                b = nl.load(v2[i, j])  # Use slicing to maintain dimensionality
                
                # Perform element-wise addition
                c = nl.add(a, b)

                # Store the result back into the output tensor
                nl.store(result[i, j], c)  # Store properly

    return result