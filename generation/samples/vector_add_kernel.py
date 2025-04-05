# Ensure neuronxcc is installed in your environment
# Run this command in your terminal if it's not installed:
# pip install neuronxcc

from neuronxcc import nki  # Ensure this is the correct import

# Define the kernel for vector addition
@nki.kernel
def vector_add_kernel(v1, v2, result):
    """
    Kernel function to add two vectors element-wise.
    
    :param v1: First input vector (tile)
    :param v2: Second input vector (tile)
    :param result: Output vector to store the sum (tile)
    """
    # Ensure the vectors are broadcastable
    assert v1.shape == v2.shape, "Input vectors must have the same shape."
    
    # Perform element-wise addition
    result[:] = nki.language.add(v1, v2)

def vector_add(v1, v2):
    """
    Adds two vectors element-wise using NKI.
    
    :param v1: List or array of numbers (first vector)
    :param v2: List or array of numbers (second vector)
    :return: List representing the sum of the two vectors
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    
    # Convert inputs to NKI compatible tiles
    v1_tile = nki.language.tile(v1)
    v2_tile = nki.language.tile(v2)
    
    # Initialize result tile
    result_tile = nki.language.tile(shape=v1_tile.shape, dtype=v1_tile.dtype)
    
    # Launch the vector addition kernel
    vector_add_kernel[v1_tile.shape](v1_tile, v2_tile, result_tile)
    
    # Return the result as a list
    return result_tile.to_list()