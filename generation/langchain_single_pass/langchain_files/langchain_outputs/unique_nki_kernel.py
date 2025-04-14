from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_unique(a_tensor):
    # Get shape and size of input tensor
    input_shape = a_tensor.shape
    
    # Calculate total number of elements
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim
    
    # Flatten the input tensor for processing
    flat_tensor = a_tensor.reshape(total_elements)
    
    # Create a temporary buffer to track unique elements
    # In worst case, all elements could be unique
    temp_unique = nl.ndarray((total_elements,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Counter to track number of unique elements found
    unique_count = nl.zeros((), dtype=nl.int32, buffer=nl.psum)
    
    # Process the tensor in tiles due to hardware limitations
    tile_size = nl.tile_size.pmax  # Maximum partition size
    num_tiles = math.ceil(total_elements / tile_size)
    
    # First pass: Find unique elements and store them
    for tile_idx in nl.affine_range(num_tiles):
        # Calculate start and end indices for current tile
        start_idx = tile_idx * tile_size
        
        # Create indices for current tile
        i_p = start_idx + nl.arange(tile_size)
        
        # Load current tile from flattened tensor
        current_tile = nl.load(flat_tensor[i_p], mask=(i_p < total_elements))
        
        # For each element in the current tile
        for i in nl.affine_range(tile_size):
            # Check if we're still within bounds of the tensor
            if start_idx + i >= total_elements:
                continue
                
            # Get the current element
            current_element = nl.load(current_tile[i])
            
            # Flag to check if element is unique
            is_unique = nl.ones((), dtype=nl.bool_, buffer=nl.psum)
            
            # Check against all previously found unique elements
            for j in nl.affine_range(unique_count):
                prev_unique = nl.load(temp_unique[j])
                if nl.equal(current_element, prev_unique):
                    is_unique = nl.zeros((), dtype=nl.bool_, buffer=nl.psum)
                    break
            
            # If element is unique, add it to our unique elements array
            if is_unique:
                nl.store(temp_unique[unique_count], current_element)
                unique_count += 1
    
    # Create the result array with the exact size needed
    result = nl.ndarray((unique_count,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy unique elements to the result array
    for i in nl.affine_range(unique_count):
        i_p = nl.arange(1)
        unique_val = nl.load(temp_unique[i])
        nl.store(result[i], unique_val)
    
    return result