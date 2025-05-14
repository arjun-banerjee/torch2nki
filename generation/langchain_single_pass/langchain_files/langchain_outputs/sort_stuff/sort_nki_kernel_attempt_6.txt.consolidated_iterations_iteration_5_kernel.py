from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimensions
    if dim < 0:
        dim = dim + len(a_tensor.shape)
    
    # Get shape of the input tensor
    shape = a_tensor.shape
    size = shape[dim]
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to result
    # We need to process the tensor in tiles to respect hardware limitations
    
    # Calculate the number of elements before and after the sort dimension
    size_before_dim = 1
    for i in range(dim):
        size_before_dim *= shape[i]
    
    size_after_dim = 1
    for i in range(dim+1, len(shape)):
        size_after_dim *= shape[i]
    
    # Calculate the maximum tile size for the partition dimension
    max_tile_size = min(nl.tile_size.pmax, size_before_dim)
    
    # Process the tensor in tiles
    trip_count = math.ceil(size_before_dim / max_tile_size)
    
    # Copy input to result first
    for p in nl.affine_range(trip_count):
        # Generate indices for current tile
        p_start = p * max_tile_size
        p_end = min((p + 1) * max_tile_size, size_before_dim)
        p_size = p_end - p_start
        
        # Create index arrays for loading and storing
        i_p = nl.arange(p_size)[:, None]
        i_d = nl.arange(size)[None, :]
        
        # For each element in the after dimensions
        for a in nl.affine_range(size_after_dim):
            # Load the slice
            slice_data = nl.load(a_tensor.reshape(size_before_dim, size, size_after_dim)[p_start + i_p, i_d, a], 
                                mask=(i_p < p_size))
            
            # Perform sorting along dimension 1 (which corresponds to dim in the original tensor)
            # Use a simple sorting algorithm (selection sort)
            for i in nl.affine_range(size - 1):
                # Find the minimum element in the unsorted portion
                min_idx = nl.full((p_size, 1), i, dtype=nl.int32)
                min_val = slice_data[:, i:i+1]
                
                for j in nl.affine_range(i+1, size):
                    # Create a mask where current value is less than min_val
                    is_smaller = nl.less(slice_data[:, j:j+1], min_val)
                    # Update min_val and min_idx where the condition is true
                    min_val = nl.where(is_smaller, slice_data[:, j:j+1], min_val)
                    min_idx = nl.where(is_smaller, nl.full((p_size, 1), j, dtype=nl.int32), min_idx)
                
                # Swap the minimum element with the first element of unsorted part
                temp = slice_data[:, i:i+1].copy()
                idx_i = nl.full((p_size, 1), i, dtype=nl.int32)
                
                # Get values at min_idx for swapping
                vals_at_min_idx = nl.zeros((p_size, 1), dtype=a_tensor.dtype, buffer=nl.sbuf)
                for k in nl.affine_range(p_size):
                    vals_at_min_idx[k, 0] = slice_data[k, min_idx[k, 0]]
                
                # Perform the swap
                slice_data[:, i:i+1] = vals_at_min_idx
                
                # Update the values at min_idx positions
                for k in nl.affine_range(p_size):
                    slice_data[k, min_idx[k, 0]] = temp[k, 0]
            
            # Store the sorted slice back
            nl.store(result.reshape(size_before_dim, size, size_after_dim)[p_start + i_p, i_d, a], 
                    value=slice_data, mask=(i_p < p_size))
    
    return result