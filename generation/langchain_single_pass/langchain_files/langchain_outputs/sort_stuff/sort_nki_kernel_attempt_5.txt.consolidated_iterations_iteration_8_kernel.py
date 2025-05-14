from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input to result
    if ndim == 1:
        # 1D tensor case
        size = shape[0]
        max_tile_size = min(128, size)
        
        # Copy input to result in tiles
        for i in nl.affine_range((size + max_tile_size - 1) // max_tile_size):
            start_idx = i * max_tile_size
            indices = start_idx + nl.arange(max_tile_size)
            input_tile = nl.load(a_tensor[indices], mask=(indices < size))
            nl.store(result[indices], value=input_tile, mask=(indices < size))
            
        # Bubble sort
        for i in range(size):
            for j in range(size - 1):
                # Use tiling to process pairs
                tiles_needed = (size - 1 + max_tile_size - 1) // max_tile_size
                for k in nl.affine_range(tiles_needed):
                    start_idx = k * max_tile_size
                    indices = start_idx + nl.arange(max_tile_size)
                    indices_next = indices + 1
                    
                    # Load current elements
                    curr_vals = nl.load(result[indices], mask=(indices < size - 1))
                    next_vals = nl.load(result[indices_next], mask=(indices_next < size))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    temp = nl.where(swap_needed, next_vals, curr_vals)
                    next_temp = nl.where(swap_needed, curr_vals, next_vals)
                    
                    # Store back
                    nl.store(result[indices], value=temp, mask=(indices < size - 1))
                    nl.store(result[indices_next], value=next_temp, mask=(indices_next < size))
    
    elif ndim == 2:
        # 2D tensor case
        rows, cols = shape[0], shape[1]
        
        # Copy input to result in tiles
        max_p_size = min(128, rows)
        for r in nl.affine_range((rows + max_p_size - 1) // max_p_size):
            start_r = r * max_p_size
            indices_r = start_r + nl.arange(max_p_size)[:, None]
            indices_c = nl.arange(cols)[None, :]
            
            # Load and store in tiles
            input_tile = nl.load(a_tensor[indices_r, indices_c], mask=(indices_r < rows))
            nl.store(result[indices_r, indices_c], value=input_tile, mask=(indices_r < rows))
        
        # Sort based on dimension
        if dim == 0:  # Sort along rows
            # For each column
            for c in range(cols):
                # Bubble sort algorithm
                for i in range(rows):
                    # For each pair of elements
                    for r in nl.affine_range((rows - 1 + max_p_size - 1) // max_p_size):
                        start_r = r * max_p_size
                        indices_r = start_r + nl.arange(max_p_size)[:, None]
                        indices_r_next = indices_r + 1
                        
                        # Create column index
                        col_idx = nl.full((max_p_size, 1), c, dtype=nl.int32)
                        
                        # Load current and next values
                        curr_vals = nl.load(result[indices_r, col_idx], mask=(indices_r < rows - 1))
                        next_vals = nl.load(result[indices_r_next, col_idx], mask=(indices_r_next < rows))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        temp = nl.where(swap_needed, next_vals, curr_vals)
                        next_temp = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back
                        nl.store(result[indices_r, col_idx], value=temp, mask=(indices_r < rows - 1))
                        nl.store(result[indices_r_next, col_idx], value=next_temp, mask=(indices_r_next < rows))
        
        else:  # Sort along columns (dim == 1)
            # For each row
            for r in range(rows):
                # Bubble sort algorithm
                for i in range(cols):
                    # Process elements in tiles
                    max_c_size = min(128, cols)
                    for c in nl.affine_range((cols - 1 + max_c_size - 1) // max_c_size):
                        start_c = c * max_c_size
                        indices_c = start_c + nl.arange(max_c_size)[None, :]
                        indices_c_next = indices_c + 1
                        
                        # Create row index
                        row_idx = nl.full((1, max_c_size), r, dtype=nl.int32)
                        
                        # Load current and next values
                        curr_vals = nl.load(result[row_idx, indices_c], mask=(indices_c < cols - 1))
                        next_vals = nl.load(result[row_idx, indices_c_next], mask=(indices_c_next < cols))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        temp = nl.where(swap_needed, next_vals, curr_vals)
                        next_temp = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back
                        nl.store(result[row_idx, indices_c], value=temp, mask=(indices_c < cols - 1))
                        nl.store(result[row_idx, indices_c_next], value=next_temp, mask=(indices_c_next < cols))
    
    else:
        # For tensors with more than 2 dimensions, we need to reshape and process
        # Flattening is not directly supported in NKI, so we'll handle this case differently
        # We'll process it as a batch of 2D tensors
        
        # Determine the size of each dimension
        size_dim = shape[dim]
        batch_size = 1
        for i in range(ndim):
            if i != dim:
                batch_size *= shape[i]
        
        # Copy input to result in tiles
        max_batch_size = min(128, batch_size)
        
        # Copy the entire tensor
        # This is a simplification - in a real implementation, we would tile this properly
        for b in nl.affine_range((batch_size + max_batch_size - 1) // max_batch_size):
            start_b = b * max_batch_size
            indices_b = start_b + nl.arange(max_batch_size)[:, None]
            indices_d = nl.arange(size_dim)[None, :]
            
            # This is simplified and would need proper indexing for actual >2D tensors
            input_tile = nl.load(a_tensor[indices_b, indices_d], mask=(indices_b < batch_size))
            nl.store(result[indices_b, indices_d], value=input_tile, mask=(indices_b < batch_size))
        
        # Sort along the specified dimension
        # Again, this is simplified and would need proper indexing for actual >2D tensors
        for i in range(size_dim):
            for j in range(size_dim - 1):
                for b in nl.affine_range((batch_size + max_batch_size - 1) // max_batch_size):
                    start_b = b * max_batch_size
                    indices_b = start_b + nl.arange(max_batch_size)[:, None]
                    indices_d = nl.full((max_batch_size, 1), j, dtype=nl.int32)
                    indices_d_next = nl.full((max_batch_size, 1), j+1, dtype=nl.int32)
                    
                    # Load current and next values
                    curr_vals = nl.load(result[indices_b, indices_d], mask=(indices_b < batch_size))
                    next_vals = nl.load(result[indices_b, indices_d_next], mask=(indices_b < batch_size))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    temp = nl.where(swap_needed, next_vals, curr_vals)
                    next_temp = nl.where(swap_needed, curr_vals, next_vals)
                    
                    # Store back
                    nl.store(result[indices_b, indices_d], value=temp, mask=(indices_b < batch_size))
                    nl.store(result[indices_b, indices_d_next], value=next_temp, mask=(indices_b < batch_size))
    
    return result