from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension
    ndim = len(a_tensor.shape)
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input to result
    if ndim == 1:
        # 1D tensor case
        size = a_tensor.shape[0]
        max_tile_size = 128  # Use a constant instead of nl.tile_size.pmax
        
        # Copy input to result first
        for i in nl.affine_range(math.ceil(size / max_tile_size)):
            start_idx = i * max_tile_size
            # Create index arrays for the current tile
            indices = nl.arange(max_tile_size)
            # Load input data with masking to handle boundary
            input_tile = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < size))
            # Store to result
            nl.store(result[start_idx + indices], input_tile, mask=(start_idx + indices < size))
        
        # Bubble sort implementation
        for i in range(size - 1):
            for j in nl.affine_range(math.ceil((size - i - 1) / max_tile_size)):
                start_idx = j * max_tile_size
                # Create index arrays for the current tile
                indices = nl.arange(max_tile_size)
                actual_indices = start_idx + indices
                
                # Load adjacent elements
                curr_vals = nl.load(result[actual_indices], mask=(actual_indices < size - i))
                next_vals = nl.load(result[actual_indices + 1], mask=(actual_indices + 1 < size - i))
                
                # Compare and swap if needed
                swap_needed = nl.greater(curr_vals, next_vals)
                new_curr = nl.where(swap_needed, next_vals, curr_vals)
                new_next = nl.where(swap_needed, curr_vals, next_vals)
                
                # Store back the results
                nl.store(result[actual_indices], new_curr, mask=(actual_indices < size - i))
                nl.store(result[actual_indices + 1], new_next, mask=(actual_indices + 1 < size - i))
    
    elif ndim == 2:
        # 2D tensor case
        rows = a_tensor.shape[0]
        cols = a_tensor.shape[1]
        max_tile_size = 128  # Use a constant instead of nl.tile_size.pmax
        
        # Copy input to result first
        for r in nl.affine_range(math.ceil(rows / max_tile_size)):
            r_start = r * max_tile_size
            r_indices = nl.arange(max_tile_size)[:, None]
            actual_r = r_start + r_indices
            
            for c in nl.affine_range(math.ceil(cols / max_tile_size)):
                c_start = c * max_tile_size
                c_indices = nl.arange(max_tile_size)[None, :]
                actual_c = c_start + c_indices
                
                # Load with masking for boundaries
                input_tile = nl.load(a_tensor[actual_r, actual_c], 
                                    mask=((actual_r < rows) & (actual_c < cols)))
                
                # Store to result
                nl.store(result[actual_r, actual_c], input_tile, 
                        mask=((actual_r < rows) & (actual_c < cols)))
        
        if dim == 0:
            # Sort along rows
            for i in range(rows - 1):
                for j in nl.affine_range(math.ceil((rows - i - 1) / max_tile_size)):
                    j_start = j * max_tile_size
                    j_indices = nl.arange(max_tile_size)[:, None]
                    actual_j = j_start + j_indices
                    
                    for c in nl.affine_range(math.ceil(cols / max_tile_size)):
                        c_start = c * max_tile_size
                        c_indices = nl.arange(max_tile_size)[None, :]
                        actual_c = c_start + c_indices
                        
                        # Load adjacent rows
                        curr_vals = nl.load(result[actual_j, actual_c], 
                                          mask=((actual_j < rows - i) & (actual_c < cols)))
                        next_vals = nl.load(result[actual_j + 1, actual_c], 
                                          mask=((actual_j + 1 < rows - i) & (actual_c < cols)))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back the results
                        nl.store(result[actual_j, actual_c], new_curr, 
                                mask=((actual_j < rows - i) & (actual_c < cols)))
                        nl.store(result[actual_j + 1, actual_c], new_next, 
                                mask=((actual_j + 1 < rows - i) & (actual_c < cols)))
        else:
            # Sort along columns (dim == 1)
            for i in range(cols - 1):
                for r in nl.affine_range(math.ceil(rows / max_tile_size)):
                    r_start = r * max_tile_size
                    r_indices = nl.arange(max_tile_size)[:, None]
                    actual_r = r_start + r_indices
                    
                    for j in nl.affine_range(math.ceil((cols - i - 1) / max_tile_size)):
                        j_start = j * max_tile_size
                        j_indices = nl.arange(max_tile_size)[None, :]
                        actual_j = j_start + j_indices
                        
                        # Load adjacent columns
                        curr_vals = nl.load(result[actual_r, actual_j], 
                                          mask=((actual_r < rows) & (actual_j < cols - i)))
                        next_vals = nl.load(result[actual_r, actual_j + 1], 
                                          mask=((actual_r < rows) & (actual_j + 1 < cols - i)))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back the results
                        nl.store(result[actual_r, actual_j], new_curr, 
                                mask=((actual_r < rows) & (actual_j < cols - i)))
                        nl.store(result[actual_r, actual_j + 1], new_next, 
                                mask=((actual_r < rows) & (actual_j + 1 < cols - i)))
    else:
        # For higher dimensions, only handling the specified dimension
        # First copy the tensor
        shape = a_tensor.shape
        
        # Calculate sizes for proper handling
        size_dim = shape[dim]  # Size of the dimension to sort along
        
        # Calculate total size before the sort dimension
        size_before = 1
        for i in range(dim):
            size_before *= shape[i]
        
        # Calculate total size after the sort dimension
        size_after = 1
        for i in range(dim + 1, ndim):
            size_after *= shape[i]
        
        # Copy input to result first
        max_tile_size = 128  # Use a constant instead of nl.tile_size.pmax
        total_elements = size_before * size_dim * size_after
        
        # Copy in chunks
        for i in nl.affine_range(math.ceil(total_elements / max_tile_size)):
            start_idx = i * max_tile_size
            indices = nl.arange(max_tile_size)
            actual_indices = start_idx + indices
            
            # Convert linear index to multi-dimensional index
            # This is just a flat copy for now
            flat_input = nl.load(a_tensor.reshape(-1)[actual_indices], 
                               mask=(actual_indices < total_elements))
            nl.store(result.reshape(-1)[actual_indices], flat_input, 
                    mask=(actual_indices < total_elements))
        
        # Perform bubble sort along the specified dimension
        # For each element in the outer dimensions
        for outer in range(size_before):
            # For each element in the inner dimensions
            for inner in range(size_after):
                # Bubble sort along the specified dimension
                for i in range(size_dim - 1):
                    for j in nl.affine_range(math.ceil((size_dim - i - 1) / max_tile_size)):
                        j_start = j * max_tile_size
                        j_indices = nl.arange(max_tile_size)
                        actual_j = j_start + j_indices
                        
                        # Calculate flat indices for the current and next elements
                        curr_flat_idx = outer * (size_dim * size_after) + actual_j * size_after + inner
                        next_flat_idx = outer * (size_dim * size_after) + (actual_j + 1) * size_after + inner
                        
                        # Load current and next elements
                        curr_vals = nl.load(result.reshape(-1)[curr_flat_idx], 
                                         mask=((actual_j < size_dim - i) & (curr_flat_idx < total_elements)))
                        next_vals = nl.load(result.reshape(-1)[next_flat_idx], 
                                         mask=((actual_j + 1 < size_dim - i) & (next_flat_idx < total_elements)))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back the results
                        nl.store(result.reshape(-1)[curr_flat_idx], new_curr, 
                               mask=((actual_j < size_dim - i) & (curr_flat_idx < total_elements)))
                        nl.store(result.reshape(-1)[next_flat_idx], new_next, 
                               mask=((actual_j + 1 < size_dim - i) & (next_flat_idx < total_elements)))
    
    return result