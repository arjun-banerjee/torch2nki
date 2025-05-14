from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if ndim == 1:
        size = shape[0]
        # Copy input to result first
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = i * nl.tile_size.pmax
            # Generate indices for current tile
            indices = start_idx + nl.arange(nl.tile_size.pmax)
            # Load data with masking to handle boundary
            tile_data = nl.load(a_tensor[indices], mask=(indices < size))
            # Store data to result
            nl.store(result[indices], value=tile_data, mask=(indices < size))
        
        # Bubble sort implementation for 1D
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Process in tiles to respect hardware limitations
                for k in nl.affine_range(math.ceil((size - 1) / nl.tile_size.pmax)):
                    start_idx = k * nl.tile_size.pmax
                    # Generate indices for current tile
                    indices = start_idx + nl.arange(nl.tile_size.pmax)
                    valid_mask = (indices < size - 1)
                    
                    # Load current and next elements
                    curr_vals = nl.load(result[indices], mask=valid_mask)
                    next_vals = nl.load(result[indices + 1], mask=valid_mask)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    new_curr = nl.where(swap_needed, next_vals, curr_vals)
                    new_next = nl.where(swap_needed, curr_vals, next_vals)
                    
                    # Store back the values
                    nl.store(result[indices], value=new_curr, mask=valid_mask)
                    nl.store(result[indices + 1], value=new_next, mask=valid_mask)
    
    # Handle 2D tensor case
    elif ndim == 2:
        rows, cols = shape
        
        # Copy input to result first
        for i in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
            start_row = i * nl.tile_size.pmax
            # Generate row indices for current tile
            row_indices = start_row + nl.arange(nl.tile_size.pmax)[:, None]
            col_indices = nl.arange(cols)[None, :]
            
            # Load data with masking to handle boundary
            tile_data = nl.load(a_tensor[row_indices, col_indices], mask=(row_indices < rows))
            # Store data to result
            nl.store(result[row_indices, col_indices], value=tile_data, mask=(row_indices < rows))
        
        # Sort along specified dimension
        if dim == 0:  # Sort along rows
            for i in nl.affine_range(cols):
                for j in nl.affine_range(rows - 1):
                    for k in nl.affine_range(math.ceil((rows - 1) / nl.tile_size.pmax)):
                        start_idx = k * nl.tile_size.pmax
                        # Generate indices for current tile
                        indices = start_idx + nl.arange(nl.tile_size.pmax)
                        valid_mask = (indices < rows - 1)
                        
                        # Load current and next elements
                        curr_vals = nl.load(result[indices, i], mask=valid_mask)
                        next_vals = nl.load(result[indices + 1, i], mask=valid_mask)
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back the values
                        nl.store(result[indices, i], value=new_curr, mask=valid_mask)
                        nl.store(result[indices + 1, i], value=new_next, mask=valid_mask)
        else:  # Sort along columns (dim == 1)
            for i in nl.affine_range(rows):
                for j in nl.affine_range(cols - 1):
                    for k in nl.affine_range(math.ceil((cols - 1) / nl.tile_size.pmax)):
                        start_idx = k * nl.tile_size.pmax
                        # Generate indices for current tile
                        indices = start_idx + nl.arange(nl.tile_size.pmax)
                        valid_mask = (indices < cols - 1)
                        
                        # Load current and next elements
                        curr_vals = nl.load(result[i, indices], mask=valid_mask)
                        next_vals = nl.load(result[i, indices + 1], mask=valid_mask)
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back the values
                        nl.store(result[i, indices], value=new_curr, mask=valid_mask)
                        nl.store(result[i, indices + 1], value=new_next, mask=valid_mask)
    
    return result