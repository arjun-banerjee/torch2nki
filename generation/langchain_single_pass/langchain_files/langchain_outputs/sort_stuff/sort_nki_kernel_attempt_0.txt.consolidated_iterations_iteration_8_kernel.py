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
    
    # First copy input tensor to result tensor
    if ndim == 1:
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        for i in nl.affine_range(math.ceil(size / max_tile_size)):
            start = i * max_tile_size
            # Generate indices for the current tile
            indices = start + nl.arange(max_tile_size)
            # Load input data with masking for boundary
            input_tile = nl.load(a_tensor[indices], mask=(indices < size))
            # Store to result tensor
            nl.store(result[indices], input_tile, mask=(indices < size))
            
        # Bubble sort implementation
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Process in tiles to respect hardware limitations
                for k in nl.affine_range(math.ceil((size - 1) / max_tile_size)):
                    start = k * max_tile_size
                    # Generate indices for the current tile
                    indices = start + nl.arange(max_tile_size)
                    # Constrain to valid range
                    valid_indices = nl.less(indices, size - 1)
                    # Load current elements
                    curr_vals = nl.load(result[indices], mask=valid_indices)
                    next_vals = nl.load(result[indices + 1], mask=valid_indices)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    new_curr = nl.where(swap_needed, next_vals, curr_vals)
                    new_next = nl.where(swap_needed, curr_vals, next_vals)
                    
                    # Store back the values
                    nl.store(result[indices], new_curr, mask=valid_indices)
                    nl.store(result[indices + 1], new_next, mask=valid_indices)
    
    elif ndim == 2:
        rows, cols = shape
        sort_dim_size = shape[dim]
        other_dim = 1 - dim  # The other dimension
        other_dim_size = shape[other_dim]
        max_tile_size = nl.tile_size.pmax
        
        # Copy input to result first
        for i in nl.affine_range(math.ceil(rows / max_tile_size)):
            start_row = i * max_tile_size
            row_indices = start_row + nl.arange(max_tile_size)[:, None]
            col_indices = nl.arange(cols)[None, :]
            
            # Load input data with masking for boundary
            input_tile = nl.load(a_tensor[row_indices, col_indices], mask=(row_indices < rows))
            # Store to result tensor
            nl.store(result[row_indices, col_indices], input_tile, mask=(row_indices < rows))
        
        # Sort along specified dimension
        if dim == 0:  # Sort along rows
            for row in nl.affine_range(cols):
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        for k in nl.affine_range(math.ceil((rows - 1) / max_tile_size)):
                            start = k * max_tile_size
                            indices = start + nl.arange(max_tile_size)
                            valid_indices = nl.less(indices, rows - 1)
                            
                            # Load current and next values
                            curr_vals = nl.load(result[indices, row], mask=valid_indices)
                            next_vals = nl.load(result[indices + 1, row], mask=valid_indices)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store back the values
                            nl.store(result[indices, row], new_curr, mask=valid_indices)
                            nl.store(result[indices + 1, row], new_next, mask=valid_indices)
        
        else:  # Sort along columns (dim == 1)
            for row in nl.affine_range(rows):
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        for k in nl.affine_range(math.ceil((cols - 1) / max_tile_size)):
                            start = k * max_tile_size
                            indices = start + nl.arange(max_tile_size)
                            valid_indices = nl.less(indices, cols - 1)
                            
                            # Load current and next values
                            curr_vals = nl.load(result[row, indices], mask=valid_indices)
                            next_vals = nl.load(result[row, indices + 1], mask=valid_indices)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store back the values
                            nl.store(result[row, indices], new_curr, mask=valid_indices)
                            nl.store(result[row, indices + 1], new_next, mask=valid_indices)
    
    else:  # Higher dimensional tensors
        # For higher dims, we can reshape and handle as 2D case
        # This is a simplified approach that works for common cases
        total_size = 1
        for i in range(ndim):
            if i != dim:
                total_size *= shape[i]
        
        # For now, we'll just support dim=-1 (last dimension) for higher dims
        if dim == ndim - 1:
            dim_size = shape[dim]
            max_tile_size = nl.tile_size.pmax
            
            # Copy input to result first
            for batch in nl.affine_range(math.ceil(total_size / max_tile_size)):
                start_batch = batch * max_tile_size
                batch_indices = start_batch + nl.arange(max_tile_size)[:, None]
                dim_indices = nl.arange(dim_size)[None, :]
                
                # Use flat indexing for the non-sort dimensions
                input_tile = nl.load(a_tensor.reshape((total_size, dim_size))[batch_indices, dim_indices], 
                                    mask=(batch_indices < total_size))
                nl.store(result.reshape((total_size, dim_size))[batch_indices, dim_indices], 
                         input_tile, mask=(batch_indices < total_size))
            
            # Sort each row along the last dimension
            for batch in nl.affine_range(total_size):
                for i in nl.affine_range(dim_size):
                    for j in nl.affine_range(dim_size - 1):
                        for k in nl.affine_range(math.ceil((dim_size - 1) / max_tile_size)):
                            start = k * max_tile_size
                            indices = start + nl.arange(max_tile_size)
                            valid_indices = nl.less(indices, dim_size - 1)
                            
                            # Load current and next values
                            curr_vals = nl.load(result.reshape((total_size, dim_size))[batch, indices], 
                                              mask=valid_indices)
                            next_vals = nl.load(result.reshape((total_size, dim_size))[batch, indices + 1], 
                                              mask=valid_indices)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store back the values
                            nl.store(result.reshape((total_size, dim_size))[batch, indices], 
                                    new_curr, mask=valid_indices)
                            nl.store(result.reshape((total_size, dim_size))[batch, indices + 1], 
                                    new_next, mask=valid_indices)
    
    return result