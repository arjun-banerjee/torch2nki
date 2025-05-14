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
    
    # Copy input to result first
    if ndim == 1:
        # 1D tensor case
        size = a_tensor.shape[0]
        # Maximum tile size for processing
        max_tile_size = 128  # Using a fixed value instead of nl.tile_size.pmax
        
        # Copy input to result in tiles
        for i in nl.affine_range((size + max_tile_size - 1) // max_tile_size):
            start_idx = i * max_tile_size
            # Generate indices for the current tile
            indices = nl.arange(max_tile_size) + start_idx
            # Load values from input tensor
            values = nl.load(a_tensor[indices], mask=(indices < size))
            # Store values to result tensor
            nl.store(result[indices], values, mask=(indices < size))
        
        # Bubble sort implementation
        for i in nl.affine_range(size):
            # For each iteration, we need to go through the array and swap adjacent elements
            num_passes = (size - i - 1 + max_tile_size - 1) // max_tile_size
            
            for j in nl.affine_range(num_passes):
                start_idx = j * max_tile_size
                # Load current segment
                indices = nl.arange(max_tile_size) + start_idx
                # Ensure we don't go beyond the array bounds
                valid_indices = (indices < (size - i)) & (indices < size)
                values = nl.load(result[indices], mask=valid_indices)
                
                # Get the next values (shifted by 1)
                next_indices = indices + 1
                next_valid = (next_indices < (size - i)) & (next_indices < size)
                next_values = nl.load(result[next_indices], mask=next_valid)
                
                # Compare and swap if needed
                swap_needed = nl.greater(values, next_values) & next_valid & valid_indices
                
                # Update values based on swap condition
                new_values = nl.where(swap_needed, next_values, values)
                new_next_values = nl.where(swap_needed, values, next_values)
                
                # Store updated values
                nl.store(result[indices], new_values, mask=valid_indices)
                nl.store(result[next_indices], new_next_values, mask=next_valid)
    
    elif ndim == 2:
        # 2D tensor case
        rows = a_tensor.shape[0]
        cols = a_tensor.shape[1]
        
        # Maximum tile size for processing
        max_tile_size = 128  # Using a fixed value instead of nl.tile_size.pmax
        
        # Copy input to result in tiles
        for i in nl.affine_range((rows + max_tile_size - 1) // max_tile_size):
            start_row = i * max_tile_size
            row_indices = nl.arange(max_tile_size)[:, None] + start_row
            col_indices = nl.arange(cols)[None, :]
            
            # Load values from input tensor
            values = nl.load(a_tensor[row_indices, col_indices], mask=(row_indices < rows))
            # Store values to result tensor
            nl.store(result[row_indices, col_indices], values, mask=(row_indices < rows))
        
        # Sort along specified dimension
        if dim == 0:
            # Sort along rows (for each column)
            for c in nl.affine_range(cols):
                # Bubble sort each column
                for i in nl.affine_range(rows):
                    for r in nl.affine_range(rows - i - 1):
                        # Load adjacent elements
                        val1 = nl.load(result[r, c])
                        val2 = nl.load(result[r+1, c])
                        
                        # Compare and swap if needed
                        if_greater = nl.greater(val1, val2)
                        temp1 = nl.where(if_greater, val2, val1)
                        temp2 = nl.where(if_greater, val1, val2)
                        
                        # Store back
                        nl.store(result[r, c], temp1)
                        nl.store(result[r+1, c], temp2)
        else:
            # Sort along columns (for each row)
            for r in nl.affine_range(rows):
                # Bubble sort each row
                for i in nl.affine_range(cols):
                    num_passes = (cols - i - 1 + max_tile_size - 1) // max_tile_size
                    
                    for j in nl.affine_range(num_passes):
                        start_col = j * max_tile_size
                        # Load current segment
                        col_indices = nl.arange(max_tile_size) + start_col
                        # Ensure we don't go beyond the array bounds
                        valid_indices = (col_indices < (cols - i)) & (col_indices < cols)
                        values = nl.load(result[r, col_indices], mask=valid_indices)
                        
                        # Get the next values (shifted by 1)
                        next_indices = col_indices + 1
                        next_valid = (next_indices < (cols - i)) & (next_indices < cols)
                        next_values = nl.load(result[r, next_indices], mask=next_valid)
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(values, next_values) & next_valid & valid_indices
                        
                        # Update values based on swap condition
                        new_values = nl.where(swap_needed, next_values, values)
                        new_next_values = nl.where(swap_needed, values, next_values)
                        
                        # Store updated values
                        nl.store(result[r, col_indices], new_values, mask=valid_indices)
                        nl.store(result[r, next_indices], new_next_values, mask=next_valid)
    else:
        # Higher dimensional tensors - handle the specified dimension
        # First, copy input to result
        shape = a_tensor.shape
        dim_size = shape[dim]
        
        # Determine the total number of slices to process
        total_slices = 1
        for i in range(ndim):
            if i != dim:
                total_slices *= shape[i]
        
        # Process each slice separately
        for slice_idx in nl.affine_range(total_slices):
            # Sort each slice along the specified dimension
            for i in nl.affine_range(dim_size):
                for j in nl.affine_range(dim_size - i - 1):
                    # For higher dimensions, we need to calculate the indices
                    # This is a simplified approach for demonstration
                    if dim == 0:
                        val1 = nl.load(result[j, slice_idx // shape[2], slice_idx % shape[2]])
                        val2 = nl.load(result[j+1, slice_idx // shape[2], slice_idx % shape[2]])
                        
                        # Compare and swap if needed
                        if_greater = nl.greater(val1, val2)
                        temp1 = nl.where(if_greater, val2, val1)
                        temp2 = nl.where(if_greater, val1, val2)
                        
                        # Store back
                        nl.store(result[j, slice_idx // shape[2], slice_idx % shape[2]], temp1)
                        nl.store(result[j+1, slice_idx // shape[2], slice_idx % shape[2]], temp2)
                    elif dim == 1:
                        val1 = nl.load(result[slice_idx // shape[2], j, slice_idx % shape[2]])
                        val2 = nl.load(result[slice_idx // shape[2], j+1, slice_idx % shape[2]])
                        
                        # Compare and swap if needed
                        if_greater = nl.greater(val1, val2)
                        temp1 = nl.where(if_greater, val2, val1)
                        temp2 = nl.where(if_greater, val1, val2)
                        
                        # Store back
                        nl.store(result[slice_idx // shape[2], j, slice_idx % shape[2]], temp1)
                        nl.store(result[slice_idx // shape[2], j+1, slice_idx % shape[2]], temp2)
                    else:  # dim == 2
                        val1 = nl.load(result[slice_idx // shape[1], slice_idx % shape[1], j])
                        val2 = nl.load(result[slice_idx // shape[1], slice_idx % shape[1], j+1])
                        
                        # Compare and swap if needed
                        if_greater = nl.greater(val1, val2)
                        temp1 = nl.where(if_greater, val2, val1)
                        temp2 = nl.where(if_greater, val1, val2)
                        
                        # Store back
                        nl.store(result[slice_idx // shape[1], slice_idx % shape[1], j], temp1)
                        nl.store(result[slice_idx // shape[1], slice_idx % shape[1], j+1], temp2)
    
    return result