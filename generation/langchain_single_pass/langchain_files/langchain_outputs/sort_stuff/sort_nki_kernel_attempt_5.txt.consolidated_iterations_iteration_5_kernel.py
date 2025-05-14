from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information statically
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input to result
    if ndim == 1:
        # For 1D tensor, sort the entire array
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Copy input to result first
        trip_count = math.ceil(size / max_tile_size)
        for p in nl.affine_range(trip_count):
            i_p = p * max_tile_size + nl.arange(max_tile_size)
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
        
        # Bubble sort implementation
        for i in range(size):
            for j in range(0, size - i - 1):
                # Process in tiles to handle large arrays
                trip_count = math.ceil((size - i - 1) / max_tile_size)
                for p in nl.affine_range(trip_count):
                    start_idx = p * max_tile_size
                    i_p = start_idx + nl.arange(max_tile_size)
                    
                    # Only process indices that are valid for this comparison
                    valid_mask = (i_p < size - i - 1) & (i_p >= 0)
                    
                    # Load current and next elements
                    curr_elements = nl.load(result[i_p], mask=valid_mask)
                    next_elements = nl.load(result[i_p + 1], mask=valid_mask)
                    
                    # Compare and swap if needed
                    swap_mask = nl.greater(curr_elements, next_elements) & valid_mask
                    
                    # Store swapped elements
                    nl.store(result[i_p], value=nl.where(swap_mask, next_elements, curr_elements), mask=valid_mask)
                    nl.store(result[i_p + 1], value=nl.where(swap_mask, curr_elements, next_elements), mask=valid_mask)
    
    elif ndim == 2:
        # For 2D tensor, sort along the specified dimension
        rows, cols = shape[0], shape[1]
        
        # Copy input to result first
        max_tile_size = nl.tile_size.pmax
        trip_count_rows = math.ceil(rows / max_tile_size)
        
        for p in nl.affine_range(trip_count_rows):
            i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
            i_f = nl.arange(cols)[None, :]
            in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < rows))
            nl.store(result[i_p, i_f], value=in_tile, mask=(i_p < rows))
        
        if dim == 0:
            # Sort along rows (for each column)
            for col in range(cols):
                for i in range(rows):
                    for j in range(0, rows - i - 1):
                        # Load current and next elements
                        curr = nl.load(result[j, col])
                        next_val = nl.load(result[j+1, col])
                        
                        # Compare and swap if needed
                        if_greater = nl.greater(curr, next_val)
                        nl.store(result[j, col], value=nl.where(if_greater, next_val, curr))
                        nl.store(result[j+1, col], value=nl.where(if_greater, curr, next_val))
        else:
            # Sort along columns (for each row)
            for row in range(rows):
                for i in range(cols):
                    for j in range(0, cols - i - 1):
                        # Process in tiles to handle large arrays
                        max_j = cols - i - 1
                        trip_count = math.ceil(max_j / max_tile_size)
                        
                        for p in nl.affine_range(trip_count):
                            start_idx = p * max_tile_size
                            j_indices = start_idx + nl.arange(max_tile_size)
                            
                            # Only process indices that are valid for this comparison
                            valid_mask = (j_indices < max_j) & (j_indices >= 0)
                            
                            # Load current and next elements
                            curr_elements = nl.load(result[row, j_indices], mask=valid_mask)
                            next_elements = nl.load(result[row, j_indices + 1], mask=valid_mask)
                            
                            # Compare and swap if needed
                            swap_mask = nl.greater(curr_elements, next_elements) & valid_mask
                            
                            # Store swapped elements
                            nl.store(result[row, j_indices], value=nl.where(swap_mask, next_elements, curr_elements), mask=valid_mask)
                            nl.store(result[row, j_indices + 1], value=nl.where(swap_mask, curr_elements, next_elements), mask=valid_mask)
    else:
        # For higher dimensional tensors, we need to reshape and handle accordingly
        # This is a simplified implementation that doesn't handle all cases
        # Copy input to result since we're not implementing higher dimensions fully
        size_p = shape[0]
        trip_count = math.ceil(size_p / nl.tile_size.pmax)
        
        # Set up the rest of the dimensions for loading/storing
        rest_dims = []
        for d in range(1, ndim):
            rest_dims.append(nl.arange(shape[d]))
        
        # Create a meshgrid for the rest of the dimensions
        if len(rest_dims) == 1:
            rest_indices = rest_dims[0][None, :]
        else:
            # For higher dimensions, we'd need a more complex approach
            # This is simplified
            rest_indices = rest_dims
            
        # Copy the input tensor to the result tensor
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None, None]
            in_tile = nl.load(a_tensor[i_p, rest_indices], mask=(i_p < size_p))
            nl.store(result[i_p, rest_indices], value=in_tile, mask=(i_p < size_p))
    
    return result