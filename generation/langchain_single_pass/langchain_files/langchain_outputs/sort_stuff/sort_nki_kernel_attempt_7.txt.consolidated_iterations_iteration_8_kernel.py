from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get the shape of the input tensor
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim
        
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy input tensor to result tensor initially
    if ndim == 1:
        # Handle 1D tensor case
        size = shape[0]
        # Calculate the number of tiles needed
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # Process the tensor in tiles
        for p in nl.affine_range(trip_count):
            # Generate indices for current tile
            i_p = p * nl.tile_size.pmax + nl.arange(min(nl.tile_size.pmax, size - p * nl.tile_size.pmax))
            
            # Load input data for current tile
            tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Bubble sort algorithm
            for i in nl.static_range(min(nl.tile_size.pmax, size - p * nl.tile_size.pmax)):
                for j in nl.static_range(min(nl.tile_size.pmax, size - p * nl.tile_size.pmax) - 1):
                    # Compare adjacent elements
                    condition = nl.greater(tile[j], tile[j+1])
                    
                    # Swap if necessary
                    temp = nl.where(condition, tile[j+1], tile[j])
                    tile = nl.where(condition & (nl.arange(min(nl.tile_size.pmax, size - p * nl.tile_size.pmax)) == j), 
                                   temp, tile)
                    
                    temp = nl.where(condition, tile[j], tile[j+1])
                    tile = nl.where(condition & (nl.arange(min(nl.tile_size.pmax, size - p * nl.tile_size.pmax)) == j+1), 
                                   temp, tile)
            
            # Store the sorted tile
            nl.store(result[i_p], value=tile, mask=(i_p < size))
    
    elif dim == ndim - 1:
        # Optimized case: sorting along the last dimension
        # This is a common case so we handle it specifically
        outer_dims_prod = 1
        for i in range(ndim - 1):
            outer_dims_prod *= shape[i]
            
        inner_dim_size = shape[dim]
        
        # Calculate the number of outer dimension tiles needed
        trip_count_outer = math.ceil(outer_dims_prod / nl.tile_size.pmax)
        
        for p_outer in nl.affine_range(trip_count_outer):
            # Generate indices for current outer dimension tile
            i_p_outer = p_outer * nl.tile_size.pmax + nl.arange(min(nl.tile_size.pmax, outer_dims_prod - p_outer * nl.tile_size.pmax))[:, None]
            i_inner = nl.arange(inner_dim_size)[None, :]
            
            # Load data for current tile
            tile = nl.load(a_tensor.reshape(outer_dims_prod, inner_dim_size)[i_p_outer, i_inner], 
                          mask=(i_p_outer < outer_dims_prod))
            
            # Bubble sort algorithm for each row
            for i in nl.static_range(inner_dim_size):
                for j in nl.static_range(inner_dim_size - 1):
                    # Create indices for comparison
                    curr_idx = j
                    next_idx = j + 1
                    
                    # Compare adjacent elements
                    condition = nl.greater(tile[:, curr_idx], tile[:, next_idx])
                    
                    # Swap if necessary using temporary variables to avoid overwriting
                    temp = tile[:, curr_idx].copy()
                    tile[:, curr_idx] = nl.where(condition, tile[:, next_idx], tile[:, curr_idx])
                    tile[:, next_idx] = nl.where(condition, temp, tile[:, next_idx])
            
            # Store the sorted tile
            nl.store(result.reshape(outer_dims_prod, inner_dim_size)[i_p_outer, i_inner], 
                    value=tile, mask=(i_p_outer < outer_dims_prod))
    
    else:
        # General case: sorting along an arbitrary dimension
        # In this case, we transpose the tensor to bring the target dimension to the end,
        # sort it, and then transpose back
        
        # First copy input to result
        for p in nl.affine_range(math.ceil(a_tensor.size / nl.tile_size.pmax)):
            start_idx = p * nl.tile_size.pmax
            indices = start_idx + nl.arange(min(nl.tile_size.pmax, a_tensor.size - start_idx))
            flat_input = a_tensor.reshape(-1)
            flat_result = result.reshape(-1)
            
            # Load input data
            tile = nl.load(flat_input[indices], mask=(indices < a_tensor.size))
            
            # Store to result
            nl.store(flat_result[indices], value=tile, mask=(indices < a_tensor.size))
        
        # Now sort each slice along the target dimension
        # We'll use a bubble sort implementation
        # For simplicity, we'll handle specific common cases
        
        # Calculate sizes and strides for iterating through the tensor
        # This is a simplified approach for 2D and 3D tensors
        if ndim == 2:
            if dim == 0:
                # Sort columns
                for col in nl.affine_range(shape[1]):
                    for i in nl.static_range(shape[0]):
                        for j in nl.static_range(shape[0] - 1):
                            # Load the two elements to compare
                            a_val = nl.load(result[j, col])
                            b_val = nl.load(result[j+1, col])
                            
                            # Check if we need to swap
                            should_swap = nl.greater(a_val, b_val)
                            
                            # Perform the swap if needed
                            if should_swap:
                                nl.store(result[j, col], value=b_val)
                                nl.store(result[j+1, col], value=a_val)
            # dim == 1 is already handled in the optimized case
        
        elif ndim == 3:
            if dim == 0:
                # Sort along first dimension
                for i2 in nl.affine_range(shape[1]):
                    for i3 in nl.affine_range(shape[2]):
                        for i in nl.static_range(shape[0]):
                            for j in nl.static_range(shape[0] - 1):
                                # Load the two elements to compare
                                a_val = nl.load(result[j, i2, i3])
                                b_val = nl.load(result[j+1, i2, i3])
                                
                                # Check if we need to swap
                                should_swap = nl.greater(a_val, b_val)
                                
                                # Perform the swap if needed
                                if should_swap:
                                    nl.store(result[j, i2, i3], value=b_val)
                                    nl.store(result[j+1, i2, i3], value=a_val)
            elif dim == 1:
                # Sort along middle dimension
                for i1 in nl.affine_range(shape[0]):
                    for i3 in nl.affine_range(shape[2]):
                        for i in nl.static_range(shape[1]):
                            for j in nl.static_range(shape[1] - 1):
                                # Load the two elements to compare
                                a_val = nl.load(result[i1, j, i3])
                                b_val = nl.load(result[i1, j+1, i3])
                                
                                # Check if we need to swap
                                should_swap = nl.greater(a_val, b_val)
                                
                                # Perform the swap if needed
                                if should_swap:
                                    nl.store(result[i1, j, i3], value=b_val)
                                    nl.store(result[i1, j+1, i3], value=a_val)
            # dim == 2 is already handled in the optimized case
    
    return result