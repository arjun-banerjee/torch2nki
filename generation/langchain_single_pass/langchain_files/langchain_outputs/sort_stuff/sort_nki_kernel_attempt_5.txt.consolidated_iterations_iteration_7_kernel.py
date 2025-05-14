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
    
    # Copy input to result first
    if ndim == 1:
        # Simple 1D case
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # First copy the input to result
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
        
        # Bubble sort the 1D array
        for i in range(size):
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                
                # Load current elements
                curr_tile = nl.load(result[i_p], mask=(i_p < size - i - 1))
                
                # Load next elements (with offset of 1)
                next_indices = i_p + 1
                next_tile = nl.load(result[next_indices], mask=(i_p < size - i - 1) & (next_indices < size))
                
                # Compare and swap if needed
                swap_mask = (i_p < size - i - 1) & (curr_tile > next_tile)
                
                # Store the smaller values at current positions
                nl.store(result[i_p], 
                         value=nl.where(curr_tile > next_tile, next_tile, curr_tile),
                         mask=swap_mask)
                
                # Store the larger values at next positions
                nl.store(result[next_indices], 
                         value=nl.where(curr_tile > next_tile, curr_tile, next_tile),
                         mask=swap_mask)
    
    elif ndim == 2:
        rows, cols = shape[0], shape[1]
        
        # Determine which dimension to sort along
        if dim == 0:  # Sort along rows
            # First copy the input to result
            for r in range(rows):
                trip_count = math.ceil(cols / nl.tile_size.pmax)
                for p in nl.affine_range(trip_count):
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    in_tile = nl.load(a_tensor[r, i_p], mask=(i_p < cols))
                    nl.store(result[r, i_p], value=in_tile, mask=(i_p < cols))
            
            # Sort each column independently
            for c in range(cols):
                # Bubble sort algorithm
                for i in range(rows):
                    for j in range(rows - i - 1):
                        # Load current and next element
                        curr_val = nl.load(result[j, c])
                        next_val = nl.load(result[j+1, c])
                        
                        # Compare and swap if needed
                        if_greater = nl.greater(curr_val, next_val)
                        
                        # Store swapped values if needed
                        nl.store(result[j, c], value=next_val, mask=if_greater)
                        nl.store(result[j+1, c], value=curr_val, mask=if_greater)
        
        else:  # Sort along columns (dim == 1)
            # First copy the input to result
            row_trip_count = math.ceil(rows / nl.tile_size.pmax)
            
            for r_p in nl.affine_range(row_trip_count):
                i_r = r_p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_c = nl.arange(cols)[None, :]
                in_tile = nl.load(a_tensor[i_r, i_c], mask=(i_r < rows))
                nl.store(result[i_r, i_c], value=in_tile, mask=(i_r < rows))
            
            # Sort each row independently
            for r in range(rows):
                # Bubble sort algorithm
                for i in range(cols):
                    trip_count = math.ceil((cols - i - 1) / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        
                        # Load current elements
                        curr_tile = nl.load(result[r, i_p], mask=(i_p < cols - i - 1))
                        
                        # Load next elements (with offset of 1)
                        next_indices = i_p + 1
                        next_tile = nl.load(result[r, next_indices], mask=(i_p < cols - i - 1))
                        
                        # Compare and swap if needed
                        swap_mask = (i_p < cols - i - 1) & (curr_tile > next_tile)
                        
                        # Store the smaller values at current positions
                        nl.store(result[r, i_p], 
                                value=nl.where(curr_tile > next_tile, next_tile, curr_tile),
                                mask=swap_mask)
                        
                        # Store the larger values at next positions
                        nl.store(result[r, next_indices], 
                                value=nl.where(curr_tile > next_tile, curr_tile, next_tile),
                                mask=swap_mask)
    
    else:  # ndim > 2
        # For higher dimensions, we reshape and handle as a batch of 2D problems
        # Determine outer dimensions (before dim) and inner dimensions (after dim)
        outer_size = 1
        for i in range(dim):
            outer_size *= shape[i]
            
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= shape[i]
            
        sort_size = shape[dim]
        
        # First copy the input to result
        for outer in range(outer_size):
            for inner in range(inner_size):
                trip_count = math.ceil(sort_size / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Create indices for the current position
                    indices = []
                    remaining_outer = outer
                    for i in range(dim):
                        dim_size = shape[i]
                        indices.append(remaining_outer % dim_size)
                        remaining_outer = remaining_outer // dim_size
                    
                    indices.append(i_p)  # Add sorting dimension
                    
                    remaining_inner = inner
                    for i in range(dim + 1, ndim):
                        dim_size = shape[i]
                        indices.append(remaining_inner % dim_size)
                        remaining_inner = remaining_inner // dim_size
                    
                    # Convert to tuple for indexing
                    if len(indices) == 1:
                        input_idx = i_p
                        result_idx = i_p
                    elif len(indices) == 2:
                        input_idx = (indices[0], indices[1])
                        result_idx = (indices[0], indices[1])
                    elif len(indices) == 3:
                        input_idx = (indices[0], indices[1], indices[2])
                        result_idx = (indices[0], indices[1], indices[2])
                    
                    # Load and store
                    if ndim == 3:
                        if dim == 0:
                            in_tile = nl.load(a_tensor[i_p, outer % shape[1], inner], mask=(i_p < sort_size))
                            nl.store(result[i_p, outer % shape[1], inner], value=in_tile, mask=(i_p < sort_size))
                        elif dim == 1:
                            in_tile = nl.load(a_tensor[outer, i_p, inner], mask=(i_p < sort_size))
                            nl.store(result[outer, i_p, inner], value=in_tile, mask=(i_p < sort_size))
                        else:  # dim == 2
                            in_tile = nl.load(a_tensor[outer, inner, i_p], mask=(i_p < sort_size))
                            nl.store(result[outer, inner, i_p], value=in_tile, mask=(i_p < sort_size))
                
                # Bubble sort along the specified dimension
                for i in range(sort_size):
                    trip_count = math.ceil((sort_size - i - 1) / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        next_indices = i_p + 1
                        
                        # Load current and next elements based on dimension
                        if ndim == 3:
                            if dim == 0:
                                curr_tile = nl.load(result[i_p, outer % shape[1], inner], mask=(i_p < sort_size - i - 1))
                                next_tile = nl.load(result[next_indices, outer % shape[1], inner], 
                                                  mask=(i_p < sort_size - i - 1) & (next_indices < sort_size))
                                
                                # Compare and swap if needed
                                swap_mask = (i_p < sort_size - i - 1) & (curr_tile > next_tile)
                                
                                # Store the smaller values at current positions
                                nl.store(result[i_p, outer % shape[1], inner], 
                                        value=nl.where(curr_tile > next_tile, next_tile, curr_tile),
                                        mask=swap_mask)
                                
                                # Store the larger values at next positions
                                nl.store(result[next_indices, outer % shape[1], inner], 
                                        value=nl.where(curr_tile > next_tile, curr_tile, next_tile),
                                        mask=swap_mask)
                            
                            elif dim == 1:
                                curr_tile = nl.load(result[outer, i_p, inner], mask=(i_p < sort_size - i - 1))
                                next_tile = nl.load(result[outer, next_indices, inner], 
                                                  mask=(i_p < sort_size - i - 1) & (next_indices < sort_size))
                                
                                # Compare and swap if needed
                                swap_mask = (i_p < sort_size - i - 1) & (curr_tile > next_tile)
                                
                                # Store the smaller values at current positions
                                nl.store(result[outer, i_p, inner], 
                                        value=nl.where(curr_tile > next_tile, next_tile, curr_tile),
                                        mask=swap_mask)
                                
                                # Store the larger values at next positions
                                nl.store(result[outer, next_indices, inner], 
                                        value=nl.where(curr_tile > next_tile, curr_tile, next_tile),
                                        mask=swap_mask)
                            
                            else:  # dim == 2
                                curr_tile = nl.load(result[outer, inner, i_p], mask=(i_p < sort_size - i - 1))
                                next_tile = nl.load(result[outer, inner, next_indices], 
                                                  mask=(i_p < sort_size - i - 1) & (next_indices < sort_size))
                                
                                # Compare and swap if needed
                                swap_mask = (i_p < sort_size - i - 1) & (curr_tile > next_tile)
                                
                                # Store the smaller values at current positions
                                nl.store(result[outer, inner, i_p], 
                                        value=nl.where(curr_tile > next_tile, next_tile, curr_tile),
                                        mask=swap_mask)
                                
                                # Store the larger values at next positions
                                nl.store(result[outer, inner, next_indices], 
                                        value=nl.where(curr_tile > next_tile, curr_tile, next_tile),
                                        mask=swap_mask)
    
    return result