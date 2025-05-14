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
    
    # Get tensor shape as a list (not tuple)
    shape_list = []
    for i in range(ndim):
        shape_list.append(a_tensor.shape[i])
    
    # First copy the input tensor to result
    if ndim == 1:
        # For 1D tensor, use simple tiling
        size = shape_list[0]
        tiles = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(tiles):
            start = p * nl.tile_size.pmax
            # Create indices for current tile
            indices = nl.arange(nl.tile_size.pmax) + start
            # Load data with masking for boundary
            data = nl.load(a_tensor[indices], mask=(indices < size))
            # Store to result
            nl.store(result[indices], data, mask=(indices < size))
            
    else:
        # For multi-dimensional tensors
        # Calculate the total size for all dimensions except the sorting dimension
        outer_size = 1
        for i in range(ndim):
            if i != dim:
                outer_size *= shape_list[i]
        
        # Calculate number of outer tiles
        outer_tiles = math.ceil(outer_size / nl.tile_size.pmax)
        
        # Calculate the size of the dimension to sort
        sort_dim_size = shape_list[dim]
        
        # Copy data to result tensor first
        for p in nl.affine_range(outer_tiles):
            outer_start = p * nl.tile_size.pmax
            
            # Load and store data for the entire sort dimension
            for s in nl.affine_range(sort_dim_size):
                # Create multi-dimensional indices based on flattened index
                flat_indices = nl.arange(nl.tile_size.pmax) + outer_start
                multi_indices = []
                
                remaining = flat_indices
                for i in range(ndim):
                    if i != dim:
                        stride = 1
                        for j in range(i+1, ndim):
                            if j != dim:
                                stride *= shape_list[j]
                        idx = (remaining // stride) % shape_list[i]
                        multi_indices.append(idx)
                    else:
                        # For the sort dimension, use the current sort index
                        multi_indices.append(nl.full(nl.tile_size.pmax, s, dtype=nl.int32))
                
                # Convert to tuple of indices for each dimension
                indices_tuple = tuple(multi_indices[i] for i in range(ndim))
                
                # Mask for valid indices
                mask = (flat_indices < outer_size)
                
                # Load data with masking
                data = nl.load(a_tensor[indices_tuple], mask=mask)
                
                # Store to result
                nl.store(result[indices_tuple], data, mask=mask)
    
    # Now perform bubble sort on the result tensor along the specified dimension
    if ndim == 1:
        # For 1D tensor, sort the entire array
        size = shape_list[0]
        
        # Bubble sort
        for i in nl.affine_range(size):
            # In bubble sort, we do size-i-1 comparisons in each pass
            for j in nl.affine_range(size-1):
                # Calculate indices for current and next elements
                curr_idx = j
                next_idx = j + 1
                
                # Skip if we're past the effective range for this pass
                if curr_idx >= size-i-1:
                    continue
                
                # Load current and next elements
                curr_val = nl.load(result[curr_idx])
                next_val = nl.load(result[next_idx])
                
                # Compare and swap if needed
                need_swap = nl.greater(curr_val, next_val)
                
                if need_swap:
                    # Swap values
                    nl.store(result[curr_idx], next_val)
                    nl.store(result[next_idx], curr_val)
    else:
        # For multi-dimensional tensors
        # Calculate the total size for all dimensions except the sorting dimension
        outer_size = 1
        for i in range(ndim):
            if i != dim:
                outer_size *= shape_list[i]
        
        # Calculate number of outer tiles
        outer_tiles = math.ceil(outer_size / nl.tile_size.pmax)
        
        # Calculate the size of the dimension to sort
        sort_dim_size = shape_list[dim]
        
        # Process each outer tile
        for p in nl.affine_range(outer_tiles):
            outer_start = p * nl.tile_size.pmax
            
            # Bubble sort for each position in the outer dimensions
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size-1):
                    # Skip if we're past the effective range for this pass
                    if j >= sort_dim_size-i-1:
                        continue
                    
                    # Create multi-dimensional indices based on flattened index
                    flat_indices = nl.arange(nl.tile_size.pmax) + outer_start
                    
                    # Create indices for current and next elements along sort dimension
                    curr_indices = []
                    next_indices = []
                    
                    remaining = flat_indices
                    for k in range(ndim):
                        if k != dim:
                            stride = 1
                            for m in range(k+1, ndim):
                                if m != dim:
                                    stride *= shape_list[m]
                            idx = (remaining // stride) % shape_list[k]
                            curr_indices.append(idx)
                            next_indices.append(idx)
                        else:
                            # For the sort dimension, use j and j+1
                            curr_indices.append(nl.full(nl.tile_size.pmax, j, dtype=nl.int32))
                            next_indices.append(nl.full(nl.tile_size.pmax, j+1, dtype=nl.int32))
                    
                    # Convert to tuple of indices for each dimension
                    curr_indices_tuple = tuple(curr_indices[k] for k in range(ndim))
                    next_indices_tuple = tuple(next_indices[k] for k in range(ndim))
                    
                    # Mask for valid indices
                    mask = (flat_indices < outer_size)
                    
                    # Load current and next elements
                    curr_val = nl.load(result[curr_indices_tuple], mask=mask)
                    next_val = nl.load(result[next_indices_tuple], mask=mask)
                    
                    # Compare elements
                    swap_mask = nl.greater(curr_val, next_val) & mask
                    
                    # Swap if needed
                    temp = nl.zeros(curr_val.shape, dtype=curr_val.dtype, buffer=nl.sbuf)
                    temp = curr_val
                    
                    # Store swapped values
                    nl.store(result[curr_indices_tuple], nl.where(swap_mask, next_val, curr_val), mask=mask)
                    nl.store(result[next_indices_tuple], nl.where(swap_mask, temp, next_val), mask=mask)
    
    return result