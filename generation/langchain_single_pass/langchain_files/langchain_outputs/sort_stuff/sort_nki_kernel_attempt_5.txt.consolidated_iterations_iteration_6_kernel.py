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
        # Handle 1D tensor case
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        trip_count = (size + max_tile_size - 1) // max_tile_size  # Ceiling division
        
        # Copy input to result in tiles
        for p in nl.affine_range(trip_count):
            start_idx = p * max_tile_size
            i_p = start_idx + nl.arange(max_tile_size)
            
            # Load one tile of data
            data_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=data_tile, mask=(i_p < size))
        
        # Perform bubble sort on the whole array
        for i in range(size):
            for p in nl.affine_range(trip_count):
                start_idx = p * max_tile_size
                i_p = start_idx + nl.arange(max_tile_size)
                
                # Load current tile
                data_tile = nl.load(result[i_p], mask=(i_p < size))
                
                # For each element in the tile, compare with next element
                # and swap if needed
                for j in range(max_tile_size - 1):
                    curr_idx = i_p + j
                    next_idx = i_p + j + 1
                    
                    # Only process valid indices
                    valid_indices = (curr_idx < size - 1) & (next_idx < size)
                    if nl.any(valid_indices):
                        curr_val = nl.load(result[curr_idx], mask=valid_indices)
                        next_val = nl.load(result[next_idx], mask=valid_indices)
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_val, next_val)
                        if nl.any(swap_needed):
                            # Swap values
                            nl.store(result[curr_idx], value=next_val, mask=valid_indices & swap_needed)
                            nl.store(result[next_idx], value=curr_val, mask=valid_indices & swap_needed)
    else:
        # For multi-dimensional tensors, we sort along the specified dimension
        # First, copy the input tensor to result
        # We'll handle this by flattening the dimensions before and after the sort dimension
        
        # Calculate sizes for processing
        outer_size = 1
        for i in range(dim):
            outer_size *= shape[i]
            
        sort_size = shape[dim]
        
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= shape[i]
        
        # Maximum tile sizes for each dimension
        max_outer_tile = min(nl.tile_size.pmax, outer_size)
        max_sort_tile = min(nl.tile_size.pmax, sort_size)
        max_inner_tile = min(nl.tile_size.pmax, inner_size)
        
        # Copy input to result in tiles
        for o in nl.affine_range((outer_size + max_outer_tile - 1) // max_outer_tile):
            for s in nl.affine_range((sort_size + max_sort_tile - 1) // max_sort_tile):
                for i in nl.affine_range((inner_size + max_inner_tile - 1) // max_inner_tile):
                    # Calculate start indices
                    outer_start = o * max_outer_tile
                    sort_start = s * max_sort_tile
                    inner_start = i * max_inner_tile
                    
                    # Generate indices
                    i_o = outer_start + nl.arange(max_outer_tile)[:, None, None]
                    i_s = sort_start + nl.arange(max_sort_tile)[None, :, None]
                    i_i = inner_start + nl.arange(max_inner_tile)[None, None, :]
                    
                    # Convert to actual tensor indices
                    indices = []
                    idx_pos = 0
                    for d in range(ndim):
                        if d < dim:
                            # This is part of outer dimensions
                            indices.append(i_o // (outer_size // shape[d]))
                        elif d == dim:
                            # This is the sort dimension
                            indices.append(i_s)
                        else:
                            # This is part of inner dimensions
                            indices.append(i_i // (inner_size // shape[d]))
                    
                    # Load data
                    valid_mask = (i_o < outer_size) & (i_s < sort_size) & (i_i < inner_size)
                    data_tile = nl.load(a_tensor[tuple(indices)], mask=valid_mask)
                    
                    # Store to result
                    nl.store(result[tuple(indices)], value=data_tile, mask=valid_mask)
        
        # Perform bubble sort for each outer and inner combination
        for o in nl.affine_range(outer_size):
            for i in nl.affine_range(inner_size):
                # For each outer-inner combination, sort the elements along sort dimension
                for pass_idx in range(sort_size):
                    for s in nl.affine_range(sort_size - 1):
                        # Calculate actual tensor indices for current and next elements
                        curr_indices = []
                        next_indices = []
                        for d in range(ndim):
                            if d < dim:
                                # Outer dimension indices
                                div = 1
                                for d2 in range(d+1, dim):
                                    div *= shape[d2]
                                curr_indices.append((o // div) % shape[d])
                                next_indices.append((o // div) % shape[d])
                            elif d == dim:
                                # Sort dimension indices
                                curr_indices.append(s)
                                next_indices.append(s + 1)
                            else:
                                # Inner dimension indices
                                div = 1
                                for d2 in range(d+1, ndim):
                                    div *= shape[d2]
                                curr_indices.append((i // div) % shape[d])
                                next_indices.append((i // div) % shape[d])
                        
                        # Load current and next values
                        valid = s < sort_size - 1
                        if valid:
                            curr_val = nl.load(result[tuple(curr_indices)])
                            next_val = nl.load(result[tuple(next_indices)])
                            
                            # Compare and swap if needed
                            if nl.greater(curr_val, next_val):
                                # Swap values
                                nl.store(result[tuple(curr_indices)], value=next_val)
                                nl.store(result[tuple(next_indices)], value=curr_val)
    
    return result