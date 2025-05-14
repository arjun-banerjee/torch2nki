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
        # Handle 1D tensor - sort along the only dimension
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Copy input to result first
        for p in nl.affine_range(math.ceil(size / max_tile_size)):
            start_idx = p * max_tile_size
            i_p = nl.arange(max_tile_size)
            
            # Load input data
            in_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < size))
            
            # Store to result
            nl.store(result[start_idx + i_p], value=in_tile, mask=(start_idx + i_p < size))
        
        # Perform bubble sort
        for i in range(size):
            for j in range(0, size - i - 1):
                for p in nl.affine_range(math.ceil(size / max_tile_size)):
                    start_idx = p * max_tile_size
                    i_p = nl.arange(max_tile_size)
                    valid_idx = start_idx + i_p
                    
                    # Mask for valid indices
                    valid_mask = (valid_idx < size) & ((valid_idx == j) | (valid_idx == j + 1))
                    
                    # Load current and next elements
                    curr_batch = nl.load(result[start_idx + i_p], mask=valid_mask)
                    
                    # Identify elements to compare
                    is_j = valid_idx == j
                    is_j_plus_1 = valid_idx == (j + 1)
                    
                    # Calculate indices for the two elements we're comparing
                    j_idx = j
                    j_plus_1_idx = j + 1
                    
                    # If both elements are in current tile, perform swap if needed
                    if (j_idx >= start_idx and j_idx < start_idx + max_tile_size and 
                        j_plus_1_idx >= start_idx and j_plus_1_idx < start_idx + max_tile_size):
                        
                        j_offset = j_idx - start_idx
                        j_plus_1_offset = j_plus_1_idx - start_idx
                        
                        el_j = curr_batch[j_offset]
                        el_j_plus_1 = curr_batch[j_plus_1_offset]
                        
                        # Check if swap is needed
                        swap_needed = nl.greater(el_j, el_j_plus_1)
                        
                        if swap_needed:
                            # Swap elements
                            curr_batch[j_offset] = el_j_plus_1
                            curr_batch[j_plus_1_offset] = el_j
                            
                            # Store the swapped elements back
                            nl.store(result[start_idx + i_p], value=curr_batch, mask=valid_mask)
    else:
        # Handle multi-dimensional tensor - sort along specified dimension
        # Determine the size of the dimension to sort along
        sort_dim_size = shape[dim]
        
        # Calculate sizes for processing
        # Product of dimensions before the sort dimension
        outer_size = 1
        for i in range(dim):
            outer_size *= shape[i]
        
        # Product of dimensions after the sort dimension
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= shape[i]
        
        # Copy input to result first
        for o in nl.affine_range(math.ceil(outer_size / nl.tile_size.pmax)):
            o_start = o * nl.tile_size.pmax
            o_indices = o_start + nl.arange(nl.tile_size.pmax)[:, None, None]
            o_mask = o_indices < outer_size
            
            for s in nl.affine_range(sort_dim_size):
                s_indices = nl.full((nl.tile_size.pmax, 1, 1), s, dtype=nl.int32)
                
                for i in nl.affine_range(math.ceil(inner_size / nl.tile_size.fmax)):
                    i_start = i * nl.tile_size.fmax
                    i_indices = i_start + nl.arange(nl.tile_size.fmax)[None, None, :]
                    i_mask = i_indices < inner_size
                    
                    # Combined mask
                    mask = o_mask & i_mask
                    
                    # Create index for loading/storing
                    if dim == 0:
                        src_tile = nl.load(a_tensor[s, o_indices[..., 0], i_indices[..., 0]], mask=mask)
                        nl.store(result[s, o_indices[..., 0], i_indices[..., 0]], value=src_tile, mask=mask)
                    elif dim == 1:
                        src_tile = nl.load(a_tensor[o_indices[..., 0], s, i_indices[..., 0]], mask=mask)
                        nl.store(result[o_indices[..., 0], s, i_indices[..., 0]], value=src_tile, mask=mask)
                    else:  # dim == 2
                        src_tile = nl.load(a_tensor[o_indices[..., 0], i_indices[..., 0], s], mask=mask)
                        nl.store(result[o_indices[..., 0], i_indices[..., 0], s], value=src_tile, mask=mask)
        
        # Perform bubble sort for each "slice" along the sort dimension
        for o in nl.affine_range(outer_size):
            for i in nl.affine_range(inner_size):
                # Bubble sort algorithm
                for s1 in range(sort_dim_size):
                    for s2 in range(0, sort_dim_size - s1 - 1):
                        # Load the two elements to compare
                        if dim == 0:
                            val1 = nl.load(result[s2, o, i])
                            val2 = nl.load(result[s2 + 1, o, i])
                            
                            # Compare and swap if needed
                            if nl.greater(val1, val2):
                                nl.store(result[s2, o, i], value=val2)
                                nl.store(result[s2 + 1, o, i], value=val1)
                        elif dim == 1:
                            val1 = nl.load(result[o, s2, i])
                            val2 = nl.load(result[o, s2 + 1, i])
                            
                            # Compare and swap if needed
                            if nl.greater(val1, val2):
                                nl.store(result[o, s2, i], value=val2)
                                nl.store(result[o, s2 + 1, i], value=val1)
                        else:  # dim == 2
                            val1 = nl.load(result[o, i, s2])
                            val2 = nl.load(result[o, i, s2 + 1])
                            
                            # Compare and swap if needed
                            if nl.greater(val1, val2):
                                nl.store(result[o, i, s2], value=val2)
                                nl.store(result[o, i, s2 + 1], value=val1)
    
    return result