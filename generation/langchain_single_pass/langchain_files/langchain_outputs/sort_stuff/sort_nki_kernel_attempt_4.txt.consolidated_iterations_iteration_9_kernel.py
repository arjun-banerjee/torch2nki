from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # If dim is negative, convert to positive index
    if dim < 0:
        dim = ndim + dim
        
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # For 1D tensor case
    if ndim == 1:
        # Copy input to result first
        size = shape[0]
        
        # Number of tiles needed
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # Copy input to result
        for p in nl.affine_range(trip_count):
            idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[idx], mask=(idx < size))
            nl.store(result[idx], in_tile, mask=(idx < size))
        
        # Bubble sort
        for i in range(size):
            for j in range(size - 1):
                for p in nl.affine_range(trip_count):
                    idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load current elements
                    curr_idx = j
                    next_idx = j + 1
                    
                    # Create masks for the elements we need
                    curr_mask = nl.equal(idx, curr_idx)
                    next_mask = nl.equal(idx, next_idx)
                    
                    # Load current segment
                    segment = nl.load(result[idx], mask=(idx < size))
                    
                    # Get current and next values
                    curr_val = segment
                    next_val = segment
                    
                    # Extract values using masks
                    curr_val_extracted = nl.where(curr_mask, curr_val, nl.zeros_like(curr_val))
                    next_val_extracted = nl.where(next_mask, next_val, nl.zeros_like(next_val))
                    
                    # Compute sum to get single values (only one element is non-zero)
                    curr_sum = nl.sum(curr_val_extracted)
                    next_sum = nl.sum(next_val_extracted)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_sum, next_sum)
                    
                    # Create new values after potential swap
                    new_curr = nl.where(swap_needed & curr_mask, next_sum, segment)
                    result_segment = nl.where(swap_needed & next_mask, curr_sum, new_curr)
                    
                    # Store result
                    nl.store(result[idx], result_segment, mask=(idx < size))
        
    # For multi-dimensional tensors
    else:
        # Transpose tensor if needed to make the sort dimension the last dimension
        if dim != ndim - 1:
            # Create a permutation that moves dim to the end
            perm = list(range(ndim))
            perm.remove(dim)
            perm.append(dim)
            
            # Reshape input and result according to permutation
            # This is a placeholder for the actual transpose logic
            # In real implementation, we would need to handle this differently
            
            # For now, we'll assume dim is the last dimension
            pass
            
        # Get sizes for outer and inner dimensions
        outer_size = 1
        for i in range(ndim - 1):
            outer_size *= shape[i]
        
        inner_size = shape[ndim - 1]
        
        # Number of tiles needed for outer dimension
        outer_trip_count = math.ceil(outer_size / nl.tile_size.pmax)
        
        # Copy input to result first
        for p_outer in nl.affine_range(outer_trip_count):
            idx_outer = p_outer * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            idx_inner = nl.arange(inner_size)[None, :]
            
            # Load input data
            in_tile = nl.load(a_tensor[idx_outer, idx_inner], mask=(idx_outer < outer_size))
            
            # Store to result
            nl.store(result[idx_outer, idx_inner], in_tile, mask=(idx_outer < outer_size))
            
        # Sort each outer slice independently
        for i in range(inner_size):
            for j in range(inner_size - 1):
                for p_outer in nl.affine_range(outer_trip_count):
                    idx_outer = p_outer * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                    
                    # Load current segment
                    curr_j = nl.full((1, 1), j, dtype=nl.int32)
                    next_j = nl.full((1, 1), j + 1, dtype=nl.int32)
                    
                    curr_val = nl.load(result[idx_outer, curr_j], mask=(idx_outer < outer_size))
                    next_val = nl.load(result[idx_outer, next_j], mask=(idx_outer < outer_size))
                    
                    # Compare and swap if needed
                    swap_condition = nl.greater(curr_val, next_val)
                    
                    # Create new values after potential swap
                    new_curr = nl.where(swap_condition, next_val, curr_val)
                    new_next = nl.where(swap_condition, curr_val, next_val)
                    
                    # Store results
                    nl.store(result[idx_outer, curr_j], new_curr, mask=(idx_outer < outer_size))
                    nl.store(result[idx_outer, next_j], new_next, mask=(idx_outer < outer_size))
                    
    return result