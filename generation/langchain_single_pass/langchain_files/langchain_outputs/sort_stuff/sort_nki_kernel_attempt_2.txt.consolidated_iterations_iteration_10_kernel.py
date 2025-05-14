from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = ndim + dim
        
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy the input tensor to result first
    if ndim == 1:
        sz = shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            nl.store(result[i_p], value=in_tile, mask=(i_p < sz))
    elif ndim == 2:
        sz_0, sz_1 = shape[0], shape[1]
        trip_count_0 = math.ceil(sz_0 / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count_0):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            i_f = nl.arange(sz_1)[None, :]
            in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_0))
            nl.store(result[i_p, i_f], value=in_tile, mask=(i_p < sz_0))
    
    # Sort the data along the specified dimension
    if ndim == 1:
        sz = shape[0]
        
        # Bubble sort algorithm - iterate over the array multiple times
        for i in nl.static_range(sz - 1):
            # In each iteration, compare adjacent elements and swap if needed
            trip_count = math.ceil((sz - 1) / nl.tile_size.pmax)
            
            for p in nl.affine_range(trip_count):
                # Calculate indices for this tile
                j_base = p * nl.tile_size.pmax
                j_indices = j_base + nl.arange(nl.tile_size.pmax)
                
                # Load current elements
                mask = (j_indices < sz - 1)
                curr_vals = nl.load(result[j_indices], mask=mask)
                
                # Load next elements
                next_indices = j_indices + 1
                next_mask = (next_indices < sz)
                next_vals = nl.load(result[next_indices], mask=next_mask)
                
                # Compare and swap if needed
                swap_mask = nl.greater(curr_vals, next_vals) & mask & next_mask
                
                # Store swapped values
                nl.store(result[j_indices], value=nl.where(swap_mask, next_vals, curr_vals), mask=mask)
                nl.store(result[next_indices], value=nl.where(swap_mask, curr_vals, next_vals), mask=next_mask)
    
    elif ndim == 2:
        sz_0, sz_1 = shape[0], shape[1]
        
        if dim == 0:
            # Sort along dimension 0
            for j in nl.static_range(sz_1):
                for i in nl.static_range(sz_0 - 1):
                    trip_count = math.ceil(sz_0 - 1 - i / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        # Calculate indices for this tile
                        k_base = p * nl.tile_size.pmax
                        k_indices = k_base + nl.arange(nl.tile_size.pmax)
                        
                        # Load current elements
                        mask = (k_indices < sz_0 - 1 - i)
                        curr_vals = nl.load(result[k_indices, j], mask=mask)
                        
                        # Load next elements
                        next_indices = k_indices + 1
                        next_mask = (next_indices < sz_0)
                        next_vals = nl.load(result[next_indices, j], mask=next_mask)
                        
                        # Compare and swap if needed
                        swap_mask = nl.greater(curr_vals, next_vals) & mask & next_mask
                        
                        # Store swapped values
                        nl.store(result[k_indices, j], value=nl.where(swap_mask, next_vals, curr_vals), mask=mask)
                        nl.store(result[next_indices, j], value=nl.where(swap_mask, curr_vals, next_vals), mask=next_mask)
        else:  # dim == 1
            # Sort along dimension 1
            for i in nl.static_range(sz_0):
                for j in nl.static_range(sz_1 - 1):
                    trip_count = math.ceil((sz_1 - 1 - j) / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        # Calculate indices for this tile
                        k_base = p * nl.tile_size.pmax
                        k_indices = k_base + nl.arange(nl.tile_size.pmax)
                        
                        # Load current elements
                        mask = (k_indices < sz_1 - 1 - j)
                        curr_vals = nl.load(result[i, k_indices], mask=mask)
                        
                        # Load next elements
                        next_indices = k_indices + 1
                        next_mask = (next_indices < sz_1)
                        next_vals = nl.load(result[i, next_indices], mask=next_mask)
                        
                        # Compare and swap if needed
                        swap_mask = nl.greater(curr_vals, next_vals) & mask & next_mask
                        
                        # Store swapped values
                        nl.store(result[i, k_indices], value=nl.where(swap_mask, next_vals, curr_vals), mask=mask)
                        nl.store(result[i, next_indices], value=nl.where(swap_mask, curr_vals, next_vals), mask=next_mask)
    
    return result