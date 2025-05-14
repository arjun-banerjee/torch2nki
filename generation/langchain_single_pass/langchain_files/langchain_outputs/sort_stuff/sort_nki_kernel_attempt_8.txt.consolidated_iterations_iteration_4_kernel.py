from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension indexing
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Get tensor shape
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    sort_dim_size = tensor_shape[dim]
    
    # First, copy input to result
    if ndims == 1:
        # For 1D tensor, use tiling to handle large tensors
        trip_count = math.ceil(tensor_shape[0] / nl.tile_size.pmax)
        for p in nl.affine_range(trip_count):
            start_idx = p * nl.tile_size.pmax
            i_p = start_idx + nl.arange(nl.tile_size.pmax)
            
            # Load, then store to result
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < tensor_shape[0]))
            nl.store(result[i_p], value=in_tile, mask=(i_p < tensor_shape[0]))
    else:
        # For multi-dimensional tensors, we need to handle the case differently
        # For simplicity, we currently only support 2D tensors with dim=0 or dim=1
        if ndims == 2:
            sz_p, sz_f = tensor_shape
            trip_count = math.ceil(sz_p / nl.tile_size.pmax)
            
            i_f = nl.arange(sz_f)[None, :]
            
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                
                # Load, then store to result
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
                nl.store(result[i_p, i_f], value=in_tile, mask=(i_p < sz_p))
    
    # Now perform bubble sort on the result tensor
    if ndims == 1:
        # For 1D tensor, use bubble sort directly
        for _ in nl.affine_range(sort_dim_size):
            for j in nl.affine_range(sort_dim_size - 1):
                # Process in tiles to handle large tensors
                trip_count = math.ceil((sort_dim_size - 1) / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    start_idx = p * nl.tile_size.pmax
                    i_p = start_idx + nl.arange(nl.tile_size.pmax)
                    
                    # Load current and next elements
                    curr_vals = nl.load(result[i_p], mask=(i_p < sort_dim_size - 1))
                    next_vals = nl.load(result[i_p + 1], mask=(i_p < sort_dim_size - 1))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    new_curr = nl.where(swap_needed, next_vals, curr_vals)
                    new_next = nl.where(swap_needed, curr_vals, next_vals)
                    
                    # Store the updated values
                    nl.store(result[i_p], value=new_curr, mask=(i_p < sort_dim_size - 1))
                    nl.store(result[i_p + 1], value=new_next, mask=(i_p < sort_dim_size - 1))
    
    elif ndims == 2:
        if dim == 0:
            # Sort along dimension 0 (rows)
            for i_f in nl.affine_range(sz_f):
                for _ in nl.affine_range(sz_p):
                    for j in nl.affine_range(sz_p - 1):
                        trip_count = math.ceil((sz_p - 1) / nl.tile_size.pmax)
                        
                        for p in nl.affine_range(trip_count):
                            start_idx = p * nl.tile_size.pmax
                            i_p = start_idx + nl.arange(nl.tile_size.pmax)
                            
                            # Load current and next elements
                            curr_vals = nl.load(result[i_p, i_f], mask=(i_p < sz_p - 1))
                            next_vals = nl.load(result[i_p + 1, i_f], mask=(i_p < sz_p - 1))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store the updated values
                            nl.store(result[i_p, i_f], value=new_curr, mask=(i_p < sz_p - 1))
                            nl.store(result[i_p + 1, i_f], value=new_next, mask=(i_p < sz_p - 1))
        else:  # dim == 1
            # Sort along dimension 1 (columns)
            for i_p in nl.affine_range(sz_p):
                for _ in nl.affine_range(sz_f):
                    for j in nl.affine_range(sz_f - 1):
                        trip_count = math.ceil((sz_f - 1) / nl.tile_size.pmax)
                        
                        for p in nl.affine_range(trip_count):
                            start_idx = p * nl.tile_size.pmax
                            i_f = start_idx + nl.arange(nl.tile_size.pmax)
                            
                            # Load current and next elements
                            curr_vals = nl.load(result[i_p, i_f], mask=(i_f < sz_f - 1))
                            next_vals = nl.load(result[i_p, i_f + 1], mask=(i_f < sz_f - 1))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store the updated values
                            nl.store(result[i_p, i_f], value=new_curr, mask=(i_f < sz_f - 1))
                            nl.store(result[i_p, i_f + 1], value=new_next, mask=(i_f < sz_f - 1))
    
    return result