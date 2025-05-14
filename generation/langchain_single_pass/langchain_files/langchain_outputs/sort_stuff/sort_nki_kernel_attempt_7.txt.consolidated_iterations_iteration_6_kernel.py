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
    
    # First, copy the input tensor to the result tensor
    if ndim == 1:
        # For 1D tensor
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            start_idx = p * nl.tile_size.pmax
            i_p = start_idx + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            data_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=data_tile, mask=(i_p < size))
            
        # Bubble sort implementation for 1D tensor
        for _ in nl.static_range(size):
            for p in nl.affine_range(trip_count):
                start_idx = p * nl.tile_size.pmax
                i_p = start_idx + nl.arange(nl.tile_size.pmax)
                
                # Load current elements
                current_tile = nl.load(result[i_p], mask=(i_p < size))
                
                # For each element (except the last one), compare with next element
                for offset in nl.static_range(nl.tile_size.pmax - 1):
                    # Create indices for current and next elements
                    curr_idx = i_p + offset
                    next_idx = i_p + offset + 1
                    
                    # Only process valid indices
                    mask = (curr_idx < size - 1) & (next_idx < size)
                    
                    if nl.any(mask):
                        # Load current and next elements
                        curr = nl.load(result[curr_idx], mask=mask)
                        next_elem = nl.load(result[next_idx], mask=mask)
                        
                        # Create comparison mask for swapping
                        swap_mask = nl.greater(curr, next_elem) & mask
                        
                        # Perform swap using where operation
                        nl.store(result[curr_idx], value=nl.where(swap_mask, next_elem, curr), mask=swap_mask)
                        nl.store(result[next_idx], value=nl.where(swap_mask, curr, next_elem), mask=swap_mask)
    
    elif ndim == 2:
        # For 2D tensor
        dim_size = shape[dim]
        other_dim = 1 - dim  # The other dimension
        other_dim_size = shape[other_dim]
        
        # Calculate trip counts for tiling
        dim_trip_count = math.ceil(dim_size / nl.tile_size.pmax)
        other_dim_trip_count = math.ceil(other_dim_size / nl.tile_size.pmax)
        
        # Copy input to result
        if dim == 0:
            # Sorting along rows
            for p in nl.affine_range(dim_trip_count):
                for f in nl.affine_range(other_dim_trip_count):
                    p_start = p * nl.tile_size.pmax
                    f_start = f * nl.tile_size.pmax
                    
                    i_p = p_start + nl.arange(nl.tile_size.pmax)[:, None]
                    i_f = f_start + nl.arange(nl.tile_size.pmax)[None, :]
                    
                    mask = (i_p < dim_size) & (i_f < other_dim_size)
                    data_tile = nl.load(a_tensor[i_p, i_f], mask=mask)
                    nl.store(result[i_p, i_f], value=data_tile, mask=mask)
        else:
            # Sorting along columns
            for p in nl.affine_range(other_dim_trip_count):
                for f in nl.affine_range(dim_trip_count):
                    p_start = p * nl.tile_size.pmax
                    f_start = f * nl.tile_size.pmax
                    
                    i_p = p_start + nl.arange(nl.tile_size.pmax)[:, None]
                    i_f = f_start + nl.arange(nl.tile_size.pmax)[None, :]
                    
                    mask = (i_p < other_dim_size) & (i_f < dim_size)
                    data_tile = nl.load(a_tensor[i_p, i_f], mask=mask)
                    nl.store(result[i_p, i_f], value=data_tile, mask=mask)
        
        # Bubble sort implementation for 2D tensor
        for _ in nl.static_range(dim_size):
            if dim == 0:
                # Sort along rows
                for f in nl.affine_range(other_dim_trip_count):
                    f_start = f * nl.tile_size.pmax
                    i_f = f_start + nl.arange(nl.tile_size.pmax)
                    
                    for p in nl.affine_range(dim_size - 1):
                        # Compare element at position p with element at position p+1
                        curr = nl.load(result[p, i_f], mask=(i_f < other_dim_size))
                        next_elem = nl.load(result[p+1, i_f], mask=(i_f < other_dim_size))
                        
                        # Create swap mask
                        swap_mask = nl.greater(curr, next_elem) & (i_f < other_dim_size)
                        
                        # Perform conditional swap
                        nl.store(result[p, i_f], value=nl.where(swap_mask, next_elem, curr), mask=swap_mask)
                        nl.store(result[p+1, i_f], value=nl.where(swap_mask, curr, next_elem), mask=swap_mask)
            else:
                # Sort along columns
                for p in nl.affine_range(other_dim_trip_count):
                    p_start = p * nl.tile_size.pmax
                    i_p = p_start + nl.arange(nl.tile_size.pmax)
                    
                    for f in nl.affine_range(dim_size - 1):
                        # Compare element at position f with element at position f+1
                        curr = nl.load(result[i_p, f], mask=(i_p < other_dim_size))
                        next_elem = nl.load(result[i_p, f+1], mask=(i_p < other_dim_size))
                        
                        # Create swap mask
                        swap_mask = nl.greater(curr, next_elem) & (i_p < other_dim_size)
                        
                        # Perform conditional swap
                        nl.store(result[i_p, f], value=nl.where(swap_mask, next_elem, curr), mask=swap_mask)
                        nl.store(result[i_p, f+1], value=nl.where(swap_mask, curr, next_elem), mask=swap_mask)
    
    return result