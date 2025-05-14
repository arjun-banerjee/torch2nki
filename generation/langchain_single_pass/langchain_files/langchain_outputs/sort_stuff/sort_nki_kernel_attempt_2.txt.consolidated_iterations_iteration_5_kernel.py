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
    
    # First copy input tensor to result
    if ndim == 1:
        # For 1D tensor, we process directly
        sz = shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data with masking for boundary
            input_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            
            # Store to result tensor
            nl.store(result[i_p], value=input_tile, mask=(i_p < sz))
    else:
        # For multi-dimensional tensor, we need to handle the specified dimension
        # First copy the entire tensor
        # Calculate total elements
        total_elements = 1
        for i in range(ndim):
            total_elements *= shape[i]
        
        # Process in tiles to handle large tensors
        max_tile = nl.tile_size.pmax
        trip_count = math.ceil(total_elements / max_tile)
        
        for p in nl.affine_range(trip_count):
            # Linear index for flat copy
            i_linear = p * max_tile + nl.arange(max_tile)
            
            # Create mask for valid elements
            mask = (i_linear < total_elements)
            
            # Load and store with masking
            # We're using flat indexing for the copy operation
            input_tile = nl.load(a_tensor.reshape(-1)[i_linear], mask=mask)
            nl.store(result.reshape(-1)[i_linear], value=input_tile, mask=mask)
    
    # Now perform the sorting along the specified dimension
    dim_size = shape[dim]
    
    # For each slice along the sort dimension
    # We need to implement bubble sort
    for _ in nl.affine_range(dim_size - 1):
        for j in nl.affine_range(dim_size - 1):
            # We need to compare adjacent elements along the sort dimension
            # and swap if necessary
            
            if ndim == 1:
                # For 1D tensor, we process in tiles
                trip_count = math.ceil(dim_size / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    # Generate tensor indices for the current tile
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load current elements
                    mask = (i_p < dim_size - 1) & (i_p == j)
                    current = nl.load(result[i_p], mask=mask)
                    
                    # Load next elements
                    next_mask = (i_p < dim_size - 1) & (i_p == j)
                    next_val = nl.load(result[i_p + 1], mask=next_mask)
                    
                    # Compare and swap if necessary
                    swap_mask = (i_p < dim_size - 1) & (i_p == j) & nl.greater(current, next_val)
                    
                    # Store swapped values
                    nl.store(result[i_p], value=next_val, mask=swap_mask)
                    nl.store(result[i_p + 1], value=current, mask=swap_mask)
            else:
                # For multi-dimensional tensor, we need to handle the specified dimension
                # This is more complex as we need to iterate through all slices
                
                # Calculate the total number of slices perpendicular to the sort dimension
                total_slices = 1
                for i in range(ndim):
                    if i != dim:
                        total_slices *= shape[i]
                
                # Process slices in tiles
                max_slice_tile = nl.tile_size.pmax
                slice_trip_count = math.ceil(total_slices / max_slice_tile)
                
                for s in nl.affine_range(slice_trip_count):
                    # Linear index for slice
                    slice_linear = s * max_slice_tile + nl.arange(max_slice_tile)
                    
                    # Create mask for valid slices
                    slice_mask = (slice_linear < total_slices) & (j < dim_size - 1)
                    
                    # Now we need to map the linear slice index back to multi-dimensional indices
                    # This is complex and would require tensor reshaping and indexing
                    # For simplicity, we'll use flat indexing and calculate offsets
                    
                    # Calculate offsets for current and next elements along the sort dimension
                    stride = 1
                    for i in range(ndim - 1, dim, -1):
                        stride *= shape[i]
                    
                    # Load current elements for this slice and position
                    offset = j * stride
                    flat_indices = slice_linear * stride + offset
                    current = nl.load(result.reshape(-1)[flat_indices], mask=slice_mask)
                    
                    # Load next elements
                    next_offset = (j + 1) * stride
                    next_flat_indices = slice_linear * stride + next_offset
                    next_val = nl.load(result.reshape(-1)[next_flat_indices], mask=slice_mask)
                    
                    # Compare and swap if necessary
                    swap_mask = slice_mask & nl.greater(current, next_val)
                    
                    # Store swapped values
                    nl.store(result.reshape(-1)[flat_indices], value=next_val, mask=swap_mask)
                    nl.store(result.reshape(-1)[next_flat_indices], value=current, mask=swap_mask)
    
    return result