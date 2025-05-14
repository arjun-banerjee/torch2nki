from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and normalize dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Convert negative dim to positive
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy input to result
    if ndim == 1:
        # For 1D case
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process in tiles to handle large tensors
        trip_count = (size + max_tile_size - 1) // max_tile_size
        
        for p in nl.affine_range(trip_count):
            # Calculate indices for current tile
            start_idx = p * max_tile_size
            
            # Create indices array for loading/storing
            idx = nl.arange(max_tile_size)
            
            # Load data with masking for boundary
            tile = nl.load(a_tensor[start_idx + idx], mask=(start_idx + idx < size))
            
            # Store to result
            nl.store(result[start_idx + idx], tile, mask=(start_idx + idx < size))
            
        # Now perform bubble sort on the entire array
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Process in tiles due to hardware limitations
                for p in nl.affine_range(trip_count):
                    start_idx = p * max_tile_size
                    
                    # Load adjacent elements with masking for boundary
                    idx = nl.arange(max_tile_size)
                    curr_idx = start_idx + idx
                    next_idx = start_idx + idx + 1
                    
                    # Only process elements where next_idx is valid (< size)
                    valid_mask = (curr_idx < size - 1) & (next_idx < size)
                    
                    curr_val = nl.load(result[curr_idx], mask=valid_mask)
                    next_val = nl.load(result[next_idx], mask=valid_mask)
                    
                    # Compare and swap if needed
                    swap_mask = nl.greater(curr_val, next_val) & valid_mask
                    
                    # Store swapped values
                    nl.store(result[curr_idx], next_val, mask=swap_mask)
                    nl.store(result[next_idx], curr_val, mask=swap_mask)
                    
    else:
        # For N-dimensional tensors, we sort along the specified dimension
        # First copy the input to result
        # Calculate total number of elements
        total_elements = 1
        for i in range(ndim):
            total_elements = total_elements * shape[i]
            
        max_tile_size = nl.tile_size.pmax
        trip_count = (total_elements + max_tile_size - 1) // max_tile_size
        
        # Copy input to result in tiles
        for p in nl.affine_range(trip_count):
            start_idx = p * max_tile_size
            idx = nl.arange(max_tile_size)
            flat_idx = start_idx + idx
            
            # Create a mask for valid indices
            valid_mask = (flat_idx < total_elements)
            
            # Calculate multi-dimensional indices
            # This is a simplified approach - we'll need to implement
            # proper multi-dimensional sorting below
            
            # First copy all data
            elements = nl.load(a_tensor.reshape(-1)[flat_idx], mask=valid_mask)
            nl.store(result.reshape(-1)[flat_idx], elements, mask=valid_mask)
        
        # Sort along the specified dimension
        dim_size = shape[dim]
        
        # Calculate stride for the sort dimension
        stride = 1
        for i in range(dim + 1, ndim):
            stride = stride * shape[i]
            
        # Calculate number of slices to sort
        num_slices = total_elements // dim_size
        
        # For each slice
        for s in nl.affine_range(num_slices):
            # Calculate base index for this slice
            base_idx = s * dim_size
            
            # Bubble sort for this slice
            for i in nl.affine_range(dim_size):
                for j in nl.affine_range(dim_size - 1):
                    # Calculate indices for current and next elements
                    curr_idx = base_idx + j
                    next_idx = base_idx + j + 1
                    
                    # Only process if next_idx is in the same slice
                    if next_idx < base_idx + dim_size:
                        # Load values
                        curr_val = nl.load(result.reshape(-1)[curr_idx])
                        next_val = nl.load(result.reshape(-1)[next_idx])
                        
                        # Compare and swap if needed
                        if nl.greater(curr_val, next_val).item():
                            nl.store(result.reshape(-1)[curr_idx], next_val)
                            nl.store(result.reshape(-1)[next_idx], curr_val)
    
    return result