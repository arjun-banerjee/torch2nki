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
    
    # First copy the input tensor to the result
    if ndim == 1:
        # Handle 1D tensor case
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for i in nl.affine_range(trip_count):
            start_idx = i * nl.tile_size.pmax
            indices = nl.arange(nl.tile_size.pmax)
            
            # Load input data
            in_tile = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < size))
            
            # Store to result
            nl.store(result[start_idx + indices], value=in_tile, mask=(start_idx + indices < size))
            
        # Bubble sort implementation on 1D tensor
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                trip_count = math.ceil((size - 1) / nl.tile_size.pmax)
                
                for k in nl.affine_range(trip_count):
                    start_idx = k * nl.tile_size.pmax
                    indices = nl.arange(nl.tile_size.pmax)
                    curr_indices = start_idx + indices
                    next_indices = start_idx + indices + 1
                    
                    # Load current and next elements
                    curr_vals = nl.load(result[curr_indices], mask=(curr_indices < size - 1))
                    next_vals = nl.load(result[next_indices], mask=(next_indices < size))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    temp = nl.copy(curr_vals)
                    curr_vals = nl.where(swap_needed, next_vals, curr_vals)
                    next_vals = nl.where(swap_needed, temp, next_vals)
                    
                    # Store back the updated values
                    nl.store(result[curr_indices], value=curr_vals, mask=(curr_indices < size - 1))
                    nl.store(result[next_indices], value=next_vals, mask=(next_indices < size))
    else:
        # Handle multi-dimensional tensor case
        # Determine sizes and strides for processing
        sort_dim_size = shape[dim]
        
        # Calculate total elements and elements per slice
        total_elements = 1
        for i in range(ndim):
            total_elements *= shape[i]
        
        elements_per_slice = total_elements // sort_dim_size
        
        # First, copy input to result
        trip_count = math.ceil(total_elements / nl.tile_size.pmax)
        
        # Copy input to result using flattened indexing
        for i in nl.affine_range(trip_count):
            start_idx = i * nl.tile_size.pmax
            indices = nl.arange(nl.tile_size.pmax)
            flat_indices = start_idx + indices
            
            # Create multi-dimensional indices
            multi_indices = []
            remaining = flat_indices
            for d in range(ndim-1, -1, -1):
                if d == 0:
                    idx = remaining
                else:
                    idx = remaining % shape[d]
                    remaining = remaining // shape[d]
                multi_indices.insert(0, idx)
            
            # Load input data - we'll do this by processing each slice separately
            in_tile = nl.load(a_tensor.reshape(-1)[flat_indices], mask=(flat_indices < total_elements))
            
            # Store to result
            nl.store(result.reshape(-1)[flat_indices], value=in_tile, mask=(flat_indices < total_elements))
        
        # Process each slice separately
        for slice_idx in nl.affine_range(elements_per_slice):
            # Bubble sort implementation for this slice
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    # Calculate base offset for this slice
                    base_offset = slice_idx
                    
                    # Load current and next elements in the sort dimension
                    curr_idx = j
                    next_idx = j + 1
                    
                    # Calculate flat indices for current and next elements
                    curr_flat_idx = base_offset * sort_dim_size + curr_idx
                    next_flat_idx = base_offset * sort_dim_size + next_idx
                    
                    # Load values
                    curr_val = nl.load(result.reshape(-1)[curr_flat_idx])
                    next_val = nl.load(result.reshape(-1)[next_flat_idx])
                    
                    # Compare and swap if needed
                    if nl.greater(curr_val, next_val).item():
                        # Swap values
                        nl.store(result.reshape(-1)[curr_flat_idx], value=next_val)
                        nl.store(result.reshape(-1)[next_flat_idx], value=curr_val)
    
    return result