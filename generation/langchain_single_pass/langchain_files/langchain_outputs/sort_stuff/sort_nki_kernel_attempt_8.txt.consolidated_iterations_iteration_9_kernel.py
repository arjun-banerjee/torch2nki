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
    
    # First, copy input to result
    if ndims == 1:
        # For 1D tensor, simple case
        size = tensor_shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process the tensor in tiles
        for p in nl.affine_range(math.ceil(size / max_tile_size)):
            # Calculate start and end indices for this tile
            start_idx = p * max_tile_size
            # Create index array for current tile
            i_p = nl.arange(max_tile_size)
            
            # Load input data with mask to handle boundary
            tile_data = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < size))
            
            # Store to result
            nl.store(result[start_idx + i_p], tile_data, mask=(start_idx + i_p < size))
        
        # Bubble sort implementation for 1D tensor
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Process the tensor in tiles
                for p in nl.affine_range(math.ceil((size - 1) / max_tile_size)):
                    tile_start = p * max_tile_size
                    i_p = nl.arange(max_tile_size)
                    
                    # Generate current indices
                    curr_idx = tile_start + i_p
                    next_idx = tile_start + i_p + 1
                    
                    # Load current and next elements with mask
                    curr_mask = (curr_idx < size - 1)
                    curr_vals = nl.load(result[curr_idx], mask=curr_mask)
                    next_vals = nl.load(result[next_idx], mask=curr_mask)
                    
                    # Compare and swap if needed
                    should_swap = nl.greater(curr_vals, next_vals)
                    new_curr = nl.where(should_swap, next_vals, curr_vals)
                    new_next = nl.where(should_swap, curr_vals, next_vals)
                    
                    # Store back the results
                    nl.store(result[curr_idx], new_curr, mask=curr_mask)
                    nl.store(result[next_idx], new_next, mask=curr_mask)
    
    else:
        # For multi-dimensional tensors, we need to sort along the specified dimension
        # First copy the input to result
        # Calculate total number of elements
        total_elements = 1
        for i in range(ndims):
            total_elements *= tensor_shape[i]
        
        # Calculate max elements per tile
        max_elements_per_tile = nl.tile_size.pmax
        
        # Process the tensor in tiles
        for p in nl.affine_range(math.ceil(total_elements / max_elements_per_tile)):
            tile_start = p * max_elements_per_tile
            
            # Calculate multi-dimensional indices from linear index
            indices = []
            remaining = nl.arange(max_elements_per_tile)
            
            # Store the input data to result tensor
            flat_input = nl.load(a_tensor.reshape(-1)[tile_start + remaining], 
                                mask=(tile_start + remaining < total_elements))
            nl.store(result.reshape(-1)[tile_start + remaining], flat_input, 
                    mask=(tile_start + remaining < total_elements))
        
        # Now sort each slice along the specified dimension
        # Calculate number of slices to process
        slice_size = tensor_shape[dim]
        num_slices = total_elements // slice_size
        
        # For each slice
        for s in nl.affine_range(num_slices):
            # Bubble sort this slice
            for i in nl.affine_range(slice_size):
                for j in nl.affine_range(slice_size - 1):
                    # Calculate linear indices for current and next elements
                    base_idx = s * slice_size
                    curr_idx = base_idx + j
                    next_idx = base_idx + j + 1
                    
                    # Load current and next elements
                    curr_val = nl.load(result.reshape(-1)[curr_idx])
                    next_val = nl.load(result.reshape(-1)[next_idx])
                    
                    # Compare and swap if needed
                    if_greater = nl.greater(curr_val, next_val)
                    new_curr = nl.where(if_greater, next_val, curr_val)
                    new_next = nl.where(if_greater, curr_val, next_val)
                    
                    # Store back the results
                    nl.store(result.reshape(-1)[curr_idx], new_curr)
                    nl.store(result.reshape(-1)[next_idx], new_next)
    
    return result