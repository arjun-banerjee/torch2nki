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
    
    # Handle 1D tensor case separately for simplicity
    if ndim == 1:
        # Copy input to result first
        for i in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
            start_idx = i * nl.tile_size.pmax
            indices = start_idx + nl.arange(nl.tile_size.pmax)
            
            # Load data from input tensor
            in_tile = nl.load(a_tensor[indices], mask=(indices < shape[0]))
            
            # Store to result tensor
            nl.store(result[indices], value=in_tile, mask=(indices < shape[0]))
        
        # Bubble sort implementation for 1D
        for i in nl.affine_range(shape[0]):
            for j in nl.affine_range(shape[0] - 1):
                # Process in tiles due to architecture limitations
                for k in nl.affine_range(math.ceil((shape[0]-j-1) / nl.tile_size.pmax)):
                    start_idx = j + k * nl.tile_size.pmax
                    indices = start_idx + nl.arange(nl.tile_size.pmax)
                    next_indices = indices + 1
                    
                    # Load current elements
                    current = nl.load(result[indices], mask=(indices < shape[0]))
                    next_elem = nl.load(result[next_indices], mask=(next_indices < shape[0]))
                    
                    # Compare and swap if needed
                    swap_mask = (indices < shape[0]-1) & (current > next_elem)
                    if nl.any(swap_mask):
                        # Store swapped elements
                        nl.store(result[indices], value=nl.where(swap_mask, next_elem, current), mask=(indices < shape[0]-1))
                        nl.store(result[next_indices], value=nl.where(swap_mask, current, next_elem), mask=(next_indices < shape[0]))
        
        return result
    
    # For multi-dimensional tensors, we need to handle the sort dimension separately
    # Determine the shape of each "slice" to sort
    slice_size = shape[dim]
    
    # Calculate the number of slices to process
    num_slices = 1
    for i in range(ndim):
        if i != dim:
            num_slices *= shape[i]
    
    # First, copy the input tensor to the result
    for p in nl.affine_range(math.ceil(num_slices / nl.tile_size.pmax)):
        p_start = p * nl.tile_size.pmax
        p_indices = p_start + nl.arange(nl.tile_size.pmax)
        
        # Convert flat indices to multi-dimensional indices for each slice
        for d in nl.affine_range(slice_size):
            # Create multi-dimensional index tuple for loading and storing
            # This is a simplified approach that works for 2D tensors
            if dim == 0:
                # If sorting along first dimension, d varies in dim 0, p_indices in dim 1
                in_tile = nl.load(a_tensor[d, p_indices], mask=(p_indices < shape[1]))
                nl.store(result[d, p_indices], value=in_tile, mask=(p_indices < shape[1]))
            else:
                # If sorting along second dimension, p_indices varies in dim 0, d in dim 1
                in_tile = nl.load(a_tensor[p_indices, d], mask=(p_indices < shape[0]))
                nl.store(result[p_indices, d], value=in_tile, mask=(p_indices < shape[0]))
    
    # Now perform bubble sort on each slice
    # For simplicity, we implement for 2D tensors with clear dimension handling
    if ndim == 2:
        if dim == 0:
            # Sort along first dimension for each column
            for j in nl.affine_range(shape[1]):  # For each column
                for i in nl.affine_range(shape[0]):  # Bubble sort passes
                    for k in nl.affine_range(shape[0] - 1):  # Compare adjacent elements
                        # Load current and next elements
                        current = nl.load(result[k, j])
                        next_elem = nl.load(result[k+1, j])
                        
                        # Compare and swap if needed
                        if current > next_elem:
                            nl.store(result[k, j], value=next_elem)
                            nl.store(result[k+1, j], value=current)
        else:
            # Sort along second dimension for each row
            for i in nl.affine_range(shape[0]):  # For each row
                for j in nl.affine_range(shape[1]):  # Bubble sort passes
                    for k in nl.affine_range(shape[1] - 1):  # Compare adjacent elements
                        # Load current and next elements
                        current = nl.load(result[i, k])
                        next_elem = nl.load(result[i, k+1])
                        
                        # Compare and swap if needed
                        if current > next_elem:
                            nl.store(result[i, k], value=next_elem)
                            nl.store(result[i, k+1], value=current)
    
    return result