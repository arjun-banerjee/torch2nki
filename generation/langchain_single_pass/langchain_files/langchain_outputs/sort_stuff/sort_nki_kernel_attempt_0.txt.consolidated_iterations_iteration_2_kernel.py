from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to the result
    # We'll sort in-place in the result tensor
    
    # Calculate the number of elements in dimensions before and after the sort dimension
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]
    
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    
    sort_size = shape[dim]
    
    # Process the tensor in tiles for outer dimensions
    outer_tile_size = min(128, outer_size)
    for outer_offset in nl.affine_range(math.ceil(outer_size / outer_tile_size)):
        # Process inner dimension tiles
        inner_tile_size = min(128, inner_size)
        for inner_offset in nl.affine_range(math.ceil(inner_size / inner_tile_size)):
            # Load the slice to be sorted into on-chip memory
            sort_buffer = nl.zeros((outer_tile_size, sort_size, inner_tile_size), 
                                  dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Generate indices for loading
            i_o = nl.arange(outer_tile_size)[:, None, None]
            i_s = nl.arange(sort_size)[None, :, None]
            i_i = nl.arange(inner_tile_size)[None, None, :]
            
            # Calculate actual indices
            actual_o = outer_offset * outer_tile_size + i_o
            actual_i = inner_offset * inner_tile_size + i_i
            
            # Load data with masking to handle boundaries
            o_mask = actual_o < outer_size
            i_mask = actual_i < inner_size
            
            # Create flattened indices for loading
            indices = []
            idx = 0
            for d in range(ndim):
                if d < dim:
                    # Dimensions before sort dimension
                    div = 1
                    for dd in range(d+1, dim):
                        div *= shape[dd]
                    dim_idx = (actual_o // div) % shape[d]
                    indices.append(dim_idx)
                elif d == dim:
                    # Sort dimension
                    indices.append(i_s)
                else:
                    # Dimensions after sort dimension
                    div = 1
                    for dd in range(d+1, ndim):
                        div *= shape[dd]
                    dim_idx = (actual_i // div) % shape[d]
                    indices.append(dim_idx)
            
            # Load data
            load_mask = o_mask & i_mask
            sort_buffer = nl.load(a_tensor[tuple(indices)], mask=load_mask)
            
            # Bubble sort algorithm for each outer/inner combination
            for i in nl.affine_range(sort_size - 1):
                for j in nl.affine_range(sort_size - i - 1):
                    # Compare adjacent elements
                    curr = sort_buffer[:, j, :]
                    next_val = sort_buffer[:, j+1, :]
                    
                    # Create mask for swapping elements
                    swap_mask = nl.greater(curr, next_val)
                    
                    # Swap elements where needed using conditional operations
                    temp = nl.zeros_like(curr)
                    
                    # Where swap_mask is true, temp = next_val, else temp = curr
                    temp = nl.where(swap_mask, next_val, curr)
                    
                    # Where swap_mask is true, next_val = curr
                    next_val_updated = nl.where(swap_mask, curr, next_val)
                    
                    # Update the sort buffer
                    sort_buffer[:, j, :] = temp
                    sort_buffer[:, j+1, :] = next_val_updated
            
            # Store the sorted data back to result tensor
            nl.store(result[tuple(indices)], value=sort_buffer, mask=load_mask)
    
    return result