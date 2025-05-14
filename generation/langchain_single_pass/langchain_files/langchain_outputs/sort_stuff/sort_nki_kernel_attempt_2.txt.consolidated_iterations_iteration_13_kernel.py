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
    
    # Copy input to result first
    if ndim == 1:
        # For 1D tensor, copy the entire tensor
        size = shape[0]
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start = i * nl.tile_size.pmax
            # Create indices for current tile
            idx = nl.arange(nl.tile_size.pmax) + start
            # Load data with masking for boundary
            data = nl.load(a_tensor[idx], mask=(idx < size))
            # Store data to result
            nl.store(result[idx], value=data, mask=(idx < size))
    else:
        # For multidimensional tensors
        # Calculate sizes before and after the sort dimension
        dim_size = shape[dim]
        
        # Calculate the product of dimensions before sort dimension
        pre_size = 1
        for i in range(dim):
            pre_size *= shape[i]
        
        # Calculate the product of dimensions after sort dimension
        post_size = 1
        for i in range(dim + 1, ndim):
            post_size *= shape[i]
        
        # Process in tiles for pre_size
        for pre_idx in nl.affine_range(math.ceil(pre_size / nl.tile_size.pmax)):
            pre_start = pre_idx * nl.tile_size.pmax
            pre_indices = nl.arange(nl.tile_size.pmax) + pre_start
            
            # Process each dimension element
            for dim_idx in nl.affine_range(dim_size):
                # Process in tiles for post_size
                for post_idx in nl.affine_range(math.ceil(post_size / nl.tile_size.pmax)):
                    post_start = post_idx * nl.tile_size.pmax
                    post_indices = nl.arange(nl.tile_size.pmax) + post_start
                    
                    # Create input indices based on dimensions
                    if ndim == 2:
                        if dim == 0:
                            # [sort_dim, other_dim]
                            input_indices = (dim_idx, post_indices)
                            mask = (post_indices < post_size)
                        else:  # dim == 1
                            # [other_dim, sort_dim]
                            input_indices = (pre_indices, dim_idx)
                            mask = (pre_indices < pre_size)
                    elif ndim == 3:
                        if dim == 0:
                            # Handle 3D with dim 0 as sort dimension
                            d1_size = shape[1]
                            d2_size = shape[2]
                            d1_idx = (post_indices // d2_size) % d1_size
                            d2_idx = post_indices % d2_size
                            input_indices = (dim_idx, d1_idx, d2_idx)
                            mask = (post_indices < post_size)
                        elif dim == 1:
                            # Handle 3D with dim 1 as sort dimension
                            d2_size = shape[2]
                            d0_idx = pre_indices
                            d2_idx = post_indices
                            input_indices = (d0_idx, dim_idx, d2_idx)
                            mask = (pre_indices < pre_size) & (post_indices < post_size)
                        else:  # dim == 2
                            # Handle 3D with dim 2 as sort dimension
                            d0_size = shape[0]
                            d1_size = shape[1]
                            d0_idx = pre_indices // d1_size
                            d1_idx = pre_indices % d1_size
                            input_indices = (d0_idx, d1_idx, dim_idx)
                            mask = (pre_indices < pre_size)
                    
                    # Load data from input tensor with appropriate masking
                    data = nl.load(a_tensor[input_indices], mask=mask)
                    # Store data to result tensor
                    nl.store(result[input_indices], value=data, mask=mask)
    
    # Perform the sort operation along the specified dimension
    # We'll use bubble sort for simplicity
    if ndim == 1:
        # For 1D tensor, sort the entire tensor
        size = shape[0]
        # Bubble sort implementation
        for i in nl.affine_range(size - 1):
            for j in nl.affine_range(size - 1 - i):
                # Load adjacent elements
                idx_j = j
                idx_j1 = j + 1
                
                # Process in tiles to handle large arrays
                val_j = nl.load(result[idx_j])
                val_j1 = nl.load(result[idx_j1])
                
                # Compare and swap if necessary
                swap_needed = nl.greater(val_j, val_j1)
                
                # Perform conditional swap
                if swap_needed:
                    nl.store(result[idx_j], value=val_j1)
                    nl.store(result[idx_j1], value=val_j)
    else:
        # For multidimensional tensors, sort along the specified dimension
        dim_size = shape[dim]
        
        # Calculate the product of dimensions before sort dimension
        pre_size = 1
        for i in range(dim):
            pre_size *= shape[i]
        
        # Calculate the product of dimensions after sort dimension
        post_size = 1
        for i in range(dim + 1, ndim):
            post_size *= shape[i]
        
        # Process each slice separately
        for pre_idx in nl.affine_range(pre_size):
            for post_idx in nl.affine_range(post_size):
                # Bubble sort implementation for this slice
                for i in nl.affine_range(dim_size - 1):
                    for j in nl.affine_range(dim_size - 1 - i):
                        # Create indices for adjacent elements
                        if ndim == 2:
                            if dim == 0:
                                # [sort_dim, other_dim]
                                idx_j = (j, post_idx)
                                idx_j1 = (j + 1, post_idx)
                            else:  # dim == 1
                                # [other_dim, sort_dim]
                                idx_j = (pre_idx, j)
                                idx_j1 = (pre_idx, j + 1)
                        elif ndim == 3:
                            if dim == 0:
                                # [sort_dim, d1, d2]
                                d1_size = shape[1]
                                d1_idx = post_idx // shape[2]
                                d2_idx = post_idx % shape[2]
                                idx_j = (j, d1_idx, d2_idx)
                                idx_j1 = (j + 1, d1_idx, d2_idx)
                            elif dim == 1:
                                # [d0, sort_dim, d2]
                                d2_idx = post_idx
                                idx_j = (pre_idx, j, d2_idx)
                                idx_j1 = (pre_idx, j + 1, d2_idx)
                            else:  # dim == 2
                                # [d0, d1, sort_dim]
                                d0_size = shape[0]
                                d0_idx = pre_idx // shape[1]
                                d1_idx = pre_idx % shape[1]
                                idx_j = (d0_idx, d1_idx, j)
                                idx_j1 = (d0_idx, d1_idx, j + 1)
                        
                        # Load adjacent elements
                        val_j = nl.load(result[idx_j])
                        val_j1 = nl.load(result[idx_j1])
                        
                        # Compare and swap if necessary
                        swap_needed = nl.greater(val_j, val_j1)
                        
                        # Perform conditional swap
                        if swap_needed:
                            nl.store(result[idx_j], value=val_j1)
                            nl.store(result[idx_j1], value=val_j)
    
    return result