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
    
    # Copy the input tensor to result first
    if ndim == 1:
        # 1D tensor - handle as a special case
        size = shape[0]
        # Copy input to result
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = i * nl.tile_size.pmax
            indices = nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < size))
            nl.store(result[start_idx + indices], value=in_tile, mask=(start_idx + indices < size))
        
        # Bubble sort implementation for 1D
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements
                idx_j = nl.full((1, 1), j, dtype=nl.int32)
                idx_j1 = nl.full((1, 1), j + 1, dtype=nl.int32)
                
                val_j = nl.load(result[idx_j])
                val_j1 = nl.load(result[idx_j1])
                
                # Compare and swap if needed
                is_greater = nl.greater(val_j, val_j1)
                
                # Create temporary values for swap
                temp_j = nl.copy(val_j1)
                temp_j1 = nl.copy(val_j)
                
                # Create final values based on comparison
                final_j = nl.where(is_greater, temp_j, val_j)
                final_j1 = nl.where(is_greater, temp_j1, val_j1)
                
                # Store back
                nl.store(result[idx_j], value=final_j)
                nl.store(result[idx_j1], value=final_j1)
    else:
        # Handle multi-dimensional tensors
        # Determine the size of the dimension to sort
        sort_dim_size = shape[dim]
        
        # Calculate total elements before and after the sort dimension
        # to handle the tensor as a 3D tensor: [before_dims, sort_dim, after_dims]
        before_dims_size = 1
        for i in range(dim):
            before_dims_size *= shape[i]
            
        after_dims_size = 1
        for i in range(dim + 1, ndim):
            after_dims_size *= shape[i]
            
        # Copy input to result
        for b in nl.affine_range(math.ceil(before_dims_size / nl.tile_size.pmax)):
            b_start = b * nl.tile_size.pmax
            b_indices = nl.arange(nl.tile_size.pmax)[:, None, None]
            
            for s in nl.affine_range(sort_dim_size):
                s_idx = nl.full((1, 1, 1), s, dtype=nl.int32)
                
                for a in nl.affine_range(math.ceil(after_dims_size / nl.tile_size.pmax)):
                    a_start = a * nl.tile_size.pmax
                    a_indices = nl.arange(nl.tile_size.pmax)[None, None, :]
                    
                    # Load and store with proper masking
                    in_tile = nl.load(a_tensor[b_start + b_indices, s_idx, a_start + a_indices], 
                                      mask=((b_start + b_indices < before_dims_size) & 
                                           (a_start + a_indices < after_dims_size)))
                    
                    nl.store(result[b_start + b_indices, s_idx, a_start + a_indices], 
                             value=in_tile,
                             mask=((b_start + b_indices < before_dims_size) & 
                                  (a_start + a_indices < after_dims_size)))
        
        # Bubble sort implementation for multi-dimensional tensor
        # Sort each "slice" along the sort dimension
        for b in nl.affine_range(before_dims_size):
            for a in nl.affine_range(after_dims_size):
                # Bubble sort this slice
                for i in nl.affine_range(sort_dim_size):
                    for j in nl.affine_range(sort_dim_size - 1):
                        # Load adjacent elements
                        b_idx = nl.full((1, 1), b, dtype=nl.int32)
                        j_idx = nl.full((1, 1), j, dtype=nl.int32)
                        j1_idx = nl.full((1, 1), j + 1, dtype=nl.int32)
                        a_idx = nl.full((1, 1), a, dtype=nl.int32)
                        
                        val_j = nl.load(result[b_idx, j_idx, a_idx])
                        val_j1 = nl.load(result[b_idx, j1_idx, a_idx])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(val_j, val_j1)
                        
                        # Create temporary values for swap
                        temp_j = nl.copy(val_j1)
                        temp_j1 = nl.copy(val_j)
                        
                        # Create final values based on comparison
                        final_j = nl.where(is_greater, temp_j, val_j)
                        final_j1 = nl.where(is_greater, temp_j1, val_j1)
                        
                        # Store back
                        nl.store(result[b_idx, j_idx, a_idx], value=final_j)
                        nl.store(result[b_idx, j1_idx, a_idx], value=final_j1)
    
    return result