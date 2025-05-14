from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # If dim is negative, convert to positive index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy input to result
    if ndim == 1:
        # Handle 1D tensor
        sz_p = shape[0]
        trip_count = math.ceil(sz_p / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < sz_p))
            
            # Store in result
            nl.store(result[i_p], value=x_tile, mask=(i_p < sz_p))
            
        # Bubble sort for 1D tensor
        for i in range(sz_p):
            for j in nl.affine_range(trip_count):
                j_start = j * nl.tile_size.pmax
                j_end = min((j+1) * nl.tile_size.pmax, sz_p-i-1)
                
                if j_start < sz_p-i-1:
                    # Load current segment
                    j_indices = j_start + nl.arange(nl.tile_size.pmax)
                    current = nl.load(result[j_indices], mask=(j_indices < j_end))
                    
                    # Load next elements
                    next_indices = j_indices + 1
                    next_elem = nl.load(result[next_indices], mask=(j_indices < j_end))
                    
                    # Compare and swap
                    swap_mask = nl.greater(current, next_elem) & (j_indices < j_end)
                    
                    # Where swap is needed, swap values
                    new_current = nl.where(swap_mask, next_elem, current)
                    new_next = nl.where(swap_mask, current, next_elem)
                    
                    # Store back
                    nl.store(result[j_indices], value=new_current, mask=(j_indices < j_end))
                    nl.store(result[next_indices], value=new_next, mask=(j_indices < j_end))
                    
    else:
        # Handle multi-dimensional tensor
        # Compute sizes before and after the dimension to sort
        pre_dim_size = 1
        for i in range(dim):
            pre_dim_size = pre_dim_size * shape[i]
            
        dim_size = shape[dim]
        
        post_dim_size = 1
        for i in range(dim+1, ndim):
            post_dim_size = post_dim_size * shape[i]
            
        # First copy the input to result
        for pre in nl.affine_range(pre_dim_size):
            for post in nl.affine_range(math.ceil(post_dim_size / nl.tile_size.pmax)):
                post_start = post * nl.tile_size.pmax
                post_indices = post_start + nl.arange(nl.tile_size.pmax)[:, None]
                
                for d in nl.affine_range(math.ceil(dim_size / nl.tile_size.pmax)):
                    d_start = d * nl.tile_size.pmax
                    d_indices = d_start + nl.arange(nl.tile_size.pmax)[None, :]
                    
                    # Create index tensors for each dimension
                    if dim == 0:
                        # For dim=0, the indices are [d_indices, post_indices]
                        x_tile = nl.load(a_tensor[d_indices, post_indices], 
                                        mask=(d_indices < dim_size) & (post_indices < post_dim_size))
                        nl.store(result[d_indices, post_indices], value=x_tile, 
                                mask=(d_indices < dim_size) & (post_indices < post_dim_size))
                    elif dim == 1:
                        # For dim=1, the indices are [pre, d_indices, post_indices]
                        x_tile = nl.load(a_tensor[pre, d_indices, post_indices], 
                                        mask=(d_indices < dim_size) & (post_indices < post_dim_size))
                        nl.store(result[pre, d_indices, post_indices], value=x_tile, 
                                mask=(d_indices < dim_size) & (post_indices < post_dim_size))
                    else:
                        # For dim>1, we need more complex indexing which is not directly supported
                        # This is a simplified approach for dim=2
                        x_tile = nl.load(a_tensor[pre, d_indices, post_indices], 
                                        mask=(d_indices < dim_size) & (post_indices < post_dim_size))
                        nl.store(result[pre, d_indices, post_indices], value=x_tile, 
                                mask=(d_indices < dim_size) & (post_indices < post_dim_size))
        
        # Bubble sort along dimension dim
        # We'll implement a simplified version for dim <= 2
        if dim <= 2:
            for i in range(dim_size):
                for j in nl.affine_range(math.ceil((dim_size - i - 1) / nl.tile_size.pmax)):
                    j_start = j * nl.tile_size.pmax
                    j_indices = j_start + nl.arange(nl.tile_size.pmax)
                    
                    for pre in nl.affine_range(pre_dim_size):
                        for post in nl.affine_range(math.ceil(post_dim_size / nl.tile_size.pmax)):
                            post_start = post * nl.tile_size.pmax
                            post_indices = post_start + nl.arange(nl.tile_size.pmax)
                            
                            # Load current and next elements
                            if dim == 0:
                                current = nl.load(result[j_indices, post_indices], 
                                                mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                                next_elem = nl.load(result[j_indices+1, post_indices], 
                                                mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                            elif dim == 1:
                                current = nl.load(result[pre, j_indices, post_indices], 
                                                mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                                next_elem = nl.load(result[pre, j_indices+1, post_indices], 
                                                mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                            else:  # dim == 2
                                current = nl.load(result[pre, post_indices, j_indices], 
                                                mask=(j_indices < dim_size-i-1) & (post_indices < pre_dim_size))
                                next_elem = nl.load(result[pre, post_indices, j_indices+1], 
                                                mask=(j_indices < dim_size-i-1) & (post_indices < pre_dim_size))
                            
                            # Compare and swap
                            swap_mask = nl.greater(current, next_elem) & (j_indices < dim_size-i-1) & (post_indices < post_dim_size)
                            
                            # Where swap is needed, swap values
                            new_current = nl.where(swap_mask, next_elem, current)
                            new_next = nl.where(swap_mask, current, next_elem)
                            
                            # Store back
                            if dim == 0:
                                nl.store(result[j_indices, post_indices], value=new_current, 
                                        mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                                nl.store(result[j_indices+1, post_indices], value=new_next, 
                                        mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                            elif dim == 1:
                                nl.store(result[pre, j_indices, post_indices], value=new_current, 
                                        mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                                nl.store(result[pre, j_indices+1, post_indices], value=new_next, 
                                        mask=(j_indices < dim_size-i-1) & (post_indices < post_dim_size))
                            else:  # dim == 2
                                nl.store(result[pre, post_indices, j_indices], value=new_current, 
                                        mask=(j_indices < dim_size-i-1) & (post_indices < pre_dim_size))
                                nl.store(result[pre, post_indices, j_indices+1], value=new_next, 
                                        mask=(j_indices < dim_size-i-1) & (post_indices < pre_dim_size))
    
    return result