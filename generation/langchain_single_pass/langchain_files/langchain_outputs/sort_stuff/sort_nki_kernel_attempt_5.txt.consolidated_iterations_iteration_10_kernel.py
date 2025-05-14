from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension
    ndim = len(a_tensor.shape)
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy input to result first
    if ndim == 1:
        # 1D tensor case
        sz = a_tensor.shape[0]
        max_tile_size = nl.tile_size.pmax
        num_tiles = (sz + max_tile_size - 1) // max_tile_size
        
        for p in nl.affine_range(num_tiles):
            start_idx = p * max_tile_size
            i_p = start_idx + nl.arange(max_tile_size)
            mask = i_p < sz
            tile = nl.load(a_tensor[i_p], mask=mask)
            nl.store(result[i_p], value=tile, mask=mask)
    else:
        # Multi-dimensional tensor case
        if dim == 0:
            # Sort along first dimension
            sz_0 = a_tensor.shape[0]
            sz_rest = 1
            for i in range(1, ndim):
                sz_rest *= a_tensor.shape[i]
            
            max_tile_size = nl.tile_size.pmax
            num_tiles = (sz_rest + max_tile_size - 1) // max_tile_size
            
            for p in nl.affine_range(num_tiles):
                start_idx = p * max_tile_size
                i_p = start_idx + nl.arange(max_tile_size)
                i_f = nl.arange(sz_0)[None, :]
                mask = i_p < sz_rest
                
                for s in range(sz_0):
                    for t in range(s):
                        idx_s = nl.full((1,), s, dtype=nl.int32)
                        idx_t = nl.full((1,), t, dtype=nl.int32)
                        
                        # Load values to compare
                        val_s = nl.load(a_tensor[idx_s, i_p], mask=mask)
                        val_t = nl.load(a_tensor[idx_t, i_p], mask=mask)
                        
                        # Compare and swap if needed
                        cond = nl.greater(val_t, val_s)
                        val_s_new = nl.where(cond, val_t, val_s)
                        val_t_new = nl.where(cond, val_s, val_t)
                        
                        # Store back
                        nl.store(result[idx_s, i_p], value=val_s_new, mask=mask)
                        nl.store(result[idx_t, i_p], value=val_t_new, mask=mask)
        else:
            # Sort along other dimensions
            # Calculate sizes before and after the target dimension
            sz_before = 1
            for i in range(dim):
                sz_before *= a_tensor.shape[i]
            
            sz_dim = a_tensor.shape[dim]
            
            sz_after = 1
            for i in range(dim + 1, ndim):
                sz_after *= a_tensor.shape[i]
            
            # Copy the input first
            max_tile_p = nl.tile_size.pmax
            num_tiles_p = (sz_before + max_tile_p - 1) // max_tile_p
            
            for p in nl.affine_range(num_tiles_p):
                start_p = p * max_tile_p
                i_p = start_p + nl.arange(max_tile_p)[:, None, None]
                mask_p = i_p < sz_before
                
                max_tile_f = 128  # Choose an appropriate tile size
                num_tiles_f = (sz_after + max_tile_f - 1) // max_tile_f
                
                for f in nl.affine_range(num_tiles_f):
                    start_f = f * max_tile_f
                    i_f = start_f + nl.arange(max_tile_f)[None, None, :]
                    mask_f = i_f < sz_after
                    
                    # Use temporary buffers for the dimension to be sorted
                    temp = nl.ndarray((max_tile_p, sz_dim, max_tile_f), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    
                    # Load all values along the dimension to be sorted
                    for d in range(sz_dim):
                        idx_d = nl.full((1,), d, dtype=nl.int32)
                        if dim == 1:  # Special case for dim=1
                            tile = nl.load(a_tensor[i_p, idx_d, i_f], mask=(mask_p & mask_f))
                        else:  # General case
                            # This is a placeholder - actual indexing would need dimension-specific code
                            # which is hard to represent generically
                            tile = nl.load(a_tensor[i_p, idx_d, i_f], mask=(mask_p & mask_f))
                        
                        # Store to temporary buffer
                        nl.store(temp[0:max_tile_p, d, 0:max_tile_f], value=tile, mask=(mask_p & mask_f))
                    
                    # Bubble sort along the target dimension
                    for i in range(sz_dim):
                        for j in range(0, sz_dim - i - 1):
                            idx_j = nl.full((1,), j, dtype=nl.int32)
                            idx_j1 = nl.full((1,), j + 1, dtype=nl.int32)
                            
                            # Load adjacent elements
                            val_j = nl.load(temp[0:max_tile_p, idx_j, 0:max_tile_f], mask=(mask_p & mask_f))
                            val_j1 = nl.load(temp[0:max_tile_p, idx_j1, 0:max_tile_f], mask=(mask_p & mask_f))
                            
                            # Compare and swap if needed
                            cond = nl.greater(val_j, val_j1)
                            val_j_new = nl.where(cond, val_j1, val_j)
                            val_j1_new = nl.where(cond, val_j, val_j1)
                            
                            # Store back
                            nl.store(temp[0:max_tile_p, idx_j, 0:max_tile_f], value=val_j_new, mask=(mask_p & mask_f))
                            nl.store(temp[0:max_tile_p, idx_j1, 0:max_tile_f], value=val_j1_new, mask=(mask_p & mask_f))
                    
                    # Store sorted values back to the result
                    for d in range(sz_dim):
                        idx_d = nl.full((1,), d, dtype=nl.int32)
                        tile = nl.load(temp[0:max_tile_p, d, 0:max_tile_f], mask=(mask_p & mask_f))
                        
                        if dim == 1:  # Special case for dim=1
                            nl.store(result[i_p, idx_d, i_f], value=tile, mask=(mask_p & mask_f))
                        else:  # General case
                            # This is a placeholder - actual indexing would need dimension-specific code
                            nl.store(result[i_p, idx_d, i_f], value=tile, mask=(mask_p & mask_f))
    
    return result