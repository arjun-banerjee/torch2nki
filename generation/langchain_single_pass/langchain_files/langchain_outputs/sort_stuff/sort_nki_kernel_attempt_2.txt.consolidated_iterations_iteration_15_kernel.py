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
    
    # First, copy the input tensor to the result
    # For 2D tensor case
    if ndim == 2:
        dim_size = shape[dim]
        other_dim = 1 - dim  # The other dimension (0 or 1)
        other_size = shape[other_dim]
        
        # Calculate tiles needed to respect hardware limitations
        other_tiles = math.ceil(other_size / nl.tile_size.pmax)
        
        # Copy input to result first
        for i in nl.affine_range(other_tiles):
            start_idx = i * nl.tile_size.pmax
            # Create indices for current tile
            if other_dim == 0:
                # When sorting along dimension 1
                i_p = nl.arange(min(nl.tile_size.pmax, other_size - start_idx))[:, None]
                i_f = nl.arange(dim_size)[None, :]
                
                # Load data from input tensor
                in_tile = nl.load(a_tensor[start_idx + i_p, i_f], 
                                  mask=(start_idx + i_p < other_size))
                
                # Store into result tensor
                nl.store(result[start_idx + i_p, i_f], value=in_tile, 
                         mask=(start_idx + i_p < other_size))
            else:
                # When sorting along dimension 0
                i_p = nl.arange(min(nl.tile_size.pmax, other_size - start_idx))[None, :]
                i_f = nl.arange(dim_size)[:, None]
                
                # Load data from input tensor
                in_tile = nl.load(a_tensor[i_f, start_idx + i_p], 
                                  mask=(start_idx + i_p < other_size))
                
                # Store into result tensor
                nl.store(result[i_f, start_idx + i_p], value=in_tile, 
                         mask=(start_idx + i_p < other_size))
                
        # Now perform bubble sort on each segment
        for i in nl.affine_range(other_tiles):
            start_idx = i * nl.tile_size.pmax
            actual_size = min(nl.tile_size.pmax, other_size - start_idx)
            
            for j in nl.static_range(dim_size - 1):
                for k in nl.static_range(dim_size - j - 1):
                    if dim == 1:
                        # Sort along dimension 1
                        i_p = nl.arange(actual_size)[:, None]
                        
                        # Load adjacent elements
                        a = nl.load(result[start_idx + i_p, k], 
                                   mask=(start_idx + i_p < other_size))
                        b = nl.load(result[start_idx + i_p, k + 1], 
                                   mask=(start_idx + i_p < other_size))
                        
                        # Compare and swap if needed
                        swap_mask = nl.greater(a, b)
                        temp = nl.zeros(a.shape, dtype=a.dtype, buffer=nl.sbuf)
                        
                        # Swap where needed
                        temp = nl.where(swap_mask, b, a)
                        nl.store(result[start_idx + i_p, k], value=temp, 
                                mask=(start_idx + i_p < other_size))
                        
                        temp = nl.where(swap_mask, a, b)
                        nl.store(result[start_idx + i_p, k + 1], value=temp, 
                                mask=(start_idx + i_p < other_size))
                    else:
                        # Sort along dimension 0
                        i_p = nl.arange(actual_size)[None, :]
                        
                        # Load adjacent elements
                        a = nl.load(result[k, start_idx + i_p], 
                                   mask=(start_idx + i_p < other_size))
                        b = nl.load(result[k + 1, start_idx + i_p], 
                                   mask=(start_idx + i_p < other_size))
                        
                        # Compare and swap if needed
                        swap_mask = nl.greater(a, b)
                        temp = nl.zeros(a.shape, dtype=a.dtype, buffer=nl.sbuf)
                        
                        # Swap where needed
                        temp = nl.where(swap_mask, b, a)
                        nl.store(result[k, start_idx + i_p], value=temp, 
                                mask=(start_idx + i_p < other_size))
                        
                        temp = nl.where(swap_mask, a, b)
                        nl.store(result[k + 1, start_idx + i_p], value=temp, 
                                mask=(start_idx + i_p < other_size))
    
    # Handle 1D tensor case
    elif ndim == 1:
        # Copy input to result first
        size = shape[0]
        tiles = math.ceil(size / nl.tile_size.pmax)
        
        for i in nl.affine_range(tiles):
            start_idx = i * nl.tile_size.pmax
            i_p = nl.arange(min(nl.tile_size.pmax, size - start_idx))
            
            # Load data from input tensor
            in_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < size))
            
            # Store into result tensor
            nl.store(result[start_idx + i_p], value=in_tile, mask=(start_idx + i_p < size))
        
        # Bubble sort the entire array
        for i in nl.static_range(size - 1):
            for j in nl.static_range(size - i - 1):
                # Process in tiles to respect hardware limitations
                for k in nl.affine_range(tiles):
                    start_idx = k * nl.tile_size.pmax
                    actual_size = min(nl.tile_size.pmax, size - start_idx)
                    
                    if j >= start_idx and j < start_idx + actual_size and j + 1 < size:
                        # Load adjacent elements
                        a = nl.load(result[j])
                        b = nl.load(result[j + 1])
                        
                        # Compare and swap if needed
                        if nl.greater(a, b).item():
                            # Swap
                            nl.store(result[j], value=b)
                            nl.store(result[j + 1], value=a)
    
    return result