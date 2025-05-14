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
    
    # Copy input tensor to result first
    # Handle 2D case specifically for simplicity
    if ndim == 2:
        # Get sizes
        size0 = shape[0]
        size1 = shape[1]
        
        # Determine which dimension to sort along
        if dim == 0:
            # Sort along dimension 0
            # Process in tiles to respect hardware limitations
            max_tile = nl.tile_size.pmax
            for j in nl.affine_range(math.ceil(size1 / max_tile)):
                j_start = j * max_tile
                j_end = min(size1, (j + 1) * max_tile)
                j_size = j_end - j_start
                
                # Load a tile from the input tensor
                i_p = nl.arange(size0)[:, None]
                i_f = nl.arange(j_size)[None, :]
                tile = nl.load(a_tensor[i_p, i_f + j_start])
                
                # Store the tile to the result tensor
                nl.store(result[i_p, i_f + j_start], value=tile)
                
                # Now sort each column using bubble sort
                for _ in nl.affine_range(size0):
                    for i in nl.affine_range(size0 - 1):
                        # Compare adjacent elements
                        curr = nl.load(result[i, i_f + j_start])
                        next_val = nl.load(result[i + 1, i_f + j_start])
                        
                        # If current > next, swap them
                        swap_mask = nl.greater(curr, next_val)
                        
                        # Only swap if needed
                        where_swap = nl.zeros((j_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        where_no_swap = nl.zeros((j_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        
                        for idx in nl.affine_range(j_size):
                            if swap_mask[0, idx]:
                                where_swap[idx] = curr[0, idx]
                                where_no_swap[idx] = next_val[0, idx]
                            else:
                                where_swap[idx] = next_val[0, idx]
                                where_no_swap[idx] = curr[0, idx]
                        
                        # Store the swapped values
                        nl.store(result[i, i_f + j_start], value=where_no_swap[None, :])
                        nl.store(result[i + 1, i_f + j_start], value=where_swap[None, :])
        else:
            # Sort along dimension 1
            # Process in tiles to respect hardware limitations
            max_tile = nl.tile_size.pmax
            for i in nl.affine_range(math.ceil(size0 / max_tile)):
                i_start = i * max_tile
                i_end = min(size0, (i + 1) * max_tile)
                i_size = i_end - i_start
                
                # Load a tile from the input tensor
                i_p = nl.arange(i_size)[:, None]
                i_f = nl.arange(size1)[None, :]
                tile = nl.load(a_tensor[i_p + i_start, i_f])
                
                # Store the tile to the result tensor
                nl.store(result[i_p + i_start, i_f], value=tile)
                
                # Now sort each row using bubble sort
                for _ in nl.affine_range(size1):
                    for j in nl.affine_range(size1 - 1):
                        # Compare adjacent elements
                        curr = nl.load(result[i_p + i_start, j])
                        next_val = nl.load(result[i_p + i_start, j + 1])
                        
                        # If current > next, swap them
                        swap_mask = nl.greater(curr, next_val)
                        
                        # Only swap if needed
                        where_swap = nl.zeros((i_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        where_no_swap = nl.zeros((i_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        
                        for idx in nl.affine_range(i_size):
                            if swap_mask[idx, 0]:
                                where_swap[idx] = curr[idx, 0]
                                where_no_swap[idx] = next_val[idx, 0]
                            else:
                                where_swap[idx] = next_val[idx, 0]
                                where_no_swap[idx] = curr[idx, 0]
                        
                        # Store the swapped values
                        nl.store(result[i_p + i_start, j], value=where_no_swap[:, None])
                        nl.store(result[i_p + i_start, j + 1], value=where_swap[:, None])
    else:
        # For 1D case
        size = shape[0]
        
        # Load input tensor to result
        i_p = nl.arange(size)
        result_tile = nl.load(a_tensor[i_p])
        nl.store(result[i_p], value=result_tile)
        
        # Sort using bubble sort
        for _ in nl.affine_range(size):
            for i in nl.affine_range(size - 1):
                # Compare adjacent elements
                curr = nl.load(result[i])
                next_val = nl.load(result[i + 1])
                
                # If current > next, swap them
                if nl.greater(curr, next_val)[0]:
                    nl.store(result[i], value=next_val)
                    nl.store(result[i + 1], value=curr)
    
    return result