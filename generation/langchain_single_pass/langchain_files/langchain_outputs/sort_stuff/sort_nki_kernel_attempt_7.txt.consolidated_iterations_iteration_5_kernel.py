from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get the shape of the input tensor
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to the result tensor
    if ndim == 1:
        # For 1D tensor, sort the entire tensor
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Calculate number of tiles needed
        num_tiles = math.ceil(size / max_tile_size)
        
        # First copy input to result
        for p in nl.affine_range(num_tiles):
            # Calculate indices
            idx = p * max_tile_size + nl.arange(max_tile_size)
            
            # Load from input, respecting tensor boundaries
            in_tile = nl.load(a_tensor[idx], mask=(idx < size))
            
            # Store to result
            nl.store(result[idx], value=in_tile, mask=(idx < size))
        
        # Bubble sort implementation for 1D tensor
        for i in nl.static_range(size):
            for j in nl.static_range(size - i - 1):
                # Process in tiles to respect hardware limitations
                for p in nl.affine_range(num_tiles):
                    # Calculate indices
                    idx = p * max_tile_size + nl.arange(max_tile_size)
                    idx_next = idx + 1
                    
                    # Load current elements
                    curr = nl.load(result[idx], mask=((idx < size) & (idx < size - i - 1)))
                    next_elem = nl.load(result[idx_next], mask=((idx_next < size) & (idx < size - i - 1)))
                    
                    # Compare and swap if needed
                    swap_mask = ((idx < size - i - 1) & (curr > next_elem))
                    temp = nl.zeros(curr.shape, dtype=curr.dtype, buffer=nl.sbuf)
                    
                    # Conditionally swap elements
                    temp = nl.select(swap_mask, next_elem, curr)
                    nl.store(result[idx], value=temp, mask=((idx < size) & (idx < size - i - 1)))
                    
                    temp = nl.select(swap_mask, curr, next_elem)
                    nl.store(result[idx_next], value=temp, mask=((idx_next < size) & (idx < size - i - 1)))
        
    elif ndim == 2:
        # For 2D tensor, sort along specified dimension
        if dim == 0:
            # Sort along rows
            rows, cols = shape[0], shape[1]
            max_tile_size = nl.tile_size.pmax
            
            # Calculate number of tiles needed
            num_tiles_rows = math.ceil(rows / max_tile_size)
            
            # First copy input to result
            for p in nl.affine_range(num_tiles_rows):
                # Calculate indices
                idx_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                idx_f = nl.arange(cols)[None, :]
                
                # Load from input, respecting tensor boundaries
                in_tile = nl.load(a_tensor[idx_p, idx_f], mask=(idx_p < rows))
                
                # Store to result
                nl.store(result[idx_p, idx_f], value=in_tile, mask=(idx_p < rows))
            
            # Sort each column
            for j in nl.static_range(cols):
                for i in nl.static_range(rows):
                    for k in nl.static_range(rows - i - 1):
                        # Process in tiles
                        for p in nl.affine_range(num_tiles_rows):
                            idx_p = p * max_tile_size + nl.arange(max_tile_size)
                            curr_mask = (idx_p == k)
                            next_mask = (idx_p == (k + 1))
                            
                            if p * max_tile_size <= k < (p + 1) * max_tile_size and k < rows - i - 1:
                                # Load current and next elements
                                curr = nl.load(result[k, j])
                                next_elem = nl.load(result[k + 1, j])
                                
                                # Compare and swap if needed
                                if nl.greater(curr, next_elem):
                                    nl.store(result[k, j], value=next_elem)
                                    nl.store(result[k + 1, j], value=curr)
                
        else:  # dim == 1
            # Sort along columns
            rows, cols = shape[0], shape[1]
            max_tile_size = nl.tile_size.pmax
            
            # Calculate number of tiles needed
            num_tiles_rows = math.ceil(rows / max_tile_size)
            
            # First copy input to result
            for p in nl.affine_range(num_tiles_rows):
                # Calculate indices
                idx_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                idx_f = nl.arange(cols)[None, :]
                
                # Load from input, respecting tensor boundaries
                in_tile = nl.load(a_tensor[idx_p, idx_f], mask=(idx_p < rows))
                
                # Store to result
                nl.store(result[idx_p, idx_f], value=in_tile, mask=(idx_p < rows))
            
            # Sort each row
            for i in nl.static_range(rows):
                # Bubble sort
                for j in nl.static_range(cols):
                    for k in nl.static_range(cols - j - 1):
                        # Load current and next elements
                        curr = nl.load(result[i, k])
                        next_elem = nl.load(result[i, k + 1])
                        
                        # Compare and swap if needed
                        if nl.greater(curr, next_elem):
                            nl.store(result[i, k], value=next_elem)
                            nl.store(result[i, k + 1], value=curr)
    else:
        # For tensors with more than 2 dimensions, we need a different approach
        # This is a simple placeholder implementation
        # Copy input to result
        sz_p = shape[0]
        sz_f = 1
        for i in range(1, ndim):
            sz_f *= shape[i]
            
        # Calculate number of tiles needed
        max_tile_size = nl.tile_size.pmax
        num_tiles = math.ceil(sz_p / max_tile_size)
        
        # Reshape to 2D for processing
        for p in nl.affine_range(num_tiles):
            # Calculate indices
            idx_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
            idx_f = nl.arange(sz_f)[None, :]
            
            # Load from input, respecting tensor boundaries
            in_tile = nl.load(a_tensor.reshape((sz_p, sz_f))[idx_p, idx_f], mask=(idx_p < sz_p))
            
            # Store to result
            nl.store(result.reshape((sz_p, sz_f))[idx_p, idx_f], value=in_tile, mask=(idx_p < sz_p))
    
    return result