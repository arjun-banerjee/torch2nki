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
    if ndim == 1:
        # For 1D tensor
        size = shape[0]
        # Process in tiles to respect architecture limitations
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for i in nl.affine_range(trip_count):
            # Calculate indices for current tile
            idx = i * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load a tile from input tensor
            in_tile = nl.load(a_tensor[idx], mask=(idx < size))
            
            # Store to result tensor
            nl.store(result[idx], value=in_tile, mask=(idx < size))
    else:
        # For multi-dimensional tensor
        # Calculate sizes for the dimension to sort along and other dimensions
        dim_size = shape[dim]
        
        # Create a tensor to hold temporary indices for each position
        # We'll process the tensor slice by slice along the sort dimension
        
        # Example for 2D: if dim=1, we process each row independently
        if dim == 0:
            # Sort along first dimension
            for i in nl.affine_range(shape[1]):
                # Process this column
                trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
                
                for j in nl.affine_range(trip_count):
                    # Calculate indices for current tile
                    idx = j * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load a tile from input tensor
                    if ndim == 2:
                        in_tile = nl.load(a_tensor[idx, i], mask=(idx < shape[0]))
                        # Store to result tensor
                        nl.store(result[idx, i], value=in_tile, mask=(idx < shape[0]))
                    else:
                        # Handle higher dimensions similarly
                        # For simplicity, we're not implementing higher dims in this example
                        pass
        else:
            # Sort along any other dimension
            # For simplicity, we'll implement the 2D case with dim=1 (sort along rows)
            if ndim == 2:
                for i in nl.affine_range(shape[0]):
                    # Process this row
                    trip_count = math.ceil(shape[1] / nl.tile_size.pmax)
                    
                    for j in nl.affine_range(trip_count):
                        # Calculate indices for current tile
                        idx = j * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        
                        # Load a tile from input tensor
                        in_tile = nl.load(a_tensor[i, idx], mask=(idx < shape[1]))
                        
                        # Store to result tensor
                        nl.store(result[i, idx], value=in_tile, mask=(idx < shape[1]))
    
    # Now perform bubble sort on the result tensor
    if ndim == 1:
        # For 1D tensor
        size = shape[0]
        # We need to do n-1 passes for bubble sort
        for i in nl.affine_range(size - 1):
            # In each pass, compare adjacent elements and swap if needed
            # We need to process in tiles due to hardware limitations
            trip_count = math.ceil((size - 1 - i) / nl.tile_size.pmax)
            
            for j in nl.affine_range(trip_count):
                # Calculate indices for current tile
                base_idx = j * nl.tile_size.pmax
                idx1 = base_idx + nl.arange(nl.tile_size.pmax)
                idx2 = idx1 + 1
                
                # Load adjacent elements
                vals1 = nl.load(result[idx1], mask=((idx1 < size - 1 - i) & (idx1 >= 0)))
                vals2 = nl.load(result[idx2], mask=((idx2 < size) & (idx1 < size - 1 - i) & (idx1 >= 0)))
                
                # Compare and swap if needed
                swap_mask = nl.greater(vals1, vals2) & (idx1 < size - 1 - i) & (idx1 >= 0) & (idx2 < size)
                
                # Create new values after potential swap
                new_vals1 = nl.zeros(vals1.shape, dtype=vals1.dtype, buffer=nl.sbuf)
                new_vals2 = nl.zeros(vals2.shape, dtype=vals2.dtype, buffer=nl.sbuf)
                
                # Where swap_mask is True, swap values
                # Where swap_mask is False, keep original values
                new_vals1 = vals2 * swap_mask + vals1 * (1 - swap_mask)
                new_vals2 = vals1 * swap_mask + vals2 * (1 - swap_mask)
                
                # Store back to result
                nl.store(result[idx1], value=new_vals1, mask=((idx1 < size - 1 - i) & (idx1 >= 0)))
                nl.store(result[idx2], value=new_vals2, mask=((idx2 < size) & (idx1 < size - 1 - i) & (idx1 >= 0)))
    
    elif ndim == 2 and dim == 1:
        # For 2D tensor, sort along dim=1 (rows)
        for i in nl.affine_range(shape[0]):
            size = shape[1]
            # Bubble sort algorithm for each row
            for j in nl.affine_range(size - 1):
                # In each pass, compare adjacent elements and swap if needed
                # We need to process in tiles due to hardware limitations
                trip_count = math.ceil((size - 1 - j) / nl.tile_size.pmax)
                
                for k in nl.affine_range(trip_count):
                    # Calculate indices for current tile
                    base_idx = k * nl.tile_size.pmax
                    idx1 = base_idx + nl.arange(nl.tile_size.pmax)
                    idx2 = idx1 + 1
                    
                    # Load adjacent elements
                    vals1 = nl.load(result[i, idx1], mask=((idx1 < size - 1 - j) & (idx1 >= 0)))
                    vals2 = nl.load(result[i, idx2], mask=((idx2 < size) & (idx1 < size - 1 - j) & (idx1 >= 0)))
                    
                    # Compare and swap if needed
                    swap_mask = nl.greater(vals1, vals2) & (idx1 < size - 1 - j) & (idx1 >= 0) & (idx2 < size)
                    
                    # Create new values after potential swap
                    new_vals1 = nl.zeros(vals1.shape, dtype=vals1.dtype, buffer=nl.sbuf)
                    new_vals2 = nl.zeros(vals2.shape, dtype=vals2.dtype, buffer=nl.sbuf)
                    
                    # Where swap_mask is True, swap values
                    # Where swap_mask is False, keep original values
                    new_vals1 = vals2 * swap_mask + vals1 * (1 - swap_mask)
                    new_vals2 = vals1 * swap_mask + vals2 * (1 - swap_mask)
                    
                    # Store back to result
                    nl.store(result[i, idx1], value=new_vals1, mask=((idx1 < size - 1 - j) & (idx1 >= 0)))
                    nl.store(result[i, idx2], value=new_vals2, mask=((idx2 < size) & (idx1 < size - 1 - j) & (idx1 >= 0)))
    
    elif ndim == 2 and dim == 0:
        # For 2D tensor, sort along dim=0 (columns)
        for i in nl.affine_range(shape[1]):
            size = shape[0]
            # Bubble sort algorithm for each column
            for j in nl.affine_range(size - 1):
                # In each pass, compare adjacent elements and swap if needed
                # We need to process in tiles due to hardware limitations
                trip_count = math.ceil((size - 1 - j) / nl.tile_size.pmax)
                
                for k in nl.affine_range(trip_count):
                    # Calculate indices for current tile
                    base_idx = k * nl.tile_size.pmax
                    idx1 = base_idx + nl.arange(nl.tile_size.pmax)
                    idx2 = idx1 + 1
                    
                    # Load adjacent elements
                    vals1 = nl.load(result[idx1, i], mask=((idx1 < size - 1 - j) & (idx1 >= 0)))
                    vals2 = nl.load(result[idx2, i], mask=((idx2 < size) & (idx1 < size - 1 - j) & (idx1 >= 0)))
                    
                    # Compare and swap if needed
                    swap_mask = nl.greater(vals1, vals2) & (idx1 < size - 1 - j) & (idx1 >= 0) & (idx2 < size)
                    
                    # Create new values after potential swap
                    new_vals1 = nl.zeros(vals1.shape, dtype=vals1.dtype, buffer=nl.sbuf)
                    new_vals2 = nl.zeros(vals2.shape, dtype=vals2.dtype, buffer=nl.sbuf)
                    
                    # Where swap_mask is True, swap values
                    # Where swap_mask is False, keep original values
                    new_vals1 = vals2 * swap_mask + vals1 * (1 - swap_mask)
                    new_vals2 = vals1 * swap_mask + vals2 * (1 - swap_mask)
                    
                    # Store back to result
                    nl.store(result[idx1, i], value=new_vals1, mask=((idx1 < size - 1 - j) & (idx1 >= 0)))
                    nl.store(result[idx2, i], value=new_vals2, mask=((idx2 < size) & (idx1 < size - 1 - j) & (idx1 >= 0)))
    
    return result