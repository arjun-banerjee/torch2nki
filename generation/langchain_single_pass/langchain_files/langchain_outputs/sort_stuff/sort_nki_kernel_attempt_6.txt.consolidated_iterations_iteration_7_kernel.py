from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimensions
    if dim < 0:
        dim = len(a_tensor.shape) + dim
        
    # Get shape of input tensor
    shape = a_tensor.shape
    sort_dim_size = shape[dim]
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input tensor to result
    # Calculate the number of tiles needed
    max_tile_size = nl.tile_size.pmax
    trip_count_p = math.ceil(shape[0] / max_tile_size)
    
    # Handle 1D tensor case
    if len(shape) == 1:
        # Copy data to result
        for p in nl.affine_range(trip_count_p):
            i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < shape[0]))
            nl.store(result[i_p], value=in_tile, mask=(i_p < shape[0]))
            
        # Bubble sort along the only dimension
        for i in nl.affine_range(sort_dim_size):
            for p in nl.affine_range(trip_count_p):
                i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                
                # Load current values
                in_tile = nl.load(result[i_p], mask=(i_p < shape[0]))
                
                # For each element in the tile
                for j in nl.affine_range(max_tile_size - 1):
                    # Skip if beyond array bounds
                    idx = p * max_tile_size + j
                    if idx >= shape[0] - 1 - i:
                        continue
                    
                    # Compare and swap if needed
                    val_j = nl.load(result[idx])
                    val_j1 = nl.load(result[idx + 1])
                    
                    # Swap if needed
                    condition = nl.greater(val_j, val_j1)
                    if condition:
                        nl.store(result[idx], value=val_j1)
                        nl.store(result[idx + 1], value=val_j)
    
    # Handle 2D tensor case
    elif len(shape) == 2:
        # Copy data to result
        trip_count_f = math.ceil(shape[1] / max_tile_size)
        
        for p in nl.affine_range(trip_count_p):
            for f in nl.affine_range(trip_count_f):
                i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                i_f = f * max_tile_size + nl.arange(max_tile_size)[None, :]
                
                in_tile = nl.load(a_tensor[i_p, i_f], mask=((i_p < shape[0]) & (i_f < shape[1])))
                nl.store(result[i_p, i_f], value=in_tile, mask=((i_p < shape[0]) & (i_f < shape[1])))
        
        # Sort along specified dimension
        if dim == 0:  # Sort along rows
            for i in nl.affine_range(shape[0]):
                for j in nl.affine_range(shape[0] - 1):
                    if j >= shape[0] - 1 - i:
                        continue
                        
                    # Load current and next row
                    row_j = nl.zeros((1, shape[1]), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    row_j1 = nl.zeros((1, shape[1]), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    
                    for f in nl.affine_range(trip_count_f):
                        i_f = f * max_tile_size + nl.arange(max_tile_size)[None, :]
                        
                        # Load rows
                        row_j_tile = nl.load(result[j:j+1, i_f], mask=(i_f < shape[1]))
                        row_j1_tile = nl.load(result[j+1:j+2, i_f], mask=(i_f < shape[1]))
                        
                        # Compare rows and swap if needed
                        for k in nl.affine_range(min(max_tile_size, shape[1])):
                            idx_f = f * max_tile_size + k
                            if idx_f >= shape[1]:
                                break
                                
                            val_j = nl.load(result[j, idx_f])
                            val_j1 = nl.load(result[j+1, idx_f])
                            
                            if nl.greater(val_j, val_j1):
                                # Swap entire rows
                                for l in nl.affine_range(trip_count_f):
                                    i_l = l * max_tile_size + nl.arange(max_tile_size)[None, :]
                                    
                                    temp_j = nl.load(result[j:j+1, i_l], mask=(i_l < shape[1]))
                                    temp_j1 = nl.load(result[j+1:j+2, i_l], mask=(i_l < shape[1]))
                                    
                                    nl.store(result[j:j+1, i_l], value=temp_j1, mask=(i_l < shape[1]))
                                    nl.store(result[j+1:j+2, i_l], value=temp_j, mask=(i_l < shape[1]))
                                break
                            elif nl.greater(val_j1, val_j):
                                break
        else:  # Sort along columns (dim == 1)
            for i in nl.affine_range(shape[1]):
                for p in nl.affine_range(trip_count_p):
                    i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                    
                    for j in nl.affine_range(shape[1] - 1):
                        if j >= shape[1] - 1 - i:
                            continue
                            
                        # Load current and next column elements for this partition
                        col_j = nl.load(result[i_p, j:j+1], mask=(i_p < shape[0]))
                        col_j1 = nl.load(result[i_p, j+1:j+2], mask=(i_p < shape[0]))
                        
                        # Compare and swap if needed
                        for k in nl.affine_range(min(max_tile_size, shape[0])):
                            idx_p = p * max_tile_size + k
                            if idx_p >= shape[0]:
                                break
                                
                            val_j = nl.load(result[idx_p, j])
                            val_j1 = nl.load(result[idx_p, j+1])
                            
                            if nl.greater(val_j, val_j1):
                                # Swap elements across the whole column
                                for l in nl.affine_range(trip_count_p):
                                    i_l = l * max_tile_size + nl.arange(max_tile_size)[:, None]
                                    
                                    temp_j = nl.load(result[i_l, j:j+1], mask=(i_l < shape[0]))
                                    temp_j1 = nl.load(result[i_l, j+1:j+2], mask=(i_l < shape[0]))
                                    
                                    nl.store(result[i_l, j:j+1], value=temp_j1, mask=(i_l < shape[0]))
                                    nl.store(result[i_l, j+1:j+2], value=temp_j, mask=(i_l < shape[0]))
                                break
                            elif nl.greater(val_j1, val_j):
                                break
                                
    return result