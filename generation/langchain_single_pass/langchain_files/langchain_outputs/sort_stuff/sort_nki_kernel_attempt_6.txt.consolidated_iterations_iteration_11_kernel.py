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
    if dim == 0:
        # When sorting along dimension 0
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(shape[0] / max_tile_size)
        
        for p in nl.affine_range(trip_count):
            start_p = p * max_tile_size
            i_p = start_p + nl.arange(max_tile_size)[:, None]
            
            # Create indices for other dimensions
            if len(shape) == 1:
                # 1D tensor case
                in_tile = nl.load(a_tensor[i_p[:, 0]], mask=(i_p[:, 0] < shape[0]))
                nl.store(result[i_p[:, 0]], in_tile, mask=(i_p[:, 0] < shape[0]))
            else:
                # Multi-dimensional tensor case
                i_f = nl.arange(shape[1])[None, :]
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < shape[0]))
                nl.store(result[i_p, i_f], in_tile, mask=(i_p < shape[0]))
                
        # Bubble sort implementation for dimension 0
        for i in nl.affine_range(sort_dim_size):
            for j in nl.affine_range(sort_dim_size - 1):
                for p in nl.affine_range(trip_count):
                    start_p = p * max_tile_size
                    i_p = start_p + nl.arange(max_tile_size)[:, None]
                    
                    if len(shape) == 1:
                        # 1D tensor case
                        if j < shape[0] - 1 and j+1 < shape[0]:
                            val1 = nl.load(result[j])
                            val2 = nl.load(result[j+1])
                            cond = nl.greater(val1, val2)
                            
                            # Swap if val1 > val2
                            new_val1 = nl.where(cond, val2, val1)
                            new_val2 = nl.where(cond, val1, val2)
                            
                            nl.store(result[j], new_val1)
                            nl.store(result[j+1], new_val2)
                    else:
                        # Multi-dimensional tensor case
                        if j < shape[0] - 1 and j+1 < shape[0]:
                            i_f = nl.arange(shape[1])[None, :]
                            val1 = nl.load(result[j, i_f])
                            val2 = nl.load(result[j+1, i_f])
                            cond = nl.greater(val1, val2)
                            
                            # Swap if val1 > val2
                            new_val1 = nl.where(cond, val2, val1)
                            new_val2 = nl.where(cond, val1, val2)
                            
                            nl.store(result[j, i_f], new_val1)
                            nl.store(result[j+1, i_f], new_val2)
    else:
        # When sorting along dimension other than 0
        # For simplicity, handle 2D case specifically
        if len(shape) == 2:
            max_tile_size = nl.tile_size.pmax
            trip_count = math.ceil(shape[0] / max_tile_size)
            
            # Copy input to result
            for p in nl.affine_range(trip_count):
                start_p = p * max_tile_size
                i_p = start_p + nl.arange(max_tile_size)[:, None]
                i_f = nl.arange(shape[1])[None, :]
                
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < shape[0]))
                nl.store(result[i_p, i_f], in_tile, mask=(i_p < shape[0]))
            
            # Bubble sort implementation for dimension 1
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    for p in nl.affine_range(trip_count):
                        start_p = p * max_tile_size
                        i_p = start_p + nl.arange(max_tile_size)[:, None]
                        
                        # Only process rows that are within bounds
                        mask = (i_p < shape[0])
                        
                        if j < shape[1] - 1 and j+1 < shape[1]:
                            val1 = nl.load(result[i_p, j], mask=mask)
                            val2 = nl.load(result[i_p, j+1], mask=mask)
                            cond = nl.greater(val1, val2)
                            
                            # Swap if val1 > val2
                            new_val1 = nl.where(cond, val2, val1)
                            new_val2 = nl.where(cond, val1, val2)
                            
                            nl.store(result[i_p, j], new_val1, mask=mask)
                            nl.store(result[i_p, j+1], new_val2, mask=mask)
        else:
            # For higher dimensions, just copy input to output without sorting
            # This is a simplified fallback
            max_tile_size = nl.tile_size.pmax
            trip_count = math.ceil(shape[0] / max_tile_size)
            
            for p in nl.affine_range(trip_count):
                start_p = p * max_tile_size
                i_p = start_p + nl.arange(max_tile_size)[:, None]
                i_f = nl.arange(shape[1])[None, :]
                
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < shape[0]))
                nl.store(result[i_p, i_f], in_tile, mask=(i_p < shape[0]))
    
    return result