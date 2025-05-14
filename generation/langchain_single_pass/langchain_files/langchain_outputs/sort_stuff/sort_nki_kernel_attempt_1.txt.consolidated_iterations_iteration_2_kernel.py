from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Convert negative dim to positive
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input to result
    if len(a_tensor.shape) == 1:
        # Handle 1D case
        size = a_tensor.shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
    elif len(a_tensor.shape) == 2:
        # Handle 2D case
        rows, cols = a_tensor.shape
        
        if dim == 0:  # Sort along rows
            trip_count = math.ceil(cols / nl.tile_size.pmax)
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(rows)[None, :]
                in_tile = nl.load(a_tensor[i_f, i_p], mask=(i_p < cols))
                nl.store(result[i_f, i_p], value=in_tile, mask=(i_p < cols))
        else:  # Sort along columns (dim == 1)
            trip_count = math.ceil(rows / nl.tile_size.pmax)
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(cols)[None, :]
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < rows))
                nl.store(result[i_p, i_f], value=in_tile, mask=(i_p < rows))
    
    # Now sort the result tensor
    if len(a_tensor.shape) == 1:
        # Sort 1D array using bubble sort
        size = a_tensor.shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # Bubble sort
        for i in range(size):
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                
                # Load current segment
                segment = nl.load(result[i_p], mask=(i_p < size - i - 1))
                
                # Load next elements
                next_indices = i_p + 1
                next_segment = nl.load(result[next_indices], mask=(next_indices < size - i))
                
                # Compare and swap if needed
                swap_mask = nl.greater(segment, next_segment) & (i_p < size - i - 1) & (next_indices < size - i)
                
                # Where swap is needed, store the swapped values
                where_to_swap = nl.load(result[i_p], mask=swap_mask)
                nl.store(result[i_p], value=nl.load(result[next_indices], mask=swap_mask), mask=swap_mask)
                nl.store(result[next_indices], value=where_to_swap, mask=swap_mask)
    
    elif len(a_tensor.shape) == 2:
        rows, cols = a_tensor.shape
        
        if dim == 0:  # Sort along rows (each column independently)
            for col in range(cols):
                for i in range(rows):
                    for j in range(rows - i - 1):
                        curr_val = nl.load(result[j, col])
                        next_val = nl.load(result[j + 1, col])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_val, next_val)
                        
                        if swap_needed:
                            nl.store(result[j, col], value=next_val)
                            nl.store(result[j + 1, col], value=curr_val)
        
        else:  # Sort along columns (each row independently)
            for row in range(rows):
                for i in range(cols):
                    trip_count = math.ceil((cols - i - 1) / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        
                        # Ensure we don't go out of bounds
                        valid_mask = (i_p < cols - i - 1)
                        
                        # Load current elements
                        curr_vals = nl.load(result[row, i_p], mask=valid_mask)
                        
                        # Load next elements
                        next_indices = i_p + 1
                        next_vals = nl.load(result[row, next_indices], mask=(next_indices < cols - i))
                        
                        # Compare and swap if needed
                        swap_mask = nl.greater(curr_vals, next_vals) & valid_mask & (next_indices < cols - i)
                        
                        # Where swap is needed, store the swapped values
                        temp = nl.load(result[row, i_p], mask=swap_mask)
                        nl.store(result[row, i_p], value=nl.load(result[row, next_indices], mask=swap_mask), mask=swap_mask)
                        nl.store(result[row, next_indices], value=temp, mask=swap_mask)
    
    return result