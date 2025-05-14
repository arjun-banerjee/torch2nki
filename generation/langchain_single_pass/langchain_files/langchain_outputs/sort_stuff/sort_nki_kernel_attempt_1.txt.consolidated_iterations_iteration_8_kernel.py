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
    
    # Handle 1D case
    if ndim == 1:
        size = shape[0]
        # Copy input to result first
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = i * nl.tile_size.pmax
            idx = nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[start_idx + idx], mask=(start_idx + idx < size))
            nl.store(result[start_idx + idx], in_tile, mask=(start_idx + idx < size))
        
        # Bubble sort
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements
                j_val = nl.load(result[j])
                j_next_val = nl.load(result[j+1])
                
                # Compare and swap if needed
                swap_needed = nl.greater(j_val, j_next_val)
                
                # If swap is needed, swap the elements
                if swap_needed:
                    nl.store(result[j], j_next_val)
                    nl.store(result[j+1], j_val)
    
    # Handle 2D case
    elif ndim == 2:
        rows, cols = shape
        
        if dim == 0:  # Sort along rows for each column
            for j in nl.affine_range(cols):
                # Copy column to result first
                for i in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                    start_idx = i * nl.tile_size.pmax
                    idx = nl.arange(nl.tile_size.pmax)
                    in_tile = nl.load(a_tensor[start_idx + idx, j], mask=(start_idx + idx < rows))
                    nl.store(result[start_idx + idx, j], in_tile, mask=(start_idx + idx < rows))
                
                # Bubble sort along column
                for k in nl.affine_range(rows):
                    for i in nl.affine_range(rows - 1):
                        # Load adjacent elements
                        i_val = nl.load(result[i, j])
                        i_next_val = nl.load(result[i+1, j])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(i_val, i_next_val)
                        
                        # If swap is needed, swap the elements
                        if swap_needed:
                            nl.store(result[i, j], i_next_val)
                            nl.store(result[i+1, j], i_val)
                            
        else:  # Sort along columns for each row
            for i in nl.affine_range(rows):
                # Copy row to result first
                for j in nl.affine_range(math.ceil(cols / nl.tile_size.pmax)):
                    start_idx = j * nl.tile_size.pmax
                    idx = nl.arange(nl.tile_size.pmax)
                    in_tile = nl.load(a_tensor[i, start_idx + idx], mask=(start_idx + idx < cols))
                    nl.store(result[i, start_idx + idx], in_tile, mask=(start_idx + idx < cols))
                
                # Bubble sort along row
                for k in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        # Load adjacent elements
                        j_val = nl.load(result[i, j])
                        j_next_val = nl.load(result[i, j+1])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(j_val, j_next_val)
                        
                        # If swap is needed, swap the elements
                        if swap_needed:
                            nl.store(result[i, j], j_next_val)
                            nl.store(result[i, j+1], j_val)
    
    return result