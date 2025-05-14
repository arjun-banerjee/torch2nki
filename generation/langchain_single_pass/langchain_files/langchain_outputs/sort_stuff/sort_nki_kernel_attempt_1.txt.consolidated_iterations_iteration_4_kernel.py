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
    
    # Handle 1D case separately
    if ndim == 1:
        size = shape[0]
        # Copy input to result first
        for p in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = p * nl.tile_size.pmax
            indices = start_idx + nl.arange(nl.tile_size.pmax)
            
            # Load a tile from input
            input_tile = nl.load(a_tensor[indices], mask=(indices < size))
            
            # Store to result
            nl.store(result[indices], input_tile, mask=(indices < size))
        
        # Bubble sort the entire array
        for i in range(size):
            for j in range(size - i - 1):
                # Load adjacent elements
                j_idx = nl.full((), j, dtype=nl.int32)
                j_plus_1_idx = nl.full((), j+1, dtype=nl.int32)
                
                val_j = nl.load(result[j_idx])
                val_j_plus_1 = nl.load(result[j_plus_1_idx])
                
                # Compare and swap if needed
                swap_needed = nl.greater(val_j, val_j_plus_1)
                
                # Conditional swap
                temp = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                temp = nl.select(swap_needed, val_j_plus_1, val_j)
                nl.store(result[j_idx], nl.select(swap_needed, val_j_plus_1, val_j))
                nl.store(result[j_plus_1_idx], nl.select(swap_needed, val_j, val_j_plus_1))
    
    else:
        # For multi-dimensional tensors, we need to sort along the specified dimension
        # First, copy the input to result
        for p in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
            # Generate indices for the first dimension
            start_idx = p * nl.tile_size.pmax
            p_indices = start_idx + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Generate indices for the second dimension
            if ndim > 1:
                f_indices = nl.arange(shape[1])[None, :]
                
                input_tile = nl.load(a_tensor[p_indices, f_indices], mask=(p_indices < shape[0]))
                nl.store(result[p_indices, f_indices], input_tile, mask=(p_indices < shape[0]))
            else:
                input_tile = nl.load(a_tensor[p_indices], mask=(p_indices < shape[0]))
                nl.store(result[p_indices], input_tile, mask=(p_indices < shape[0]))
        
        # For 2D tensors, sort each row or column based on dim
        if ndim == 2:
            if dim == 0:  # Sort along columns
                for col in range(shape[1]):
                    col_idx = nl.full((), col, dtype=nl.int32)
                    
                    # Bubble sort this column
                    for i in range(shape[0]):
                        for j in range(shape[0] - i - 1):
                            # Load adjacent elements
                            j_idx = nl.full((), j, dtype=nl.int32)
                            j_plus_1_idx = nl.full((), j+1, dtype=nl.int32)
                            
                            val_j = nl.load(result[j_idx, col_idx])
                            val_j_plus_1 = nl.load(result[j_plus_1_idx, col_idx])
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(val_j, val_j_plus_1)
                            
                            # Conditional swap
                            nl.store(result[j_idx, col_idx], nl.select(swap_needed, val_j_plus_1, val_j))
                            nl.store(result[j_plus_1_idx, col_idx], nl.select(swap_needed, val_j, val_j_plus_1))
            
            else:  # Sort along rows (dim == 1)
                for row in range(shape[0]):
                    row_idx = nl.full((), row, dtype=nl.int32)
                    
                    # Bubble sort this row
                    for i in range(shape[1]):
                        for j in range(shape[1] - i - 1):
                            # Load adjacent elements
                            j_idx = nl.full((), j, dtype=nl.int32)
                            j_plus_1_idx = nl.full((), j+1, dtype=nl.int32)
                            
                            val_j = nl.load(result[row_idx, j_idx])
                            val_j_plus_1 = nl.load(result[row_idx, j_plus_1_idx])
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(val_j, val_j_plus_1)
                            
                            # Conditional swap
                            nl.store(result[row_idx, j_idx], nl.select(swap_needed, val_j_plus_1, val_j))
                            nl.store(result[row_idx, j_plus_1_idx], nl.select(swap_needed, val_j, val_j_plus_1))
    
    return result