from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if ndim == 1:
        size = shape[0]
        # Copy input to result
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start = i * nl.tile_size.pmax
            indices = start + nl.arange(nl.tile_size.pmax)
            data = nl.load(a_tensor[indices], mask=(indices < size))
            nl.store(result[indices], data, mask=(indices < size))
            
        # Bubble sort implementation for 1D tensor
        for i in range(size):
            for j in range(size - i - 1):
                # Load adjacent elements
                idx1 = nl.arange(1)
                idx2 = nl.arange(1) + 1
                
                # Mask for valid indices
                mask1 = (j < size - i - 1)
                mask2 = (j + 1 < size - i - 1)
                
                val1 = nl.load(result[j:j+1])
                val2 = nl.load(result[j+1:j+2])
                
                # Swap if val1 > val2
                cond = nl.greater(val1, val2)
                temp1 = nl.where(cond, val2, val1)
                temp2 = nl.where(cond, val1, val2)
                
                # Store back
                nl.store(result[j:j+1], temp1)
                nl.store(result[j+1:j+2], temp2)
                
    # Handle 2D tensor case
    elif ndim == 2:
        rows, cols = shape
        
        if dim == 0:  # Sort along rows
            # First copy input to result
            for i in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                start_row = i * nl.tile_size.pmax
                row_indices = start_row + nl.arange(nl.tile_size.pmax)[:, None]
                col_indices = nl.arange(cols)[None, :]
                
                data = nl.load(a_tensor[row_indices, col_indices], mask=(row_indices < rows))
                nl.store(result[row_indices, col_indices], data, mask=(row_indices < rows))
                
            # Sort each column independently
            for j in range(cols):
                for i in range(rows):
                    for k in range(rows - i - 1):
                        # Load adjacent elements
                        val1 = nl.load(result[k:k+1, j:j+1])
                        val2 = nl.load(result[k+1:k+2, j:j+1])
                        
                        # Swap if val1 > val2
                        cond = nl.greater(val1, val2)
                        temp1 = nl.where(cond, val2, val1)
                        temp2 = nl.where(cond, val1, val2)
                        
                        # Store back
                        nl.store(result[k:k+1, j:j+1], temp1)
                        nl.store(result[k+1:k+2, j:j+1], temp2)
                        
        else:  # Sort along columns (dim=1)
            # First copy input to result
            for i in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                start_row = i * nl.tile_size.pmax
                row_indices = start_row + nl.arange(nl.tile_size.pmax)[:, None]
                col_indices = nl.arange(cols)[None, :]
                
                data = nl.load(a_tensor[row_indices, col_indices], mask=(row_indices < rows))
                nl.store(result[row_indices, col_indices], data, mask=(row_indices < rows))
                
            # Sort each row independently
            for i in range(rows):
                for j in range(cols):
                    for k in range(cols - j - 1):
                        # Load adjacent elements
                        val1 = nl.load(result[i:i+1, k:k+1])
                        val2 = nl.load(result[i:i+1, k+1:k+2])
                        
                        # Swap if val1 > val2
                        cond = nl.greater(val1, val2)
                        temp1 = nl.where(cond, val2, val1)
                        temp2 = nl.where(cond, val1, val2)
                        
                        # Store back
                        nl.store(result[i:i+1, k:k+1], temp1)
                        nl.store(result[i:i+1, k+1:k+2], temp2)
    
    # Handle higher dimensional tensors
    else:
        # Calculate number of elements to process based on dim
        dim_size = shape[dim]
        
        # Calculate stride and total size
        stride = 1
        for i in range(dim + 1, ndim):
            stride *= shape[i]
            
        outer_size = 1
        for i in range(dim):
            outer_size *= shape[i]
            
        # Loop through each outer dimension slice and sort
        for i in range(outer_size):
            for j in range(dim_size):
                for k in range(dim_size - j - 1):
                    # Calculate indices for the two adjacent elements to compare
                    idx1 = i * dim_size * stride + k * stride
                    idx2 = i * dim_size * stride + (k + 1) * stride
                    
                    # Handle as 1D for simplicity
                    flat_idx1 = nl.arange(stride) + idx1
                    flat_idx2 = nl.arange(stride) + idx2
                    
                    # Load the elements
                    val1 = nl.load(result.reshape(-1)[flat_idx1])
                    val2 = nl.load(result.reshape(-1)[flat_idx2])
                    
                    # Compare the first elements (assuming they determine the order)
                    cond = nl.greater(val1[0:1], val2[0:1])
                    
                    # Swap if needed
                    temp1 = nl.where(cond, val2, val1)
                    temp2 = nl.where(cond, val1, val2)
                    
                    # Store back
                    nl.store(result.reshape(-1)[flat_idx1], temp1)
                    nl.store(result.reshape(-1)[flat_idx2], temp2)
    
    return result