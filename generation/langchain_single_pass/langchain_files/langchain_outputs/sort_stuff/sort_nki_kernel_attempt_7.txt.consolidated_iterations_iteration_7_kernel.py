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
        # For 1D tensor, sort the entire array
        size = shape[0]
        
        # Calculate the number of tiles needed
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # Copy input to result first
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], value=x_tile, mask=(i_p < size))
        
        # Selection sort algorithm
        for i in nl.static_range(size - 1):
            # Find minimum element in the unsorted part
            min_val = nl.load(result[i])
            min_idx = i
            
            for j in nl.static_range(i + 1, size):
                curr_val = nl.load(result[j])
                # Update min_val and min_idx if current element is smaller
                is_smaller = nl.less(curr_val, min_val)
                min_val = nl.where(is_smaller, curr_val, min_val)
                min_idx = nl.where(is_smaller, j, min_idx)
            
            # Swap the found minimum element with the first element
            if min_idx != i:
                temp = nl.load(result[i])
                nl.store(result[i], value=min_val)
                nl.store(result[min_idx], value=temp)
    
    elif ndim == 2:
        # For 2D tensor, sort along specified dimension
        sz_dim0, sz_dim1 = shape
        
        if dim == 0:
            # Sort along rows (dimension 0)
            # Process in tiles to respect hardware limitations
            trip_count = math.ceil(sz_dim1 / nl.tile_size.pmax)
            
            # Copy input to result first
            for p in nl.affine_range(trip_count):
                i_f = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                i_p = nl.arange(sz_dim0)[:, None]
                
                x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_f < sz_dim1))
                nl.store(result[i_p, i_f], value=x_tile, mask=(i_f < sz_dim1))
            
            # Sort each column independently
            for col in nl.static_range(sz_dim1):
                for i in nl.static_range(sz_dim0 - 1):
                    # Find minimum element in the unsorted part of this column
                    min_val = nl.load(result[i, col])
                    min_idx = i
                    
                    for j in nl.static_range(i + 1, sz_dim0):
                        curr_val = nl.load(result[j, col])
                        # Update min_val and min_idx if current element is smaller
                        is_smaller = nl.less(curr_val, min_val)
                        min_val = nl.where(is_smaller, curr_val, min_val)
                        min_idx = nl.where(is_smaller, j, min_idx)
                    
                    # Swap the found minimum element with the first element
                    if min_idx != i:
                        temp = nl.load(result[i, col])
                        nl.store(result[i, col], value=min_val)
                        nl.store(result[min_idx, col], value=temp)
        
        else:  # dim == 1
            # Sort along columns (dimension 1)
            # Process in tiles to respect hardware limitations
            trip_count = math.ceil(sz_dim0 / nl.tile_size.pmax)
            
            # Copy input to result first
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                i_f = nl.arange(sz_dim1)[None, :]
                
                x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_dim0))
                nl.store(result[i_p, i_f], value=x_tile, mask=(i_p < sz_dim0))
            
            # Sort each row independently
            for row in nl.static_range(sz_dim0):
                for i in nl.static_range(sz_dim1 - 1):
                    # Find minimum element in the unsorted part of this row
                    min_val = nl.load(result[row, i])
                    min_idx = i
                    
                    for j in nl.static_range(i + 1, sz_dim1):
                        curr_val = nl.load(result[row, j])
                        # Update min_val and min_idx if current element is smaller
                        is_smaller = nl.less(curr_val, min_val)
                        min_val = nl.where(is_smaller, curr_val, min_val)
                        min_idx = nl.where(is_smaller, j, min_idx)
                    
                    # Swap the found minimum element with the first element
                    if min_idx != i:
                        temp = nl.load(result[row, i])
                        nl.store(result[row, i], value=min_val)
                        nl.store(result[row, min_idx], value=temp)
    
    else:  # ndim > 2
        # For higher dimensions, we reshape the tensor to handle it as a 2D tensor
        # where the specified dimension is one of the dimensions and all other dimensions
        # are flattened into the other dimension
        
        # Copy input to result first
        flat_shape = 1
        for i in range(ndim):
            if i != dim:
                flat_shape *= shape[i]
        
        if dim == 0:
            # Sort along the first dimension
            sz_dim0 = shape[0]
            
            # Process in tiles to respect hardware limitations
            trip_count = math.ceil(flat_shape / nl.tile_size.pmax)
            
            # Copy input to result
            for p in nl.affine_range(trip_count):
                i_f = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                i_p = nl.arange(sz_dim0)[:, None]
                
                # Handle multi-dimensional indexing here by linearizing all dimensions except dim
                # This part is simplified and assumes we can load/store directly
                x_tile = nl.load(a_tensor[i_p, i_f % shape[1]], mask=(i_f < flat_shape))
                nl.store(result[i_p, i_f % shape[1]], value=x_tile, mask=(i_f < flat_shape))
        
        else:
            # Sort along a non-first dimension
            sz_dim = shape[dim]
            
            # Process in tiles to respect hardware limitations
            trip_count = math.ceil(flat_shape / nl.tile_size.pmax)
            
            # Copy input to result
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                i_f = nl.arange(sz_dim)[None, :]
                
                # Handle multi-dimensional indexing here by linearizing all dimensions except dim
                # This part is simplified and assumes we can load/store directly
                x_tile = nl.load(a_tensor[i_p % shape[0], i_f], mask=(i_p < flat_shape))
                nl.store(result[i_p % shape[0], i_f], value=x_tile, mask=(i_p < flat_shape))
    
    return result