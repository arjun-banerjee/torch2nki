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
    
    # First copy input tensor to result tensor (we'll sort in-place)
    if ndim == 1:
        # For 1D tensor, directly sort the entire tensor
        sort_dim_size = shape[0]
        max_tile_size = min(nl.tile_size.pmax, sort_dim_size)
        
        # Process in tiles
        trip_count = math.ceil(sort_dim_size / max_tile_size)
        
        for p in nl.affine_range(trip_count):
            # Calculate indices for current tile
            start_idx = p * max_tile_size
            
            # Create indices for loading data
            indices = start_idx + nl.arange(max_tile_size)
            
            # Load data with mask to handle boundary
            data_tile = nl.load(a_tensor[indices], mask=(indices < sort_dim_size))
            
            # Bubble sort within the tile
            for i in nl.affine_range(max_tile_size):
                for j in nl.affine_range(max_tile_size - 1):
                    # Compare adjacent elements
                    condition = nl.less(j + 1, max_tile_size - i)
                    mask = condition & (indices[j] < sort_dim_size) & (indices[j+1] < sort_dim_size)
                    
                    # Get values to compare
                    val_j = data_tile[j]
                    val_j_plus_1 = data_tile[j+1]
                    
                    # Check if swap is needed
                    swap_needed = nl.greater(val_j, val_j_plus_1)
                    
                    # Conditionally swap values
                    data_tile = nl.where(swap_needed & mask, 
                                         nl.where(nl.equal(nl.arange(max_tile_size), j), 
                                                  val_j_plus_1, 
                                                  nl.where(nl.equal(nl.arange(max_tile_size), j+1), 
                                                           val_j, 
                                                           data_tile)),
                                         data_tile)
            
            # Store the sorted data back
            nl.store(result[indices], value=data_tile, mask=(indices < sort_dim_size))
    
    elif dim == ndim - 1:
        # For sorting along the last dimension
        outer_dims_size = 1
        for i in range(ndim - 1):
            outer_dims_size *= shape[i]
        
        sort_dim_size = shape[dim]
        max_tile_size = min(nl.tile_size.pmax, sort_dim_size)
        
        # Process each outer dimension slice
        for outer_idx in nl.affine_range(outer_dims_size):
            # Calculate multi-dimensional indices for outer dimensions
            outer_indices = []
            remaining = outer_idx
            for i in range(ndim - 1):
                dim_size = shape[i]
                idx = remaining // math.prod([shape[j] for j in range(i+1, ndim-1)]) if i < ndim-2 else remaining
                remaining = remaining % math.prod([shape[j] for j in range(i+1, ndim-1)]) if i < ndim-2 else 0
                outer_indices.append(idx)
            
            # Load entire slice to sort
            slice_data = nl.zeros((sort_dim_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Load data in tiles
            for p in nl.affine_range(math.ceil(sort_dim_size / max_tile_size)):
                start_idx = p * max_tile_size
                indices = start_idx + nl.arange(max_tile_size)
                
                # Create index tuple for loading
                idx_tuple = []
                for i in range(ndim - 1):
                    idx_tuple.append(outer_indices[i])
                idx_tuple.append(indices)
                
                # Load data with mask
                tile_data = nl.load(a_tensor[tuple(idx_tuple)], mask=(indices < sort_dim_size))
                
                # Store into temporary buffer
                slice_data[indices] = tile_data
            
            # Bubble sort the entire slice
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    # Compare adjacent elements
                    j_val = slice_data[j]
                    j_next_val = slice_data[j+1]
                    
                    # Check if swap is needed
                    swap_needed = nl.greater(j_val, j_next_val)
                    
                    # Conditionally swap
                    temp = j_val
                    slice_data = nl.where(swap_needed & (j < sort_dim_size - i - 1),
                                         nl.where(nl.equal(nl.arange(sort_dim_size), j),
                                                 j_next_val,
                                                 nl.where(nl.equal(nl.arange(sort_dim_size), j+1),
                                                          temp,
                                                          slice_data)),
                                         slice_data)
            
            # Store sorted data back to result
            for p in nl.affine_range(math.ceil(sort_dim_size / max_tile_size)):
                start_idx = p * max_tile_size
                indices = start_idx + nl.arange(max_tile_size)
                
                # Create index tuple for storing
                idx_tuple = []
                for i in range(ndim - 1):
                    idx_tuple.append(outer_indices[i])
                idx_tuple.append(indices)
                
                # Get tile data from sorted slice
                tile_data = slice_data[indices]
                
                # Store with mask
                nl.store(result[tuple(idx_tuple)], value=tile_data, mask=(indices < sort_dim_size))
    
    else:
        # For sorting along any other dimension, we need to handle it differently
        # This is a simplified implementation for 2D tensors sorting along dim 0
        if ndim == 2 and dim == 0:
            rows, cols = shape
            
            # Sort columns independently
            for col in nl.affine_range(cols):
                # Load the entire column
                col_data = nl.zeros((rows,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                
                # Load column data in tiles
                for p in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                    start_idx = p * nl.tile_size.pmax
                    indices = start_idx + nl.arange(nl.tile_size.pmax)
                    
                    # Load data with mask
                    tile_data = nl.load(a_tensor[indices, col], mask=(indices < rows))
                    
                    # Store into temporary buffer
                    col_data[indices] = tile_data
                
                # Bubble sort the column
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        # Compare adjacent elements
                        j_val = col_data[j]
                        j_next_val = col_data[j+1]
                        
                        # Check if swap is needed
                        swap_needed = nl.greater(j_val, j_next_val)
                        
                        # Conditionally swap
                        temp = j_val
                        col_data = nl.where(swap_needed & (j < rows - i - 1),
                                          nl.where(nl.equal(nl.arange(rows), j),
                                                  j_next_val,
                                                  nl.where(nl.equal(nl.arange(rows), j+1),
                                                          temp,
                                                          col_data)),
                                          col_data)
                
                # Store sorted column back
                for p in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                    start_idx = p * nl.tile_size.pmax
                    indices = start_idx + nl.arange(nl.tile_size.pmax)
                    
                    # Get tile data from sorted column
                    tile_data = col_data[indices]
                    
                    # Store with mask
                    nl.store(result[indices, col], value=tile_data, mask=(indices < rows))
    
    return result