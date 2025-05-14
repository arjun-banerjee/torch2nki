from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension indexing
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to result tensor
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    
    # For 1D tensor case
    if ndims == 1:
        size = tensor_shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process in tiles to respect hardware limitations
        for start in nl.affine_range(math.ceil(size / max_tile_size)):
            start_idx = start * max_tile_size
            # Load a tile of data
            indices = nl.arange(max_tile_size)
            tile_data = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < size))
            
            # Bubble sort on the loaded tile
            for i in nl.affine_range(max_tile_size):
                for j in nl.affine_range(max_tile_size - 1):
                    val_j = nl.copy(tile_data[j])
                    val_j1 = nl.copy(tile_data[j + 1])
                    
                    # Compare and swap if needed
                    cond = nl.greater(val_j, val_j1)
                    tile_data[j] = nl.where(cond, val_j1, val_j)
                    tile_data[j + 1] = nl.where(cond, val_j, val_j1)
            
            # Store the sorted tile back
            nl.store(result[start_idx + indices], tile_data, mask=(start_idx + indices < size))
        
        return result
    
    # For multi-dimensional tensors
    # We need to process each slice along the sort dimension
    
    # Calculate number of slices
    slice_size = tensor_shape[dim]
    num_slices = 1
    for i in range(ndims):
        if i != dim:
            num_slices *= tensor_shape[i]
    
    # Process each slice
    slice_indices = []
    for i in range(ndims):
        if i == dim:
            slice_indices.append(slice(None))
        else:
            slice_indices.append(0)
    
    # First, copy the entire tensor to result
    for start in nl.affine_range(math.ceil(num_slices / nl.tile_size.pmax)):
        start_idx = start * nl.tile_size.pmax
        # Process a batch of slices
        for slice_idx in nl.affine_range(min(nl.tile_size.pmax, num_slices - start_idx)):
            current_slice_idx = start_idx + slice_idx
            
            # Convert flat index to multi-dimensional indices
            multi_idx = [0] * ndims
            temp_idx = current_slice_idx
            for i in range(ndims-1, -1, -1):
                if i != dim:
                    multi_idx[i] = temp_idx % tensor_shape[i]
                    temp_idx //= tensor_shape[i]
            
            # Load the slice
            slice_data = nl.zeros((slice_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Load data for this slice
            for k in nl.affine_range(math.ceil(slice_size / nl.tile_size.pmax)):
                k_start = k * nl.tile_size.pmax
                k_indices = nl.arange(nl.tile_size.pmax)
                
                # Create indices for loading
                load_indices = [0] * ndims
                for i in range(ndims):
                    if i == dim:
                        load_indices[i] = k_start + k_indices
                    else:
                        load_indices[i] = multi_idx[i]
                
                # Load data for this part of the slice
                part_data = nl.load(a_tensor[tuple(load_indices)], 
                                   mask=(k_start + k_indices < slice_size))
                
                # Store into our slice buffer
                nl.store(slice_data[k_start + k_indices], part_data, 
                        mask=(k_start + k_indices < slice_size))
            
            # Perform bubble sort on the slice
            for i in nl.affine_range(slice_size):
                for j in nl.affine_range(slice_size - 1):
                    val_j = nl.copy(slice_data[j])
                    val_j1 = nl.copy(slice_data[j + 1])
                    
                    # Compare and swap if needed
                    cond = nl.greater(val_j, val_j1)
                    slice_data[j] = nl.where(cond, val_j1, val_j)
                    slice_data[j + 1] = nl.where(cond, val_j, val_j1)
            
            # Store the sorted slice back to result
            for k in nl.affine_range(math.ceil(slice_size / nl.tile_size.pmax)):
                k_start = k * nl.tile_size.pmax
                k_indices = nl.arange(nl.tile_size.pmax)
                
                # Create indices for storing
                store_indices = [0] * ndims
                for i in range(ndims):
                    if i == dim:
                        store_indices[i] = k_start + k_indices
                    else:
                        store_indices[i] = multi_idx[i]
                
                # Load from our slice buffer
                part_data = nl.load(slice_data[k_start + k_indices], 
                                   mask=(k_start + k_indices < slice_size))
                
                # Store to the result tensor
                nl.store(result[tuple(store_indices)], part_data, 
                        mask=(k_start + k_indices < slice_size))
    
    return result