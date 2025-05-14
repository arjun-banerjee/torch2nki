from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension indexing
    if dim < 0:
        dim = dim + len(a_tensor.shape)
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input tensor to result
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    
    # Special case for 1D tensor
    if ndims == 1:
        sort_dim_size = tensor_shape[0]
        
        # Calculate the number of tiles needed
        tile_size = min(nl.tile_size.pmax, sort_dim_size)
        trip_count = math.ceil(sort_dim_size / tile_size)
        
        # Copy input to result first
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            start_idx = p * tile_size
            # Create index array for current tile
            i_p = nl.arange(tile_size)
            
            # Load input data, with masking to handle boundary
            x_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < sort_dim_size))
            
            # Store to result
            nl.store(result[start_idx + i_p], value=x_tile, mask=(start_idx + i_p < sort_dim_size))
        
        # Bubble sort implementation
        for i in nl.affine_range(sort_dim_size):
            for j in nl.affine_range(sort_dim_size - 1):
                # Load adjacent elements
                idx1 = j
                idx2 = j + 1
                
                # Process in tiles if needed
                tile_idx1 = idx1 // tile_size
                offset_idx1 = idx1 % tile_size
                
                tile_idx2 = idx2 // tile_size
                offset_idx2 = idx2 % tile_size
                
                # Load elements
                elem1 = nl.load(result[idx1])
                elem2 = nl.load(result[idx2])
                
                # Compare and swap if needed
                swap_needed = nl.greater(elem1, elem2)
                
                # Create temporary values for swapping
                temp1 = nl.where(swap_needed, elem2, elem1)
                temp2 = nl.where(swap_needed, elem1, elem2)
                
                # Store back
                nl.store(result[idx1], value=temp1)
                nl.store(result[idx2], value=temp2)
    
    # Handle multi-dimensional tensors
    else:
        # Get the size of the dimension to sort along
        sort_dim_size = tensor_shape[dim]
        
        # First copy input to result
        # We need to process the tensor in tiles
        if dim == 0:  # Sorting along first dimension
            # Calculate free dimension size (product of all other dimensions)
            free_dim_size = 1
            for i in range(1, ndims):
                free_dim_size *= tensor_shape[i]
            
            # Calculate the number of tiles needed for partition dimension
            tile_p_size = min(nl.tile_size.pmax, tensor_shape[0])
            trip_count_p = math.ceil(tensor_shape[0] / tile_p_size)
            
            # Calculate the number of tiles needed for free dimension
            tile_f_size = min(nl.tile_size.fmax, free_dim_size)
            trip_count_f = math.ceil(free_dim_size / tile_f_size)
            
            # Copy input to result
            for p in nl.affine_range(trip_count_p):
                for f in nl.affine_range(trip_count_f):
                    start_p = p * tile_p_size
                    start_f = f * tile_f_size
                    
                    # Create index arrays
                    i_p = nl.arange(tile_p_size)[:, None]
                    i_f = nl.arange(tile_f_size)[None, :]
                    
                    # Reshape indices to match tensor dimensions
                    idx_p = start_p + i_p
                    
                    # Load data with masking
                    mask_p = idx_p < tensor_shape[0]
                    mask_f = (start_f + i_f) < free_dim_size
                    
                    # For multidimensional tensors, we need to convert linear indices to multi-dimensional
                    # This is complex in NKI, so we'll handle each case separately
                    if ndims == 2:
                        x_tile = nl.load(a_tensor[idx_p, start_f + i_f], mask=(mask_p & mask_f))
                        nl.store(result[idx_p, start_f + i_f], value=x_tile, mask=(mask_p & mask_f))
            
            # Bubble sort implementation for dim=0
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    for f in nl.affine_range(free_dim_size):
                        if ndims == 2:
                            # 2D tensor case
                            elem1 = nl.load(result[j, f])
                            elem2 = nl.load(result[j+1, f])
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(elem1, elem2)
                            
                            # Create temporary values for swapping
                            temp1 = nl.where(swap_needed, elem2, elem1)
                            temp2 = nl.where(swap_needed, elem1, elem2)
                            
                            # Store back
                            nl.store(result[j, f], value=temp1)
                            nl.store(result[j+1, f], value=temp2)
                            
        else:  # Sorting along dimension > 0
            # For simplicity, we'll handle the 2D case with dim=1
            if ndims == 2 and dim == 1:
                # Calculate the number of tiles needed
                tile_p_size = min(nl.tile_size.pmax, tensor_shape[0])
                trip_count_p = math.ceil(tensor_shape[0] / tile_p_size)
                
                tile_f_size = min(nl.tile_size.fmax, tensor_shape[1])
                
                # Copy input to result
                for p in nl.affine_range(trip_count_p):
                    start_p = p * tile_p_size
                    
                    # Create index arrays
                    i_p = nl.arange(tile_p_size)[:, None]
                    i_f = nl.arange(tensor_shape[1])[None, :]
                    
                    # Load data with masking
                    mask_p = (start_p + i_p) < tensor_shape[0]
                    
                    x_tile = nl.load(a_tensor[start_p + i_p, i_f], mask=mask_p)
                    nl.store(result[start_p + i_p, i_f], value=x_tile, mask=mask_p)
                
                # Bubble sort implementation for each row independently
                for p in nl.affine_range(trip_count_p):
                    start_p = p * tile_p_size
                    end_p = min(start_p + tile_p_size, tensor_shape[0])
                    
                    # For each row in the current tile
                    for row in nl.affine_range(end_p - start_p):
                        actual_row = start_p + row
                        
                        # Bubble sort for this row
                        for i in nl.affine_range(sort_dim_size):
                            for j in nl.affine_range(sort_dim_size - 1):
                                # Load adjacent elements
                                elem1 = nl.load(result[actual_row, j])
                                elem2 = nl.load(result[actual_row, j+1])
                                
                                # Compare and swap if needed
                                swap_needed = nl.greater(elem1, elem2)
                                
                                # Create temporary values for swapping
                                temp1 = nl.where(swap_needed, elem2, elem1)
                                temp2 = nl.where(swap_needed, elem1, elem2)
                                
                                # Store back
                                nl.store(result[actual_row, j], value=temp1)
                                nl.store(result[actual_row, j+1], value=temp2)
    
    return result