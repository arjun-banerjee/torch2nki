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
    size = shape[dim]
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to result
    # We need to handle tensors of all sizes using proper tiling
    if len(shape) == 1:
        # For 1D tensor, sort the entire array
        trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
        
        # Copy input to result first
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < shape[0]))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < shape[0]))
        
        # Bubble sort implementation for 1D tensor
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements
                j_idx = nl.arange(size - 1)
                j_mask = j_idx == j
                
                elem1 = nl.load(result[j_idx], mask=j_mask)
                elem2 = nl.load(result[j_idx + 1], mask=j_mask)
                
                # Compare and swap if necessary
                swap_needed = nl.greater(elem1, elem2)
                if_greater = nl.where(swap_needed, elem2, elem1)
                if_less = nl.where(swap_needed, elem1, elem2)
                
                # Store back
                nl.store(result[j_idx], value=if_greater, mask=j_mask)
                nl.store(result[j_idx + 1], value=if_less, mask=j_mask)
    
    elif len(shape) == 2:
        # For 2D tensor, sort along specified dimension
        if dim == 0:
            # Sort along first dimension (columns)
            trip_count_f = math.ceil(shape[1] / nl.tile_size.pmax)
            
            # Copy input to result first
            for f in nl.affine_range(trip_count_f):
                i_f = f * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                i_p = nl.arange(shape[0])[:, None]
                
                # Load input data
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_f < shape[1]))
                
                # Store to result
                nl.store(result[i_p, i_f], value=in_tile, mask=(i_f < shape[1]))
            
            # Process each column separately
            for f in nl.affine_range(shape[1]):
                # Bubble sort implementation for each column
                for i in nl.affine_range(shape[0]):
                    for j in nl.affine_range(shape[0] - 1):
                        # Load adjacent elements
                        j_idx = nl.arange(shape[0] - 1)
                        j_mask = j_idx == j
                        
                        elem1 = nl.load(result[j_idx, f], mask=j_mask)
                        elem2 = nl.load(result[j_idx + 1, f], mask=j_mask)
                        
                        # Compare and swap if necessary
                        swap_needed = nl.greater(elem1, elem2)
                        if_greater = nl.where(swap_needed, elem2, elem1)
                        if_less = nl.where(swap_needed, elem1, elem2)
                        
                        # Store back
                        nl.store(result[j_idx, f], value=if_greater, mask=j_mask)
                        nl.store(result[j_idx + 1, f], value=if_less, mask=j_mask)
        
        else:  # dim == 1
            # Sort along second dimension (rows)
            trip_count_p = math.ceil(shape[0] / nl.tile_size.pmax)
            
            # Copy input to result first
            for p in nl.affine_range(trip_count_p):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(shape[1])[None, :]
                
                # Load input data
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < shape[0]))
                
                # Store to result
                nl.store(result[i_p, i_f], value=in_tile, mask=(i_p < shape[0]))
            
            # Process each row separately
            for p in nl.affine_range(shape[0]):
                # Bubble sort implementation for each row
                for i in nl.affine_range(shape[1]):
                    for j in nl.affine_range(shape[1] - 1):
                        # Load adjacent elements
                        j_idx = nl.arange(shape[1] - 1)
                        j_mask = j_idx == j
                        
                        elem1 = nl.load(result[p, j_idx], mask=j_mask)
                        elem2 = nl.load(result[p, j_idx + 1], mask=j_mask)
                        
                        # Compare and swap if necessary
                        swap_needed = nl.greater(elem1, elem2)
                        if_greater = nl.where(swap_needed, elem2, elem1)
                        if_less = nl.where(swap_needed, elem1, elem2)
                        
                        # Store back
                        nl.store(result[p, j_idx], value=if_greater, mask=j_mask)
                        nl.store(result[p, j_idx + 1], value=if_less, mask=j_mask)
    
    return result