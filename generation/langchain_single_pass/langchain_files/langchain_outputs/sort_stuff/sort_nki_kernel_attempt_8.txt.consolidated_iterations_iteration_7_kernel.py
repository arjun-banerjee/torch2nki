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
    
    # Get tensor shape
    tensor_shape = a_tensor.shape
    
    # For simplicity, first copy input to result
    if len(tensor_shape) == 1:
        # Handle 1D tensor case
        size = tensor_shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            offset = p * nl.tile_size.pmax
            i_p = nl.arange(nl.tile_size.pmax)
            
            # Load input data with masking for boundary
            input_tile = nl.load(a_tensor[offset + i_p], mask=(offset + i_p < size))
            
            # Store to result
            nl.store(result[offset + i_p], value=input_tile, mask=(offset + i_p < size))
    else:
        # Handle multi-dimensional tensor case
        # We'll implement for 2D tensors but the approach generalizes
        sz_p, sz_f = tensor_shape[0], tensor_shape[1]
        trip_count_p = math.ceil(sz_p / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count_p):
            offset_p = p * nl.tile_size.pmax
            i_p = nl.arange(nl.tile_size.pmax)[:, None]
            i_f = nl.arange(sz_f)[None, :]
            
            # Load input data with masking for boundary
            input_tile = nl.load(a_tensor[offset_p + i_p, i_f], mask=(offset_p + i_p < sz_p))
            
            # Store to result
            nl.store(result[offset_p + i_p, i_f], value=input_tile, mask=(offset_p + i_p < sz_p))
    
    # Now perform bubble sort on the result tensor along the specified dimension
    if len(tensor_shape) == 1:
        # Sort 1D tensor
        size = tensor_shape[0]
        # Bubble sort implementation
        for i in nl.affine_range(size):
            # Last i elements are already in place
            for j in nl.affine_range(size - 1):
                # We need to handle hardware limitations by processing in tiles
                trip_count = math.ceil((size - 1) / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    offset = p * nl.tile_size.pmax
                    i_p = nl.arange(nl.tile_size.pmax)
                    
                    # Calculate actual indices with masking
                    indices = offset + i_p
                    valid_mask = (indices < (size - 1 - i)) & (indices >= j)
                    
                    # Load adjacent elements
                    a = nl.load(result[indices], mask=valid_mask)
                    b = nl.load(result[indices + 1], mask=valid_mask)
                    
                    # Compare and swap if needed
                    condition = nl.greater(a, b)
                    new_a = nl.where(condition, b, a)
                    new_b = nl.where(condition, a, b)
                    
                    # Store back
                    nl.store(result[indices], value=new_a, mask=valid_mask)
                    nl.store(result[indices + 1], value=new_b, mask=valid_mask)
    else:
        # Sort multi-dimensional tensor along specified dimension
        if dim == 0:
            # Sort along first dimension
            sz_p, sz_f = tensor_shape[0], tensor_shape[1]
            
            # For each column
            for col in nl.affine_range(sz_f):
                # Bubble sort implementation for this column
                for i in nl.affine_range(sz_p):
                    # Last i elements are already in place
                    trip_count = math.ceil((sz_p - 1) / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        offset = p * nl.tile_size.pmax
                        i_p = nl.arange(nl.tile_size.pmax)
                        
                        # Calculate actual indices with masking
                        indices = offset + i_p
                        valid_mask = (indices < (sz_p - 1 - i))
                        
                        # Load adjacent elements
                        a = nl.load(result[indices, col], mask=valid_mask)
                        b = nl.load(result[indices + 1, col], mask=valid_mask)
                        
                        # Compare and swap if needed
                        condition = nl.greater(a, b)
                        new_a = nl.where(condition, b, a)
                        new_b = nl.where(condition, a, b)
                        
                        # Store back
                        nl.store(result[indices, col], value=new_a, mask=valid_mask)
                        nl.store(result[indices + 1, col], value=new_b, mask=valid_mask)
        else:
            # Sort along second dimension (dim == 1)
            sz_p, sz_f = tensor_shape[0], tensor_shape[1]
            
            # For each row
            for row in nl.affine_range(sz_p):
                # Bubble sort implementation for this row
                for i in nl.affine_range(sz_f):
                    # Last i elements are already in place
                    trip_count = math.ceil((sz_f - 1) / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        offset = p * nl.tile_size.pmax
                        i_p = nl.arange(nl.tile_size.pmax)
                        
                        # Calculate actual indices with masking
                        indices = offset + i_p
                        valid_mask = (indices < (sz_f - 1 - i))
                        
                        # Load adjacent elements
                        a = nl.load(result[row, indices], mask=valid_mask)
                        b = nl.load(result[row, indices + 1], mask=valid_mask)
                        
                        # Compare and swap if needed
                        condition = nl.greater(a, b)
                        new_a = nl.where(condition, b, a)
                        new_b = nl.where(condition, a, b)
                        
                        # Store back
                        nl.store(result[row, indices], value=new_a, mask=valid_mask)
                        nl.store(result[row, indices + 1], value=new_b, mask=valid_mask)
    
    return result