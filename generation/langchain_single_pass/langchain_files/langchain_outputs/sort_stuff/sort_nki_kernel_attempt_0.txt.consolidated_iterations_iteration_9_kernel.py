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
        # Copy input to result first
        for i in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
            start = i * nl.tile_size.pmax
            end = min(start + nl.tile_size.pmax, shape[0])
            length = end - start
            
            # Create indices for this tile
            indices = nl.arange(length)
            
            # Load input data
            data = nl.load(a_tensor[start:end])
            
            # Bubble sort implementation for 1D
            for i in nl.affine_range(length):
                for j in nl.affine_range(length - 1):
                    # Load current and next values
                    curr_val = data[j]
                    next_val = data[j+1]
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_val, next_val)
                    
                    # Update values using where to perform conditional swap
                    data = nl.store(data, nl.where(swap_needed, next_val, curr_val), indices=[j])
                    data = nl.store(data, nl.where(swap_needed, curr_val, next_val), indices=[j+1])
            
            # Store sorted result
            nl.store(result[start:end], data)
    
    # Handle 2D or higher tensor case
    else:
        # Determine sizes for processing
        sort_dim_size = shape[dim]
        
        # Handle based on which dimension to sort
        if dim == 0:
            # Sort along first dimension
            for i in nl.affine_range(shape[1]):
                # Extract column
                col_data = nl.zeros((shape[0],), dtype=a_tensor.dtype, buffer=nl.sbuf)
                
                # Load column data in chunks if needed
                for j in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
                    start = j * nl.tile_size.pmax
                    end = min(start + nl.tile_size.pmax, shape[0])
                    length = end - start
                    
                    # Load chunk of column
                    chunk = nl.load(a_tensor[start:end, i])
                    
                    # Store into column buffer
                    for k in nl.affine_range(length):
                        col_data = nl.store(col_data, chunk[k], indices=[start + k])
                
                # Sort column using bubble sort
                for j in nl.affine_range(shape[0]):
                    for k in nl.affine_range(shape[0] - 1):
                        # Load current and next values
                        curr_val = nl.load(col_data[k:k+1])
                        next_val = nl.load(col_data[k+1:k+2])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_val, next_val)
                        
                        # Update values using where to perform conditional swap
                        tmp_curr = nl.where(swap_needed, next_val, curr_val)
                        tmp_next = nl.where(swap_needed, curr_val, next_val)
                        
                        col_data = nl.store(col_data, tmp_curr, indices=[k])
                        col_data = nl.store(col_data, tmp_next, indices=[k+1])
                
                # Store sorted column back to result
                for j in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
                    start = j * nl.tile_size.pmax
                    end = min(start + nl.tile_size.pmax, shape[0])
                    length = end - start
                    
                    # Extract chunk from sorted column
                    chunk = nl.zeros((length,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    for k in nl.affine_range(length):
                        chunk = nl.store(chunk, nl.load(col_data[start+k:start+k+1]), indices=[k])
                    
                    # Store chunk to result
                    nl.store(result[start:end, i], chunk)
        
        else:  # dim == 1 or higher
            # Sort along second or higher dimension
            # For simplicity, we'll handle the dim == 1 case directly here
            if dim == 1 and ndim == 2:
                for i in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
                    start = i * nl.tile_size.pmax
                    end = min(start + nl.tile_size.pmax, shape[0])
                    length = end - start
                    
                    # Load row data
                    row_data = nl.load(a_tensor[start:end, :])
                    
                    # Sort each row using bubble sort
                    for j in nl.affine_range(length):  # For each row in this batch
                        for k in nl.affine_range(shape[1]):  # Outer loop of bubble sort
                            for l in nl.affine_range(shape[1] - 1):  # Inner loop of bubble sort
                                # Compare and swap if needed
                                curr_val = row_data[j, l]
                                next_val = row_data[j, l+1]
                                
                                swap_needed = nl.greater(curr_val, next_val)
                                
                                # Update values using where to perform conditional swap
                                tmp_curr = nl.where(swap_needed, next_val, curr_val)
                                tmp_next = nl.where(swap_needed, curr_val, next_val)
                                
                                row_data = nl.store(row_data, tmp_curr, indices=[j, l])
                                row_data = nl.store(row_data, tmp_next, indices=[j, l+1])
                    
                    # Store sorted rows
                    nl.store(result[start:end, :], row_data)
            
            else:
                # Copy input to result for dimensions we don't handle yet
                # In a real implementation, this would need to handle higher dimensions
                for i in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
                    start = i * nl.tile_size.pmax
                    end = min(start + nl.tile_size.pmax, shape[0])
                    
                    if ndim == 2:
                        data = nl.load(a_tensor[start:end, :])
                        nl.store(result[start:end, :], data)
                    else:
                        # For higher dimensions, copy as is
                        # This is a placeholder; in reality, you'd need to handle this case
                        data = nl.load(a_tensor[start:end, ...])
                        nl.store(result[start:end, ...], data)
    
    return result