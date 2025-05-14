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
    
    # Copy input tensor to result tensor initially
    # For 1D tensor
    if ndim == 1:
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(size / max_tile_size)
        
        # First, copy the input to result
        for i in nl.affine_range(trip_count):
            start = i * max_tile_size
            indices = nl.arange(max_tile_size)
            in_tile = nl.load(a_tensor[start + indices], mask=(start + indices < size))
            nl.store(result[start + indices], value=in_tile, mask=(start + indices < size))
        
        # Bubble sort implementation
        for i in range(size):
            for j in nl.affine_range(trip_count):
                start = j * max_tile_size
                indices = nl.arange(max_tile_size)
                valid_indices = start + indices
                
                # Load current segment
                segment = nl.load(result[valid_indices], mask=(valid_indices < size))
                
                # For each valid position except the last one in the segment
                for k in range(max_tile_size - 1):
                    if start + k + 1 >= size:
                        break
                        
                    # Compare adjacent elements
                    curr = segment[k]
                    next_val = segment[k + 1]
                    
                    # Swap if needed
                    condition = nl.greater(curr, next_val)
                    segment[k] = nl.where(condition, next_val, curr)
                    segment[k + 1] = nl.where(condition, curr, next_val)
                
                # Store the updated segment
                nl.store(result[valid_indices], value=segment, mask=(valid_indices < size))
                
                # Handle boundary between segments
                if j < trip_count - 1 and start + max_tile_size < size:
                    # Load last element of current segment and first element of next segment
                    last_elem = nl.load(result[start + max_tile_size - 1])
                    next_start = (j + 1) * max_tile_size
                    first_elem = nl.load(result[next_start])
                    
                    # Compare and swap if needed
                    if nl.greater(last_elem, first_elem):
                        nl.store(result[start + max_tile_size - 1], value=first_elem)
                        nl.store(result[next_start], value=last_elem)
    
    # For 2D tensor
    elif ndim == 2:
        rows, cols = shape
        
        # If sorting along dimension 0 (rows)
        if dim == 0:
            max_tile_size = nl.tile_size.pmax
            for c in nl.affine_range(cols):
                # Copy column to result
                for i in nl.affine_range(math.ceil(rows / max_tile_size)):
                    start = i * max_tile_size
                    indices = nl.arange(max_tile_size)
                    valid_indices = start + indices
                    in_tile = nl.load(a_tensor[valid_indices, c], mask=(valid_indices < rows))
                    nl.store(result[valid_indices, c], value=in_tile, mask=(valid_indices < rows))
                
                # Sort each column
                for i in range(rows):
                    for j in nl.affine_range(math.ceil(rows / max_tile_size)):
                        start = j * max_tile_size
                        indices = nl.arange(max_tile_size)
                        valid_indices = start + indices
                        
                        # Load current segment of the column
                        segment = nl.load(result[valid_indices, c], mask=(valid_indices < rows))
                        
                        # For each valid position except the last one in the segment
                        for k in range(max_tile_size - 1):
                            if start + k + 1 >= rows:
                                break
                                
                            # Compare adjacent elements
                            curr = segment[k]
                            next_val = segment[k + 1]
                            
                            # Swap if needed
                            condition = nl.greater(curr, next_val)
                            segment[k] = nl.where(condition, next_val, curr)
                            segment[k + 1] = nl.where(condition, curr, next_val)
                        
                        # Store the updated segment
                        nl.store(result[valid_indices, c], value=segment, mask=(valid_indices < rows))
                        
                        # Handle boundary between segments
                        if j < math.ceil(rows / max_tile_size) - 1 and start + max_tile_size < rows:
                            # Load last element of current segment and first element of next segment
                            last_elem = nl.load(result[start + max_tile_size - 1, c])
                            next_start = (j + 1) * max_tile_size
                            first_elem = nl.load(result[next_start, c])
                            
                            # Compare and swap if needed
                            if nl.greater(last_elem, first_elem):
                                nl.store(result[start + max_tile_size - 1, c], value=first_elem)
                                nl.store(result[next_start, c], value=last_elem)
        
        # If sorting along dimension 1 (columns)
        else:  # dim == 1
            max_tile_size = nl.tile_size.pmax
            for r in nl.affine_range(rows):
                # Copy row to result
                for i in nl.affine_range(math.ceil(cols / max_tile_size)):
                    start = i * max_tile_size
                    indices = nl.arange(max_tile_size)
                    valid_indices = start + indices
                    in_tile = nl.load(a_tensor[r, valid_indices], mask=(valid_indices < cols))
                    nl.store(result[r, valid_indices], value=in_tile, mask=(valid_indices < cols))
                
                # Sort each row
                for i in range(cols):
                    for j in nl.affine_range(math.ceil(cols / max_tile_size)):
                        start = j * max_tile_size
                        indices = nl.arange(max_tile_size)
                        valid_indices = start + indices
                        
                        # Load current segment of the row
                        segment = nl.load(result[r, valid_indices], mask=(valid_indices < cols))
                        
                        # For each valid position except the last one in the segment
                        for k in range(max_tile_size - 1):
                            if start + k + 1 >= cols:
                                break
                                
                            # Compare adjacent elements
                            curr = segment[k]
                            next_val = segment[k + 1]
                            
                            # Swap if needed
                            condition = nl.greater(curr, next_val)
                            segment[k] = nl.where(condition, next_val, curr)
                            segment[k + 1] = nl.where(condition, curr, next_val)
                        
                        # Store the updated segment
                        nl.store(result[r, valid_indices], value=segment, mask=(valid_indices < cols))
                        
                        # Handle boundary between segments
                        if j < math.ceil(cols / max_tile_size) - 1 and start + max_tile_size < cols:
                            # Load last element of current segment and first element of next segment
                            last_elem = nl.load(result[r, start + max_tile_size - 1])
                            next_start = (j + 1) * max_tile_size
                            first_elem = nl.load(result[r, next_start])
                            
                            # Compare and swap if needed
                            if nl.greater(last_elem, first_elem):
                                nl.store(result[r, start + max_tile_size - 1], value=first_elem)
                                nl.store(result[r, next_start], value=last_elem)
    
    return result