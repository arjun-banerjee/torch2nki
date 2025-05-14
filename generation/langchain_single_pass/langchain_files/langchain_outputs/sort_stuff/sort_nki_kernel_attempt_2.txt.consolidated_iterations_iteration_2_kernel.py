from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to the result
    # We need to process in tiles to handle large tensors
    
    # Calculate the total size of all dimensions except the sorting dimension
    outer_size = 1
    for i in range(ndim):
        if i != dim:
            outer_size *= shape[i]
    
    # Calculate number of tiles needed for outer dimensions
    trip_count = math.ceil(outer_size / nl.tile_size.pmax)
    
    # Get the size of the dimension to sort
    dim_size = shape[dim]
    
    # Create a function to get the indices for a flat index
    def get_indices(flat_idx, shape, sort_dim):
        indices = []
        temp_idx = flat_idx
        for i in range(len(shape) - 1, -1, -1):
            if i != sort_dim:
                idx_i = temp_idx % shape[i]
                temp_idx = temp_idx // shape[i]
                indices.insert(0, idx_i)
            else:
                indices.insert(sort_dim, 0)  # Placeholder for sort dimension
        return indices
    
    # Process the tensor in tiles
    for p in nl.affine_range(trip_count):
        # Calculate starting index for this tile
        start_idx = p * nl.tile_size.pmax
        
        # Process each element in the sorting dimension
        for i in range(dim_size):
            # Load the current slice of the tensor
            i_p = nl.arange(nl.tile_size.pmax)
            
            # Create indices for loading
            indices = []
            for d in range(ndim):
                if d == dim:
                    indices.append(i)
                else:
                    # For other dimensions, we need to calculate the appropriate indices
                    # based on the flat index p * nl.tile_size.pmax + i_p
                    indices.append((start_idx + i_p) % shape[d])
            
            # Load the slice
            slice_data = nl.load(a_tensor[tuple(indices)], mask=(start_idx + i_p < outer_size))
            
            # Store to result
            nl.store(result[tuple(indices)], slice_data, mask=(start_idx + i_p < outer_size))
    
    # Now perform bubble sort along the specified dimension
    for _ in range(dim_size - 1):
        for j in range(dim_size - 1):
            for p in nl.affine_range(trip_count):
                # Calculate starting index for this tile
                start_idx = p * nl.tile_size.pmax
                
                # Create indices for the current elements
                indices_curr = []
                for d in range(ndim):
                    if d == dim:
                        indices_curr.append(j)
                    else:
                        indices_curr.append((start_idx + nl.arange(nl.tile_size.pmax)) % shape[d])
                
                # Create indices for the next elements
                indices_next = []
                for d in range(ndim):
                    if d == dim:
                        indices_next.append(j + 1)
                    else:
                        indices_next.append((start_idx + nl.arange(nl.tile_size.pmax)) % shape[d])
                
                # Load current and next elements
                i_p = nl.arange(nl.tile_size.pmax)
                curr_data = nl.load(result[tuple(indices_curr)], mask=(start_idx + i_p < outer_size))
                next_data = nl.load(result[tuple(indices_next)], mask=(start_idx + i_p < outer_size))
                
                # Compare and swap if necessary
                swap_needed = nl.greater(curr_data, next_data)
                temp = nl.zeros(curr_data.shape, dtype=curr_data.dtype, buffer=nl.sbuf)
                
                # Where swap_needed is True, use next_data, otherwise use curr_data
                temp = nl.where(swap_needed, next_data, curr_data)
                nl.store(result[tuple(indices_curr)], temp, mask=(start_idx + i_p < outer_size))
                
                # Where swap_needed is True, use curr_data, otherwise use next_data
                temp = nl.where(swap_needed, curr_data, next_data)
                nl.store(result[tuple(indices_next)], temp, mask=(start_idx + i_p < outer_size))
    
    return result