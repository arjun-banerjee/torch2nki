from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dim index
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to the result tensor
    # We need to process in tiles due to hardware limitations
    tensor_shape = a_tensor.shape
    rank = len(tensor_shape)
    
    # Calculate the sort dimension length
    sort_dim_size = tensor_shape[dim]
    
    # For simplicity, we handle 1D and 2D tensors explicitly
    if rank == 1:
        # For 1D tensor, we sort the entire array using bubble sort
        # Process in tiles of maximum size
        trip_count = math.ceil(tensor_shape[0] / nl.tile_size.pmax)
        
        # First copy the data to result
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < tensor_shape[0]))
            nl.store(result[i_p], value=x_tile, mask=(i_p < tensor_shape[0]))
        
        # Bubble sort implementation
        for i in range(sort_dim_size):
            for j in range(sort_dim_size - i - 1):
                # Load the current pair of elements
                val1 = nl.load(result[j])
                val2 = nl.load(result[j+1])
                
                # Compare and swap if needed
                is_greater = nl.greater(val1, val2)
                
                # If val1 > val2, swap them
                if is_greater:
                    nl.store(result[j], value=val2)
                    nl.store(result[j+1], value=val1)
                
    elif rank == 2:
        # For 2D tensor, we sort along the specified dimension
        if dim == 0:
            # Sort along dimension 0 (rows)
            # This means we sort each column independently
            trip_count = math.ceil(tensor_shape[1] / nl.tile_size.pmax)
            
            # First copy the data to result
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(tensor_shape[0])[None, :]
                x_tile = nl.load(a_tensor[i_f, i_p], mask=(i_p < tensor_shape[1]))
                nl.store(result[i_f, i_p], value=x_tile, mask=(i_p < tensor_shape[1]))
            
            # Sort each column
            for col in range(tensor_shape[1]):
                for i in range(sort_dim_size):
                    for j in range(sort_dim_size - i - 1):
                        # Load the current pair of elements
                        val1 = nl.load(result[j, col])
                        val2 = nl.load(result[j+1, col])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(val1, val2)
                        
                        # If val1 > val2, swap them
                        if is_greater:
                            nl.store(result[j, col], value=val2)
                            nl.store(result[j+1, col], value=val1)
        else:
            # Sort along dimension 1 (columns)
            # This means we sort each row independently
            trip_count = math.ceil(tensor_shape[0] / nl.tile_size.pmax)
            
            # First copy the data to result
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(tensor_shape[1])[None, :]
                x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < tensor_shape[0]))
                nl.store(result[i_p, i_f], value=x_tile, mask=(i_p < tensor_shape[0]))
            
            # Sort each row
            for row in range(tensor_shape[0]):
                for i in range(sort_dim_size):
                    for j in range(sort_dim_size - i - 1):
                        # Load the current pair of elements
                        val1 = nl.load(result[row, j])
                        val2 = nl.load(result[row, j+1])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(val1, val2)
                        
                        # If val1 > val2, swap them
                        if is_greater:
                            nl.store(result[row, j], value=val2)
                            nl.store(result[row, j+1], value=val1)
    
    return result