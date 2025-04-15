from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_norm(a_tensor):
    # Get the shape of the input tensor
    vector_length = a_tensor.shape[0]
    
    # Initialize result scalar
    result = nl.ndarray((), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Initialize sum of squares accumulator
    sum_of_squares = nl.zeros((), dtype=nl.float32, buffer=nl.psum)
    
    # Calculate the number of tiles needed
    trip_count = math.ceil(vector_length / nl.tile_size.pmax)
    
    # Process the vector in tiles to respect hardware limitations
    for p in nl.affine_range(trip_count):
        # Calculate indices for the current tile
        start_idx = p * nl.tile_size.pmax
        
        # Create indices for the current tile
        indices = start_idx + nl.arange(nl.tile_size.pmax)
        
        # Load tile with masking to handle boundary conditions
        tile_data = nl.load(a_tensor[indices], mask=(indices < vector_length))
        
        # Square each element in the tile
        squared_tile = nl.square(tile_data)
        
        # Sum the squared values in the tile and accumulate to total sum
        tile_sum = nl.sum(squared_tile, axis=0, mask=(indices < vector_length))
        sum_of_squares += tile_sum
    
    # Calculate the square root of the sum to get the norm
    norm_value = nl.sqrt(sum_of_squares)
    
    # Store the result
    nl.store(result, norm_value)
    
    return result