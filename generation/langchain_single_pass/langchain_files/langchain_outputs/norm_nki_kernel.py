from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_norm(vector):
    # Initialize result to store the final scalar output
    result = nl.ndarray((), dtype=vector.dtype, buffer=nl.shared_hbm)
    
    # Get the size of the input vector
    vector_size = vector.shape[0]
    
    # Determine the number of tiles needed
    trip_count = math.ceil(vector_size / nl.tile_size.pmax)
    
    # Initialize a variable to accumulate the sum of squares
    sum_of_squares = nl.zeros((), dtype=vector.dtype)
    
    # Process the vector in tiles
    for i in nl.affine_range(trip_count):
        # Calculate start index for this tile
        start_idx = i * nl.tile_size.pmax
        
        # Generate indices for the current tile
        indices = start_idx + nl.arange(nl.tile_size.pmax)
        
        # Load the current tile with proper masking to handle edge cases
        tile = nl.load(vector[indices], mask=(indices < vector_size))
        
        # Square each element in the tile
        squared_tile = nl.square(tile)
        
        # Sum the squares in this tile and accumulate
        tile_sum = nl.sum(squared_tile, axis=0, mask=(indices < vector_size))
        sum_of_squares += tile_sum
    
    # Calculate the square root of the sum of squares
    norm_value = nl.sqrt(sum_of_squares)
    
    # Store the result
    nl.store(result, norm_value)
    
    return result