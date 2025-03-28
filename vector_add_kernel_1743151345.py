from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor, out_tensor):
    """
    Adds two vectors element-wise using NKI.
    
    Args:
        a_tensor: First input vector
        b_tensor: Second input vector 
        out_tensor: Output vector to store results
    """
    # Get vector length
    vector_length = a_tensor.shape[0]
    
    # Calculate number of iterations needed based on max partition size
    trip_count = (vector_length + nl.tile_size.pmax - 1) // nl.tile_size.pmax
    
    # Process vector in chunks
    for p in nl.affine_range(trip_count):
        # Calculate indices for current chunk
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load chunks from both vectors, masking to handle non-aligned sizes
        a_tile = nl.load(a_tensor[i_p], mask=(i_p < vector_length))
        b_tile = nl.load(b_tensor[i_p], mask=(i_p < vector_length))
        
        # Add the chunks
        result_tile = nl.add(a_tile, b_tile)
        
        # Store result back to output tensor
        nl.store(out_tensor[i_p], value=result_tile, mask=(i_p < vector_length))
        
    return out_tensor