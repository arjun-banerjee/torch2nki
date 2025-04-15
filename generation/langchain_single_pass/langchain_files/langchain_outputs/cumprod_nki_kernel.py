from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_cumprod(a_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Get tensor shape information
    if len(a_tensor.shape) == 1:
        sz = a_tensor.shape[0]
        
        # Calculate the number of tiles needed
        trip_count = math.ceil(sz/nl.tile_size.pmax)
        
        # Process the tensor in tiles
        product = nl.ones(1, dtype=a_tensor.dtype, buffer=nl.sbuf)
        
        for idx in nl.affine_range(sz):
            # Load single element
            val = nl.load(a_tensor[idx])
            
            # Compute cumulative product
            product = nl.multiply(product, val)
            
            # Store the result
            nl.store(result[idx], product)
            
    else:
        sz_p, sz_f = a_tensor.shape
        
        # Calculate the number of tiles needed
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        # Generate the free dimension index array
        i_f = nl.arange(sz_f)[None, :]
        
        # Process the tensor in tiles
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Create mask for valid indices
            mask = (i_p < sz_p)
            
            # Initialize running product for this row
            running_products = nl.ones((nl.tile_size.pmax, sz_f), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # For each position in the row
            for idx in nl.affine_range(sz_f):
                # Skip first element as it's the starting value
                if idx == 0:
                    # Load first element directly
                    curr_vals = nl.load(a_tensor[i_p, idx], mask=mask)
                    nl.store(running_products[i_p[:, 0], idx], curr_vals, mask=mask)
                else:
                    # Load current value
                    curr_vals = nl.load(a_tensor[i_p, idx], mask=mask)
                    
                    # Get previous running product
                    prev_products = nl.load(running_products[i_p[:, 0], idx-1], mask=mask)
                    
                    # Multiply with previous running product
                    new_products = nl.multiply(prev_products, curr_vals)
                    
                    # Store in running product buffer
                    nl.store(running_products[i_p[:, 0], idx], new_products, mask=mask)
            
            # Store all final products back to result
            for idx in nl.affine_range(sz_f):
                curr_products = nl.load(running_products[i_p[:, 0], idx], mask=mask)
                nl.store(result[i_p, idx], curr_products, mask=mask)
    
    return result