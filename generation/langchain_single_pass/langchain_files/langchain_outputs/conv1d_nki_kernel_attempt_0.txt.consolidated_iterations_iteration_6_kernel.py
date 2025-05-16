from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Extract dimensions
    batch_size, in_channels, input_length = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    
    # Calculate output length
    output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Initialize result array
    result = nl.ndarray((batch_size, out_channels, output_length), dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Calculate how many tiles we need for processing
    max_batch_tile = min(nl.tile_size.pmax, batch_size)
    max_out_chan_tile = min(nl.tile_size.pmax, out_channels)
    max_out_len_tile = min(nl.tile_size.pmax, output_length)
    
    # Batch processing
    for b in nl.affine_range(math.ceil(batch_size / max_batch_tile)):
        b_start = b * max_batch_tile
        b_size = min(max_batch_tile, batch_size - b_start)
        
        # Out channel processing
        for oc in nl.affine_range(math.ceil(out_channels / max_out_chan_tile)):
            oc_start = oc * max_out_chan_tile
            oc_size = min(max_out_chan_tile, out_channels - oc_start)
            
            # Output length processing
            for ol in nl.affine_range(math.ceil(output_length / max_out_len_tile)):
                ol_start = ol * max_out_len_tile
                ol_size = min(max_out_len_tile, output_length - ol_start)
                
                # Create index arrays for this tile
                i_b = nl.arange(b_size)[:, None, None]
                i_oc = nl.arange(oc_size)[None, :, None]
                i_ol = nl.arange(ol_size)[None, None, :]
                
                # Initialize output tile with zeros
                output_tile = nl.zeros((b_size, oc_size, ol_size), dtype=x.dtype)
                
                # Process each input channel group
                for g in nl.affine_range(groups):
                    g_start = g * in_channels_per_group
                    g_end = (g + 1) * in_channels_per_group
                    
                    # Process each input channel within the group
                    for ic in nl.affine_range(in_channels_per_group):
                        # Process each kernel position
                        for k in nl.affine_range(kernel_size):
                            # Calculate input position with dilation
                            in_pos = ol_start * stride + k * dilation - padding
                            
                            # Skip computation if outside valid range
                            if (in_pos >= 0) and (in_pos < input_length):
                                # Load input values for this position
                                x_vals = nl.load(x[b_start + i_b[:, 0, 0], 
                                                 g_start + ic, 
                                                 in_pos + i_ol[0, 0, :] * stride])
                                
                                # Load weight values
                                w_vals = nl.load(weight[oc_start + i_oc[0, :, 0], ic, k])
                                
                                # Compute convolution for this position
                                contrib = nl.multiply(x_vals[:, None, :], w_vals[None, :, None])
                                output_tile = nl.add(output_tile, contrib)
                
                # Add bias if provided
                if bias is not None:
                    bias_vals = nl.load(bias[oc_start + i_oc[0, :, 0]])
                    output_tile = nl.add(output_tile, bias_vals[None, :, None])
                
                # Store result
                nl.store(result[b_start + i_b[:, 0, 0], 
                               oc_start + i_oc[0, :, 0], 
                               ol_start + i_ol[0, 0, :]], 
                        value=output_tile)
    
    return result