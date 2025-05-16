from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Extract dimensions
    batch_size, in_channels, input_length = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    
    # Calculate output size
    output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Initialize result array
    result = nl.ndarray((batch_size, out_channels, output_length), dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Process batch and output channels in tiles to respect hardware limitations
    for b in nl.affine_range(batch_size):
        for oc_start in nl.affine_range(math.ceil(out_channels / nl.tile_size.pmax)):
            # Calculate actual output channels for this tile
            oc_size = min(nl.tile_size.pmax, out_channels - oc_start * nl.tile_size.pmax)
            oc_indices = oc_start * nl.tile_size.pmax + nl.arange(oc_size)[:, None]
            
            # Process output length in tiles if needed
            for ol_start in nl.affine_range(math.ceil(output_length / nl.tile_size.pmax)):
                # Calculate actual output length for this tile
                ol_size = min(nl.tile_size.pmax, output_length - ol_start * nl.tile_size.pmax)
                ol_indices = ol_start * nl.tile_size.pmax + nl.arange(ol_size)[None, :]
                
                # Initialize accumulator for output
                output_tile = nl.zeros((oc_size, ol_size), dtype=x.dtype)
                
                # Process input channels by groups
                for g in nl.affine_range(groups):
                    # Calculate channel ranges for this group
                    ic_start = g * in_channels_per_group
                    ic_end = (g + 1) * in_channels_per_group
                    oc_start_g = g * (out_channels // groups)
                    oc_end_g = (g + 1) * (out_channels // groups)
                    
                    # Filter output channels for this group and tile
                    oc_mask = (oc_indices >= oc_start_g) & (oc_indices < oc_end_g)
                    
                    # Only process if we have output channels in this group
                    if any(oc_mask):
                        # Map to local indices within this group
                        oc_local = oc_indices[oc_mask] - oc_start_g
                        
                        # Process each input channel in the group
                        for ic in nl.affine_range(in_channels_per_group):
                            # Process each position in the kernel
                            for k in nl.affine_range(kernel_size):
                                # Calculate input positions with padding, stride, and dilation
                                in_pos = ol_indices * stride - padding + k * dilation
                                
                                # Only compute for valid input positions
                                valid_pos = (in_pos >= 0) & (in_pos < input_length)
                                
                                if any(valid_pos):
                                    # Load weight values for this output channel and kernel position
                                    w_values = nl.load(weight[oc_start_g + oc_local, ic, k], 
                                                     mask=oc_mask)
                                    
                                    # Load input values
                                    x_values = nl.load(x[b, ic_start + ic, in_pos], 
                                                     mask=valid_pos)
                                    
                                    # Multiply and accumulate to output
                                    output_tile += nl.multiply(w_values[:, None], x_values[None, :], 
                                                             mask=(oc_mask[:, None] & valid_pos[None, :]))
                
                # Add bias if provided
                if bias is not None:
                    bias_values = nl.load(bias[oc_indices], mask=(oc_indices < out_channels))
                    output_tile = nl.add(output_tile, bias_values[:, None])
                
                # Store result
                nl.store(result[b, oc_indices, ol_indices], 
                        value=output_tile, 
                        mask=((oc_indices < out_channels)[:, None] & (ol_indices < output_length)[None, :]))
    
    return result