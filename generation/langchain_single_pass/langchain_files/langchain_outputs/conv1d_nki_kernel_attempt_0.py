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
    
    # Determine maximum tile sizes while respecting hardware limitations
    max_p_size = nl.tile_size.pmax  # Usually 128
    
    # Process convolution in tiles
    for b in range(batch_size):
        for oc_start in range(0, out_channels, max_p_size):
            oc_end = min(oc_start + max_p_size, out_channels)
            oc_size = oc_end - oc_start
            
            for ol_start in range(0, output_length, max_p_size):
                ol_end = min(ol_start + max_p_size, output_length)
                ol_size = ol_end - ol_start
                
                # Initialize accumulator for this output tile
                out_tile = nl.zeros((oc_size, ol_size), dtype=x.dtype, buffer=nl.sbuf)
                
                # Process each group separately
                for g in range(groups):
                    ic_start_group = g * in_channels_per_group
                    ic_end_group = (g + 1) * in_channels_per_group
                    oc_start_group = g * (out_channels // groups)
                    oc_end_group = (g + 1) * (out_channels // groups)
                    
                    # Adjust output channel range for current group
                    oc_start_in_group = max(0, oc_start - oc_start_group)
                    oc_end_in_group = min(oc_end_group - oc_start_group, oc_end - oc_start_group)
                    
                    if oc_start_in_group >= oc_end_in_group:
                        continue  # Skip if this group doesn't contribute to current output tile
                    
                    # Process each input channel in the group
                    for ic in range(ic_start_group, ic_end_group):
                        # For each position in the kernel
                        for k in range(kernel_size):
                            # Calculate corresponding input position for each output position
                            for ol in range(ol_size):
                                # Convert output position to actual output index
                                out_idx = ol_start + ol
                                
                                # Calculate input index considering stride, padding and dilation
                                in_idx = out_idx * stride - padding + k * dilation
                                
                                # Skip if outside input bounds
                                if in_idx < 0 or in_idx >= input_length:
                                    continue
                                
                                # Load input value
                                x_val = nl.load(x[b, ic, in_idx])
                                
                                # Get corresponding weight
                                oc_offset = oc_start - oc_start_group
                                w_val = nl.load(weight[oc_start_group + oc_offset + nl.arange(oc_size), ic % in_channels_per_group, k])
                                
                                # Multiply and accumulate
                                product = nl.multiply(w_val, x_val)
                                
                                # Update output tile
                                out_tile[:, ol] = nl.add(out_tile[:, ol], product)
                
                # Add bias if provided
                if bias is not None:
                    bias_vals = nl.load(bias[oc_start:oc_end])
                    for ol in range(ol_size):
                        out_tile[:, ol] = nl.add(out_tile[:, ol], bias_vals)
                
                # Store output tile to result
                for oc_idx in range(oc_size):
                    for ol_idx in range(ol_size):
                        nl.store(result[b, oc_start + oc_idx, ol_start + ol_idx], out_tile[oc_idx, ol_idx])
    
    return result