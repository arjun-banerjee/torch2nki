from neuronxcc import nki
import neuronxcc.nki.language as nl


def main():

    #import kernel code
    from transpose import nki_transpose_nd
    import numpy as np 
    import torch

    # Create random 1D tensors
    np.random.seed(0)
    a_small = torch.rand((5, 6, 2, 8, 3))
    permutation = (0, 1, 4, 2, 3) # new shape is 


    # Convert PyTorch tensors to NumPy for NKI
    # This example uses the kernel's expected shape: (lhs_rows, lhs_cols) and (lhs_cols, rhs_cols).
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_transpose_nd, 
        np.array(a_small),
        np.array(permutation),
    )

    # Compare with PyTorch reference to check correctness
    output_torch = torch.permute(a_small, permutation)
    print(output_torch)
    output_torch = output_torch.flatten(1)

    print(torch.tensor(output_nki).shape, output_torch.shape)

    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("NKI and Torch match!")
    else:
        print("NKI and Torch differ!")
        print(f"Torch output: {output_torch}")
        print(f"NKI output: {output_nki}")

if __name__ == "__main__":
    main()



