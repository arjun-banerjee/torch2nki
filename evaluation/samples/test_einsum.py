from neuronxcc import nki
import neuronxcc.nki.language as nl


def main():

    #import kernel code
    from einsum import nki_einsum_kernel_naive
    import numpy as np 
    import torch

    # Create random 1D tensors
    np.random.seed(0)
    lhs_small = torch.rand((5,4,3,3))
    rhs_small = torch.rand((3,3,4))
    contract_str = "ijlk,klm->ijm"
    einsum_inputs = [lhs_small, rhs_small]


    # Convert PyTorch tensors to NumPy for NKI
    # This example uses the kernel's expected shape: (lhs_rows, lhs_cols) and (lhs_cols, rhs_cols).
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_einsum_kernel_naive, 
        einsum_inputs,
        contract_str
    )

    # Compare with PyTorch reference to check correctness
    output_torch = torch.einsum(contract_str, *einsum_inputs) 


    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("NKI and Torch match!")
    else:
        print("NKI and Torch differ!")
        print(f"Torch output: {output_torch}")
        print(f"NKI output: {output_nki}")

if __name__ == "__main__":
    main()



