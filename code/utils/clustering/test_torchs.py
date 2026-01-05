#!/usr/bin/env python3
# Minimal parity check runner (argparse version) with memory totals

import argparse
import time
import gc
import torch
import numpy as np

def storage_bytes(t: torch.Tensor) -> int:
    """Return storage size in bytes for a tensor's underlying storage."""
    return t.untyped_storage().nbytes()

def gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)

def fmt_gib(nbytes: int) -> str:
    return f"{gib(nbytes):.3f} GiB"

def maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def print_cuda_peak(device: str, tag: str):
    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(device=device)
        print(f"[{tag}] CUDA peak allocated: {fmt_gib(peak)}")

def run_exp1(bs, num_init, n, k, d, device, dtype):
    """
    Version 1: flatten (bs, num_init) -> B and do (B, n, k)^T @ (B, n, d)
    """
    # Optional: reset CUDA stats for clean peak measurement
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device=device)

    # Allocate inputs
    tensor1 = torch.rand((bs, n, d), device=device, dtype=dtype)[:, None, :, :].expand(bs, num_init, n, d)
    tensor2 = torch.rand((bs, num_init, n, k), device=device, dtype=dtype)



    B = bs * num_init
    t0 = time.time()
    maybe_sync()
 
    print(f"[exp1] storage tensor1: {fmt_gib(storage_bytes(tensor1))}")
    print(f"[exp1] storage tensor2: {fmt_gib(storage_bytes(tensor2))}")
    # Views (no new storage)
    print("is contiguous:", tensor1.is_contiguous(), tensor2.is_contiguous())
    X = tensor1.reshape(B, n, d) # view
    W = tensor2.reshape(B, n, k).contiguous()   # view
    
    print("is contiguous after reshape:", X.is_contiguous(), W.is_contiguous())
    s1 = storage_bytes(X)
    s2 = storage_bytes(W)
    print(f"[exp1] storage tensor1: {fmt_gib(s1)}")
    print(f"[exp1] storage tensor2: {fmt_gib(s2)}")

    # Compute result
    res = W.transpose(1, 2) @ X    # (B, k, d) new allocation
    s_res = storage_bytes(res)
    print(f"[exp1] res shape      : {tuple(res.shape)}")
    print(f"[exp1] storage res    : {fmt_gib(s_res)}")

    maybe_sync()
    t1 = time.time()
    print(f"[exp1] time           : {t1 - t0:.4f} s")

    # Total live tensor storage (inputs + result)
    total_live = s1 + s2 + s_res
    print(f"[exp1] TOTAL storage  : {fmt_gib(total_live)}  (inputs + result)")

    # CUDA peak memory (includes temporaries / allocator overhead)
    print_cuda_peak(device, "exp1")

    # Cleanup
    del res, X, W, tensor1, tensor2
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

def run_exp2(bs, num_init, n, k, d, device, dtype):
    """
    Version 2: keep 4D shape and use batched matmul
               (bs, num_init, n, k)^T x (bs, num_init, n, d) -> (bs, num_init, k, d)
    """
    # Optional: reset CUDA stats for clean peak measurement
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device=device)

    # Allocate inputs
    tensor1 = torch.rand((bs, n, d), device=device, dtype=dtype)
    tensor2 = torch.rand((bs, num_init, n, k), device=device, dtype=dtype)

    s1 = storage_bytes(tensor1)
    s2 = storage_bytes(tensor2)
    print(f"[exp2] storage tensor1: {fmt_gib(s1)}")
    print(f"[exp2] storage tensor2: {fmt_gib(s2)}")

    t0 = time.time()
    maybe_sync()

    # Compute result (new allocation)
    res = torch.matmul(tensor2.transpose(2, 3), tensor1)  # (bs, num_init, k, d)
    s_res = storage_bytes(res)
    print(f"[exp2] res shape      : {tuple(res.shape)}")
    print(f"[exp2] storage res    : {fmt_gib(s_res)}")

    maybe_sync()
    t1 = time.time()
    print(f"[exp2] time           : {t1 - t0:.4f} s")

    # Total live tensor storage (inputs + result)
    total_live = s1 + s2 + s_res
    print(f"[exp2] TOTAL storage  : {fmt_gib(total_live)}  (inputs + result)")

    # CUDA peak memory (includes temporaries / allocator overhead)
    print_cuda_peak(device, "exp2")

    # Cleanup
    del res, tensor1, tensor2
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_exp3(bs, num_init, n, k, d, device, dtype):
    """
    Version 2: keep 4D shape and use batched matmul
               (bs, num_init, n, k)^T x (bs, num_init, n, d) -> (bs, num_init, k, d)
    """
    # Optional: reset CUDA stats for clean peak measurement
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device=device)

    # Allocate inputs
    tensor1 = torch.rand((bs, n, d), device=device, dtype=dtype)
    tensor2 = torch.rand((bs, num_init, n, k), device=device, dtype=dtype)

    s1 = storage_bytes(tensor1)
    s2 = storage_bytes(tensor2)
    print(f"[exp3] storage tensor1: {fmt_gib(s1)}")
    print(f"[exp3] storage tensor2: {fmt_gib(s2)}")

    t0 = time.time()
    maybe_sync()

    # Compute result (new allocation)
    res = torch.einsum('bmnk, bnd -> bmkd', tensor2, tensor1)  # (bs, num_init, k, d)
    # res = torch.matmul(tensor2.transpose(2, 3), tensor1)  # (bs, num_init, k, d)
    s_res = storage_bytes(res)
    print(f"[exp3] res shape      : {tuple(res.shape)}")
    print(f"[exp3] storage res    : {fmt_gib(s_res)}")

    maybe_sync()
    t1 = time.time()
    print(f"[exp3] time           : {t1 - t0:.4f} s")

    # Total live tensor storage (inputs + result)
    total_live = s1 + s2 + s_res
    print(f"[exp3] TOTAL storage  : {fmt_gib(total_live)}  (inputs + result)")

    # CUDA peak memory (includes temporaries / allocator overhead)
    print_cuda_peak(device, "exp2")

    # Cleanup
    del res, tensor1, tensor2
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Run one KMeans parity experiment (method 1, 2 or 3) with memory totals."
    )
    parser.add_argument("--method", "-m", type=int, choices=[1, 2, 3], required=True,
                        help="1 = flattened batch '@'; 2 = 4D batched matmul")
    parser.add_argument("--bs", type=int, default=11, help="batch size (outer)")
    parser.add_argument("--num-init", type=int, default=10, help="number of inits")
    parser.add_argument("--n", type=int, default=10_000, help="number of samples n")
    parser.add_argument("--k", type=int, default=100, help="number of clusters k")
    parser.add_argument("--d", type=int, default=1000, help="dimension d")
    parser.add_argument("--device", type=str, default="cpu",
                        help="device, e.g. 'cpu', 'cuda', or 'cuda:0'")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"], help="tensor dtype")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    args = parser.parse_args()

    # Device / dtype
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Optional: clear allocator state before starting
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        maybe_sync()

    # Run the chosen method
    if args.method == 1:
        run_exp1(args.bs, args.num_init, args.n, args.k, args.d, device, dtype)
    elif args.method == 2:
        run_exp2(args.bs, args.num_init, args.n, args.k, args.d, device, dtype)
    else:
        run_exp3(args.bs, args.num_init, args.n, args.k, args.d, device, dtype)

if __name__ == "__main__":
    main()
