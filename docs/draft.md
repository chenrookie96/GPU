## Overview 
### DeepEp基本介绍
DeepEP is a high-performance communication library for Mixture-of-Experts (MoE) and expert parallelism (EP), providing throughput‑oriented and latency‑optimized GPU all‑to‑all (dispatch / combine) primitives with optional FP8 / BF16 support.It adapts DeepEP to work with Moore Threads GPUs.

This edition adds preliminary AMD GPU (ROCm) enablement:

xGMI / Infinity Fabric intranode + InfiniBand / RoCE RDMA internode with HIPIFY kernels.
Unified Python API with the upstream deepseek-ai/DeepEP
Some NVIDIA-only optimizations (e.g., PTX loads) are not mirrored; AMD path focuses on correctness + baseline overlap.

The logical flow is:
```text
                 MoE Framework
                      │
                      ▼
                   DeepEP
                      │
              ┌───────┴───────┐
              ▼               ▼
          dispatch          combine
              │               │
              ▼               ▼
          ACE Mode / MCCL / MUSA runtime
                      │
                      ▼
                   GPU
```
### V1 legacy 和 V2 的实现
```text
                    MoE Framework
                DeepSeek / Qwen / vLLM
                           │
                           ▼
                       DeepEP
              ┌────────────────────────┐
              │ MoE Dispatch / Combine │
              │ Token Routing / Buffer │
              │ Communication Schedule │
              └───────────┬────────────┘
                          │
                GPU-initiated communication
                          │
              ┌───────────┴────────────┐
              │                        │
             V1                       V2
              │                        │
         NVSHMEM backend         NCCL Gin backend
              │                        │
              │                  NCCL Device API
              │                        │
              └────────────┬───────────┘
                           │
                    Communication
                       transport
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
          Scale-up                  Scale-out
          节点内                     跨节点
              │                         │
              ▼                         ▼
         GPU P2P/Fabric                RDMA
              │                         │
         PCIe / GPU互联             NIC / IB / RoCE
```
DeepEP 本身做了什么修改

V1适配实现：NVSHMEM -> MVSHMEM

V2适配实现：NCCL -> MCCL,MCCL 提供了什么

MUSA SDK / Runtime 提供了什么

##  Quick start
### Requirements
- MooreThreads GPU (Compute Capability 3.1)
- Python 3.8 and above
- MUSA 4.0.0 and above
- PyTorch 2.1 and above
- MTLink for intranode communication
- RDMA network for internode communication

### IDownload and install MTSHMEM dependency
DeepEP (AMD version) depends on rocSHMEM. Please take a look at rocSHMEM Installation Guide for instructions.

### Development
```bash
git clone https://github.com/ROCm/DeepEP
cd DeepEP

## Network configurations

DeepEP (AMD) currently supports the following NIC families (selected via --nic):

cx7   : NVIDIA/Mellanox ConnectX (mlx5 driver) — e.g., mlx5_0, mlx5_1
thor2 : Broadcom / Thor2 RDMA (bnxt_re driver) — e.g., bnxt_re0 ... bnxt_re7
io    : AMD Pensando AI NIC

Tip: use cx7 if you see mlx5_* under /sys/class/infiniband, and thor2 if you see bnxt_re*.

You can confirm your NIC driver names via:
ibdev2netdev
ls /sys/class/infiniband


# To use DeepEP without MPI, please make sure rocSHMEM was built with this flag -DUSE_EXTERNAL_MPI=OFF
# Pass the NIC_TYPE, which can be one of: cx7, thor2, io. The default is cx7.
# Then install DeepEP using this command
python3 setup.py --variant rocm --nic <NIC_TYPE> build develop --user

# To use DeepEP with MPI, please proceed with these commands
# Export OMPI dir in the next command (e.g., it's $BUILD_DIR/ompi in third-party/README.md)
export OMPI_DIR=<ompi_dir>
# Pass the NIC_TYPE, which can be one of: cx7, thor2, io. The default is cx7.
# Then install DeepEP using this command
python3 setup.py --variant rocm --enable-mpi --nic <NIC_TYPE> build develop --user

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils.py`
# according to your own cluster settings, and launch into multiple nodes
python3 tests/test_intranode.py
# set the requiered ROCSHMEM number of contexts.
export ROCSHMEM_MAX_NUM_CONTEXTS=64
python3 tests/test_internode.py
# Set the required ROCSHMEM heap size (for example, for DeepSeek models) 
export ROCSHMEM_HEAP_SIZE=2147483648
# set the requiered ROCSHMEM number of contexts.
export ROCSHMEM_MAX_NUM_CONTEXTS=144
python3 tests/test_low_latency.py
```
### Installation

### Network configurations 

## V1 Case1

## V2 Case2 

## DeepEP ACE Mode Programming Guide


