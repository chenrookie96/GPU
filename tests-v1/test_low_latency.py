import argparse
import random
import time
import torch,torch_musa
import torch.distributed as dist
from functools import partial

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def add_compute_noise():
    """Add random GEMM computations to simulate real-world workload"""
    num_gemm = random.randint(1, 3)  # Random number of GEMM operations (1-3)
    for _ in range(num_gemm):
        # Random matrix sizes to add variability
        m = random.randint(512, 2048)
        n = random.randint(512, 2048)
        k = random.randint(512, 2048)
        mat_a = torch.randn((m, k), dtype=torch.float32, device='musa')
        mat_b = torch.randn((k, n), dtype=torch.float32, device='musa')
        _ = mat_a @ mat_b
    torch.musa.synchronize()

def report_precision_error( rank: int, seed: int, diff: float, threshold: float, zero_copy: bool, round_scale: bool ):
    msg = (
        f"[PRECISION ERROR] "
        f"rank={rank}, seed={seed}, "
        f"diff={diff:.6e}, threshold={threshold:.6e}, "
        f"zero_copy={zero_copy}, round_scale={round_scale}"
    )
    print(msg, flush=True)


def print_performance_stats(perf_data: torch.Tensor, title: str):
    """
    打印性能统计信息
    
    Args:
        perf_data: 形状为 [num_ranks, 8] 的性能数据tensor
                   列顺序: [dispatch_bw, dispatch_e2e, dispatch_send, dispatch_recv,
                           combine_bw, combine_e2e, combine_send, combine_recv]
        title: 打印的标题字符串
    """
    dispatch_bw   = perf_data[:, 0]
    dispatch_e2e  = perf_data[:, 1]
    dispatch_send = perf_data[:, 2]
    dispatch_recv = perf_data[:, 3]
    combine_bw    = perf_data[:, 4]
    combine_e2e   = perf_data[:, 5]
    combine_send  = perf_data[:, 6]
    combine_recv  = perf_data[:, 7]
    # calc avg/min/max（us）
    stats = {
        "dispatch_bw": (
            dispatch_bw.mean().item(),
            dispatch_bw.min().item(),
            dispatch_bw.max().item()
        ),
        "dispatch_e2e": (
            dispatch_e2e.mean().item() * 1e6,
            dispatch_e2e.min().item() * 1e6,
            dispatch_e2e.max().item() * 1e6
        ),
        "dispatch_send": (
            dispatch_send.mean().item() * 1e6,
            dispatch_send.min().item() * 1e6,
            dispatch_send.max().item() * 1e6
        ),
        "dispatch_recv": (
            dispatch_recv.mean().item() * 1e6,
            dispatch_recv.min().item() * 1e6,
            dispatch_recv.max().item() * 1e6
        ),
        "combine_bw": (
            combine_bw.mean().item(),
            combine_bw.min().item(),
            combine_bw.max().item()
        ),
        "combine_e2e": (
            combine_e2e.mean().item() * 1e6,
            combine_e2e.min().item() * 1e6,
            combine_e2e.max().item() * 1e6
        ),
        "combine_send": (
            combine_send.mean().item() * 1e6,
            combine_send.min().item() * 1e6,
            combine_send.max().item() * 1e6
        ),
        "combine_recv": (
            combine_recv.mean().item() * 1e6,
            combine_recv.min().item() * 1e6,
            combine_recv.max().item() * 1e6
        )
    }
    # show info
    print("\n" + "="*60)
    print(title)
    print("="*60)
    for name, (avg, min_val, max_val) in stats.items():
        unit = "GB/s" if "bw" in name else "us  "
        print(f"{name:>15}  avg: {avg:>8.2f} {unit},  min: {min_val:>8.2f} {unit},  max: {max_val:>8.2f} {unit}")


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, 
              topk_mode: str = 'random', show_avg: bool = False, seed: int = 0,suppress_display: bool = False,
              num_tests: int =10):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='musa') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='musa').to(torch.bfloat16).view(-1, 1)
    if topk_mode == 'random':
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='musa').abs() + 1
        # Normal routing: use topk to select experts
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
        # Randomly mask some positions
        for i in range(10):
            topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1
    elif topk_mode == 'uniform':
        if (num_topk * num_tokens) % num_ranks != 0:
            error_msg = f"num_topk * num_tokens ({num_topk * num_tokens}) must be divisible by num_ranks ({num_ranks}) when using uniform routing"
            print(f"[rank {rank}] ERROR: {error_msg}", flush=True, file=sys.stderr)
            assert False, error_msg
        step = num_experts // num_ranks
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='musa').abs() + 1
        topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.long, device='musa')
        flat = topk_idx.view(-1)
        total = num_tokens * num_topk
        filled = 0
        n = 0
        while filled < total:
            if n >= step:
                n = 0
            k = 0
            while True:
                idx = n + k * step
                if idx >= num_experts:
                    break
                flat[filled] = idx
                filled += 1
                if filled >= total:
                    break
                k += 1
            n += 1
    elif topk_mode == 'no-routing':
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='musa').abs() + 1
        topk_idx = torch.full((num_tokens, num_topk), -1, dtype=torch.long, device='musa')
    else:
        raise ValueError(f'Invalid topk_mode: {topk_mode}')
    
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='musa').abs()


    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    precision_error_count = 0
    for return_recv_hook in (False, True):
        for dispatch_use_fp8 in (False, True):
            for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                for use_ue8m0 in (False, True) if round_scale else (False, ):
                    if rank == 0:
                        print(f'[testing] Running with return_recv_hook={return_recv_hook}, dispatch_use_fp8={dispatch_use_fp8}, round_scale={round_scale}, use_ue8m0={use_ue8m0} ...', flush=True, end='')
                    num_times += 1
                    for i in range((num_times % 2) + 1):
                        cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='musa')
                        dist.barrier()
                        packed_recv_x, packed_recv_count, handle, event, hook = \
                            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                                        use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                        async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                        hook() if return_recv_hook else event.current_stream_wait()
                    packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                    simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                        if dispatch_use_fp8 else packed_recv_x.clone()
                    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='musa')
                    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
                    for i in range(num_local_experts if do_check else 0):
                        expert_id = rank * num_local_experts + i
                        recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                        recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                        # Check expert indices
                        int_mask = (2 ** 32) - 1
                        num_valid_tokens = recv_count.item()
                        assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                        assert num_valid_tokens == (recv_layout_range & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                        assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

                        # Check received data
                        recv_x = recv_x[:num_valid_tokens]
                        recv_src_info = recv_src_info[:num_valid_tokens]
                        
                        # Skip validation if no valid tokens
                        if num_valid_tokens > 0:
                            recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                            assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                            if round_scale:
                                assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 1e-2
                            else:
                                assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                            for j in range(num_ranks):
                                begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                                if not round_scale:
                                    assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                if count > 0:
                                    assert (recv_x[begin_idx:begin_idx + count][:-128] - j).sum().item() == 0
                        if dispatch_use_fp8:
                            hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                            hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                        else:
                            hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

                    # Check combine correctness
                    for zero_copy in (False, True):
                        if zero_copy:
                            buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                        out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='musa')
                        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                             async_finish=not return_recv_hook, zero_copy=zero_copy,
                                                                             return_recv_hook=return_recv_hook, out=out)
                        hook() if return_recv_hook else event.current_stream_wait()
                        if do_check:
                            diff = calc_diff(x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                            assert torch.isnan(combined_x).sum().item() == 0
                            threshold = 1e-3 if round_scale else 1e-4
                            if diff > threshold:
                                precision_error_count += 1
                                report_precision_error(rank = rank, seed = seed, diff = diff, threshold = threshold, zero_copy = zero_copy, round_scale = round_scale)                   
                            hash_value ^= hash_tensor(combined_x)
                        dist.barrier()
                        precision_error_count_tensor = torch.tensor([precision_error_count], device='musa')
                        dist.reduce(precision_error_count_tensor, dst=0, op=dist.ReduceOp.SUM)
                        if rank == 0:
                            total_precision_errors = precision_error_count_tensor.item()
                            if total_precision_errors == 0:
                                print("[CHECK] PASSED")
                            else:
                                print(f"[CHECK] finished with {total_precision_errors} precision warnings")
                            # 重置本地计数
                            precision_error_count = 0
    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        if zero_copy:
            buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             zero_copy=zero_copy, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, zero_copy=False, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    # show avg/min/max gathered from all processes in hook mode
    show_avg_gather_val = show_avg
    e2e_dispatch_info = None
    e2e_combine_info = None
    true_dispatch_t = None
    true_combine_t = None

    # Separate profiling
    for return_recv_hook in (False, True):
        dist.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, zero_copy=True, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                                             suppress_kineto_output=True, num_tests=num_tests, num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
            if show_avg_gather_val:
                e2e_dispatch_info = [num_dispatch_comm_bytes / 1e9 / dispatch_t, dispatch_t]  # (bw, time)
                e2e_combine_info  = [num_combine_comm_bytes  / 1e9 / combine_t,  combine_t]   # (bw, time)
        else:
            print(f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                  f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us', flush=True)
            if show_avg_gather_val:
                true_dispatch_t = dispatch_t  # (send_time, recv_time)
                true_combine_t = combine_t    # (send_time, recv_time)
    
    # gather all processes dispatch/combine time
    perf_data = None
    if show_avg_gather_val:
        local_tensor = torch.tensor(
            [e2e_dispatch_info[0], e2e_dispatch_info[1], true_dispatch_t[0], true_dispatch_t[1], 
             e2e_combine_info[0], e2e_combine_info[1], true_combine_t[0], true_combine_t[1]],
            dtype=torch.float32,
            device='musa'
        )
        # main process tensor（shape: [num_ranks, 8]）
        if rank == 0:
            all_process_data = torch.empty((num_ranks, 8), dtype=torch.float32, device='musa')
        else:
            all_process_data = None
        # gather all process data
        dist.gather(local_tensor, gather_list=[all_process_data[i] for i in range(num_ranks)] if rank == 0 else None, dst=0)

        # Return performance data for pressure test aggregation
        perf_data = all_process_data if rank == 0 else None
        # show info in main process (suppress in pressure test mode)
        if rank == 0 and not suppress_display:
            print_performance_stats(all_process_data, f">>> Profiling Info: Ranks {num_ranks}")
    return hash_value, perf_data


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks, allow_nvlink_for_low_latency_mode=not args.disable_nvlink)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, 
              topk_mode=args.topk_mode, show_avg=not args.suppress_show_avg, seed=1, num_tests=args.num_tests)

    do_pressure_test = args.pressure_test
    pressure_test_rounds = args.pressure_test_rounds if do_pressure_test else 0
    pressure_test_perf_data = []  # Collect performance data for all seeds
    for seed in range(pressure_test_rounds):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        ref_hash, perf_data = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, 
                             show_avg=do_pressure_test, seed=seed, suppress_display=do_pressure_test, num_tests=args.num_tests)
        # Collect performance data from all 20 validation rounds for averaging
        validation_perf_data_list = []
        if do_pressure_test:
            # Collect perf_data from first call
            if perf_data is not None:
                validation_perf_data_list.append(perf_data)
            
            # Collect perf_data from 20 validation rounds
            for i in range(20):
                hash_val, validation_perf_data = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, 
                                           show_avg=do_pressure_test, seed=seed, suppress_display=True, num_tests=args.num_tests)
                assert hash_val == ref_hash, f'Error: seed={seed}'
                if validation_perf_data is not None:
                    validation_perf_data_list.append(validation_perf_data)
            
            # Calculate average across all 21 rounds (1 initial + 20 validation)
            if len(validation_perf_data_list) > 0 and rank == 0:
                # Stack all perf_data: [num_rounds, num_ranks, 8]
                stacked_perf_data = torch.stack(validation_perf_data_list, dim=0)
                # Average across rounds: [num_ranks, 8]
                avg_perf_data = stacked_perf_data.mean(dim=0)
                
                print_performance_stats(
                    avg_perf_data,
                    f">>> Pressure Test Profiling Info: Seed {seed}, Ranks {num_ranks} (averaged over {len(validation_perf_data_list)} rounds)"
                )
                # Collect performance data for averaging across seeds
                pressure_test_perf_data.append(avg_perf_data)
        else:
            for i in range(20):
                # Add noise based on noise_mode parameter
                if args.noise_mode in ['sleep', 'both']:
                    # Add random sleep to desynchronize different ranks (1-20ms)
                    sleep_ms = random.uniform(1e-3, 20e-3)  # Random sleep between 1ms and 20ms
                    time.sleep(sleep_ms)
                if args.noise_mode in ['gemm', 'both']:
                    # Add computational noise (GEMM operations)
                    add_compute_noise()
                hash_val, _ = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, 
                            topk_mode=args.topk_mode,show_avg=False, seed=seed, num_tests=args.num_tests)
                assert hash_val == ref_hash, f'Error: seed={seed}'

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                       help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288,
                       help='Number of experts (default: 288)')
    parser.add_argument("--pressure-test", action='store_true',
                        help='Whether to do pressure test')
    parser.add_argument("--suppress-show-avg", action='store_true',
                        help='Disable show average performance statistics')
    parser.add_argument("--disable-nvlink", action='store_true',
                        default=False, help='Whether to disable NVLink for testing')
    parser.add_argument("--topk-mode", type=str, choices=["random", "uniform","no-routing"], default="random",
                        help=(
                            "Routing mode for selecting top-k experts:\n"
                            "  - random (default): randomly select top-k experts per token.\n"
                            "  - uniform: route each token uniformly to each destination expert.\n"
                            "  - no-routing: disable expert routing (all tokens have topk_idx=-1, not sent tokens to any experts)."
                        ))
    parser.add_argument("--noise-mode", type=str, choices=["gemm", "sleep", "both", "none"], default="none",
                        help=(
                            "Noise mode for pressure testing:\n"
                            "  - gemm: Add random GEMM computations between API calls.\n"
                            "  - sleep: Add random sleep (1-20ms) to desynchronize different ranks.\n"
                            "  - both: Add both GEMM computations and random sleep.\n"
                            "  - none (default): No noise added."
                        ))
    parser.add_argument("--pressure-test-rounds", type=int, default=int(1e9),
                        help='Number of rounds for pressure test (default: 1000000000)')
    parser.add_argument("--num-tests", type=int, default=10,
                        help='Number of iterations for Kineto profiling (default: 10)')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
