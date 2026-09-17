import argparse
import random
import torch,torch_musa
import torch.distributed as dist
from functools import partial
from typing import Literal, Set

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import calc_diff, per_token_cast_back, hash_tensor
from deep_ep.utils.testing import bench, bench_kineto


def simulate_failure_and_skip(rank: int, api: Literal["dispatch", "combine", "clean"], expected_masked_ranks: Set[int]):
    # Simulates rank failure when the rank first calls the corresponding communication API
    failed_api_ranks = {
        # API -> rank to fail (rank fails when it first calls the corresponding communication API)
        'dispatch': 1,
        'combine': 3,
        'clean': 5
    }
    if rank in expected_masked_ranks:
        # Rank already failed
        return True
    if api in failed_api_ranks.keys():
        expected_masked_ranks.add(failed_api_ranks[api])
        if failed_api_ranks[api] == rank:
            print(f"Rank {rank} failed when first calling {api} communication API, exit...", flush=True)
            return True
    return False


def query_mask_buffer_and_check(api: Literal["dispatch", "combine", "clean"], buffer: deep_ep.Buffer, mask_status: torch.Tensor,
                                expected_masked_ranks: Set[int]):
    buffer.low_latency_query_mask_buffer(mask_status)
    assert set(mask_status.nonzero().squeeze(-1).tolist()) == expected_masked_ranks


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


def test_main(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              use_logfmt: bool = False,
              shrink_test: bool = False,
              seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='musa') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='musa').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for _ in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='musa') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='musa') * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='musa').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='musa').abs()

    # Randomly mask some positions
    for _ in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='musa')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    # For failure simulation and shrink testing
    mask_status = torch.zeros((num_ranks, ), dtype=torch.int, device='musa')
    expected_masked_ranks = set()

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        if rank == 0:
                            print(f'[testing] Running with return_recv_hook={return_recv_hook}, dispatch_use_fp8={dispatch_use_fp8}, round_scale={round_scale}, use_ue8m0={use_ue8m0} ...', flush=True, end='')
                        if shrink_test and simulate_failure_and_skip(rank, "dispatch", expected_masked_ranks):
                            break
                        num_times += 1
                        for _ in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='musa')
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()
                        if shrink_test:
                            query_mask_buffer_and_check("dispatch", buffer, mask_status, expected_masked_ranks)
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert cumulative_local_expert_recv_stats[i].item(
                            ) == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                            assert num_valid_tokens == (
                                recv_layout_range
                                & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item(
                            ), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status==0].sum().item()}'

                            if num_valid_tokens == 0:
                                continue
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]
                                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                                if round_scale:
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                                else:
                                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                for j in range(num_ranks):
                                    if shrink_test and mask_status[j]:
                                        continue
                                    begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                                    if not round_scale:
                                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                            else:
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

                        # Check combine correctness
                        if shrink_test and simulate_failure_and_skip(rank, "combine", expected_masked_ranks):
                            break
                        for zero_copy in (False, ) if use_logfmt else (False, True):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='musa')
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                                                 topk_idx,
                                                                                 topk_weights,
                                                                                 handle,
                                                                                 use_logfmt=use_logfmt,
                                                                                 async_finish=not return_recv_hook,
                                                                                 zero_copy=zero_copy,
                                                                                 return_recv_hook=return_recv_hook,
                                                                                 out=out)
                            hook() if return_recv_hook else event.current_stream_wait()
                            if shrink_test:
                                query_mask_buffer_and_check("combine", buffer, mask_status, expected_masked_ranks)
                            if do_check:
                                if shrink_test:
                                    owner_by_expert = (torch.arange(num_experts, device='musa') // num_local_experts)
                                    fail_owner_mask = (mask_status == 1).index_select(0, owner_by_expert)
                                    valid_topk_idx = topk_idx >= 0
                                    failed_topk_idx = torch.zeros_like(topk_idx, device='musa', dtype=torch.bool)
                                    failed_topk_idx[valid_topk_idx] = fail_owner_mask.index_select(0, topk_idx[valid_topk_idx])
                                    topk_idx[failed_topk_idx] = -1
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                assert torch.isnan(combined_x).sum().item() == 0
                                if not round_scale:
                                    assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
                                hash_value ^= hash_tensor(combined_x)

                        # Clean buffer API
                        if shrink_test:
                            if simulate_failure_and_skip(rank, "clean", expected_masked_ranks):
                                break

                            buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts)
                            query_mask_buffer_and_check("clean", buffer, mask_status, expected_masked_ranks)
                        if rank == 0:
                            print('passed')

    if shrink_test:
        return

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                             topk_idx,
                                                             topk_weights,
                                                             handle,
                                                             use_logfmt=use_logfmt,
                                                             return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(
        f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
        f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
        flush=True)

    # show avg/min/max gathered from all processes in hook mode
    show_avg_gather_val = True
    e2e_dispatch_info = None
    e2e_combine_info = None
    true_dispatch_t = None
    true_combine_t = None

    # Separate profiling
    for return_recv_hook in (False, True):
        dist.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'),
                                             barrier_comm_profiling=True,
                                             suppress_kineto_output=True,
                                             num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            print(
                f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us',
                flush=True)
            if show_avg_gather_val:
                e2e_dispatch_info = [num_dispatch_comm_bytes / 1e9 / dispatch_t, dispatch_t]  # (bw, time)
                e2e_combine_info  = [num_combine_comm_bytes  / 1e9 / combine_t,  combine_t]   # (bw, time)
        else:
            print(
                f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us',
                flush=True)
            if show_avg_gather_val:
                true_dispatch_t = dispatch_t  # (send_time, recv_time)
                true_combine_t = combine_t    # (send_time, recv_time)
    
    # gather all processes dispatch/combine time
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
        if rank == 0:
            print_performance_stats(all_process_data, f">>> Profiling Info: Ranks {num_ranks}")
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl,
                            enable_shrink=args.shrink_test)
    test_main(num_tokens,
              hidden,
              num_experts,
              num_topk,
              rank,
              num_ranks,
              group,
              buffer,
              use_logfmt=args.use_logfmt,
              shrink_test=args.shrink_test,
              seed=1)

    do_pressure_test = args.pressure_test
    for seed in range(int(1e9) if do_pressure_test else 0):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        ref_hash = test_main(num_tokens,
                             hidden,
                             num_experts,
                             num_topk,
                             rank,
                             num_ranks,
                             group,
                             buffer,
                             use_logfmt=args.use_logfmt,
                             seed=seed)
        for _ in range(20):
            assert test_main(num_tokens,
                             hidden,
                             num_experts,
                             num_topk,
                             rank,
                             num_ranks,
                             group,
                             buffer,
                             use_logfmt=args.use_logfmt,
                             seed=seed) == ref_hash, f'Error: seed={seed}'

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    # dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288, help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true', help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true', help='Whether to do pressure test')
    parser.add_argument("--shrink-test", action='store_true', help='Whether to simulate failure and test shrink mode')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
