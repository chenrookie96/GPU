#pragma once

#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <musa_runtime.h>
#if 0
#include <deep_ep/common/comm.cuh>
#else
#include <nccl.h>
#endif
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
}

#if 0
template <int64_t kNumTimeoutCycles, typename timeout_print_t>
__device__ __forceinline__ void check_signal(
    const handle::NCCLGin& gin,
    const ncclGinSignal_t& signal_idx,
    const int64_t& target,
    const timeout_print_t& timeout_print) {
    const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin.gin._ginHandle) + gin.gin.contextId;
    const auto signal_ptr = reinterpret_cast<int64_t*>(
        __ldg(reinterpret_cast<int64_t*>(&gdaki->signals_table.buffer))) + signal_idx;
    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
        if (signal >= target)
            return true;

        if (is_last_check)
            timeout_print();
        return false;
    });
}

template <int kNumSMs,
          int kNumSmemBytes,
          int kNumStages = 2,
          int kNumTMABytesPerStage = math::constexpr_align<int, false>(
              (kNumSmemBytes - kNumStages * sizeof(ptx::mbarrier)) / kNumStages, ptx::kNumTMAAlignBytes),
          int kNumTMABlocksPerStage = kNumTMABytesPerStage / ptx::kNumTMAAlignBytes>
__device__ __forceinline__ void tma_copy(
    void* src_ptr, void* dst_ptr,
    const int64_t& num_bytes, const int& sm_idx) {
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffers = smem;
    const auto mbarriers = reinterpret_cast<ptx::mbarrier*>(smem + kNumStages * kNumTMABytesPerStage);
    EP_STATIC_ASSERT(kNumTMABytesPerStage > 0, "Invalid shared memory bytes");
    EP_STATIC_ASSERT(kNumStages >= 2, "Need at least 2 stages for pipelining");

    // Init mbarriers
    ptx::arrival_phase phases[kNumStages];
    #pragma unroll
    for (int s = 0; s < kNumStages; ++ s)
        phases[s] = 0, ptx::mbarrier_init_with_fence(mbarriers + s, 1);

    // Work partitioning across SMs
    EP_DEVICE_ASSERT(num_bytes % ptx::kNumTMAAlignBytes == 0);
    const auto num_tma_blocks = num_bytes / ptx::kNumTMAAlignBytes;
    const auto num_tma_blocks_per_sm = math::ceil_div<int64_t>(num_tma_blocks, kNumSMs);
    const auto start_block_idx = sm_idx * num_tma_blocks_per_sm;
    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_sm, num_tma_blocks);
    const auto num_iterations = math::ceil_div<int64_t>(end_block_idx - start_block_idx, kNumTMABlocksPerStage);

    auto get_iter_info = [&](const int64_t& iter_idx) {
        const auto i = start_block_idx + iter_idx * kNumTMABlocksPerStage;
        const auto offset = i * ptx::kNumTMAAlignBytes;
        const auto num_transaction_bytes =
            std::min<int>(kNumTMABlocksPerStage, end_block_idx - i) * ptx::kNumTMAAlignBytes;
        return std::make_pair(offset, num_transaction_bytes);
    };

    // Fill pipeline: issue loads for the first kNumStages iterations
    for (int64_t iter_idx = 0; iter_idx < kNumStages and iter_idx < num_iterations; ++ iter_idx) {
        const auto [load_offset, num_load_bytes] = get_iter_info(iter_idx);
        ptx::tma_load_1d(
            tma_buffers + iter_idx * kNumTMABytesPerStage,
            math::advance_ptr(src_ptr, load_offset),
            mbarriers + iter_idx, num_load_bytes);
        ptx::mbarrier_arrive_and_set_tx(mbarriers + iter_idx, num_load_bytes);
    }

    for (int64_t iter_idx = 0; iter_idx < num_iterations; ++ iter_idx) {
        const auto stage_idx = static_cast<int>(iter_idx % kNumStages);
        const auto [store_offset, num_store_bytes] = get_iter_info(iter_idx);

        // Wait this stage's load and issue store
        ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);
        ptx::tma_store_1d(
            math::advance_ptr(dst_ptr, store_offset),
            tma_buffers + stage_idx * kNumTMABytesPerStage,
            num_store_bytes);
        ptx::tma_store_commit();

        // Prefetch: wait until this stage's store is completed, then issue next load
        const auto next_iter_idx = iter_idx + kNumStages;
        if (next_iter_idx < num_iterations) {
            ptx::tma_store_wait();
            const auto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);
            ptx::tma_load_1d(
                tma_buffers + stage_idx * kNumTMABytesPerStage,
                math::advance_ptr(src_ptr, load_offset),
                mbarriers + stage_idx, num_load_bytes);
            ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);
        }
    }

    // Drain all outstanding stores
    ptx::tma_store_wait();
}
#else
__device__ __forceinline__ bool wait_signal(
    const ncclGin& gin, ncclGinSignal_t signal_idx, uint64_t target,
    int64_t timeout_cycles, int rank_idx, int peer_rank_idx, int slot_idx,
    int64_t count, int64_t bytes, const char* operation) {
    if (target == 0)
        return true;
    const uint64_t start = clock64();
    uint64_t current = 0;
    while (true) {
        current = gin.readSignal(signal_idx, 64, musa::memory_order_acquire);
        if (current >= target)
            return true;
        if (timeout_cycles > 0 && static_cast<int64_t>(clock64() - start) > timeout_cycles)
            return false;
    }
}

constexpr int kTMECopyBlockBytes = 64 * 1024;
constexpr int kTMECopyBarrierId = 1;

__device__ __forceinline__ bool is_tme_copy_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 31u) == 0;
}

__device__ __forceinline__ void thread_copy_bytes(uint8_t* dst, const uint8_t* src, int64_t bytes) {
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < bytes; i += static_cast<int64_t>(blockDim.x))
        dst[i] = src[i];
}

__device__ __forceinline__ void copy_bytes(void* dst, const void* src, int64_t bytes) {
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    if (bytes <= 0)
        return;
    if (bytes < 32 || !is_tme_copy_aligned(dst_bytes) || !is_tme_copy_aligned(src_bytes)) {
        thread_copy_bytes(dst_bytes, src_bytes, bytes);
        return;
    }

    __shared__ __align__(256) uint8_t smem[kTMECopyBlockBytes];
    __shared__ unsigned phase;
    __musa::async_barrier bar(kTMECopyBarrierId);
    if (threadIdx.x == 0) {
        bar.init_arrival(1, 0);
        phase = 0;
    }
    __syncthreads();

    int64_t copied = 0;
    while (copied < bytes) {
        const int64_t remaining = bytes - copied;
        const uint32_t tme_bytes = static_cast<uint32_t>(
            std::min<int64_t>(remaining, kTMECopyBlockBytes) & ~int64_t(31));
        if (tme_bytes == 0)
            break;

        if (threadIdx.x == 0) {
            __musa::memcpy_async_blk(
                smem, src_bytes + copied, tme_bytes, bar,
                __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B, __musa::SZ_NONE);
            phase = bar.arrive();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            bar.wait(phase);
            __musa::memcpy_blk(
                smem, dst_bytes + copied, tme_bytes,
                __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B);
            __musa_tme_store_commit();
            __musa_tme_store_read_wait();
            __musa::memcpy_idf_l2();
        }
        __syncthreads();
        copied += tme_bytes;
    }

    if (copied < bytes)
        thread_copy_bytes(dst_bytes + copied, src_bytes + copied, bytes - copied);
}

__device__ __forceinline__ ncclGin make_gin(const ncclDevComm_t& dev_comm) {
    return ncclGin(dev_comm, 0);
}

__device__ __forceinline__ void get_block_range(
    int64_t num_bytes, int block_idx, int num_blocks, int64_t* begin, int64_t* end) {
    constexpr int64_t kGranularity = 128;
    const int64_t num_granules = (num_bytes + kGranularity - 1) / kGranularity;
    const int64_t granules_per_block = (num_granules + num_blocks - 1) / num_blocks;
    const int64_t bytes_per_block = granules_per_block * kGranularity;
    const int64_t block_begin = static_cast<int64_t>(block_idx) * bytes_per_block;
    *begin = block_begin < num_bytes ? block_begin : num_bytes;
    const int64_t block_end = *begin + bytes_per_block;
    *end = block_end < num_bytes ? block_end : num_bytes;
}
#endif

template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_send_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* x, const int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int dst_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
#if 0
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto [local_idx_in_dst, dst_idx_in_local] = get_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    const auto send_count_ptr = workspace_layout.get_pp_send_count_ptr(dst_idx_in_local);
    const auto send_count = __ldg(send_count_ptr);
    const auto slot_idx = send_count % num_max_inflight_tensors;
    auto send_buffer_ptr = math::advance_ptr(
        buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Wait buffer slot release and do TMA
    if (ptx::elect_one_sync()) {
        check_signal<kNumTimeoutCycles>(
            gin,
            static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),
            send_count - num_max_inflight_tensors + 1,
            // TODO: print more info, and control the SM who prints it
            []() { printf("DeepEP PP send timeout, recv buffer is full"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // Issue RDMA put
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        gin.put<ncclTeamTagWorld>(
            recv_buffer_ptr,
            send_buffer_ptr,
            num_x_bytes, dst_rank_idx,
            0,
            // TODO: is this signal highly optimized?
            ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));
        *send_count_ptr += 1;
    }
#else
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto [local_idx_in_dst, dst_idx_in_local] = get_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);
    const auto send_count_ptr = workspace_layout.get_pp_send_count_ptr(dst_idx_in_local);
    const auto send_count = __ldg(send_count_ptr);
    const int slot_idx = static_cast<int>(send_count % num_max_inflight_tensors);
    auto* send_ptr = reinterpret_cast<uint8_t*>(buffer) +
        ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes;
    auto* recv_ptr = reinterpret_cast<uint8_t*>(buffer) +
        (local_idx_in_dst * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes;
    const auto gin = make_gin(nccl_dev_comm);

    const uint64_t release_target = send_count < num_max_inflight_tensors ? 0 :
        static_cast<uint64_t>(send_count - num_max_inflight_tensors + 1);
    __shared__ int ok;
    if (threadIdx.x == 0)
        ok = wait_signal(
            gin, static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),
            release_target,
            kNumTimeoutCycles, rank_idx, dst_rank_idx, slot_idx, send_count, num_x_bytes, "send");
    __syncthreads();
    if (!ok)
        return;

    int64_t begin, end;
    get_block_range(num_x_bytes, sm_idx, kNumSMs, &begin, &end);
    if (end > begin)
        copy_bytes(send_ptr + begin, static_cast<uint8_t*>(x) + begin, end - begin);
    cooperative_groups::this_grid().sync();

    if (sm_idx == 0 && threadIdx.x == 0) {
        __threadfence_system();
        gin.put(ncclTeamWorld(nccl_dev_comm), dst_rank_idx,
                nccl_window, static_cast<uint8_t*>(recv_ptr) - static_cast<uint8_t*>(workspace),
                nccl_window, static_cast<uint8_t*>(send_ptr) - static_cast<uint8_t*>(workspace),
                num_x_bytes,
                ncclGin_SignalInc{static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)},
                mcclGin_None{}, mcclCoopThread{});
        *send_count_ptr = send_count + 1;
    }
#endif
}

template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* x, int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int src_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
#if 0
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto [src_idx_in_local, local_idx_in_src] = get_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    const auto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);
    const auto recv_count = __ldg(recv_count_ptr);
    const auto slot_idx = recv_count % num_max_inflight_tensors;
    const auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Copy from the buffer into a new tensor
    if (ptx::elect_one_sync()) {
        check_signal<kNumTimeoutCycles>(
            gin,
            static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks),
            recv_count + 1,
            // TODO: print more info, and control the SM who prints it
            []() { printf("DeepEP PP recv timeout, recv buffer is empty\n"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // TODO: add a comment
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        gin.signal<ncclTeamTagWorld>(
            src_rank_idx, ncclGin_SignalInc(static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_src + 2))
        );
        *recv_count_ptr += 1;
    }
#else
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto [src_idx_in_local, local_idx_in_src] = get_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);
    const auto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);
    const auto recv_count = __ldg(recv_count_ptr);
    const int slot_idx = static_cast<int>(recv_count % num_max_inflight_tensors);
    auto* recv_ptr = reinterpret_cast<uint8_t*>(buffer) +
        (src_idx_in_local * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes;
    const auto gin = make_gin(nccl_dev_comm);

    __shared__ int ok;
    if (threadIdx.x == 0)
        ok = wait_signal(
            gin, static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks), recv_count + 1,
            kNumTimeoutCycles, rank_idx, src_rank_idx, slot_idx, recv_count, num_x_bytes, "recv");
    __syncthreads();
    if (!ok)
        return;

    int64_t begin, end;
    get_block_range(num_x_bytes, sm_idx, kNumSMs, &begin, &end);
    if (end > begin)
        copy_bytes(static_cast<uint8_t*>(x) + begin, recv_ptr + begin, end - begin);
    cooperative_groups::this_grid().sync();

    if (sm_idx == 0 && threadIdx.x == 0) {
        __threadfence_system();
        gin.signal(ncclTeamWorld(nccl_dev_comm), src_rank_idx,
                   ncclGin_SignalInc{static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_src + 2)},
                   mcclCoopThread{});
        *recv_count_ptr = recv_count + 1;
    }
#endif
}

} // namespace deep_ep::elastic
