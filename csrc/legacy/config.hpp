#pragma once

#include <deep_ep/common/exception.cuh>

#include "../kernels/legacy/api.cuh"

namespace deep_ep::legacy {

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
out_ptr_t advance_ptr(in_ptr_t &ptr, size_t count) {
    out_ptr_t saved = reinterpret_cast<out_ptr_t>(ptr);
    ptr = reinterpret_cast<in_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    return saved;
}

template <int T>
void transpose(std::vector<int>& v) {
    if (v.size() != T * T) {
        EP_HOST_ASSERT(false);
    }

    for (int i = 0; i < T; ++i) {
        for (int j = i + 1; j < T; ++j) {
            std::swap(v[i * T + j], v[j * T + i]);
        }
    }
}

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms,
           int num_max_nvl_chunked_send_tokens,
           int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens,
           int num_max_rdma_chunked_recv_tokens)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        EP_HOST_ASSERT(num_sms >= 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);

        // Ceil up RDMA buffer size
        this->num_max_rdma_chunked_recv_tokens = align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);
        // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_ranks <= LEGACY_NUM_MAX_NVL_PEERS or num_sms % 2 == 0);
        const auto num_rdma_ranks = std::max(num_ranks / LEGACY_NUM_MAX_NVL_PEERS, 1);
        const auto num_nvl_ranks = std::min(num_ranks, LEGACY_NUM_MAX_NVL_PEERS);
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * internode::get_source_meta_bytes();
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);
        num_bytes = align_up<size_t>(num_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
        return num_bytes;
    }

    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
        // Legacy mode
        if (num_ranks <= LEGACY_NUM_MAX_NVL_PEERS)
            return 0;

        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_sms % 2 == 0);
        const int num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        num_bytes += num_channels * num_rdma_ranks * (LEGACY_NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * internode::get_source_meta_bytes() * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;
        num_bytes = align_up<size_t>(num_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
        return num_bytes;
    }
};

struct LowLatencyBuffer {
    int num_clean_int = 0;

    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;

    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;

    void* combine_rdma_send_buffer_data_start = nullptr;
    size_t num_bytes_per_combine_msg = 0;

    std::pair<int*, int> clean_meta() {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};

struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffers[2];

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
        const int num_scales = hidden / 128;

        // Dispatch and combine layout:
        //  - 2 symmetric odd/even send buffer
        //  - 2 symmetric odd/even receive buffers
        //  - 2 symmetric odd/even signaling buffers

        // Message sizes
        // NOTES: you should add a control `int4` for combine messages if you want to do data transformation
        // NOTES: `num_scales * sizeof(nv_bfloat162)` means the per-128-channel min/max
        EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);
        size_t num_bytes_per_dispatch_msg = align_up<size_t>(sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float)), LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
        size_t num_bytes_per_combine_msg = align_up<size_t>(num_scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16), LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);

        // Send buffer
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
        EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
        total_bytes += send_buffer_bytes * 2;

        // Symmetric receive buffers
        // TODO: optimize memory usages
        size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
        EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
        total_bytes += recv_buffer_bytes * 2;

        // Symmetric signaling buffers and extra global atomic grid barrier sync counter
        size_t dispatch_recv_count_buffer_bytes = align_up<size_t>((num_experts + 1) * sizeof(int), LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);
        size_t signaling_buffer_bytes_aligned = align_up<size_t>(signaling_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
        total_bytes += signaling_buffer_bytes_aligned * 2;

        // Assign pointers
        // NOTES: we still leave some space for distinguishing dispatch/combine buffer,
        // so you may see some parameters are duplicated
        for (int i = 0; i < 2; ++i) {
            buffers[i] = {static_cast<int>(signaling_buffer_bytes / sizeof(int)),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          num_bytes_per_combine_msg};
        }
    }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    auto num_bytes = LowLatencyLayout(nullptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts).total_bytes;
    num_bytes = align_up<size_t>(num_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);
    return num_bytes;
}

struct IntranodeDispatchAceBuffer {
    void*    x;
    float*   x_scales;
    int64_t* topk_idx;
    float*   topk_weights;
};

struct IntranodeDispatchLayout {
    size_t total_bytes = 0;
    IntranodeDispatchAceBuffer buffer;

    IntranodeDispatchLayout(void* ace_buffer, size_t num_tokens, size_t hidden, size_t num_topk, size_t num_scales, bool with_topk, bool use_fp8) {
        size_t num_bytes_x = 0;
        size_t num_bytes_x_scales = 0;
        size_t num_bytes_topk_idx = 0;
        size_t num_bytes_topk_weights = 0;
        if (use_fp8) {
            num_bytes_x = num_tokens * num_topk * hidden * torch::elementSize(torch::kFloat8_e4m3fn);
            total_bytes += num_bytes_x;

            num_bytes_x_scales = num_tokens * num_topk * num_scales * sizeof(float);
            total_bytes += num_bytes_x_scales;
        } else {
            num_bytes_x = num_tokens * num_topk * hidden * torch::elementSize(torch::kBFloat16);
            total_bytes += num_bytes_x;
        }

        if (with_topk) {
            num_bytes_topk_idx = num_tokens * num_topk * num_topk * sizeof(int64_t);
            total_bytes += num_bytes_topk_idx;

            num_bytes_topk_weights = num_tokens * num_topk * num_topk * sizeof(float);
            total_bytes += num_bytes_topk_weights;
        }

        auto tmp = ace_buffer;
        buffer = {
            advance_ptr(tmp, num_bytes_x),  // x
            use_fp8 ? advance_ptr<float*>(tmp, num_bytes_x_scales) : nullptr,  // x_scales
            with_topk ? advance_ptr<int64_t*>(tmp, num_bytes_topk_idx) : nullptr,  // topk_idx
            with_topk ? advance_ptr<float*>(tmp, num_bytes_topk_weights) : nullptr  // topk_weights
        };
    }
};

struct IntranodeCombineAceBuffer {
    void*  x;
    void*  topk_weights;

    IntranodeCombineAceBuffer() : x(nullptr), topk_weights(nullptr) {}
};

struct IntranodeCombineLayout {
    size_t total_bytes = 0;
    size_t num_bytes_x = 0;
    size_t num_bytes_topk_weights = 0;

    IntranodeCombineLayout(size_t num_tokens, size_t hidden, size_t num_topk, bool with_topk) {
        num_bytes_x = num_tokens * num_topk * hidden * torch::elementSize(torch::kBFloat16);
        total_bytes += num_bytes_x;

        if (with_topk) {
            num_bytes_topk_weights = num_tokens * num_topk * num_topk * sizeof(float);
            total_bytes += num_bytes_topk_weights;
        }
    }
};



struct IntranodeAceAlltoallDispatcher {
private:
    int rank_;
    int num_ranks_;  // NEW: add num_ranks
    int batch_count;

    void* src_ptr[MAX_ACE_BATCH_SIZE] = {nullptr};
    void* dst_ptr[MAX_ACE_BATCH_SIZE] = {nullptr};
    size_t sendcounts[MAX_ACE_BATCH_SIZE * LEGACY_NUM_MAX_NVL_PEERS] = {0};
    size_t recvcounts[MAX_ACE_BATCH_SIZE * LEGACY_NUM_MAX_NVL_PEERS] = {0};
    size_t global_sdispls[MAX_ACE_BATCH_SIZE * LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_NVL_PEERS] = {0};
    size_t rdispls[MAX_ACE_BATCH_SIZE * LEGACY_NUM_MAX_NVL_PEERS] = {0};
    mcclDataType_t dtypes[MAX_ACE_BATCH_SIZE] = {mcclFloat32};

    using mcclBatchedAlltoallvFunc_t = mcclResult_t (*)(void**, const size_t*, const size_t*,
                                        void**, const size_t*, const size_t*, mcclDataType_t*,
                                        const int, mcclComm_t, musaStream_t);
    mcclBatchedAlltoallvFunc_t mcclBatchedAlltoallvFunc = nullptr;
public:
    IntranodeAceAlltoallDispatcher(int rank, int num_ranks) : rank_(rank), num_ranks_(num_ranks), batch_count(0) {
        constexpr const char* mcclLibName = "libmccl.so";
        constexpr const char* batchedFuncName = "mcclBatchAllToAllv";
        void* handle = dlopen(mcclLibName, RTLD_LAZY);
        if (!handle) {
            printf("Failed to open %s: %s\n", mcclLibName, dlerror());
        } else {
            mcclBatchedAlltoallvFunc = (mcclBatchedAlltoallvFunc_t)dlsym(handle, batchedFuncName);
            const char* err = dlerror();
            if (err != nullptr) {
                mcclBatchedAlltoallvFunc = nullptr;
            }

            dlclose(handle);
        }
    }

    void append(const torch::Tensor& input, const torch::Tensor& output, size_t hidden,
        const std::vector<int>& global_stride_list) {
        size_t sum = 0;
        // #pragma unroll
        for(int i = 0; i < num_ranks_;i++){
            sendcounts[batch_count * num_ranks_ + i] = global_stride_list[rank_ * num_ranks_ + i] * hidden;
            recvcounts[batch_count * num_ranks_ + i] = global_stride_list[i * num_ranks_ + rank_] * hidden;
            rdispls[batch_count * num_ranks_ + i] = sum;
            sum += global_stride_list[i * num_ranks_ + rank_] * hidden;
        }

        // #pragma unroll
        for(int i = 0 ; i < num_ranks_; i++){
            size_t sum = 0;
            for(int j = 0; j < num_ranks_; j++){
                global_sdispls[batch_count * num_ranks_ * num_ranks_ + i * num_ranks_ + j] = sum;
                sum += global_stride_list[i * num_ranks_ + j] * hidden;
            }
        }

        mcclDataType_t dtype = mcclBfloat16;
        if(input.scalar_type() == torch::kBFloat16){
            dtype = mcclBfloat16;
        }else if(input.scalar_type() == torch::kFloat32){
            dtype = mcclFloat32;
        }else if(input.scalar_type() == torch::kInt64){
            dtype = mcclInt64;
        }else if (input.scalar_type() == torch::kFloat8_e4m3fn) {
#if MCCL_VERSION_CODE >= MCCL_VERSION(2, 28, 9)
            dtype = mcclFloat8e4m3;
#else
            dtype = mcclFp8E4M3;
#endif
        }else{
            assert(false);
        }
        dtypes[batch_count] = dtype;

        src_ptr[batch_count] = static_cast<void*>(input.data_ptr());
        dst_ptr[batch_count] = static_cast<void*>(output.data_ptr());
        batch_count++;
        return;
    }

    void launch(mcclComm_t ace_comm, musaStream_t stream) {
        if (mcclBatchedAlltoallvFunc == nullptr) {
            for (int i = 0; i < batch_count; i++) {
                NCCL_CHECK(mcclAllToAllv(src_ptr[i],
                                         &sendcounts[i * num_ranks_],
                                         &global_sdispls[i * num_ranks_ * num_ranks_],
                                         dst_ptr[i],
                                         &recvcounts[i * num_ranks_],
                                         &rdispls[i * num_ranks_],
                                         dtypes[i],
                                         ace_comm,
                                         stream));
            }
        } else {
            NCCL_CHECK(mcclBatchedAlltoallvFunc(src_ptr,
                                                sendcounts,
                                                global_sdispls,
                                                dst_ptr,
                                                recvcounts,
                                                rdispls,
                                                dtypes,
                                                batch_count,
                                                ace_comm,
                                                stream));
        }
        batch_count = 0;
    }
};

}  // namespace deep_ep
