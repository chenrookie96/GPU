#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <torch/python.h>

#include <chrono>
#include <memory>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../utils/event.hpp"
#include "../utils/shared_memory.hpp"
#include "../kernels/legacy/api.cuh"
#include "../kernels/backend/api.cuh"
#include "config.hpp"

namespace deep_ep::legacy {

struct Buffer {
    EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    // Low-latency mode buffer
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Shrink mode buffer
    bool enable_shrink = false;
    int* mask_buffer_ptr = nullptr;
    int* sync_buffer_ptr = nullptr;

    // Device info and communication
    int device_id;
    int num_device_sms;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    shared_memory::MemHandle ipc_handles[LEGACY_NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    shared_memory::SharedMemoryAllocator shared_memory_allocator;

    // Use Ace
    bool use_ace;
    uint32_t num_ace_buffer;
    mcclComm_t ace_comm{nullptr};

    // Workspace for make_row_id_map kernel (prefix sum)
    int* row_id_workspace = nullptr;
    int* row_id_workspace_prefix = nullptr;
    size_t row_id_workspace_capacity = 0;

    // Pre-allocated buffers for block_sums and block_prefix 
    int* row_id_block_sums = nullptr;
    int* row_id_block_prefix = nullptr;
    size_t row_id_scan_buffer_capacity = 0;

    //Host-side ACE info
    volatile int* global_stride = nullptr;
    int *global_stride_mapped = nullptr;
    void* ace_dispatch_buffer[MAX_ACE_BUFFER];
    IntranodeCombineAceBuffer  ace_combine_buffer[MAX_ACE_BUFFER];
    IntranodeAceAlltoallDispatcher* ace_dispatcher = nullptr;

    uint64_t alloc_token_num = 0;
    uint64_t alloc_hidden_size = 0;
    uint64_t alloc_num_topk = 0;
    uint64_t alloc_num_scales = 0;

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_nvl_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink,
           bool use_fabric,
           bool use_ace = false,
           uint32_t num_ace_buffer = 1,
           uint64_t token_num = 0,
           uint64_t hidden_size = 0,
           uint64_t num_topk = 0)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)),
      shared_memory_allocator(use_fabric),
      use_ace(use_ace),
      num_ace_buffer(num_ace_buffer),
      alloc_token_num(token_num),
      alloc_hidden_size(hidden_size),
      alloc_num_topk(num_topk) {
        // Metadata memory
        int64_t barrier_signal_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int);
        int64_t buffer_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(void*);
        int64_t barrier_signal_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int*);

        // Common checks
        EP_STATIC_ASSERT(LEGACY_NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
        EP_HOST_ASSERT(num_nvl_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
        EP_HOST_ASSERT(num_rdma_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
        EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
        EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
        EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_RDMA_PEERS or low_latency_mode));
        EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);
        if (num_rdma_bytes > 0)
            EP_HOST_ASSERT(num_ranks > LEGACY_NUM_MAX_NVL_PEERS or low_latency_mode);

        // Get ranks
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
        rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;
        num_rdma_ranks = std::max(1, num_ranks / LEGACY_NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, LEGACY_NUM_MAX_NVL_PEERS);

        // Get device info
        cudaDeviceProp device_prop = {};
        CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        num_device_sms = device_prop.multiProcessorCount;

        // Number of per-channel bytes cannot be large
        EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
        EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

        if (num_nvl_bytes > 0) {
            // Local IPC: alloc local memory and set local IPC handles
            shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                           num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
            shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
            buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

            // Set barrier signals
            barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
            barrier_signal_ptrs_gpu =
                reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

            // No need to synchronize, will do a full device sync during `sync`
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
        }

        // Create 32 MiB workspace
        CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, LEGACY_NUM_WORKSPACE_BYTES));
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(workspace, 0, LEGACY_NUM_WORKSPACE_BYTES, comm_stream));

        // MoE counter
        CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
        *moe_recv_counter = -1;

        // MoE expert-level counter
        CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * LEGACY_NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
        for (int i = 0; i < LEGACY_NUM_MAX_LOCAL_EXPERTS; ++i)
            moe_recv_expert_counter[i] = -1;

        // MoE RDMA-level counter
        if (num_rdma_ranks > 0) {
            CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
            CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
            *moe_recv_rdma_counter = -1;
        }

        if (use_ace) {
            CUDA_RUNTIME_CHECK(musaMallocHost(&global_stride, sizeof(int) * LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_NVL_PEERS, musaHostAllocMapped));
            CUDA_RUNTIME_CHECK(musaHostGetDevicePointer(&global_stride_mapped, const_cast<int*>(global_stride), 0));
            for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_NVL_PEERS; i++)
            {
                global_stride[i] = 0;
            }

            // Allocate workspace for make_row_id_map kernel
            // Max size: 256K elements * sizeof(int) = 1 MB per buffer
            constexpr size_t MAX_WORKSPACE_ELEMENTS = 128 * 1024;
            constexpr size_t MAX_SCAN_BLOCKS = 1024;
            constexpr size_t TOTAL_WORKSPACE_SIZE = 512 * 1024;

            row_id_workspace_capacity = MAX_WORKSPACE_ELEMENTS;
            row_id_scan_buffer_capacity = MAX_SCAN_BLOCKS;

            // Allocate single large buffer for all workspace needs
            CUDA_RUNTIME_CHECK(musaMalloc(&row_id_workspace, row_id_workspace_capacity * sizeof(int)));
            CUDA_RUNTIME_CHECK(musaMalloc(&row_id_workspace_prefix, TOTAL_WORKSPACE_SIZE * sizeof(int)));
            CUDA_RUNTIME_CHECK(musaMemset(row_id_workspace, 0, row_id_workspace_capacity * sizeof(int)));
            CUDA_RUNTIME_CHECK(musaMemset(row_id_workspace_prefix, 0, TOTAL_WORKSPACE_SIZE * sizeof(int)));

            // Use fixed offsets within workspace_prefix buffer for block_sums and block_prefix
            // This avoids separate malloc/free operations
            row_id_block_sums = row_id_workspace_prefix + MAX_WORKSPACE_ELEMENTS;
            row_id_block_prefix = row_id_workspace_prefix + MAX_WORKSPACE_ELEMENTS + MAX_SCAN_BLOCKS;
            ace_dispatcher = new IntranodeAceAlltoallDispatcher(rank, num_ranks);
        }
    }

    ~Buffer() noexcept(false) {
        if (not explicitly_destroy) {
            destroy();
        } else if (not destroyed) {
            printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
            fflush(stdout);
        }
    }

    bool is_available() const {
        return available;
    }

    bool is_internode_available() const {
        return is_available() and num_ranks > LEGACY_NUM_MAX_NVL_PEERS;
    }

    int get_num_rdma_ranks() const {
        return num_rdma_ranks;
    }

    int get_rdma_rank() const {
        return rdma_rank;
    }

    int get_root_rdma_rank(bool global) const {
        return global ? nvl_rank : 0;
    }

    int get_local_device_id() const {
        return device_id;
    }

    pybind11::bytearray get_local_ipc_handle() const {
        const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
        return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
    }

    pybind11::bytearray get_local_nvshmem_unique_id() const {
        EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
        const auto unique_id = nvshmem::get_unique_id();
        return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
    }

    pybind11::bytearray get_ace_unique_id() const {
        mcclUniqueId id;
        NCCL_CHECK(mcclGetUniqueId(&id));
        return pybind11::bytearray(reinterpret_cast<const char*>(&id), sizeof(id));
    }

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
        torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
        auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
        auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
        auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
        return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;
    }

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt,
              const std::optional<pybind11::bytearray>& ace_root_unique_id) {
        EP_HOST_ASSERT(not is_available());

        // Sync IPC handles
        if (num_nvl_bytes > 0) {
            EP_HOST_ASSERT(num_ranks == device_ids.size());
            EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
            for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
                EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
                auto handle_str = std::string(all_gathered_handles[offset + i].value());
                EP_HOST_ASSERT(handle_str.size() == sizeof(shared_memory::MemHandle));
                if (offset + i != rank) {
                    std::memcpy(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle));
                    shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
                    barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
                } else {
                    EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle)) == 0);
                }
            }

            // Copy all buffer and barrier signal pointers to GPU
            CUDA_RUNTIME_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        }

        // Sync NVSHMEM handles and allocate memory
        if (num_rdma_bytes > 0) {
            // Initialize NVSHMEM
            EP_HOST_ASSERT(root_unique_id_opt.has_value());
            std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
            auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
            std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
            auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
            auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
            EP_HOST_ASSERT(nvshmem_rank == nvshmem::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks,
                                                         low_latency_mode ? LEGACY_NUM_MAX_NVL_PEERS : 0));

            // Allocate
            rdma_buffer_ptr = nvshmem::alloc(num_rdma_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);

            // Clean buffer (mainly for low-latency mode)
            CUDA_RUNTIME_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

            // Allocate and clean shrink buffer
            if (enable_shrink) {
                int num_mask_buffer_bytes = num_ranks * sizeof(int);
                int num_sync_buffer_bytes = num_ranks * sizeof(int);
                mask_buffer_ptr = static_cast<int*>(nvshmem::alloc(num_mask_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES));
                sync_buffer_ptr = static_cast<int*>(nvshmem::alloc(num_sync_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES));
                CUDA_RUNTIME_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
                CUDA_RUNTIME_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
            }

            // Barrier
            nvshmem::barrier(true);
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        }
        if(use_ace){
            EP_HOST_ASSERT(ace_root_unique_id.has_value());
            mcclUniqueId id_tmp;
            auto ace_root_unique_id_str = ace_root_unique_id->cast<std::string>();
            std::memcpy(&id_tmp, ace_root_unique_id_str.data(), sizeof(id_tmp));
            NCCL_CHECK(mcclAceCommInitRank(&ace_comm, num_ranks, id_tmp, rank));

            alloc_token_num = alloc_token_num == 0 ? DEFAULT_ACE_TOKEN_NUM : alloc_token_num;
            alloc_hidden_size = alloc_hidden_size == 0 ? DEFAULT_ACE_HIDDEN_SIZE : alloc_hidden_size;
            alloc_num_topk = alloc_num_topk == 0 ? DEFAULT_ACE_NUM_TOPK : alloc_num_topk;
            alloc_num_scales = alloc_hidden_size / 128;

            IntranodeCombineLayout combine_layout(alloc_token_num, alloc_hidden_size, alloc_num_topk, true);

            uint64_t num_dispatch_input_bytes_per_zcopy_buffer = 
                alloc_token_num * alloc_num_topk * alloc_hidden_size * torch::elementSize(torch::kBFloat16) + // x
                alloc_token_num * alloc_num_topk * alloc_num_topk* torch::elementSize(torch::kFloat32) + // topk_weights
                alloc_token_num * alloc_num_topk * alloc_num_topk * torch::elementSize(torch::kInt64) + // topk_idx
                alloc_token_num * alloc_num_topk * alloc_num_scales * torch::elementSize(torch::kFloat32); // x_scales

            EP_HOST_ASSERT(num_ace_buffer >= 1);
            EP_HOST_ASSERT(num_ace_buffer <= MAX_ACE_BUFFER);
            for(int i = 0; i < num_ace_buffer; i++){
                alloc_ace_buffer(ace_dispatch_buffer[i], num_dispatch_input_bytes_per_zcopy_buffer);
                
                alloc_ace_buffer(ace_combine_buffer[i].x, combine_layout.num_bytes_x);
                alloc_ace_buffer(ace_combine_buffer[i].topk_weights, combine_layout.num_bytes_topk_weights);
            }
        }

        // Ready to use
        available = true;
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);

        // Synchronize
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        if (num_nvl_bytes > 0) {
            // Barrier
            intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

            // Close remote IPC
            if (is_available()) {
                for (int i = 0; i < num_nvl_ranks; ++i)
                    if (i != nvl_rank)
                        shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
            }

            // Free local buffer and error flag
            shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
        }

        // Free NVSHMEM
        if (is_available() and num_rdma_bytes > 0) {
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
            nvshmem::barrier(true);
            nvshmem::free(rdma_buffer_ptr);
            if (enable_shrink) {
                nvshmem::free(mask_buffer_ptr);
                nvshmem::free(sync_buffer_ptr);
            }
            nvshmem::finalize();
        }

        // Free workspace and MoE counter
        CUDA_RUNTIME_CHECK(cudaFree(workspace));
        CUDA_RUNTIME_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

        // Free chunked mode staffs
        CUDA_RUNTIME_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

        // Free row_id_map workspace
        if (row_id_workspace) {
            CUDA_RUNTIME_CHECK(musaFree(row_id_workspace));
        }
        if (row_id_workspace_prefix) {
            CUDA_RUNTIME_CHECK(musaFree(row_id_workspace_prefix));
        }

        if (use_ace) {
            CUDA_RUNTIME_CHECK(musaFreeHost(const_cast<int*>(global_stride)));
            delete ace_dispatcher;
        }

        destroyed = true;
        available = false;
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>> get_dispatch_layout(
        const torch::Tensor& topk_idx,
        int num_experts,
        const std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream) {
        EP_HOST_ASSERT(topk_idx.dim() == 2);
        EP_HOST_ASSERT(topk_idx.is_contiguous());
        EP_HOST_ASSERT(num_experts > 0);

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::cuda::setCurrentCUDAStream(comm_stream);
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
        auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
        auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
        auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
        if (is_internode_available())
            num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                    num_tokens_per_rank.data_ptr<int>(),
                                    num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                    num_tokens_per_expert.data_ptr<int>(),
                                    is_token_in_rank.data_ptr<bool>(),
                                    num_tokens,
                                    num_topk,
                                    num_ranks,
                                    num_experts,
                                    comm_stream);

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
            for (auto& to : {num_tokens_per_rdma_rank}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
    }

    std::tuple<torch::Tensor,
              std::optional<torch::Tensor>,
              std::optional<torch::Tensor>,
              std::optional<torch::Tensor>,
              std::vector<int>,
              torch::Tensor,
              std::vector<int>,
              int,
              std::optional<EventHandle>>
    intranode_ace_dispatch(const torch::Tensor& x,
                                  const std::optional<torch::Tensor>& x_scales,
                                  const std::optional<torch::Tensor>& topk_idx,
                                  const std::optional<torch::Tensor>& topk_weights,
                                  const std::optional<torch::Tensor>& num_tokens_per_rank,
                                  const std::optional<torch::Tensor>& num_tokens_per_expert,
                                  const std::optional<torch::Tensor>& cached_row_id_map,
                                  const std::optional<std::vector<int>>& cached_global_stride_list,
                                  const int cached_num_recv_tokens,
                                  const int cached_num_experts,
                                  int expert_alignment,
                                  int buffer_idx,
                                  std::optional<EventHandle>& previous_event,
                                  bool async,
                                  bool allocate_on_comm_stream)
    { 
        bool with_topk = topk_idx.has_value() && topk_weights.has_value();
        bool cached_mode = cached_row_id_map.has_value();

        if(cached_mode) {
            EP_HOST_ASSERT(cached_row_id_map->scalar_type() == torch::kInt64);
            EP_HOST_ASSERT(cached_global_stride_list.has_value());
            EP_HOST_ASSERT(!with_topk);
        }

        auto compute_stream = at::musa::getCurrentMUSAStream();
        auto num_experts = cached_mode ? cached_num_experts : static_cast<int>(num_tokens_per_expert->size(0));
        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
        auto num_local_experts = num_experts / num_ranks;
        auto num_topk = with_topk ? static_cast<int>(topk_idx->size(1)) : 1;

        EP_HOST_ASSERT(x.size(0) <= alloc_token_num);
        EP_HOST_ASSERT(x.size(1) <= alloc_hidden_size);
        EP_HOST_ASSERT(static_cast<uint64_t>(num_topk) <= alloc_num_topk);

        int num_scales = 0;
        if (x_scales.has_value()) {
            num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        }

        // NOTES: do not allocate tensors upfront!
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::musa::setCurrentMUSAStream(comm_stream);
        }

        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        auto row_id_map_non_trans = cached_mode ? cached_row_id_map.value() : torch::empty({num_tokens, num_experts},
                                                torch::dtype(torch::kInt64).device(torch::kMUSA));
        auto routing_map = std::optional<torch::Tensor>();
        if(!cached_mode) {
            routing_map = torch::empty({num_tokens, num_experts}, 
                                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kMUSA));
            
            const int num_words = (num_experts + 31) / 32;  // get_num_words equivalent
            size_t total_size = static_cast<size_t>(num_tokens) * num_words * sizeof(uint32_t);
            CUDA_RUNTIME_CHECK(musaMemsetAsync(routing_map->data_ptr(), 0, total_size, comm_stream));
            
            intranode_ace::fused_unique_routing_kernel_launch(
                topk_idx->data_ptr(), routing_map->data_ptr(),
                num_tokens, num_topk, num_experts, num_ranks, comm_stream);
            
            intranode_ace::make_row_id_map_kernel_launch(
                routing_map->data_ptr(), row_id_map_non_trans.data_ptr(),
                num_tokens, num_experts,
                row_id_workspace, row_id_workspace_prefix, row_id_workspace_capacity,
                row_id_block_sums, row_id_block_prefix, row_id_scan_buffer_capacity,
                comm_stream);
        }

        auto [token_output, scales_output, topk_idx_output, topk_weights_output] = 
            get_ace_dispatch_buffer(alloc_token_num, alloc_hidden_size, alloc_num_topk, alloc_num_scales, with_topk, x_scales.has_value(), buffer_idx);

        EP_HOST_ASSERT(alloc_token_num > 0);
        EP_HOST_ASSERT(alloc_num_topk > 0);

        // Use unified FP8 permute kernel when scales are present (FP8 mode)
        if (x_scales.has_value()) {
            intranode_ace::moe_permute_fp8_unified_kernel_launch(
                token_output.data_ptr(), x.data_ptr(),
                scales_output->data_ptr(), x_scales->data_ptr(),
                row_id_map_non_trans.data_ptr(),
                topk_idx_output.has_value() ? topk_idx_output->data_ptr() : nullptr,
                topk_weights_output.has_value() ? topk_weights_output->data_ptr() : nullptr,
                topk_idx.has_value() ? topk_idx->data_ptr() : nullptr, 
                topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
                num_tokens, num_experts, hidden, num_scales,
                num_local_experts, num_topk, alloc_token_num * alloc_num_topk,
                comm_stream,
                static_cast<int>(x.scalar_type()),
                static_cast<int>(x_scales->scalar_type()));
        } else {
            // Non-FP8 mode: use standard permute kernel
            intranode_ace::moe_permute_mask_kernel_launch(
                token_output.data_ptr(), x.data_ptr(), row_id_map_non_trans.data_ptr(),
                topk_idx_output.has_value() ? topk_idx_output->data_ptr() : nullptr,
                topk_weights_output.has_value() ? topk_weights_output->data_ptr() : nullptr,
                topk_idx.has_value() ? topk_idx->data_ptr() : nullptr, 
                topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
                num_tokens, num_experts, hidden,
                num_local_experts, num_topk, alloc_token_num * alloc_num_topk,
                comm_stream, static_cast<int>(x.scalar_type()));
        }

        int num_recv_tokens = -1;
        std::vector<int> num_recv_tokens_per_expert_list;
        std::vector<int> global_stride_list;
        if(cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;
            global_stride_list = cached_global_stride_list.value();
        } else {

            *moe_recv_counter = -1;

            for (int i = 0; i < num_local_experts; ++ i)
                moe_recv_expert_counter[i] = -1;

            for(int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_NVL_PEERS; i++)
                global_stride[i] = -1;
            intranode_ace::ace_notify_dispatch(moe_recv_counter_mapped, global_stride_mapped, moe_recv_expert_counter_mapped, num_tokens_per_rank->data_ptr<int>(),
                                        num_tokens_per_expert->data_ptr<int>(), buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks, num_experts, expert_alignment, comm_stream);

            
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_ranks * num_ranks and ready; ++i)
                    ready &= global_stride[i] >= 0;

                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > LEGACY_NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
            global_stride_list = std::vector<int>(global_stride, global_stride + num_ranks * num_ranks);
        }

        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());

        auto recv_x_scales = std::optional<torch::Tensor>();
        if (x_scales.has_value()) {
            recv_x_scales = x_scales->dim() == 1 ?
                            torch::empty({num_recv_tokens}, x_scales->options()) :
                            torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        }

        auto recv_topk_idx = std::optional<torch::Tensor>();
        auto recv_topk_weights = std::optional<torch::Tensor>();
        if (with_topk) {
            recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            ace_dispatcher->append(topk_idx_output.value(), recv_topk_idx.value(), num_topk, global_stride_list);
            ace_dispatcher->append(topk_weights_output.value(), recv_topk_weights.value(), num_topk, global_stride_list);
        }

        ace_dispatcher->append(token_output, recv_x, hidden, global_stride_list);
        if (x_scales.has_value()) {
            ace_dispatcher->append(scales_output.value(), recv_x_scales.value(), num_scales, global_stride_list);
        }
        ace_dispatcher->launch(ace_comm, comm_stream);

        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            for (auto& t: {x, recv_x, row_id_map_non_trans, token_output}) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
            for (auto& to: {x_scales, topk_idx, topk_weights, scales_output, topk_idx_output, topk_weights_output, recv_topk_idx, recv_topk_weights, recv_x_scales, routing_map, num_tokens_per_rank, num_tokens_per_expert}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if(allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::musa::setCurrentMUSAStream(compute_stream);

        return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, row_id_map_non_trans, global_stride_list, num_recv_tokens, event};
    }


    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    intranode_ace_combine(const torch::Tensor& input_x,
                                  const std::optional<torch::Tensor>& input_topk_weights,
                                  std::vector<int>& global_stride_list,
                                  const torch::Tensor& row_id_map_non_trans,
                                  int num_tokens,
                                  int num_experts,
                                  std::optional<EventHandle>& previous_event,
                                  bool async,
                                  bool allocate_on_comm_stream)
    {
        auto x = input_x;
        auto topk_weights = input_topk_weights;
        if (x.data_ptr() == 0){
            /*No tokens need to be combined,get raw combine buffer*/
            auto [raw_x, raw_topk_weights] = get_ace_combine_buffer(alloc_token_num, alloc_hidden_size, alloc_num_topk, true, 0);
            x = raw_x;
            if (input_topk_weights.has_value()) {
                topk_weights = raw_topk_weights;
            }
        }
        int num_recv_tokens = 0;
        int hidden = static_cast<int>(x.size(1));
        auto compute_stream = at::musa::getCurrentMUSAStream();

        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::musa::setCurrentMUSAStream(comm_stream);
        }

        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        for(int i = 0; i < num_ranks; i++){
            num_recv_tokens += global_stride_list[rank * num_ranks + i];
        }

        // Transpose global_stride_list matrix (num_ranks x num_ranks)
        for(int i = 0; i < num_ranks; i++) {
            for(int j = i + 1; j < num_ranks; j++) {
                std::swap(global_stride_list[i * num_ranks + j], global_stride_list[j * num_ranks + i]);
            }
        }

        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        ace_dispatcher->append(x, recv_x, hidden, global_stride_list);

        // Handle topk_weights if provided
        int num_topk = 0;
        auto combined_topk_weights = std::optional<torch::Tensor>();
        auto unpermuted_topk_weights = std::optional<torch::Tensor>();

        if (topk_weights.has_value()) {
            num_topk = static_cast<int>(topk_weights->size(1));
            combined_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            ace_dispatcher->append(topk_weights.value(), combined_topk_weights.value(), num_topk, global_stride_list);
            unpermuted_topk_weights = torch::empty({num_tokens, num_topk}, topk_weights->options());
        }

        ace_dispatcher->launch(ace_comm, comm_stream);

        auto output = torch::empty({num_tokens, hidden}, x.options());
        intranode_ace::moe_unpermute_mask_kernel_launch(
            output.data_ptr(), recv_x.data_ptr(), row_id_map_non_trans.data_ptr(),
            combined_topk_weights.has_value() ? combined_topk_weights->data_ptr() : nullptr,
            unpermuted_topk_weights.has_value() ? unpermuted_topk_weights->data_ptr() : nullptr,
            num_tokens, num_experts, hidden, num_topk,
            comm_stream, static_cast<int>(x.scalar_type()));

        // Synchronize before returning to Python layer (which uses compute_stream by default)
        // Wait streams
      std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            std::vector<const torch::Tensor*> tensors = {&x, &recv_x, &row_id_map_non_trans, &output};
            if (topk_weights.has_value()) {
                tensors.push_back(&topk_weights.value());
            }
            if (unpermuted_topk_weights.has_value()) {
                tensors.push_back(&unpermuted_topk_weights.value());
            }
            for (auto* t : tensors) {
                t->record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t->record_stream(compute_stream);
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::musa::setCurrentMUSAStream(compute_stream);

        return {recv_x, output, unpermuted_topk_weights, event};
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream) {
        bool cached_mode = cached_rank_prefix_matrix.has_value();

        // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
        EP_HOST_ASSERT(config.num_sms % 2 == 0);
        int num_channels = config.num_sms / 2;
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
            EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank.has_value());
            EP_HOST_ASSERT(num_tokens_per_expert.has_value());
        }

        // Type checks
        EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
        } else {
            EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        }

        // Shape and contiguous checks
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
        EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
            EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
            EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
            EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
        } else {
            EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= LEGACY_NUM_MAX_LOCAL_EXPERTS);
            EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
            EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        }

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
        auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

        // Top-k checks
        int num_topk = 0;
        topk_idx_t* topk_idx_ptr = nullptr;
        float* topk_weights_ptr = nullptr;
        EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
        if (topk_idx.has_value()) {
            num_topk = static_cast<int>(topk_idx->size(1));
            EP_HOST_ASSERT(num_experts > 0);
            EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
            EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
            EP_HOST_ASSERT(num_topk == topk_weights->size(1));
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
            topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // FP8 scales checks
        float* x_scales_ptr = nullptr;
        int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
        if (x_scales.has_value()) {
            EP_HOST_ASSERT(x.element_size() == 1);
            EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
            EP_HOST_ASSERT(x_scales->dim() == 2);
            EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
            num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
            x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
            scale_token_stride = static_cast<int>(x_scales->stride(0));
            scale_hidden_stride = static_cast<int>(x_scales->stride(1));
        }

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::cuda::setCurrentCUDAStream(comm_stream);
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        // Create handles (only return for non-cached mode)
        int num_recv_tokens = -1;
        auto rank_prefix_matrix = torch::Tensor();
        auto channel_prefix_matrix = torch::Tensor();
        std::vector<int> num_recv_tokens_per_expert_list;

        // Barrier or send sizes
        // To clean: channel start/end offset, head and tail
        int num_memset_int = num_channels * num_ranks * 4;
        if (cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;
            rank_prefix_matrix = cached_rank_prefix_matrix.value();
            channel_prefix_matrix = cached_channel_prefix_matrix.value();

            // Copy rank prefix matrix and clean flags
            intranode::cached_notify_dispatch(
                rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks, comm_stream);
        } else {
            rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
            channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

            // Send sizes
            // Meta information:
            //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
            //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
            // NOTES: no more token dropping in this version
            *moe_recv_counter = -1;
            for (int i = 0; i < num_local_experts; ++i)
                moe_recv_expert_counter[i] = -1;
            EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
            intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                       moe_recv_counter_mapped,
                                       num_ranks,
                                       num_tokens_per_expert->data_ptr<int>(),
                                       moe_recv_expert_counter_mapped,
                                       num_experts,
                                       num_tokens,
                                       is_token_in_rank.data_ptr<bool>(),
                                       channel_prefix_matrix.data_ptr<int>(),
                                       rank_prefix_matrix.data_ptr<int>(),
                                       num_memset_int,
                                       expert_alignment,
                                       buffer_ptrs_gpu,
                                       barrier_signal_ptrs_gpu,
                                       rank,
                                       comm_stream,
                                       num_channels);

            if (num_worst_tokens > 0) {
                // No CPU sync, just allocate the worst case
                num_recv_tokens = num_worst_tokens;

                // Must be forward with top-k stuffs
                EP_HOST_ASSERT(topk_idx.has_value());
                EP_HOST_ASSERT(topk_weights.has_value());
            } else {
                // Synchronize total received tokens and tokens per expert
                auto start_time = std::chrono::high_resolution_clock::now();
                while (true) {
                    // Read total count
                    num_recv_tokens = static_cast<int>(*moe_recv_counter);

                    // Read per-expert count
                    bool ready = (num_recv_tokens >= 0);
                    for (int i = 0; i < num_local_experts and ready; ++i)
                        ready &= moe_recv_expert_counter[i] >= 0;

                    if (ready)
                        break;

                    // Timeout check
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                        LEGACY_NUM_CPU_TIMEOUT_SECS)
                        throw std::runtime_error("DeepEP error: CPU recv timeout");
                }
                num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
            }
        }

        // Allocate new tensors
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
        auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
             recv_x_scales = std::optional<torch::Tensor>();
        auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Assign pointers
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        float* recv_x_scales_ptr = nullptr;
        if (topk_idx.has_value()) {
            recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }
        if (x_scales.has_value()) {
            recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                                 : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
            recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
        }

        // Dispatch
        EP_HOST_ASSERT(
            num_ranks * num_ranks * sizeof(int) +                                                                     // Size prefix matrix
                num_channels * num_ranks * sizeof(int) +                                                              // Channel start offset
                num_channels * num_ranks * sizeof(int) +                                                              // Channel end offset
                num_channels * num_ranks * sizeof(int) * 2 +                                                          // Queue head and tail
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +  // Data buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                     // Source index buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // Top-k index buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +        // Top-k weight buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer
            <= num_nvl_bytes);
        intranode::dispatch(recv_x.data_ptr(),
                            recv_x_scales_ptr,
                            recv_src_idx.data_ptr<int>(),
                            recv_topk_idx_ptr,
                            recv_topk_weights_ptr,
                            recv_channel_prefix_matrix.data_ptr<int>(),
                            send_head.data_ptr<int>(),
                            x.data_ptr(),
                            x_scales_ptr,
                            topk_idx_ptr,
                            topk_weights_ptr,
                            is_token_in_rank.data_ptr<bool>(),
                            channel_prefix_matrix.data_ptr<int>(),
                            num_tokens,
                            num_worst_tokens,
                            static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                            num_topk,
                            num_experts,
                            num_scales,
                            scale_token_stride,
                            scale_hidden_stride,
                            buffer_ptrs_gpu,
                            rank,
                            num_ranks,
                            comm_stream,
                            config.num_sms,
                            config.num_max_nvl_chunked_send_tokens,
                            config.num_max_nvl_chunked_recv_tokens);

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            for (auto& t : {x,
                            is_token_in_rank,
                            rank_prefix_matrix,
                            channel_prefix_matrix,
                            recv_x,
                            recv_src_idx,
                            recv_channel_prefix_matrix,
                            send_head}) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
            for (auto& to : {x_scales,
                             topk_idx,
                             topk_weights,
                             num_tokens_per_rank,
                             num_tokens_per_expert,
                             cached_channel_prefix_matrix,
                             cached_rank_prefix_matrix,
                             recv_topk_idx,
                             recv_topk_weights,
                             recv_x_scales}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // Return values
        return {recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
                event};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_idx,
        const torch::Tensor& rank_prefix_matrix,
        const torch::Tensor& channel_prefix_matrix,
        const torch::Tensor& send_head,
        const Config& config,
        const std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream) {
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
        EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                       rank_prefix_matrix.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                       channel_prefix_matrix.scalar_type() == torch::kInt32);

        // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
        EP_HOST_ASSERT(config.num_sms % 2 == 0);
        int num_channels = config.num_sms / 2;

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
        auto num_recv_tokens = static_cast<int>(send_head.size(0));
        EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
        EP_HOST_ASSERT(send_head.size(1) == num_ranks);
        EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
        EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
        EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::cuda::setCurrentCUDAStream(comm_stream);
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        int num_topk = 0;
        auto recv_topk_weights = std::optional<torch::Tensor>();
        float* topk_weights_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
            num_topk = static_cast<int>(topk_weights->size(1));
            topk_weights_ptr = topk_weights->data_ptr<float>();
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }

        // Launch barrier and reset queue head and tail
        EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
        intranode::cached_notify_combine(buffer_ptrs_gpu,
                                         send_head.data_ptr<int>(),
                                         num_channels,
                                         num_recv_tokens,
                                         num_channels * num_ranks * 2,
                                         barrier_signal_ptrs_gpu,
                                         rank,
                                         num_ranks,
                                         comm_stream);

        // Assign bias pointers
        auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++i)
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }

        // Combine data
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                           num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                           num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                           num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                       <= num_nvl_bytes);
        intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                           recv_x.data_ptr(),
                           recv_topk_weights_ptr,
                           x.data_ptr(),
                           topk_weights_ptr,
                           bias_ptrs[0],
                           bias_ptrs[1],
                           src_idx.data_ptr<int>(),
                           rank_prefix_matrix.data_ptr<int>(),
                           channel_prefix_matrix.data_ptr<int>(),
                           send_head.data_ptr<int>(),
                           num_tokens,
                           num_recv_tokens,
                           hidden,
                           num_topk,
                           buffer_ptrs_gpu,
                           rank,
                           num_ranks,
                           comm_stream,
                           config.num_sms,
                           config.num_max_nvl_chunked_send_tokens,
                           config.num_max_nvl_chunked_recv_tokens);

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
            for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        return {recv_x, recv_topk_weights, event};
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       int cached_num_rdma_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                       const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream) {
        // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
        // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
        // unless we release GIL here.
        pybind11::gil_scoped_release release;

        const int num_channels = config.num_sms / 2;
        EP_HOST_ASSERT(config.num_sms % 2 == 0);
        EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= LEGACY_NUM_MAX_RDMA_PEERS);

        bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank.has_value());
            EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
            EP_HOST_ASSERT(num_tokens_per_expert.has_value());
        }

        // Type checks
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        }

        // Shape and contiguous checks
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                           cached_rdma_channel_prefix_matrix->size(1) == num_channels);
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                           cached_gbl_channel_prefix_matrix->size(1) == num_channels);
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
            EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
            EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
            EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
            EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= LEGACY_NUM_MAX_LOCAL_EXPERTS);
        }

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
             hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
        auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

        // Top-k checks
        int num_topk = 0;
        topk_idx_t* topk_idx_ptr = nullptr;
        float* topk_weights_ptr = nullptr;
        EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
        if (topk_idx.has_value()) {
            num_topk = static_cast<int>(topk_idx->size(1));
            EP_HOST_ASSERT(num_experts > 0);
            EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
            EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
            EP_HOST_ASSERT(num_topk == topk_weights->size(1));
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
            topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // FP8 scales checks
        float* x_scales_ptr = nullptr;
        int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
        if (x_scales.has_value()) {
            EP_HOST_ASSERT(x.element_size() == 1);
            EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
            EP_HOST_ASSERT(x_scales->dim() == 2);
            EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
            num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
            x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
            scale_token_stride = static_cast<int>(x_scales->stride(0));
            scale_hidden_stride = static_cast<int>(x_scales->stride(1));
        }

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::cuda::setCurrentCUDAStream(comm_stream);
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        // Create handles (only return for non-cached mode)
        int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
        auto rdma_channel_prefix_matrix = torch::Tensor();
        auto recv_rdma_rank_prefix_sum = torch::Tensor();
        auto gbl_channel_prefix_matrix = torch::Tensor();
        auto recv_gbl_rank_prefix_sum = torch::Tensor();
        std::vector<int> num_recv_tokens_per_expert_list;

        // Barrier or send sizes
        if (cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;
            num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
            rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
            recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
            gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
            recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

            // Just a barrier and clean flags
            internode::cached_notify(hidden_int4,
                                     num_scales,
                                     num_topk,
                                     num_topk,
                                     num_ranks,
                                     num_channels,
                                     0,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     rdma_buffer_ptr,
                                     config.num_max_rdma_chunked_recv_tokens,
                                     buffer_ptrs_gpu,
                                     config.num_max_nvl_chunked_recv_tokens,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     comm_stream,
                                     config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                     num_nvl_bytes,
                                     true,
                                     low_latency_mode);
        } else {
            rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
            recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
            gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
            recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

            // Send sizes
            *moe_recv_counter = -1;
            *moe_recv_rdma_counter = -1;
            for (int i = 0; i < num_local_experts; ++i)
                moe_recv_expert_counter[i] = -1;
            internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                       moe_recv_counter_mapped,
                                       num_ranks,
                                       num_tokens_per_rdma_rank->data_ptr<int>(),
                                       moe_recv_rdma_counter_mapped,
                                       num_tokens_per_expert->data_ptr<int>(),
                                       moe_recv_expert_counter_mapped,
                                       num_experts,
                                       is_token_in_rank.data_ptr<bool>(),
                                       num_tokens,
                                       num_worst_tokens,
                                       num_channels,
                                       hidden_int4,
                                       num_scales,
                                       num_topk,
                                       expert_alignment,
                                       rdma_channel_prefix_matrix.data_ptr<int>(),
                                       recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                       gbl_channel_prefix_matrix.data_ptr<int>(),
                                       recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                       rdma_buffer_ptr,
                                       config.num_max_rdma_chunked_recv_tokens,
                                       buffer_ptrs_gpu,
                                       config.num_max_nvl_chunked_recv_tokens,
                                       barrier_signal_ptrs_gpu,
                                       rank,
                                       comm_stream,
                                       config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                       num_nvl_bytes,
                                       low_latency_mode);

            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    LEGACY_NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
                }
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }

        // Allocate new tensors
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
             recv_x_scales = std::optional<torch::Tensor>();
        auto recv_src_meta = std::optional<torch::Tensor>();
        auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
        auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
        auto send_rdma_head = std::optional<torch::Tensor>();
        auto send_nvl_head = std::optional<torch::Tensor>();
        if (not cached_mode) {
            recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
            recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
            recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
            send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
            send_nvl_head = torch::empty({num_rdma_recv_tokens, LEGACY_NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
        }

        // Assign pointers
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        float* recv_x_scales_ptr = nullptr;
        if (topk_idx.has_value()) {
            recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }
        if (x_scales.has_value()) {
            recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                                 : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
            recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
        }

        // Launch data dispatch
        // NOTES: the buffer size checks are moved into the `.cu` file
        internode::dispatch(recv_x.data_ptr(),
                            recv_x_scales_ptr,
                            recv_topk_idx_ptr,
                            recv_topk_weights_ptr,
                            cached_mode ? nullptr : recv_src_meta->data_ptr(),
                            x.data_ptr(),
                            x_scales_ptr,
                            topk_idx_ptr,
                            topk_weights_ptr,
                            cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                            cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                            cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                            cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                            rdma_channel_prefix_matrix.data_ptr<int>(),
                            recv_rdma_rank_prefix_sum.data_ptr<int>(),
                            gbl_channel_prefix_matrix.data_ptr<int>(),
                            recv_gbl_rank_prefix_sum.data_ptr<int>(),
                            is_token_in_rank.data_ptr<bool>(),
                            num_tokens,
                            num_worst_tokens,
                            hidden_int4,
                            num_scales,
                            num_topk,
                            num_experts,
                            scale_token_stride,
                            scale_hidden_stride,
                            rdma_buffer_ptr,
                            config.num_max_rdma_chunked_send_tokens,
                            config.num_max_rdma_chunked_recv_tokens,
                            buffer_ptrs_gpu,
                            config.num_max_nvl_chunked_send_tokens,
                            config.num_max_nvl_chunked_recv_tokens,
                            rank,
                            num_ranks,
                            cached_mode,
                            comm_stream,
                            num_channels,
                            low_latency_mode);

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            for (auto& t : {x,
                            is_token_in_rank,
                            recv_x,
                            rdma_channel_prefix_matrix,
                            recv_rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            recv_gbl_rank_prefix_sum}) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
            for (auto& to : {x_scales,
                             topk_idx,
                             topk_weights,
                             num_tokens_per_rank,
                             num_tokens_per_rdma_rank,
                             num_tokens_per_expert,
                             cached_rdma_channel_prefix_matrix,
                             cached_recv_rdma_rank_prefix_sum,
                             cached_gbl_channel_prefix_matrix,
                             cached_recv_gbl_rank_prefix_sum,
                             recv_topk_idx,
                             recv_topk_weights,
                             recv_x_scales,
                             recv_rdma_channel_prefix_matrix,
                             recv_gbl_channel_prefix_matrix,
                             send_rdma_head,
                             send_nvl_head,
                             recv_src_meta}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // Return values
        return {recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
                event};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_meta,
        const torch::Tensor& is_combined_token_in_rank,
        const torch::Tensor& rdma_channel_prefix_matrix,
        const torch::Tensor& rdma_rank_prefix_sum,
        const torch::Tensor& gbl_channel_prefix_matrix,
        const torch::Tensor& combined_rdma_head,
        const torch::Tensor& combined_nvl_head,
        const Config& config,
        const std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream) {
        const int num_channels = config.num_sms / 2;
        EP_HOST_ASSERT(config.num_sms % 2 == 0);

        // Shape and contiguous checks
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
        EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
        EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                       is_combined_token_in_rank.scalar_type() == torch::kBool);
        EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                       rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                       rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                       gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                       combined_rdma_head.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
             hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
        auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
        EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
        EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
        EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
        EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
        EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                       combined_rdma_head.size(1) == num_rdma_ranks);
        EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == LEGACY_NUM_MAX_NVL_PEERS);

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);
            at::cuda::setCurrentCUDAStream(comm_stream);
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }

        // Top-k checks
        int num_topk = 0;
        auto combined_topk_weights = std::optional<torch::Tensor>();
        float* topk_weights_ptr = nullptr;
        float* combined_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
            num_topk = static_cast<int>(topk_weights->size(1));
            topk_weights_ptr = topk_weights->data_ptr<float>();
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
        }

        // Extra check for avoid-dead-lock design
        EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
        EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

        // Launch barrier and reset queue head and tail
        internode::cached_notify(hidden_int4,
                                 0,
                                 0,
                                 num_topk,
                                 num_ranks,
                                 num_channels,
                                 num_combined_tokens,
                                 combined_rdma_head.data_ptr<int>(),
                                 rdma_channel_prefix_matrix.data_ptr<int>(),
                                 rdma_rank_prefix_sum.data_ptr<int>(),
                                 combined_nvl_head.data_ptr<int>(),
                                 rdma_buffer_ptr,
                                 config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu,
                                 config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu,
                                 rank,
                                 comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes,
                                 false,
                                 low_latency_mode);

        // Assign bias pointers
        auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++i)
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }

        // Launch data combine
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                           combined_x.data_ptr(),
                           combined_topk_weights_ptr,
                           is_combined_token_in_rank.data_ptr<bool>(),
                           x.data_ptr(),
                           topk_weights_ptr,
                           bias_ptrs[0],
                           bias_ptrs[1],
                           combined_rdma_head.data_ptr<int>(),
                           combined_nvl_head.data_ptr<int>(),
                           src_meta.data_ptr(),
                           rdma_channel_prefix_matrix.data_ptr<int>(),
                           rdma_rank_prefix_sum.data_ptr<int>(),
                           gbl_channel_prefix_matrix.data_ptr<int>(),
                           num_tokens,
                           num_combined_tokens,
                           hidden,
                           num_topk,
                           rdma_buffer_ptr,
                           config.num_max_rdma_chunked_send_tokens,
                           config.num_max_rdma_chunked_recv_tokens,
                           buffer_ptrs_gpu,
                           config.num_max_nvl_chunked_send_tokens,
                           config.num_max_nvl_chunked_recv_tokens,
                           rank,
                           num_ranks,
                           comm_stream,
                           num_channels,
                           low_latency_mode);

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            for (auto& t : {x,
                            src_meta,
                            is_combined_token_in_rank,
                            rdma_channel_prefix_matrix,
                            rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            combined_x,
                            combined_rdma_head,
                            combined_nvl_head}) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
            for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // Return values
        return {combined_x, combined_topk_weights, event};
    }

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        EP_HOST_ASSERT(low_latency_mode);

        auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
        auto clean_meta_0 = layout.buffers[0].clean_meta();
        auto clean_meta_1 = layout.buffers[1].clean_meta();

        auto check_boundary = [=, this](void* ptr, size_t num_bytes) {
            auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
            EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
        };
        check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
        check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

        internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                               clean_meta_0.second,
                                               clean_meta_1.first,
                                               clean_meta_1.second,
                                               rank,
                                               num_ranks,
                                               mask_buffer_ptr,
                                               sync_buffer_ptr,
                                               at::cuda::getCurrentCUDAStream());
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const torch::Tensor& x,
                         const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank,
                         int num_experts,
                         bool use_fp8,
                         bool round_scale,
                         bool use_ue8m0,
                         bool async,
                         bool return_recv_hook) {
        EP_HOST_ASSERT(low_latency_mode);

        // Tensor checks
        // By default using `ptp128c` FP8 cast
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
        EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
        EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(num_experts % num_ranks == 0);

        // Diagnosis tensors
        if (cumulative_local_expert_recv_stats.has_value()) {
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
        }
        if (dispatch_wait_recv_cost_stats.has_value()) {
            EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
            EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
            EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
        }

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
        auto num_topk = static_cast<int>(topk_idx.size(1));
        auto num_local_experts = num_experts / num_ranks;

        // Buffer control
        LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
        EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
        auto buffer = layout.buffers[low_latency_buffer_idx];
        auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

        // Wait previous tasks to be finished
        // NOTES: the hook mode will always use the default stream
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
        EP_HOST_ASSERT(not(async and return_recv_hook));
        if (not return_recv_hook)
            stream_wait(launch_stream, compute_stream);

        // Allocate packed tensors
        auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                          x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
        auto packed_recv_src_info =
            torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        // Allocate column-majored scales
        auto packed_recv_x_scales = std::optional<torch::Tensor>();
        void* packed_recv_x_scales_ptr = nullptr;
        EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

        if (use_fp8) {
            // TODO: support unaligned cases
            EP_HOST_ASSERT(hidden % 512 == 0);
            if (not use_ue8m0) {
                packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                    torch::dtype(torch::kFloat32).device(torch::kCUDA));
            } else {
                EP_HOST_ASSERT(round_scale);
                packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                    torch::dtype(torch::kInt).device(torch::kCUDA));
            }
            packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
            packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
        }

        // Kernel launch
        auto next_clean_meta = next_buffer.clean_meta();
        auto launcher = [=, this](int phases) {
            internode_ll::dispatch(
                packed_recv_x.data_ptr(),
                packed_recv_x_scales_ptr,
                packed_recv_src_info.data_ptr<int>(),
                packed_recv_layout_range.data_ptr<int64_t>(),
                packed_recv_count.data_ptr<int>(),
                mask_buffer_ptr,
                cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                buffer.dispatch_rdma_recv_data_buffer,
                buffer.dispatch_rdma_recv_count_buffer,
                buffer.dispatch_rdma_send_buffer,
                x.data_ptr(),
                topk_idx.data_ptr<topk_idx_t>(),
                next_clean_meta.first,
                next_clean_meta.second,
                num_tokens,
                hidden,
                num_max_dispatch_tokens_per_rank,
                num_topk,
                num_experts,
                rank,
                num_ranks,
                use_fp8,
                round_scale,
                use_ue8m0,
                workspace,
                num_device_sms,
                launch_stream,
                phases);
        };
        launcher(return_recv_hook ? LEGACY_LOW_LATENCY_SEND_PHASE : (LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE));

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
            // so in Python API, we must wrap all tensors into the event handle.
            event = EventHandle(launch_stream);
        } else if (not return_recv_hook) {
            stream_wait(compute_stream, launch_stream);
        }

        // Receiver callback
        std::optional<std::function<void()>> recv_hook = std::nullopt;
        if (return_recv_hook)
            recv_hook = [=]() { launcher(LEGACY_LOW_LATENCY_RECV_PHASE); };

        // Return values
        return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
    }

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& src_info,
        const torch::Tensor& layout_range,
        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_logfmt,
        bool zero_copy,
        bool async,
        bool return_recv_hook,
        const std::optional<torch::Tensor>& out = std::nullopt) {
        EP_HOST_ASSERT(low_latency_mode);

        // Tensor checks
        EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
        EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
        EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
        EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
        EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
        EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
        EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
        EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
        EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

        if (combine_wait_recv_cost_stats.has_value()) {
            EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
            EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
            EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
        }

        auto hidden = static_cast<int>(x.size(2));
        auto num_topk = static_cast<int>(topk_weights.size(1));
        auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

        // Buffer control
        LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
        EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
        auto buffer = layout.buffers[low_latency_buffer_idx];
        auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

        // Wait previous tasks to be finished
        // NOTES: the hook mode will always use the default stream
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
        EP_HOST_ASSERT(not(async and return_recv_hook));
        if (not return_recv_hook)
            stream_wait(launch_stream, compute_stream);

        // Allocate output tensor
        torch::Tensor combined_x;
        if (out.has_value()) {
            EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
            EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
            EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
            combined_x = out.value();
        } else {
            combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        }

        // Kernel launch
        auto next_clean_meta = next_buffer.clean_meta();
        auto launcher = [=, this](int phases) {
            internode_ll::combine(combined_x.data_ptr(),
                                  buffer.combine_rdma_recv_data_buffer,
                                  buffer.combine_rdma_recv_flag_buffer,
                                  buffer.combine_rdma_send_buffer,
                                  x.data_ptr(),
                                  topk_idx.data_ptr<topk_idx_t>(),
                                  topk_weights.data_ptr<float>(),
                                  src_info.data_ptr<int>(),
                                  layout_range.data_ptr<int64_t>(),
                                  mask_buffer_ptr,
                                  combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                  next_clean_meta.first,
                                  next_clean_meta.second,
                                  num_combined_tokens,
                                  hidden,
                                  num_max_dispatch_tokens_per_rank,
                                  num_topk,
                                  num_experts,
                                  rank,
                                  num_ranks,
                                  use_logfmt,
                                  workspace,
                                  num_device_sms,
                                  launch_stream,
                                  phases,
                                  zero_copy);
        };
        launcher(return_recv_hook ? LEGACY_LOW_LATENCY_SEND_PHASE : (LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE));

        // Wait streams
        std::optional<EventHandle> event;
        if (async) {
            // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
            // so in Python API, we must wrap all tensors into the event handle.
            event = EventHandle(launch_stream);
        } else if (not return_recv_hook) {
            stream_wait(compute_stream, launch_stream);
        }

        // Receiver callback
        std::optional<std::function<void()>> recv_hook = std::nullopt;
        if (return_recv_hook)
            recv_hook = [=]() { launcher(LEGACY_LOW_LATENCY_RECV_PHASE); };

        // Return values
        return {combined_x, event, recv_hook};
    }

    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
        LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

        auto buffer = layout.buffers[low_latency_buffer_idx];
        auto dtype = torch::kBFloat16;
        auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

        EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
        return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                                {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                                torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
    }

    void alloc_ace_buffer(void*& ptr, size_t size)
    {
    #if MCCL_VERSION_CODE >= MCCL_VERSION(2, 28, 9)
        mcclWindow_t handle = nullptr;
    #else
        void *handle = nullptr;
    #endif
        NCCL_CHECK(mcclMemAlloc(&ptr, size));
        CUDA_RUNTIME_CHECK(musaMemset(ptr, 0, size));
        NCCL_CHECK(mcclCommWindowRegister(ace_comm, ptr, size, &handle, MCCL_WIN_COLL_SYMMETRIC));
    }


    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>>
    get_ace_dispatch_buffer(size_t num_tokens, size_t hidden, uint32_t num_topk, uint32_t num_scales, bool with_topk, bool use_fp8, uint32_t buffer_id) {

        EP_HOST_ASSERT(alloc_token_num >= num_tokens);
        EP_HOST_ASSERT(alloc_hidden_size >= hidden);
        EP_HOST_ASSERT(alloc_num_topk >= num_topk);

        num_tokens = std::max(num_tokens, static_cast<size_t>(1)); // avoid zero-dim tensor
        num_topk = std::max(num_topk, static_cast<uint32_t>(1)); // avoid zero-dim tensor

        deep_ep::legacy::IntranodeDispatchLayout layout(ace_dispatch_buffer[buffer_id], num_tokens, hidden, num_topk, num_scales, with_topk, use_fp8);

        auto options_token = use_fp8 ? torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kMUSA) :
                                      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kMUSA);
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMUSA);
        auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kMUSA);

        auto x = torch::from_blob(layout.buffer.x, {static_cast<long>(num_tokens * num_topk), static_cast<long>(hidden)}, options_token);
        auto x_scales = use_fp8 ? std::optional<torch::Tensor>(torch::from_blob(layout.buffer.x_scales,
                                {static_cast<long>(num_tokens * num_topk), static_cast<long>(num_scales)}, options_float)) : std::optional<torch::Tensor>();
        auto topk_idx = with_topk ? std::optional<torch::Tensor>(torch::from_blob(layout.buffer.topk_idx,
                                {static_cast<long>(num_tokens * num_topk), static_cast<long>(num_topk)}, options_int64)) : std::optional<torch::Tensor>();
        auto topk_weights = with_topk ? std::optional<torch::Tensor>(torch::from_blob(layout.buffer.topk_weights,
                                {static_cast<long>(num_tokens * num_topk), static_cast<long>(num_topk)}, options_float)) : std::optional<torch::Tensor>();

        EP_HOST_ASSERT(x.data_ptr() != nullptr);
        if (use_fp8) {
            EP_HOST_ASSERT(x_scales->data_ptr() != nullptr);
        }
        if (with_topk) {
            EP_HOST_ASSERT(topk_idx->data_ptr() != nullptr);
            EP_HOST_ASSERT(topk_weights->data_ptr() != nullptr);
        }


        return {x, x_scales, topk_idx, topk_weights};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>>
    get_ace_combine_buffer(size_t num_tokens, size_t hidden, uint32_t num_topk, bool with_topk, uint32_t buffer_id) {
        auto& buffer = ace_combine_buffer[buffer_id];

        if (alloc_token_num * alloc_num_topk < num_tokens) {
            fprintf(stderr, "\n"
                            "************************************************************************************\n"
                            "* [DeepEP] FATAL ERROR: num_tokens (%zu) EXCEEDS allocated buffer capacity (%zu) *\n"
                            "* PLEASE INCREASE THE token_num ARGUMENT IN DEEPEP BUFFER CONSTRUCTOR              *\n"
                            "* LOCATION: fused_a2a.py(Megatron)                                                 *\n"
                            "************************************************************************************\n\n",
                    num_tokens, alloc_token_num * alloc_num_topk);
        }
        EP_HOST_ASSERT(alloc_token_num * alloc_num_topk >= num_tokens);
        EP_HOST_ASSERT(alloc_hidden_size >= hidden);
        EP_HOST_ASSERT(alloc_num_topk >= num_topk);

        num_tokens = std::max(num_tokens, static_cast<size_t>(1)); // avoid zero-dim tensor
        num_topk = std::max(num_topk, static_cast<uint32_t>(1)); // avoid zero-dim tensor

        auto options_token = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kMUSA);
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMUSA);

        auto x = torch::from_blob(buffer.x, {static_cast<long>(num_tokens), static_cast<long>(hidden)}, options_token);
        auto topk_weights = with_topk ? std::optional<torch::Tensor>(torch::from_blob(buffer.topk_weights, {static_cast<long>(num_tokens), static_cast<long>(num_topk)}, options_float)) : std::optional<torch::Tensor>();

        EP_HOST_ASSERT(x.data_ptr() != nullptr);
        if (with_topk) {
            EP_HOST_ASSERT(topk_weights->data_ptr() != nullptr);
        }

        return {x, topk_weights};

    }

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask) const {
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
        EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
        internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
    }

    void low_latency_query_mask_buffer(const torch::Tensor& mask_status) const {
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
        EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

        internode_ll::query_mask_buffer(
            mask_buffer_ptr, num_ranks, static_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
    }

    void low_latency_clean_mask_buffer() const {
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
        internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &get_low_latency_rdma_size_hint);

    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    pybind11::class_<Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool, bool, uint32_t, uint64_t, uint64_t, uint64_t>())
        .def("is_available", &Buffer::is_available)
        .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &Buffer::get_local_nvshmem_unique_id)
        .def("get_ace_unique_id", &Buffer::get_ace_unique_id)
        .def("get_local_buffer_tensor", &Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &Buffer::get_comm_stream)
        .def("sync", &Buffer::sync)
        .def("destroy", &Buffer::destroy)
        .def("get_dispatch_layout", &Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &Buffer::intranode_dispatch)
        .def("intranode_ace_dispatch", &Buffer::intranode_ace_dispatch)
        .def("intranode_ace_combine", &Buffer::intranode_ace_combine)
        .def("intranode_combine", &Buffer::intranode_combine)
        .def("internode_dispatch", &Buffer::internode_dispatch)
        .def("internode_combine", &Buffer::internode_combine)
        .def("clean_low_latency_buffer", &Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &Buffer::low_latency_dispatch)
        .def("low_latency_combine", &Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &Buffer::get_next_low_latency_combine_buffer)
        .def("get_ace_dispatch_buffer", &Buffer::get_ace_dispatch_buffer)
        .def("get_ace_combine_buffer", &Buffer::get_ace_combine_buffer);
}

}  // namespace deep_ep::legacy
