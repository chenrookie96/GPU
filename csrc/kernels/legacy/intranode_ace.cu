#include "compiled.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include <stdexcept>


namespace deep_ep::legacy {
namespace intranode_ace {

template<int kNumRanks>
__global__ void
ace_notify_dispatch(int *moe_recv_counter_mapped,int *global_stride_mapped, int *moe_recv_expert_counter_mapped, const int* num_tokens_per_rank,
                    const int* num_tokens_per_expert, void** buffer_ptrs, int** barrier_signal_ptrs, int rank, int num_experts, int expert_alignment)
{
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32; 

    // Barrier first
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    int *per_rank_buffer, *per_expert_buffer;
    if(thread_id < kNumRanks) {
        per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);
        per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
    }

    int num_experts_per_rank = num_experts / kNumRanks;
    if(thread_id < kNumRanks) {
        #pragma unroll
        for(int i = 0; i < kNumRanks; ++i){
            per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
        }
        #pragma unroll
        for (int i = 0; i < num_experts_per_rank; ++ i)
            per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
    }

    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
    if (thread_id < kNumRanks) {
        #pragma unroll
        for(int i = 0; i < kNumRanks; ++i){
            global_stride_mapped[i * kNumRanks + thread_id] = local_per_rank_buffer[i * kNumRanks + thread_id];
        }

        #pragma unroll
        for (int i = 1; i < kNumRanks; ++ i)
            local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
        if (thread_id == rank)
            *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
    }

    auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
    if (thread_id < num_experts_per_rank) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++ i)
            sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
        sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
        moe_recv_expert_counter_mapped[thread_id] = sum;
    }
    __syncthreads();

    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);        
}


void ace_notify_dispatch(int *moe_recv_counter_mapped,int *global_stride_mapped,int *moe_recv_expert_counter_mapped, const int* num_tokens_per_rank,
                        const int* num_tokens_per_expert, void** buffer_ptrs, int** barrier_signal_ptrs, int rank, 
                        int num_ranks,int num_experts,int expert_alignment, musaStream_t stream) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks) \
    LAUNCH_KERNEL(&cfg, ace_notify_dispatch<ranks>, \
        moe_recv_counter_mapped, global_stride_mapped, moe_recv_expert_counter_mapped, num_tokens_per_rank, \
        num_tokens_per_expert, buffer_ptrs, barrier_signal_ptrs, rank, num_experts, expert_alignment); \
    break

    constexpr int kNumThreads = 128;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}


// ============================================================================
// fused_unique_routing.cu - Fused Unique Routing Kernel
// ============================================================================

// Bit-packing helper functions
__device__ __forceinline__ void set_bit(uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;
    data[word_idx] |= (1U << bit_pos);
}

__device__ __forceinline__ int get_bit(const uint32_t* data, int bit_idx) {
    int word_idx = bit_idx / 32;
    int bit_pos = bit_idx % 32;
    return (data[word_idx] >> bit_pos) & 1U;
}

inline int get_num_words(int num_bits) {
    return (num_bits + 31) / 32;
}

// Original version (kept for backward compatibility)
template <typename IdxDtype, int BLOCK_SIZE = 512>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
fused_unique_routing_kernel_optimized(
    const IdxDtype* __restrict__ topk_idx,      // [num_tokens, num_topk]
    int* __restrict__ routing_map,              // [num_tokens, num_experts]
    const int num_tokens,
    const int num_topk,
    const int num_experts,
    const int experts_per_rank
) {
    const int token_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (token_id >= num_tokens) return;
    
    const IdxDtype* topk_row = topk_idx + token_id * num_topk;
    int* routing_row = routing_map + token_id * num_experts;
    
    unsigned long long seen_mask = 0;
    
    if (num_topk == 8 && sizeof(IdxDtype) == 8) {
        const longlong2* topk_vec2 = reinterpret_cast<const longlong2*>(topk_row);
        
        longlong2 p0 = __ldg(&topk_vec2[0]);
        longlong2 p1 = __ldg(&topk_vec2[1]);
        longlong2 p2 = __ldg(&topk_vec2[2]);
        longlong2 p3 = __ldg(&topk_vec2[3]);
        
        IdxDtype e0 = static_cast<IdxDtype>(p0.x);
        IdxDtype e1 = static_cast<IdxDtype>(p0.y);
        IdxDtype e2 = static_cast<IdxDtype>(p1.x);
        IdxDtype e3 = static_cast<IdxDtype>(p1.y);
        IdxDtype e4 = static_cast<IdxDtype>(p2.x);
        IdxDtype e5 = static_cast<IdxDtype>(p2.y);
        IdxDtype e6 = static_cast<IdxDtype>(p3.x);
        IdxDtype e7 = static_cast<IdxDtype>(p3.y);

        #define PROCESS_EXPERT(expert) \
            if (expert >= 0 && expert < num_experts) { \
                int rank_id = expert / experts_per_rank; \
                unsigned long long rank_bit = 1ULL << rank_id; \
                int is_new = (seen_mask & rank_bit) == 0; \
                seen_mask |= rank_bit; \
                routing_row[expert] |= is_new; \
            }
        
        PROCESS_EXPERT(e0)
        PROCESS_EXPERT(e1)
        PROCESS_EXPERT(e2)
        PROCESS_EXPERT(e3)
        PROCESS_EXPERT(e4)
        PROCESS_EXPERT(e5)
        PROCESS_EXPERT(e6)
        PROCESS_EXPERT(e7)
        
        #undef PROCESS_EXPERT
    } else {
        #pragma unroll 8
        for (int k = 0; k < num_topk; k++) {
            IdxDtype expert_id = __ldg(&topk_row[k]); 
            
            if (expert_id >= 0 && expert_id < num_experts) {
                int rank_id = expert_id / experts_per_rank;
                unsigned long long rank_bit = 1ULL << rank_id;
                int is_new = (seen_mask & rank_bit) == 0;
                seen_mask |= rank_bit;
                routing_row[expert_id] = is_new;
            }
        }
    }
}

// BIT-PACKED VERSION: Optimized version with 32x memory reduction
template <typename IdxDtype, int BLOCK_SIZE = 512>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
fused_unique_routing_kernel_bitpacked(
    const IdxDtype* __restrict__ topk_idx,
    uint32_t* __restrict__ routing_map_packed,
    const int num_tokens,
    const int num_topk,
    const int num_experts,
    const int num_words,
    const int experts_per_rank
) {
    const int token_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (token_id >= num_tokens) return;
    
    const IdxDtype* topk_row = topk_idx + token_id * num_topk;
    uint32_t* routing_row = routing_map_packed + token_id * num_words;
    
    unsigned long long seen_mask = 0;
    
    if (num_topk == 8 && sizeof(IdxDtype) == 8) {
        const longlong2* topk_vec2 = reinterpret_cast<const longlong2*>(topk_row);
        
        longlong2 p0 = __ldg(&topk_vec2[0]);
        longlong2 p1 = __ldg(&topk_vec2[1]);
        longlong2 p2 = __ldg(&topk_vec2[2]);
        longlong2 p3 = __ldg(&topk_vec2[3]);
        
        IdxDtype e0 = static_cast<IdxDtype>(p0.x);
        IdxDtype e1 = static_cast<IdxDtype>(p0.y);
        IdxDtype e2 = static_cast<IdxDtype>(p1.x);
        IdxDtype e3 = static_cast<IdxDtype>(p1.y);
        IdxDtype e4 = static_cast<IdxDtype>(p2.x);
        IdxDtype e5 = static_cast<IdxDtype>(p2.y);
        IdxDtype e6 = static_cast<IdxDtype>(p3.x);
        IdxDtype e7 = static_cast<IdxDtype>(p3.y);

        #define PROCESS_EXPERT_BP(expert) \
            if (expert >= 0 && expert < num_experts) { \
                int rank_id = expert / experts_per_rank; \
                unsigned long long rank_bit = 1ULL << rank_id; \
                int is_new = (seen_mask & rank_bit) == 0; \
                seen_mask |= rank_bit; \
                if (is_new) { set_bit(routing_row, expert); } \
            }
        
        PROCESS_EXPERT_BP(e0) PROCESS_EXPERT_BP(e1) PROCESS_EXPERT_BP(e2) PROCESS_EXPERT_BP(e3)
        PROCESS_EXPERT_BP(e4) PROCESS_EXPERT_BP(e5) PROCESS_EXPERT_BP(e6) PROCESS_EXPERT_BP(e7)
        
        #undef PROCESS_EXPERT_BP
    } else {
        #pragma unroll 8
        for (int k = 0; k < num_topk; k++) {
            IdxDtype expert_id = __ldg(&topk_row[k]); 
            if (expert_id >= 0 && expert_id < num_experts) {
                int rank_id = expert_id / experts_per_rank;
                unsigned long long rank_bit = 1ULL << rank_id;
                int is_new = (seen_mask & rank_bit) == 0;
                seen_mask |= rank_bit;
                if (is_new) { set_bit(routing_row, expert_id); }
            }
        }
    }
}

void fused_unique_routing_kernel_launch(
    const void* topk_idx_ptr,
    void* routing_map_ptr,
    int num_tokens,
    int num_topk,
    int num_experts,
    int num_ranks,
    musaStream_t stream
) {
    if(num_tokens == 0) return;
    const int64_t* topk_idx = static_cast<const int64_t*>(topk_idx_ptr);
    
    const int experts_per_rank = num_experts / num_ranks;
    constexpr int BLOCK_SIZE = 512;
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Use bit-packed version for better performance
    const int num_words = get_num_words(num_experts);
    uint32_t* routing_map_packed = static_cast<uint32_t*>(routing_map_ptr);
    
    
    fused_unique_routing_kernel_bitpacked<int64_t, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        topk_idx,
        routing_map_packed,
        num_tokens,
        num_topk,
        num_experts,
        num_words,
        experts_per_rank
    );
    
    CUDA_RUNTIME_CHECK(musaGetLastError());
}

// ============================================================================
// make_row_id_map.cu - Row ID Map Generation Kernels
// ============================================================================

// Pass 1: Block-level cumsum for each expert (BIT-PACKED VERSION)
template <typename IdxDtype, int BLOCK_SIZE = 256>
__global__ void make_row_id_map_pass1_kernel(
    const uint32_t* __restrict__ routing_map_packed,   // [num_tokens, num_words] BIT-PACKED
    int* __restrict__ workspace,                       // [num_experts * num_blocks]
    const int num_tokens,
    const int num_experts,
    const int num_words
) {
    const int expert_id = blockIdx.x;
    const int block_id = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (expert_id >= num_experts) return;
    
    const int block_start = block_id * BLOCK_SIZE;
    const int token_id = block_start + tid;
    
    // Read from bit-packed routing map
    int expert_token_mask = 0;
    if (token_id < num_tokens) {
        const uint32_t* routing_row = routing_map_packed + token_id * num_words;
        expert_token_mask = get_bit(routing_row, expert_id);
    }
    
    // Block-level inclusive scan using warp-level primitives + shared memory
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ int smem[NUM_WARPS];
    
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    
    // Warp-level inclusive scan
    int val = expert_token_mask;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n;
    }
    
    // Store warp sums
    if (lane_id == 31) {
        smem[warp_id] = val;
    }
   __syncthreads_lm();
    
    // Scan warp sums to get exclusive prefix for each warp
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        int warp_sum = smem[lane_id];
        int inclusive_sum = warp_sum;
        
        // Inclusive scan
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(0xffffffff, inclusive_sum, offset);
            if (lane_id >= offset) inclusive_sum += n;
        }
        
        // Convert to exclusive: shift right by 1, first element becomes 0
        int exclusive_sum = __shfl_up_sync(0xffffffff, inclusive_sum, 1);
        if (lane_id == 0) exclusive_sum = 0;
        
        smem[lane_id] = exclusive_sum;
    }
   __syncthreads_lm();
    
    // Add warp prefix to get block-level inclusive scan
    int block_prefix = smem[warp_id];
    val += block_prefix;
    
    // Save block total to workspace (last thread in block)
    if (tid == BLOCK_SIZE - 1) {
        int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
        workspace[expert_id * num_blocks + block_id] = val;
    }
}

// Warp-level exclusive scan helper function
__device__ __forceinline__ int warp_exclusive_scan(int val) {
    const int lane_id = threadIdx.x & 31;
    
    // Warp-level inclusive scan using shuffle
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n;
    }
    
    // Convert to exclusive: shift right by 1
    int exclusive_val = __shfl_up_sync(0xffffffff, val, 1);
    if (lane_id == 0) exclusive_val = 0;
    
    return exclusive_val;
}

// GPU Exclusive Prefix Sum: Single-block scan with two-level Warp Scan (Optimized)
template <int BLOCK_SIZE = 1024>
__global__ void workspace_scan_single_block(
    const int* __restrict__ input,
    int* __restrict__ output,
    const int size
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    
    __shared__ int warp_sums[NUM_WARPS];
    
    // Load data
    int val = (tid < size) ? input[tid] : 0;
    
    // Level 1: Intra-warp exclusive scan
    int exclusive_val = warp_exclusive_scan(val);
    int inclusive_val = exclusive_val + val;
    
    // Last thread in each warp stores the inclusive sum
    if (lane_id == 31) {
        warp_sums[warp_id] = inclusive_val;
    }
    __syncthreads_lm();
    
    // Level 2: Warp 0 scans the warp sums
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        int warp_sum = warp_sums[lane_id];
        int warp_prefix = warp_exclusive_scan(warp_sum);
        warp_sums[lane_id] = warp_prefix;
    }
    __syncthreads_lm();
    
    // Add warp prefix to get final exclusive scan
    int warp_prefix = warp_sums[warp_id];
    int final_exclusive = exclusive_val + warp_prefix;
    
    // Write output
    if (tid < size) {
        output[tid] = final_exclusive;
    }
}

// GPU Exclusive Prefix Sum: Multi-block scan with two-level Warp Scan (Optimized)
template <int BLOCK_SIZE = 1024>
__global__ void workspace_scan_block_sums(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ block_sums,
    const int size
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_idx = bid * BLOCK_SIZE + tid;
    
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    
    __shared__ int warp_sums[NUM_WARPS];
    
    // Load data
    int val = (global_idx < size) ? input[global_idx] : 0;
    
    // Level 1: Intra-warp exclusive scan
    int exclusive_val = warp_exclusive_scan(val);
    int inclusive_val = exclusive_val + val;
    
    // Last thread in each warp stores the inclusive sum
    if (lane_id == 31) {
        warp_sums[warp_id] = inclusive_val;
    }
    __syncthreads_lm();
    
    // Level 2: Warp 0 scans the warp sums
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        int warp_sum = warp_sums[lane_id];
        int warp_prefix = warp_exclusive_scan(warp_sum);
        warp_sums[lane_id] = warp_prefix;
    }
    __syncthreads_lm();
    
    // Add warp prefix to get final exclusive scan
    int warp_prefix = warp_sums[warp_id];
    int final_exclusive = exclusive_val + warp_prefix;
    
    // Store output
    if (global_idx < size) {
        output[global_idx] = final_exclusive;
    }
    
    // Store block sum (last thread stores total inclusive sum)
    if (tid == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[bid] = final_exclusive + val;
    }
}

// Add block prefix to finalize scan
template <int BLOCK_SIZE = 1024>
__global__ void workspace_scan_add_block_prefix(
    int* __restrict__ output,
    const int* __restrict__ block_prefix,
    const int size
) {
    const int global_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (global_idx < size && blockIdx.x > 0) {
        output[global_idx] += block_prefix[blockIdx.x];
    }
}

// Pass 2: Write final results (BIT-PACKED VERSION with int32 output for memory efficiency)
template <int BLOCK_SIZE = 256>
__global__ void make_row_id_map_pass2_kernel(
    const uint32_t* __restrict__ routing_map_packed,     // [num_tokens, num_words] BIT-PACKED
    int32_t* __restrict__ row_id_map_non_trans,          // [num_tokens, num_experts] - OPTIMIZED: int32 instead of int64
    const int* __restrict__ workspace_prefix,            // [num_experts * num_blocks] (prefix sum)
    const int num_tokens,
    const int num_experts,
    const int num_words
) {
    const int expert_id = blockIdx.x;
    const int block_id = blockIdx.y;
    const int tid = threadIdx.x;
    
    const bool valid_expert = (expert_id < num_experts);
    const int block_start = block_id * BLOCK_SIZE;
    const int token_id = block_start + tid;
    const bool valid_token = (token_id < num_tokens);
    const bool is_valid = valid_expert && valid_token;
    
    // Read from bit-packed routing map
    int expert_token_mask = 0;
    if (is_valid) {
        const uint32_t* routing_row = routing_map_packed + token_id * num_words;
        expert_token_mask = get_bit(routing_row, expert_id);
    }
    
    // Recalculate row_id_within_block using the same logic as Pass 1
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ int smem[NUM_WARPS];
    
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    
    // Warp-level inclusive scan
    int val = expert_token_mask;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n;
    }
    
    // Store warp sums
    if (lane_id == 31) {
        smem[warp_id] = val;
    }
    // BARRIER: All threads must reach here
   __syncthreads_lm();
    
    // Scan warp sums to get exclusive prefix for each warp
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        int warp_sum = smem[lane_id];
        int inclusive_sum = warp_sum;
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(0xffffffff, inclusive_sum, offset);
            if (lane_id >= offset) inclusive_sum += n;
        }
        
        // Convert to exclusive
        int exclusive_sum = __shfl_up_sync(0xffffffff, inclusive_sum, 1);
        if (lane_id == 0) exclusive_sum = 0;
        
        smem[lane_id] = exclusive_sum;
    }
    // BARRIER: All threads must reach here
   __syncthreads_lm();
    
    // Add warp prefix to get block-level inclusive scan (only if valid)
    int block_prefix = smem[warp_id];
    val += block_prefix;
    
    // Calculate and write output only for valid threads
    if (is_valid) {
        // Calculate row_id_within_block
        int row_id_within_block = val * expert_token_mask;
        
        // Get prefix sum for this block
        int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int chunk_idx = expert_id * num_blocks + block_id;
        int prefix_sum = ldg(&workspace_prefix[chunk_idx]);
        
        // Adjust row_id: if 0 (not used), set to -1; otherwise add prefix_sum and subtract 1
        int32_t final_row_id = (row_id_within_block == 0) ? -1 : (row_id_within_block + prefix_sum - 1);
        
        // Write to row_id_map_non_trans (using int32 for 2x memory savings vs int64)
        row_id_map_non_trans[token_id * num_experts + expert_id] = final_row_id;
    }
}


void make_row_id_map_kernel_launch(
    const void* routing_map_ptr,
    void* row_id_map_non_trans_ptr,
    int num_tokens,
    int num_experts,
    int* workspace_buffer,
    int* workspace_prefix_buffer,
    size_t workspace_capacity,
    int* block_sums_buffer,
    int* block_prefix_buffer,
    size_t scan_buffer_capacity,
    musaStream_t stream
) {
    if(num_tokens == 0) return;
    const uint32_t* routing_map_packed = static_cast<const uint32_t*>(routing_map_ptr);
    int32_t* row_id_map_non_trans = static_cast<int32_t*>(row_id_map_non_trans_ptr);  // OPTIMIZED: int32 instead of int64
    
    const int num_words = get_num_words(num_experts);
    
    constexpr int BLOCK_SIZE = 256;
    constexpr int SCAN_BLOCK_SIZE = 1024;
    const int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int workspace_count = num_experts * num_blocks;
    
    EP_DEVICE_ASSERT(workspace_count <= workspace_capacity and "Workspace buffer too small for make_row_id_map");
    
    int* workspace = workspace_buffer;
    int* workspace_prefix = workspace_prefix_buffer;
    
    // Pass 1: Block-level cumsum (bit-packed input)
    dim3 grid1(num_experts, num_blocks);
    dim3 block1(BLOCK_SIZE);
    
    make_row_id_map_pass1_kernel<int32_t, BLOCK_SIZE><<<grid1, block1, 0, stream>>>(
        routing_map_packed,
        workspace,
        num_tokens,
        num_experts,
        num_words
    );
    
    CUDA_RUNTIME_CHECK(musaGetLastError());
    
    // Pass 1.5: GPU exclusive prefix sum on workspace
    if (workspace_count <= SCAN_BLOCK_SIZE) {
        // Small case: single block scan
        workspace_scan_single_block<SCAN_BLOCK_SIZE><<<1, SCAN_BLOCK_SIZE, 0, stream>>>(
            workspace,
            workspace_prefix,
            workspace_count
        );
    } else {
        // Large case: multi-level scan
        const int num_scan_blocks = (workspace_count + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
        
        // Verify scan buffers are large enough
        EP_DEVICE_ASSERT(num_scan_blocks <= scan_buffer_capacity and "Scan buffer too small for make_row_id_map");
        
        int* block_sums = block_sums_buffer;
        int* block_prefix = block_prefix_buffer;
        
        // Level 1: Scan each block
        workspace_scan_block_sums<SCAN_BLOCK_SIZE><<<num_scan_blocks, SCAN_BLOCK_SIZE, 0, stream>>>(
            workspace,
            workspace_prefix,
            block_sums,
            workspace_count
        );
        
        if (num_scan_blocks > 1) {
            // Level 2: Scan block sums
            workspace_scan_single_block<SCAN_BLOCK_SIZE><<<1, SCAN_BLOCK_SIZE, 0, stream>>>(
                block_sums,
                block_prefix,
                num_scan_blocks
            );
            
            // Level 3: Add block prefix to finalize
            workspace_scan_add_block_prefix<SCAN_BLOCK_SIZE><<<num_scan_blocks, SCAN_BLOCK_SIZE, 0, stream>>>(
                workspace_prefix,
                block_prefix,
                workspace_count
            );
        }
    }
    
    CUDA_RUNTIME_CHECK(musaGetLastError());
    
    // Pass 2: Write final results (bit-packed input, int32 output)
    dim3 grid2(num_experts, num_blocks);
    dim3 block2(BLOCK_SIZE);
    
    make_row_id_map_pass2_kernel<BLOCK_SIZE><<<grid2, block2, 0, stream>>>(
        routing_map_packed,
        row_id_map_non_trans,
        workspace_prefix,
        num_tokens,
        num_experts,
        num_words
    );
    
    CUDA_RUNTIME_CHECK(musaGetLastError());
}

// ============================================================================
// moe_permute.cu - MoE Permute and Unpermute Kernels
// ============================================================================

struct uint64_vec {
    uint4 data[4];
};

struct uint32_vec {
    uint4 data[2];
};

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<64> {
    using Type = uint64_vec;
    static_assert(sizeof(Type) == 64, "uint64_vec size mismatch");
};

template <>
struct BytesToType<32> {
    using Type = uint32_vec;
    static_assert(sizeof(Type) == 32, "uint32_vec size mismatch");
};

template <>
struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16, "uint4 size mismatch");
};

template <>
struct BytesToType<8> {
    using Type = uint2;
    static_assert(sizeof(Type) == 8, "uint2 size mismatch");
};

template <>
struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4, "uint32_t size mismatch");
};

template <>
struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2, "uint16_t size mismatch");
};

template <>
struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1, "uint8_t size mismatch");
};

template <typename Elt_type, uint32_t NUM_ELT>
struct Vec {
    enum { BYTES = NUM_ELT * sizeof(Elt_type) };
    
    using Vec_type = typename BytesToType<BYTES>::Type;
    using type = Elt_type;
    
    using Alias_type = union {
        Vec_type vec;
        Elt_type elt[NUM_ELT];
    };
    
    Alias_type data;
    
    __device__ Vec() = default;
    
    __device__ Vec(const Elt_type& num) {
        #pragma unroll
        for (int i = 0; i < NUM_ELT; i++) {
            this->data.elt[i] = num;
        }
    }
    
    __device__ Vec& operator=(const Elt_type& num) {
        #pragma unroll
        for (int i = 0; i < NUM_ELT; i++) {
            this->data.elt[i] = num;
        }
        return *this;
    }
    
    inline __device__ void load_from(const void* base_ptr, size_t idx = 0) {
        this->data.vec = static_cast<const Vec_type*>(base_ptr)[idx];
    }
    
    inline __device__ void store_to(void* base_ptr, size_t idx = 0) const {
        static_cast<Vec_type*>(base_ptr)[idx] = this->data.vec;
    }
    
    inline __device__ void load_from_elts(const void* base_ptr, size_t idx = 0,
                                          size_t count = NUM_ELT) {
        const Elt_type* elt_ptr = static_cast<const Elt_type*>(base_ptr) + idx;
        if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
            #pragma unroll
            for (int it = 0; it < NUM_ELT; it++) {
                this->data.elt[it] = (it < count ? elt_ptr[it] : Elt_type(0.f));
            }
        } else {
            this->load_from(elt_ptr);
        }
    }

    inline __device__ void store_to_elts(void* base_ptr, size_t idx = 0,
                                         size_t count = NUM_ELT) const {
        Elt_type* elt_ptr = static_cast<Elt_type*>(base_ptr) + idx;
        if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
            #pragma unroll
            for (int it = 0; it < NUM_ELT; it++) {
                if (it < count) {
                    elt_ptr[it] = this->data.elt[it];
                }
            }
        } else {
            this->store_to(elt_ptr);
        }
    }
    
    inline __device__ void clear() {
        #pragma unroll
        for (int it = 0; it < NUM_ELT; it++) {
            this->data.elt[it] = Elt_type(0.f);
        }
    }
};

// OPTIMIZED: Use int32_t for row_id_map (2x memory savings vs int64_t)
template <typename SrcDtype, bool WITH_TOPK = true>
__global__ void moe_permute_mask_kernel_tme(
    void* output,
    MUtensorDescriptor in_dev_tensorDesc,
    MUtensorDescriptor map_dev_tensorDesc,
    const int num_tokens,
    const int num_experts,
    const int hidden_size,
    const int num_experts_per_rank,
    const int num_topk,
    int64_t* topk_idx,
    float* topk_weights,
    int64_t* out_topk_idx,
    float* out_topk_weights) {
    
    using IdxDtype = int32_t;  // OPTIMIZED: int32 instead of int64
    using IdxVec = Vec<IdxDtype, 4>;
    int tidx = threadIdx.x;
    int token_id = blockIdx.x;
    
    if (token_id >= num_tokens) return;
    
    int trans_count = hidden_size * sizeof(SrcDtype);
    int trans_count_map = num_experts * sizeof(IdxDtype);
    extern __shared__ __align__(128) char shared_array[];
    SrcDtype* smem = reinterpret_cast<SrcDtype*>(shared_array);
    const size_t hidden_size_aligned =
        (hidden_size * sizeof(SrcDtype) + 127) / 128 * 128 / sizeof(SrcDtype);
    IdxDtype* smem_map = reinterpret_cast<IdxDtype*>(smem + hidden_size_aligned);
    
    __musa::async_barrier bar(1);
    __musa::async_barrier bar_map(2);
    bar.init_arrival(1);
    bar_map.init_arrival(1);
    __syncthreads_lm();
    
    int ld_dim = hidden_size;
    int ld_pos = token_id * hidden_size;
    int ld_dim_map = num_experts;
    int ld_pos_map = token_id * num_experts;
    
    __musa::memcpy_async(bar_map, smem_map, &map_dev_tensorDesc, ld_dim_map, ld_pos_map,
                         trans_count_map, 0, 3, 1);
    unsigned phase_id_map = bar_map.arrive();
    
    __musa::memcpy_async(bar, smem, &in_dev_tensorDesc, ld_dim, ld_pos, trans_count, 0, 3, 1);
    unsigned phase_id = bar.arrive();
    
    bar_map.wait(phase_id_map);
    bar.wait(phase_id);
    
    IdxVec dst_row_vec;
    for (int expert_id = 0; expert_id < num_experts; expert_id += 4) {
        dst_row_vec.load_from(smem_map + expert_id);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (dst_row_vec.data.elt[i] != -1) {
                int st_dim = hidden_size;
                int st_slot = dst_row_vec.data.elt[i];
                int st_pos = st_slot * hidden_size;
                
                __musa::memcpy_blk(
                    smem, reinterpret_cast<SrcDtype*>(output) + st_pos,
                    st_dim * sizeof(SrcDtype));

                if constexpr (WITH_TOPK) {
                    if (tidx < num_topk) {
                        int current_expert = expert_id + i;
                        int responsible_rank = current_expert / num_experts_per_rank;
                        int recv_expert_begin = responsible_rank * num_experts_per_rank;
                        int recv_expert_end = recv_expert_begin + num_experts_per_rank;

                        auto idx_value = __ldg(topk_idx + token_id * num_topk + tidx);
                        idx_value = (idx_value >= recv_expert_begin && idx_value < recv_expert_end)
                                  ? idx_value - recv_expert_begin
                                  : -1;
                        out_topk_idx[st_slot * num_topk + tidx] = idx_value;

                        auto weight_value = __ldg(topk_weights + token_id * num_topk + tidx);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        out_topk_weights[st_slot * num_topk + tidx] = weight_value;
                    }
                }
            }
        }
    }

    __musa::memcpy_idf_l2();
}


// OPTIMIZED: Fused FP8 permute kernel with int32 row_id_map
template <typename TokenDtype, typename ScaleDtype, bool WITH_TOPK = true>
__global__ void moe_permute_fp8_fused_kernel_tme(
    void* output_tokens,
    MUtensorDescriptor token_in_tensorDesc,
    void* output_scales,
    MUtensorDescriptor scale_in_tensorDesc,
    MUtensorDescriptor map_dev_tensorDesc,
    const int num_tokens,
    const int num_experts,
    const int hidden_size,
    const int num_scales,
    const int num_experts_per_rank,
    const int num_topk,
    int64_t* topk_idx,
    float* topk_weights,
    int64_t* out_topk_idx,
    float* out_topk_weights) {
    
    using IdxDtype = int32_t;  // OPTIMIZED: int32 instead of int64
    using IdxVec = Vec<IdxDtype, 4>;
    int tidx = threadIdx.x;
    int token_id = blockIdx.x;
    
    if (token_id >= num_tokens) return;
    
    // Calculate memory requirements
    int trans_count_token = hidden_size * sizeof(TokenDtype);
    int trans_count_scale = num_scales * sizeof(ScaleDtype);
    int trans_count_map = num_experts * sizeof(IdxDtype);
    
    // Shared memory layout: [tokens][scales][map]
    extern __shared__ __align__(128) char shared_array[];
    TokenDtype* smem_token = reinterpret_cast<TokenDtype*>(shared_array);
    const size_t token_size_aligned = (hidden_size * sizeof(TokenDtype) + 127) / 128 * 128 / sizeof(TokenDtype);
    
    ScaleDtype* smem_scale = reinterpret_cast<ScaleDtype*>(smem_token + token_size_aligned);
    const size_t scale_size_aligned = (num_scales * sizeof(ScaleDtype) + 127) / 128 * 128 / sizeof(ScaleDtype);
    
    IdxDtype* smem_map = reinterpret_cast<IdxDtype*>(smem_scale + scale_size_aligned);
    
    // Initialize barriers
    __musa::async_barrier bar_token(1);
    __musa::async_barrier bar_scale(2);
    __musa::async_barrier bar_map(3);
    bar_token.init_arrival(1);
    bar_scale.init_arrival(1);
    bar_map.init_arrival(1);
    __syncthreads_lm();
    
    // Calculate positions
    int ld_pos_token = token_id * hidden_size;
    int ld_pos_scale = token_id * num_scales;
    int ld_pos_map = token_id * num_experts;
    
    // Launch all async copies in parallel
    __musa::memcpy_async(bar_map, smem_map, &map_dev_tensorDesc, num_experts, ld_pos_map,
                         trans_count_map, 0, 3, 1);
    unsigned phase_id_map = bar_map.arrive();
    
    __musa::memcpy_async(bar_token, smem_token, &token_in_tensorDesc, hidden_size, ld_pos_token,
                         trans_count_token, 0, 3, 1);
    unsigned phase_id_token = bar_token.arrive();
    
    __musa::memcpy_async(bar_scale, smem_scale, &scale_in_tensorDesc, num_scales, ld_pos_scale,
                         trans_count_scale, 0, 3, 1);
    unsigned phase_id_scale = bar_scale.arrive();
    
    // Wait for all data to be loaded
    bar_map.wait(phase_id_map);
    bar_token.wait(phase_id_token);
    bar_scale.wait(phase_id_scale);
    
    // Process permutation using shared row_id_map
    IdxVec dst_row_vec;
    for (int expert_id = 0; expert_id < num_experts; expert_id += 4) {
        dst_row_vec.load_from(smem_map + expert_id);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (dst_row_vec.data.elt[i] != -1) {
                int st_slot = dst_row_vec.data.elt[i];
                
                // Copy tokens
                int st_pos_token = st_slot * hidden_size;
                __musa::memcpy_blk(
                    smem_token, reinterpret_cast<TokenDtype*>(output_tokens) + st_pos_token,
                    hidden_size * sizeof(TokenDtype));
                
                // Copy scales
                int st_pos_scale = st_slot * num_scales;
                __musa::memcpy_blk(
                    smem_scale, reinterpret_cast<ScaleDtype*>(output_scales) + st_pos_scale,
                    num_scales * sizeof(ScaleDtype));

                // Process topk_idx and topk_weights (only once, not per scale)
                if constexpr (WITH_TOPK) {
                    if (tidx < num_topk) {
                        int current_expert = expert_id + i;
                        int responsible_rank = current_expert / num_experts_per_rank;
                        int recv_expert_begin = responsible_rank * num_experts_per_rank;
                        int recv_expert_end = recv_expert_begin + num_experts_per_rank;

                        auto idx_value = __ldg(topk_idx + token_id * num_topk + tidx);
                        idx_value = (idx_value >= recv_expert_begin && idx_value < recv_expert_end)
                                  ? idx_value - recv_expert_begin
                                  : -1;
                        out_topk_idx[st_slot * num_topk + tidx] = idx_value;

                        auto weight_value = __ldg(topk_weights + token_id * num_topk + tidx);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        out_topk_weights[st_slot * num_topk + tidx] = weight_value;
                    }
                }
            }
        }
    }

    __musa::memcpy_idf_l2();
}

// OPTIMIZED: Use int32_t for row_id_map (2x memory savings vs int64_t)
template <typename Dtype, int vlen, bool WITH_TOPK = false>
__global__ void moe_unpermute_mask_kernel_tme(
    const Dtype* __restrict__ in_ptr,
    Dtype* __restrict__ out_ptr,
    const int32_t* __restrict__ row_id_map_ptr,
    const float* __restrict__ topk_weights_ptr,
    float* __restrict__ out_topk_weights_ptr,
    const int num_tokens,
    const int num_experts,
    const int hidden_size,
    const int num_topk,
    const int stride_input_token,
    const int stride_input_hidden,
    const int stride_output_token,
    const int stride_output_hidden) {
    
    using IdxDtype = int32_t;  // OPTIMIZED: int32 instead of int64
    using DtypeVec = Vec<Dtype, vlen>;
    using ComputeVec = Vec<float, vlen>;
    
    int token_id = blockIdx.y;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx_vlen = (blockIdx.x * blockDim.x + threadIdx.x) * vlen;
    const int block_id = blockIdx.x + blockIdx.y * gridDim.x;

    extern __shared__ IdxDtype smem[];

    __shared__ IdxDtype valid_experts[MAX_ACE_NUM_RANK_ROWS];
    __shared__ int valid_expert_count;

    if(threadIdx.x < MAX_ACE_NUM_RANK_ROWS){
        valid_experts[threadIdx.x] = -1;
    }
    if(threadIdx.x == 0){
        valid_expert_count = 0;
    }
    __syncthreads_lm();

    // All threads scan experts and load to shared memory
    for (int expert_id = threadIdx.x; expert_id < num_experts; expert_id += blockDim.x){
        memcpy_global2shared(smem + expert_id, const_cast<IdxDtype*>(row_id_map_ptr + token_id * num_experts + expert_id), 1);

        if(smem[expert_id] != -1){
            int pos = atomicAdd(&valid_expert_count, 1);
            if(pos < MAX_ACE_NUM_RANK_ROWS){
                valid_experts[pos] = smem[expert_id];
            }
        }
    }
    __syncthreads_lm();

    ComputeVec acc_vec = 0.0f;
    if (tidx_vlen < hidden_size) {
        #pragma unroll
        for(int i = 0; i < MAX_ACE_NUM_RANK_ROWS; i++){
            if(valid_experts[i] != -1){
                int src_offset =
                    valid_experts[i] * stride_input_token + tidx_vlen * stride_input_hidden;
                DtypeVec src_val_vec = (Dtype)(0.0f);
                src_val_vec.load_from(in_ptr + src_offset);

                #pragma unroll
                for (int j = 0; j < vlen; j++) {
                    acc_vec.data.elt[j] += (float)src_val_vec.data.elt[j];
                }
            }
        }
        int dst_offset = token_id * stride_output_token + tidx_vlen * stride_output_hidden;
        #pragma unroll
        for (int i = 0; i < vlen; i++) {
            out_ptr[dst_offset + i] = (Dtype)acc_vec.data.elt[i];
        }
    }
    
    // Reduce topk_weights if WITH_TOPK is true
    // Accumulate weights from all valid_experts (each expert contributes only its own position)
    if constexpr (WITH_TOPK) {
        if (blockIdx.x == 0 && threadIdx.x < num_topk) {
            float weight_sum = 0.0f;
            #pragma unroll
            for(int i = 0; i < MAX_ACE_NUM_RANK_ROWS; i++){
                if(valid_experts[i] != -1){
                    int src_offset = valid_experts[i] * num_topk + threadIdx.x;
                    weight_sum += topk_weights_ptr[src_offset];
                }
            }
            out_topk_weights_ptr[token_id * num_topk + threadIdx.x] = weight_sum;
        }
    }
}

void moe_permute_mask_kernel_launch(
    void* output, const void* input, const void* row_id_map,
    void* out_topk_idx, void* out_topk_weights,
    const void* topk_idx, const void* topk_weights,
    int num_tokens, int num_experts, int hidden_size,
    int num_experts_per_rank, int num_topk,
    int num_out_tokens,
    musaStream_t stream,
    int dtype) { 
    if(num_tokens == 0) return;
    
    MUtensorDescriptor intensorDesc;
    MUtensorDescriptor maptensorDesc;
    
    MUtensorDescriptorDataType tensorDataType;
    switch (dtype) {
        case SCALAR_TYPE_BFLOAT16:
            tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;  // 2 bytes
            break;
        case SCALAR_TYPE_FLOAT8_E4M3FN:
        case SCALAR_TYPE_FLOAT8_E5M2:
            tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;  // 1 byte
            break;
        case SCALAR_TYPE_FLOAT32:
            tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32;  // 4 bytes
            break;
        default:
            throw std::runtime_error("Unsupported dtype for moe_permute_mask");
    }
    MUtensorDescriptorDataType mapDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32;  // OPTIMIZED: int32 instead of int64
    uint32_t tensorRank = 1;
    
    // Input tensor: [num_tokens * hidden_size]
    const uint64_t in_globalDim[5] = {static_cast<uint64_t>(hidden_size) * num_tokens, 1, 1, 1, 1};
    const uint64_t in_globalStrides[4] = {0, 0, 0, 0};
    
    // Map tensor: [num_tokens * num_experts]
    const uint64_t map_globalDim[5] = {static_cast<uint64_t>(num_experts) * num_tokens, 1, 1, 1, 1};
    const uint64_t map_globalStrides[4] = {0, 0, 0, 0};
    
    MUtensorDescriptorInterleave interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
    uint64_t oobConstantFill = 0;
    
    MUresult res;
    res = muTensorDescriptorEncode(&intensorDesc, tensorDataType, tensorRank,
                                    const_cast<void*>(input), in_globalDim, in_globalStrides,
                                    interleave, oobConstantFill);
    if (res != MUSA_SUCCESS) {
        throw std::runtime_error("Failed to encode input tensor descriptor");
    }
    
    res = muTensorDescriptorEncode(&maptensorDesc, mapDataType, tensorRank,
                                    const_cast<void*>(row_id_map), map_globalDim, map_globalStrides,
                                    interleave, oobConstantFill);
    if (res != MUSA_SUCCESS) {
        throw std::runtime_error("Failed to encode map tensor descriptor");
    }

    const int block_x = 32; 
    const int grid_x = num_tokens;
    dim3 block(block_x, 1);
    dim3 grid(grid_x, 1);
    
    // Check if topk processing is needed
    bool has_topk = (topk_idx != nullptr && topk_weights != nullptr);
    
    switch (dtype) {
        case SCALAR_TYPE_BFLOAT16: {
            size_t hidden_size_aligned = (hidden_size * sizeof(__mt_bfloat16) + 127) / 128 * 128;
            int smem_size = hidden_size_aligned + num_experts * sizeof(int32_t);  // OPTIMIZED: int32 instead of int64
            if (has_topk) {
                moe_permute_mask_kernel_tme<__mt_bfloat16, true><<<grid, block, smem_size, stream>>>(
                    output, intensorDesc, maptensorDesc,
                    num_tokens, num_experts, hidden_size,
                    num_experts_per_rank, num_topk,
                    static_cast<int64_t*>(const_cast<void*>(topk_idx)),
                    static_cast<float*>(const_cast<void*>(topk_weights)),
                    static_cast<int64_t*>(out_topk_idx),
                    static_cast<float*>(out_topk_weights)
                );
            } else {
                moe_permute_mask_kernel_tme<__mt_bfloat16, false><<<grid, block, smem_size, stream>>>(
                    output, intensorDesc, maptensorDesc,
                    num_tokens, num_experts, hidden_size,
                    num_experts_per_rank, num_topk,
                    static_cast<int64_t*>(const_cast<void*>(topk_idx)),
                    static_cast<float*>(const_cast<void*>(topk_weights)),
                    static_cast<int64_t*>(out_topk_idx),
                    static_cast<float*>(out_topk_weights)
                );
            }
            break;
        }
        case SCALAR_TYPE_FLOAT32: {
            size_t hidden_size_aligned = (hidden_size * sizeof(float) + 127) / 128 * 128;
            int smem_size = hidden_size_aligned + num_experts * sizeof(int32_t);  // OPTIMIZED: int32 instead of int64
            if (has_topk) {
                moe_permute_mask_kernel_tme<float, true><<<grid, block, smem_size, stream>>>(
                    output, intensorDesc, maptensorDesc,
                    num_tokens, num_experts, hidden_size,
                    num_experts_per_rank, num_topk,
                    static_cast<int64_t*>(const_cast<void*>(topk_idx)),
                    static_cast<float*>(const_cast<void*>(topk_weights)),
                    static_cast<int64_t*>(out_topk_idx),
                    static_cast<float*>(out_topk_weights)
                );
            } else {
                moe_permute_mask_kernel_tme<float, false><<<grid, block, smem_size, stream>>>(
                    output, intensorDesc, maptensorDesc,
                    num_tokens, num_experts, hidden_size,
                    num_experts_per_rank, num_topk,
                    static_cast<int64_t*>(const_cast<void*>(topk_idx)),
                    static_cast<float*>(const_cast<void*>(topk_weights)),
                    static_cast<int64_t*>(out_topk_idx),
                    static_cast<float*>(out_topk_weights)
                );
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for moe_permute_mask kernel launch");
    }
    
    CUDA_RUNTIME_CHECK(musaGetLastError());
}

void moe_permute_fp8_unified_kernel_launch(
    void* output_tokens, const void* input_tokens,
    void* output_scales, const void* input_scales,
    const void* row_id_map,
    void* out_topk_idx, void* out_topk_weights,
    const void* topk_idx, const void* topk_weights,
    int num_tokens, int num_experts, int hidden_size, int num_scales,
    int num_experts_per_rank, int num_topk,
    int num_out_tokens,
    musaStream_t stream,
    int token_dtype, int scale_dtype) {
    if(num_tokens == 0) return;
    
    // Use fused kernel for FP8 types 
    if ((token_dtype == SCALAR_TYPE_FLOAT8_E4M3FN || token_dtype == SCALAR_TYPE_FLOAT8_E5M2) && scale_dtype == SCALAR_TYPE_FLOAT32) {
     //if (token_dtype == 0) {
        // Encode tensor descriptors
        MUtensorDescriptor token_in_tensorDesc;
        MUtensorDescriptor scale_in_tensorDesc;
        MUtensorDescriptor map_tensorDesc;
        
        MUtensorDescriptorDataType token_dataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
        MUtensorDescriptorDataType scale_dataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32;
        MUtensorDescriptorDataType map_dataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32;  // OPTIMIZED: int32 instead of int64
        uint32_t tensorRank = 1;
        MUtensorDescriptorInterleave interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
        uint64_t oobConstantFill = 0;
        
        // Token tensors
        const uint64_t token_in_globalDim[5] = {static_cast<uint64_t>(hidden_size) * num_tokens, 1, 1, 1, 1};
        const uint64_t strides[4] = {0, 0, 0, 0};
        
        muTensorDescriptorEncode(&token_in_tensorDesc, token_dataType, tensorRank,
                                const_cast<void*>(input_tokens), token_in_globalDim, strides,
                                interleave, oobConstantFill);
        // Scale tensors
        const uint64_t scale_in_globalDim[5] = {static_cast<uint64_t>(num_scales) * num_tokens, 1, 1, 1, 1};
        
        muTensorDescriptorEncode(&scale_in_tensorDesc, scale_dataType, tensorRank,
                                const_cast<void*>(input_scales), scale_in_globalDim, strides,
                                interleave, oobConstantFill);
        // Map tensor
        const uint64_t map_globalDim[5] = {static_cast<uint64_t>(num_experts) * num_tokens, 1, 1, 1, 1};
        muTensorDescriptorEncode(&map_tensorDesc, map_dataType, tensorRank,
                                const_cast<void*>(row_id_map), map_globalDim, strides,
                                interleave, oobConstantFill);
        
        // Calculate shared memory size
        size_t token_size_aligned = (hidden_size * sizeof(__mt_fp8_e4m3) + 127) / 128 * 128;
        size_t scale_size_aligned = (num_scales * sizeof(float) + 127) / 128 * 128;
        size_t map_size = num_experts * sizeof(int32_t);  // OPTIMIZED: int32 instead of int64
        int smem_size = token_size_aligned + scale_size_aligned + map_size;
        
        // Launch fused kernel
        const int block_x = 32;
        const int grid_x = num_tokens;
        dim3 block(block_x, 1);
        dim3 grid(grid_x, 1);
        
        // Check if topk processing is needed
        bool has_topk = (topk_idx != nullptr && topk_weights != nullptr);
        
        if (has_topk) {
            moe_permute_fp8_fused_kernel_tme<__mt_fp8_e4m3, float, true><<<grid, block, smem_size, stream>>>(
                output_tokens, token_in_tensorDesc,
                output_scales, scale_in_tensorDesc,
                map_tensorDesc,
                num_tokens, num_experts, hidden_size, num_scales,
                num_experts_per_rank, num_topk,
                static_cast<int64_t*>(const_cast<void*>(topk_idx)),
                static_cast<float*>(const_cast<void*>(topk_weights)),
                static_cast<int64_t*>(out_topk_idx),
                static_cast<float*>(out_topk_weights)
            );
        } else {
            moe_permute_fp8_fused_kernel_tme<__mt_fp8_e4m3, float, false><<<grid, block, smem_size, stream>>>(
                output_tokens, token_in_tensorDesc,
                output_scales, scale_in_tensorDesc,
                map_tensorDesc,
                num_tokens, num_experts, hidden_size, num_scales,
                num_experts_per_rank, num_topk,
                static_cast<int64_t*>(const_cast<void*>(topk_idx)),
                static_cast<float*>(const_cast<void*>(topk_weights)),
                static_cast<int64_t*>(out_topk_idx),
                static_cast<float*>(out_topk_weights)
            );
        }
        
        CUDA_RUNTIME_CHECK(musaGetLastError());
    } else {
        // Fallback to sequential launches for other combinations
        moe_permute_mask_kernel_launch(
            output_tokens, input_tokens, row_id_map,
            out_topk_idx, out_topk_weights,
            topk_idx, topk_weights,
            num_tokens, num_experts, hidden_size,
            num_experts_per_rank, num_topk, num_out_tokens,
            stream, token_dtype);

        moe_permute_mask_kernel_launch(
            output_scales, input_scales, row_id_map,
            nullptr, nullptr,
            topk_idx, topk_weights,
            num_tokens, num_experts, num_scales,
            num_experts_per_rank, num_topk, num_out_tokens,
            stream, scale_dtype);
    }
}

void moe_unpermute_mask_kernel_launch(
    void* output, const void* input, const void* row_id_map,
    const void* topk_weights, void* out_topk_weights,
    int num_tokens, int num_experts, int hidden_size, int num_topk,
    musaStream_t stream,
    int dtype) {
    if(num_tokens == 0) return;
    
    const int stride_input_token = hidden_size;
    const int stride_input_hidden = 1;
    
    const int stride_output_token = hidden_size;
    const int stride_output_hidden = 1;
    
    int smem_size = num_experts * sizeof(int32_t);  // OPTIMIZED: int32 instead of int64
    
    bool has_topk = (topk_weights != nullptr && out_topk_weights != nullptr);
    
    if (dtype == SCALAR_TYPE_BFLOAT16) {
        constexpr int vlen = 16 / sizeof(__mt_bfloat16);  // vlen = 8
        
        int block_x = 128;
        int grid_x = (hidden_size + block_x * vlen - 1) / (block_x * vlen);
        int grid_y = num_tokens;
        
        dim3 block(block_x, 1);
        dim3 grid(grid_x, grid_y);
        
        if (has_topk) {
            moe_unpermute_mask_kernel_tme<__mt_bfloat16, vlen, true><<<grid, block, smem_size, stream>>>(
                static_cast<const __mt_bfloat16*>(input),
                static_cast<__mt_bfloat16*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                static_cast<const float*>(topk_weights),
                static_cast<float*>(out_topk_weights),
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        } else {
            moe_unpermute_mask_kernel_tme<__mt_bfloat16, vlen, false><<<grid, block, smem_size, stream>>>(
                static_cast<const __mt_bfloat16*>(input),
                static_cast<__mt_bfloat16*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                nullptr,
                nullptr,
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        }
    } else if (dtype == SCALAR_TYPE_FLOAT8_E4M3FN) {
        constexpr int vlen = 16 / sizeof(__mt_fp8_e4m3);  // vlen = 16
        
        int block_x = 128;
        int grid_x = (hidden_size + block_x * vlen - 1) / (block_x * vlen);
        int grid_y = num_tokens;
        
        dim3 block(block_x, 1);
        dim3 grid(grid_x, grid_y);
        
        if (has_topk) {
            moe_unpermute_mask_kernel_tme<__mt_fp8_e4m3, vlen, true><<<grid, block, smem_size, stream>>>(
                static_cast<const __mt_fp8_e4m3*>(input),
                static_cast<__mt_fp8_e4m3*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                static_cast<const float*>(topk_weights),
                static_cast<float*>(out_topk_weights),
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        } else {
            moe_unpermute_mask_kernel_tme<__mt_fp8_e4m3, vlen, false><<<grid, block, smem_size, stream>>>(
                static_cast<const __mt_fp8_e4m3*>(input),
                static_cast<__mt_fp8_e4m3*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                nullptr,
                nullptr,
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        }
    } else if (dtype == SCALAR_TYPE_FLOAT8_E5M2) {
        constexpr int vlen = 16 / sizeof(__mt_fp8_e5m2);  // vlen = 16
        
        int block_x = 128;
        int grid_x = (hidden_size + block_x * vlen - 1) / (block_x * vlen);
        int grid_y = num_tokens;
        
        dim3 block(block_x, 1);
        dim3 grid(grid_x, grid_y);
        
        if (has_topk) {
            moe_unpermute_mask_kernel_tme<__mt_fp8_e5m2, vlen, true><<<grid, block, smem_size, stream>>>(
                static_cast<const __mt_fp8_e5m2*>(input),
                static_cast<__mt_fp8_e5m2*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                static_cast<const float*>(topk_weights),
                static_cast<float*>(out_topk_weights),
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        } else {
            moe_unpermute_mask_kernel_tme<__mt_fp8_e5m2, vlen, false><<<grid, block, smem_size, stream>>>(
                static_cast<const __mt_fp8_e5m2*>(input),
                static_cast<__mt_fp8_e5m2*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                nullptr,
                nullptr,
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        }
    } else if (dtype == SCALAR_TYPE_FLOAT32) {
        constexpr int vlen = 16 / sizeof(float);  // vlen = 4
        
        int block_x = 128;
        int grid_x = (hidden_size + block_x * vlen - 1) / (block_x * vlen);
        int grid_y = num_tokens;
        
        dim3 block(block_x, 1);
        dim3 grid(grid_x, grid_y);
        
        if (has_topk) {
            moe_unpermute_mask_kernel_tme<float, vlen, true><<<grid, block, smem_size, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                static_cast<const float*>(topk_weights),
                static_cast<float*>(out_topk_weights),
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        } else {
            moe_unpermute_mask_kernel_tme<float, vlen, false><<<grid, block, smem_size, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                static_cast<int32_t*>(const_cast<void*>(row_id_map)),
                nullptr,
                nullptr,
                num_tokens,
                num_experts,
                hidden_size,
                num_topk,
                stride_input_token,
                stride_input_hidden,
                stride_output_token,
                stride_output_hidden
            );
        }
    } else {
        throw std::runtime_error("Unsupported dtype for moe_unpermute_mask");
    }
    
    CUDA_RUNTIME_CHECK(musaGetLastError());
}

} // namespace intranode_ace
} // namespace deep_ep::legacy
