#pragma once

#include <deep_ep/common/compiled.cuh>

#define LEGACY_NUM_MAX_NVL_PEERS 8
#define LEGACY_NUM_MAX_RDMA_PEERS 20
#define LEGACY_NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define LEGACY_NUM_MAX_LOCAL_EXPERTS 1024
#define LEGACY_NUM_BUFFER_ALIGNMENT_BYTES 256

#define LEGACY_LOW_LATENCY_SEND_PHASE 1
#define LEGACY_LOW_LATENCY_RECV_PHASE 2

#define LEGACY_FINISHED_SUM_TAG 1024
#define LEGACY_NUM_WAIT_NANOSECONDS 500

#define LEGACY_NUM_CPU_TIMEOUT_SECS 100
#define LEGACY_NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s

// Scalar type definitions for dtype identification
#define SCALAR_TYPE_BFLOAT16 15
#define SCALAR_TYPE_FLOAT8_E4M3FN 23
#define SCALAR_TYPE_FLOAT8_E5M2 24
#define SCALAR_TYPE_FLOAT32 6

constexpr uint32_t MAX_ACE_BATCH_SIZE = 4;
// Defaults used when the caller does not provide ACE allocation dimensions.
// Runtime token/hidden sizes are validated against the allocation selected by
// the caller and are not capped by these values.
constexpr uint64_t DEFAULT_ACE_TOKEN_NUM = 8192;
constexpr uint64_t DEFAULT_ACE_HIDDEN_SIZE = 7168;
// Default allocation for callers that do not provide alloc_num_topk.  The
// actual raw top-k value is runtime-configured and is not capped here.
constexpr uint64_t DEFAULT_ACE_NUM_TOPK = 8;
// ACE dispatch deduplicates experts by rank, so one token contributes at most
// one aggregate row per NVLink peer to the unpermute kernel.
constexpr uint64_t MAX_ACE_NUM_RANK_ROWS = LEGACY_NUM_MAX_NVL_PEERS;
constexpr uint64_t DEFAULT_ACE_NUM_SCALES = DEFAULT_ACE_HIDDEN_SIZE / 128;
constexpr uint64_t NUM_COMBINE_INPUT_BYTES_PER_ZCOPY_BUFFER  = 8 * DEFAULT_ACE_TOKEN_NUM * 16384;
constexpr uint64_t NUM_DISPATCH_INPUT_BYTES_PER_ZCOPY_BUFFER = 8 * DEFAULT_ACE_TOKEN_NUM * 16384;
constexpr uint32_t MAX_ACE_BUFFER = 4;
