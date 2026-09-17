#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  \
        }                                                                                                                             \
        for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += 32)                                           \
            ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                                                                               \
    }

namespace deep_ep::legacy {

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
    using vec_t = int8_t;
};
template <>
struct VecInt<2> {
    using vec_t = int16_t;
};
template <>
struct VecInt<4> {
    using vec_t = int;
};
template <>
struct VecInt<8> {
    using vec_t = int64_t;
};
template <>
struct VecInt<16> {
    using vec_t = int4;
};

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

    __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }
};

__device__ __forceinline__ void trap() {
#if USE_MUSA
#else
    asm("trap;");
#endif
}

__device__ __forceinline__ void memory_fence() {
#if USE_MUSA
    // __threadfence_system();
    asm volatile(".mtx\n" "mbar.acq_rel.sys;" ::: "memory");
#else
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_gpu() {
#if USE_MUSA
    // __threadfence();
    asm volatile(".mtx\n" "mbar.acq_rel.gpu;" ::: "memory");
#else
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_cta() {
#if USE_MUSA
    // __threadfence_block();
    asm volatile(".mtx\n" "mbar.acq_rel.cta;" ::: "memory");
#else
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void st_relaxed_sys_global(const int* ptr, int val) {
#if USE_MUSA
    // atomicExch((int *)__musa_ptr_gen_to_global((void*)ptr), val);
    __lsu_st_cache_hint(const_cast<int*>(ptr), val, 0, 2, 1, 1);
#else
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
#if USE_MUSA
    // __musa_fence_rel();
    // atomicExch((int *)__musa_ptr_gen_to_global((void*)ptr), val);
    asm volatile(".mtx\n" "st2.release.sys [%0].global, %1.s32;" :: "R"(ptr), "R"(val) : "memory");
#else
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
#if USE_MUSA
    // *(int *)ptr = val;
    asm volatile(".mtx\n" "st2.release.cta [%0].shared, %1.s32;" :: "R"(ptr), "R"(val) : "memory");
#else
    asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
#if USE_MUSA
    // ret = volatile_load((int *)ptr);
    asm volatile(".mtx\n" "ld1.acquire.sys %0.s32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t* ptr) {
    uint64_t ret;
#if USE_MUSA
    // ret = volatile_load((uint64_t *)ptr);
    asm volatile(".mtx\n" "ld1.acquire.sys %0.u64, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    int ret;
#if USE_MUSA
    // ret = volatile_load((int *)ptr);
    asm volatile(".mtx\n" "ld1.acquire.gpu %0.s32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    int ret;
#if USE_MUSA
    // __musa_fence_rel();
    // ret = atomicAdd((int *)ptr, value);
    asm volatile(".mtx\n" "atom3.release.system.global.add %0.s32, [%1], %2.s32, none;"
                 : "=R"(ret) : "R"(ptr), "R"(value));
#else
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
#endif
    return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
#if USE_MUSA
    // __musa_fence_rel();
    // ret = atomicAdd((int *)ptr, value);
    asm volatile(".mtx\n" "atom3.relaxed.gpu.global.add %0.s32, [%1], %2.s32, none;"
                 : "=R"(ret) : "R"(ptr), "R"(value));
#else
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
#endif
    return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int* ptr) {
    int ret;
#if USE_MUSA
    // ret = *ptr;
    // __threadfence_block();
    asm volatile(".mtx\n" "ld1.acquire.cta %0.s32, [%1].shared;" : "=R"(ret) :"R"(ptr));
#else
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t* ptr) {
#if USE_MUSA
    // uint8_t ret = volatile_load((uint8_t *)ptr);
    uint8_t ret;
    asm volatile(".mtx\n" "ld1.relaxed.gpu %0.b8, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
    uint16_t ret;
#if USE_MUSA
    // ret = volatile_load((uint16_t *)ptr);
    asm volatile(".mtx\n" "ld1.relaxed.gpu %0.b16, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t* ptr) {
    uint32_t ret;
#if USE_MUSA
    // ret = volatile_load((uint32_t *)ptr);
    asm volatile(".mtx\n" "ld1.relaxed.gpu %0.b32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
    uint64_t ret;
#if USE_MUSA
    // ret = volatile_load((uint64_t *)ptr);
    asm volatile(".mtx\n" "ld1.relaxed.gpu %0.b64, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    int ret;
#if USE_MUSA
    // ret = volatile_load((int *)ptr);
    ret = __lsu_ld_cache_hint(ptr, 0, 2, 1, 1, true);
#else
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float* ptr) {
    float ret;
#if USE_MUSA
    // ret = volatile_load((float *)ptr);
    ret = __lsu_ld_cache_hint(ptr, 0, 2, 1, 1, true);
#else
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t* ptr) {
    int64_t ret;
#if USE_MUSA
    // ret = volatile_load((int64_t *)ptr);
    ret = __lsu_ld_cache_hint(ptr, 0, 2, 1, 1, true);
#else
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t* ptr) {
    int64_t ret;
#if USE_MUSA
    // ret = volatile_load((uint64_t *)ptr);
    ret = __lsu_ld_cache_hint(ptr, 0, 2, 1, 1, true);
#else
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t* ptr) {
#if USE_MUSA
    uint8_t ret;
    // ret = __lsu_ld_cache_hint(ptr, 0, 0, 1, 1);
    asm volatile(".mtx\n" "ld1.cv %0.u8, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    uint16_t ret;
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
    return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
    int ret;
#if USE_MUSA
    // ret = __lsu_ld_cache_hint(ptr, 0, 0, 1, 1);
    asm volatile(".mtx\n" "ld1.cv %0.s32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t* ptr) {
    int64_t ret;
#if USE_MUSA
    // ret = __lsu_ld_cache_hint(ptr, 0, 0, 1, 1);
    asm volatile(".mtx\n" "ld1.cv %0.s64, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(const float* ptr) {
    float ret;
#if USE_MUSA
    // ret = __lsu_ld_cache_hint(ptr, 0, 0, 1, 1);
    asm volatile(".mtx\n" "ld1.cv %0.f32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
#endif
    return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2* ptr) {
    int2 ret;
#if USE_MUSA
    // ret = __lsu_ld_cache_hint(ptr, 0, 0, 1, 1);
    asm volatile(".mtx\n" "ld1.cv.v2 %0.s32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
#endif
    return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
    int4 ret;
#if USE_MUSA
    // ret = __lsu_ld_cache_hint(ptr, 0, 0, 1, 1);
    asm volatile(".mtx\n" "ld1.cv.v4 %0.s32, [%1].global;" : "=R"(ret) : "R"(ptr));
#else
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
#if USE_MUSA
    volatile_store(val,(uint8_t *)ptr);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
#if USE_MUSA
    volatile_store(val,(uint16_t *)ptr);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
#if USE_MUSA
    volatile_store(val,(uint32_t *)ptr);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int* ptr, int val) {
#if USE_MUSA
    volatile_store(val,(int *)ptr);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
#if USE_MUSA
    volatile_store(val,(int4 *)ptr);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
#if USE_MUSA
    asm volatile(".mtx\n" "st2.release.gpu [%0].global, %1.b32;" :: "R"(ptr), "R"(val) : "memory");
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
#if USE_MUSA
    asm volatile(".mtx\n" "st2.release.gpu [%0].global, %1.b32;" :: "R"(ptr), "R"(val) : "memory");
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
#if USE_MUSA
    asm volatile(".mtx\n" "st2.release.gpu [%0].global, %1.b64;" :: "R"(ptr), "R"(val) : "memory");
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#endif
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__ __forceinline__ void st_na_global(const int* ptr, const int& value) {
#if USE_MUSA
    // asm volatile(".mtx\n" "st2.volatile [%0].global, %1.s32;" :: "R"(ptr), "R"(value) : "memory");
    asm volatile(".mtx\n" "st2 [%0].global, %1.s32;" :: "R"(ptr), "R"(value) : "memory");
#else
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t* ptr, const int64_t& value) {
#if USE_MUSA
    // asm volatile(".mtx\n" "st2.volatile [%0].global, %1.s64;" :: "R"(ptr), "R"(value) : "memory");
    asm volatile(".mtx\n" "st2 [%0].global, %1.s64;" :: "R"(ptr), "R"(value) : "memory");
#else
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(const float* ptr, const float& value) {
#if USE_MUSA
    // asm volatile(".mtx\n" "st2.volatile [%0].global, %1.f32;" :: "R"(ptr), "R"(value) : "memory");
    asm volatile(".mtx\n" "st2 [%0].global, %1.f32;" :: "R"(ptr), "R"(value) : "memory");
#else
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
#if USE_MUSA
    // asm volatile(".mtx\n" "st2.v4 [%0].global, %1.s32;" :: "R"(ptr), "R"(value) : "memory");
    __lsu_st_cache_hint(const_cast<int4*>(ptr), const_cast<int4&>(value), 4, 2, 1, 1);
#else
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
#endif
}

__device__ __forceinline__ int32_t ldg(const int32_t *ptr) {
    int32_t ret = __lsu_ld_cache_hint(ptr, 3, 3, 0, 0);
    return ret;
}

__device__ __forceinline__ int64_t ldg(const int64_t *ptr) {
    int64_t ret = __lsu_ld_cache_hint(ptr, 3, 3, 0, 0);
    return ret;
}

__device__ __forceinline__ int4 ldg(const int4 *ptr) {
    int4 ret = __lsu_ld_cache_hint(ptr, 3, 3, 0, 0);
    return ret;
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_normal(const dtype_t *ptr) {
    auto ret = ld_normal(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_normal(const uint8_t *ptr) {
    return *ptr;
}

template <>
__device__  __forceinline__ int ld_normal(const int *ptr) {
    return *((int *)ptr);
}

template <>
__device__  __forceinline__ int64_t ld_normal(const int64_t *ptr) {
    return *((int64_t *)ptr);
}

template <>
__device__  __forceinline__ float ld_normal(const float *ptr) {
    return *((float *)ptr);
}

template <>
__device__  __forceinline__ int2 ld_normal(const int2 *ptr) {
    return *((int2 *)ptr);
}

template <>
__device__  __forceinline__ int4 ld_normal(const int4 *ptr) {
    return *((int4 *)ptr);
}

template <typename dtype_t>
__device__  __forceinline__ void st_volatile_global(const dtype_t *ptr, const dtype_t& value) {
    st_volatile_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                       *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__  __forceinline__ void st_volatile_global(const int *ptr, const int& value) {
    volatile_store(value,(int *)ptr);
}

template <>
__device__  __forceinline__ void st_volatile_global(const int4 *ptr, const int4& value) {
    volatile_store(value,(int4 *)ptr);
}

template <>
__device__  __forceinline__ void st_volatile_global(const int64_t *ptr, const int64_t& value) {
    volatile_store(value,(int64_t *)ptr);
}

template <>
__device__  __forceinline__ void st_volatile_global(const float *ptr, const float& value) {
    volatile_store(value,(float *)ptr);
}

template <typename dtype_t>
__device__  __forceinline__ void st_normal(const dtype_t *ptr, const dtype_t& value) {
    st_normal(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
              *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template<>
__device__  __forceinline__ void st_normal(const int *ptr, const int& value) {
    *(int *)ptr = value;
}

template<>
__device__  __forceinline__ void st_normal(const int64_t *ptr, const int64_t& value) {
    *(int64_t *)ptr = value;
}

template<>
__device__  __forceinline__ void st_normal(const float *ptr, const float& value) {
    *(float *)ptr = value;
}

template<>
__device__  __forceinline__ void st_normal(const int4 *ptr, const int4& value) {
    *(int4 *)ptr = value;
}

__device__ __forceinline__ float log2f_approx(const float& x) {
    float ret;
#if USE_MUSA
#else
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
#endif
    return ret;
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
    float ret;
#if USE_MUSA
#else
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
#endif
    return ret;
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
#if USE_MUSA
    asm volatile(".mtx\n" "mov1 %0.s32, laneid;" : "=R"(lane_id));
#else
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
#endif
    return lane_id;
}

__device__ __forceinline__ uint32_t elect_one_sync() {
#if !defined(DISABLE_SM90_FEATURES) && !USE_MUSA
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
}

// TMA PTX instructions
#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ void fence_barrier_init() {
#if USE_MUSA
#else
    asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
#endif
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
#if USE_MUSA
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_int_ptr));
#endif
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
#if USE_MUSA
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbar_int_ptr));
#endif
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
#if USE_MUSA
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    const auto wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::"r"(mbar_int_ptr),
        "r"(wait),
        "r"(0x989680));
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
#endif
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
#if USE_MUSA
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes), "r"(mbar_int_ptr));
#endif
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
#if USE_MUSA
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(mbar_int_ptr));
#endif
}

__device__ __forceinline__ void tma_store_fence() {
#if USE_MUSA
#else
    asm volatile("fence.proxy.async.shared::cta;");
#endif
}

__device__ __forceinline__ void fence_view_async_shared() {
#if USE_MUSA
#else
    asm volatile (
        "{\n\t"
        "fence.proxy.async.shared::cta; \n"
        "}"
        ::
        : "memory");
#endif
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(
    const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes, bool evict_first = true) {
#if USE_MUSA
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "r"(num_bytes),
        "r"(mbar_int_ptr),
        "l"(cache_hint)
        : "memory");
#endif
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes, bool evict_first = true) {
#if USE_MUSA
#else
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(gmem_ptr),
                 "r"(smem_int_ptr),
                 "r"(num_bytes),
                 "l"(cache_hint)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;");
#endif
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
#if USE_MUSA
#else
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
#endif
}

#endif

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

__forceinline__ __device__ float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd(barrier_signal_ptrs[rank] + thread_id, LEGACY_FINISHED_SUM_TAG);
        atomicSub(barrier_signal_ptrs[thread_id] + rank, LEGACY_FINISHED_SUM_TAG);
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = clock64();
    while (true) {
        auto value = ld_volatile_global(barrier_signal_ptrs[rank] + thread_id);
        if ( value <= 0 )
            break;

#ifdef ENABLE_TIMEOUT_CHECK
        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
#endif
    }
    }
    __syncthreads();
}

__forceinline__ __device__ void grid_barrier(int num_blocks, int* barrier_signal_ptr) {
    auto thread_id = static_cast<int>(threadIdx.x);
    __syncthreads();
    if (thread_id == 0) {
        __threadfence_system_noflush();

        atomicAdd(barrier_signal_ptr, 1);
        while (ld_volatile_global(barrier_signal_ptr) != num_blocks);
    }
    __syncthreads();
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    int ret;
#if USE_MUSA
    asm volatile(".mtx\n" "atom3.acq.wg.shared.cas %0.b32, [%1], %2.b32, %3.b32;"
                 : "=R"(ret) : "R"(addr), "R"(x), "R"(y) : "memory");
#else
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");
#endif
    return ret;
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    int ret;
#if USE_MUSA
    asm volatile(".mtx\n" "atom3.release.wg.shared.exch %0.b32, [%1], %2.b32, none;"
                 : "=R"(ret) : "R"(addr), "R"(x) : "memory");
#else
    asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");
#endif
    return ret;
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
        ;
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    atomic_exch_cta_release(mutex, 0);
}

// Operation functors
template <typename T>
struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
    __device__ T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
    __device__ T operator()(T a, T b) const { return a | b; }
};

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                         kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, __shfl_xor_sync(mask, value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, __shfl_xor_sync(mask, value, 1));
    }
    return value;
}

// Convenience aliases
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

}  // namespace deep_ep::legacy
