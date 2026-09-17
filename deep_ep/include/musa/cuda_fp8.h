#pragma once
#if defined(__MUSACC__)
#include<musa_fp8.h>
using __nv_fp8_storage_t = __mt_fp8_storage_t;
using __nv_fp8x2_storage_t = __mt_fp8x2_storage_t;
using __nv_fp8_e4m3 = __mt_fp8_e4m3;
using __nv_fp8x2_e4m3 = __mt_fp8x2_e4m3;
using __nv_fp8x4_e4m3 = __mt_fp8x4_e4m3;
using __nv_fp8_e5m2 = __mt_fp8_e5m2;
using __nv_fp8x2_e5m2 = __mt_fp8x2_e5m2;
using __nv_fp8x4_e5m2 = __mt_fp8x4_e5m2;
using __nv_fp8_e8m0 = __mt_fp8_e8m0;
using __nv_fp8x2_e8m0 = __mt_fp8x2_e8m0;
using __nv_fp8x4_e8m0 = __mt_fp8x4_e8m0;
#else
using __nv_fp8_e4m3 = char;
using __nv_fp8x2_e4m3 = short;
using __nv_fp8x4_e4m3 = int;
using __nv_fp8_e5m2 = char;
using __nv_fp8x2_e5m2 = short;
using __nv_fp8x4_e5m2 = int;
using __nv_fp8_e8m0 = char;
using __nv_fp8x2_e8m0 = short;
using __nv_fp8x4_e8m0 = int;
#endif
#define __nv_cvt_float2_to_fp8x2 __musa_cvt_float2_to_fp8x2
#define __NV_SATFINITE __MT_SATFINITE
#define __NV_E4M3 __MT_E4M3
