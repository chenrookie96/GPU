#pragma once
#if defined(__MUSACC__)
#include<musa_bf16.h>
using __nv_bfloat16 = __mt_bfloat16;
using __nv_bfloat162 = __mt_bfloat162;
using nv_bfloat16 = __mt_bfloat16;
using nv_bfloat162 = __mt_bfloat162;
#else
using __nv_bfloat16 = short;
using __nv_bfloat162 = int;
using nv_bfloat16 = short;
using nv_bfloat162 = int;
#endif
#define CUDART_ZERO_BF16 0
#define CUDART_INF_BF16 32640
