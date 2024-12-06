/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * PTX intrinsics
 */


#pragma once

#include "util_type.cuh"
#include "util_arch.cuh"
#include "util_namespace.cuh"
#include "util_debug.cuh"


CUB_NAMESPACE_BEGIN


/**
 * \addtogroup UtilPtx
 * @{
 */


/******************************************************************************
 * PTX helper macros
 ******************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Register modifier for pointer-types (for inlining PTX assembly)
 */
#if defined(_WIN64) || defined(__LP64__)
    #define __CUB_LP64__ 1
    // 64-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "l"
    #define _CUB_ASM_PTR_SIZE_ "u64"
#else
    #define __CUB_LP64__ 0
    // 32-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "r"
    #define _CUB_ASM_PTR_SIZE_ "u32"
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS


/******************************************************************************
 * Inlined PTX intrinsics
 ******************************************************************************/

/**
 * \brief Shift-right then add.  Returns (\p x >> \p shift) + \p addend.
 */
__device__ __forceinline__ unsigned int SHR_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
#ifdef USE_GPU_FUSION_PTX
    unsigned int ret;
    asm ("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
    return ret;
#else //USE_GPU_FUSION_PTX
    return (x >> shift) + addend;
#endif //USE_GPU_FUSION_PTX
}


/**
 * \brief Shift-left then add.  Returns (\p x << \p shift) + \p addend.
 */
__device__ __forceinline__ unsigned int SHL_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
#ifdef USE_GPU_FUSION_PTX
    unsigned int ret;
    asm ("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
    return ret;
#else //USE_GPU_FUSION_PTX
    return (x << shift) + addend;
#endif //USE_GPU_FUSION_PTX
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Bitfield-extract.
 */
template <typename UnsignedBits, int BYTE_LEN>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<BYTE_LEN>      /*byte_len*/)
{
#ifdef USE_GPU_FUSION_PTX
    unsigned int bits;
    asm ("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else //USE_GPU_FUSION_PTX
    const unsigned int MASK = (1U << num_bits) - 1;
    return (source >> bit_start) & MASK;
#endif //USE_GPU_FUSION_PTX
}


/**
 * Bitfield-extract for 64-bit types.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<8>             /*byte_len*/)
{
    const unsigned long long MASK = (1ull << num_bits) - 1;
    return (source >> bit_start) & MASK;
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Bitfield-extract.  Extracts \p num_bits from \p source starting at bit-offset \p bit_start.  The input \p source may be an 8b, 16b, 32b, or 64b unsigned integer type.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits source,
    unsigned int bit_start,
    unsigned int num_bits)
{
    return BFE(source, bit_start, num_bits, Int2Type<sizeof(UnsignedBits)>());
}


/**
 * \brief Bitfield insert.  Inserts the \p num_bits least significant bits of \p y into \p x at bit-offset \p bit_start.
 */
__device__ __forceinline__ void BFI(
    unsigned int &ret,
    unsigned int x,
    unsigned int y,
    unsigned int bit_start,
    unsigned int num_bits)
{
#ifdef USE_GPU_FUSION_PTX
    asm ("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(y), "r"(x), "r"(bit_start), "r"(num_bits));
#else //USE_GPU_FUSION_PTX
    y <<= bit_start;
    unsigned int MASK_Y = ((1 << num_bits) - 1) << bit_start;
    unsigned int MASK_X = ~MASK_Y;
    ret = (x & MASK_X) | (y & MASK_Y);
#endif //USE_GPU_FUSION_PTX
}


/**
 * \brief Three-operand add.  Returns \p x + \p y + \p z.
 */
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
#ifdef USE_GPU_FUSION_PTX
    asm ("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
    return x;
#else //USE_GPU_FUSION_PTX
    return x + y + z;
#endif //USE_GPU_FUSION_PTX
}


/**
 * \brief Byte-permute. Pick four arbitrary bytes from two 32-bit registers, and reassemble them into a 32-bit destination register.  For SM2.0 or later.
 *
 * \par
 * The bytes in the two source registers \p a and \p b are numbered from 0 to 7:
 * {\p b, \p a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}. For each of the four bytes
 * {b3, b2, b1, b0} selected in the return value, a 4-bit selector is defined within
 * the four lower "nibbles" of \p index: {\p index } = {n7, n6, n5, n4, n3, n2, n1, n0}
 *
 * \par Snippet
 * The code snippet below illustrates byte-permute.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     int a        = 0x03020100;
 *     int b        = 0x07060504;
 *     int index    = 0x00007531;
 *
 *     int selected = PRMT(a, b, index);    // 0x07050301
 *
 * \endcode
 *
 */
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
#ifdef USE_GPU_FUSION_PTX
    int ret;
    asm ("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
    return ret;
#else //USE_GPU_FUSION_PTX
    return __byte_perm(a, b, index);
#endif //USE_GPU_FUSION_PTX
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Sync-threads barrier.
 */
__device__ __forceinline__ void BAR(int count)
{
#ifdef USE_GPU_FUSION_PTX
    asm volatile("bar.sync 1, %0;" : : "r"(count));
#else //USE_GPU_FUSION_PTX
    __syncthreads_count(count);
#endif //USE_GPU_FUSION_PTX
}

/**
 * CTA barrier
 */
__device__  __forceinline__ void CTA_SYNC()
{
    __syncthreads();
}


/**
 * CTA barrier with predicate
 */
__device__  __forceinline__ int CTA_SYNC_AND(int p)
{
    return __syncthreads_and(p);
}


/**
 * CTA barrier with predicate
 */
__device__  __forceinline__ int CTA_SYNC_OR(int p)
{
    return __syncthreads_or(p);
}


/**
 * Warp barrier
 */
__device__  __forceinline__ void WARP_SYNC(unsigned long long member_mask)
{
// #ifdef CUB_USE_COOPERATIVE_GROUPS
//     __syncwarp(member_mask); //bug 
  // __syncwarp(member_mask);
    // __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    // __builtin_amdgcn_wave_barrier();
    // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
// #endif
}


/**
 * Warp any
 */
__device__  __forceinline__ int WARP_ANY(int predicate, unsigned long long member_mask)
{
// #ifdef CUB_USE_COOPERATIVE_GROUPS
//     return __any_sync(member_mask, predicate);
// #else
//     return ::__any(predicate);
// #endif
 return ::__any(predicate);
}


/**
 * Warp any
 */
__device__  __forceinline__ int WARP_ALL(int predicate, unsigned long long member_mask)
{
// #ifdef CUB_USE_COOPERATIVE_GROUPS
//     return __all_sync(member_mask, predicate);
// #else
//     return ::__all(predicate);
// #endif
 return ::__all(predicate);
}


/**
 * Warp ballot
 */
__device__  __forceinline__ unsigned long long WARP_BALLOT(int predicate, unsigned long long member_mask)
{
// #ifdef CUB_USE_COOPERATIVE_GROUPS
//     return __ballot64(predicate);
// #else
//     return __ballot(predicate);
// #endif
return __ballot64(predicate);
}

// __device__  __forceinline__ unsigned long long WARP_BALLOT(int predicate, unsigned long long member_mask)
// {
//     auto&& ret = __ballot64(predicate);
//     return *(unsigned long long*)&ret;
// }

/**
 * Warp synchronous shfl_up
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ 
unsigned int SHFL_UP_SYNC(unsigned int word, int src_offset, int flags, unsigned long long member_mask)
{
#ifdef CUB_USE_COOPERATIVE_GROUPS
    asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
#else
    asm volatile("shfl.up.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags));
#endif
    return word;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ 
unsigned int SHFL_UP_SYNC(unsigned int word, int src_offset, int first_thread, int width, unsigned long long member_mask)  
{
    return __shfl_up_sync(member_mask, word, src_offset, width);
}
#endif //USE_GPU_FUSION_PTX

/**
 * Warp synchronous shfl_down
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ 
unsigned int SHFL_DOWN_SYNC(unsigned int word, int src_offset, int flags, unsigned long long member_mask)
{
#ifdef CUB_USE_COOPERATIVE_GROUPS
    asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
#else
    asm volatile("shfl.down.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags));
#endif
    return word;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ 
unsigned int SHFL_DOWN_SYNC(unsigned int word, int src_offset, int last_thread, int width, unsigned long long member_mask)
{
    return __shfl_down_sync(member_mask, word, src_offset, width);
}
#endif //USE_GPU_FUSION_PTX

/**
 * Warp synchronous shfl_idx
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ 
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, int flags, unsigned long long member_mask)
{
#ifdef CUB_USE_COOPERATIVE_GROUPS
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
        : "=r"(word) : "r"(word), "r"(src_lane), "r"(flags), "r"(member_mask));
#else
    asm volatile("shfl.idx.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_lane), "r"(flags));
#endif
    return word;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ 
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, int width, unsigned long long member_mask)
{
    return __shfl_sync(member_mask, word, src_lane, width);
}
#endif //USE_GPU_FUSION_PTX
/**
 * Warp synchronous shfl_idx
 */
__device__ __forceinline__ 
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, unsigned long long member_mask)
{
#ifdef CUB_USE_COOPERATIVE_GROUPS
  return __shfl_sync(member_mask, word, src_lane);
#else
  return __shfl(word, src_lane);
#endif
}

/**
 * Floating point multiply. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
#ifdef USE_GPU_FUSION_PTX
    float d;
    asm ("mul.rz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
#else //USE_GPU_FUSION_PTX
    return __fmul_rz(a, b);
#endif //USE_GPU_FUSION_PTX
}


/**
 * Floating point multiply-add. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
#ifdef USE_GPU_FUSION_PTX
    float d;
    asm ("fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
#else //USE_GPU_FUSION_PTX
    return __fma_rz(a, b, c);
#endif //USE_GPU_FUSION_PTX
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Terminates the calling thread
 */
__device__ __forceinline__ void ThreadExit() {
    //asm volatile("exit;");
    //asm volatile("ret;");
}    


/**
 * \brief  Abort execution and generate an interrupt to the host CPU
 */
__device__ __forceinline__ void ThreadTrap() {
    //asm volatile("trap;");
}


/**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
__device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
            ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
            threadIdx.x;
}


/**
 * \brief Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
    unsigned int ret;
#ifdef USE_GPU_FUSION_PTX
    asm("mov.u32 %0, %%laneid;" : "=r"(ret));
#else //USE_GPU_FUSION_PTX
    ret = threadIdx.x % warpSize;
#endif //USE_GPU_FUSION_PTX
    return ret;
}


/**
 * \brief Returns the warp ID of the calling thread.  Warp ID is guaranteed to be unique among warps, but may not correspond to a zero-based ranking within the thread block.
 */
__device__ __forceinline__ unsigned int WarpId()
{
    unsigned int ret;
#ifdef USE_GPU_FUSION_PTX
    asm("mov.u32 %0, %%warpid;" : "=r"(ret));
#else //USE_GPU_FUSION_PTX
    ret = threadIdx.x / warpSize;
#endif //USE_GPU_FUSION_PTX
    return ret;
}

/**
 * \brief Returns the warp lane mask of all lanes less than the calling thread
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned int LaneMaskLt()
{
    unsigned int ret;
    asm ("mov.u32 %0, %%lanemask_lt;" : "=r"(ret) );
    return ret;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned long long LaneMaskLt()
{
    return __lanemask_lt();
}
#endif //USE_GPU_FUSION_PTX
/**
 * \brief Returns the warp lane mask of all lanes less than or equal to the calling thread
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned int LaneMaskLe()
{
    unsigned int ret;
    asm ("mov.u32 %0, %%lanemask_le;" : "=r"(ret) );
    return ret;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned long long LaneMaskLe()
{
    return ~__lanemask_gt();
}
#endif //USE_GPU_FUSION_PTX
/**
 * \brief Returns the warp lane mask of all lanes greater than the calling thread
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned int LaneMaskGt()
{
    unsigned int ret;
    asm ("mov.u32 %0, %%lanemask_gt;" : "=r"(ret) );
    return ret;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned long long LaneMaskGt()
{
    return __lanemask_gt();
}
#endif //USE_GPU_FUSION_PTX
/**
 * \brief Returns the warp lane mask of all lanes greater than or equal to the calling thread
 */
#ifdef USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned int LaneMaskGe()
{
    unsigned int ret;
    asm ("mov.u32 %0, %%lanemask_ge;" : "=r"(ret) );
    return ret;
}
#else //USE_GPU_FUSION_PTX
__device__ __forceinline__ unsigned long long LaneMaskGe()
{
    return ~__lanemask_lt();
}
#endif //USE_GPU_FUSION_PTX
/** @} */       // end group UtilPtx




/**
 * \brief Shuffle-up for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><em>i</em>-<tt>src_offset</tt></sub>.  For thread lanes \e i < src_offset, the thread's own \p input is returned to the thread. ![](shfl_up_logo.png)
 * \ingroup WarpModule
 *
 * \tparam LOGICAL_WARP_THREADS     The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 * \tparam T                        <b>[inferred]</b> The input/output element type
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * predecessor of its predecessor.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleUp<32>(thread_data, 2, 0, 0xffffffff);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{1.0, 2.0, 1.0, 2.0, 3.0, ..., 30.0}</tt>.
 *
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleUp(
    T               input,              ///< [in] The value to broadcast
    int             src_offset,         ///< [in] The relative down-offset of the peer to read from
    int             first_thread,       ///< [in] Index of first lane in logical warp (typically 0)
    unsigned long long    member_mask)        ///< [in] 32-bit mask of participating warp lanes
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        
        //SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
        SHFL_C = (64 - LOGICAL_WARP_THREADS) << 8
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
 
    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
#ifdef USE_GPU_FUSION_PTX
    shuffle_word = SHFL_UP_SYNC((unsigned int)input_alias[0], src_offset, first_thread | SHFL_C, member_mask);
#else //USE_GPU_FUSION_PTX
    shuffle_word = SHFL_UP_SYNC((unsigned int)input_alias[0], src_offset, first_thread, LOGICAL_WARP_THREADS, member_mask);
#endif //USE_GPU_FUSION_PTX
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
#ifdef USE_GPU_FUSION_PTX
        shuffle_word = SHFL_UP_SYNC((unsigned int)input_alias[WORD], src_offset, first_thread | SHFL_C, member_mask);
#else //USE_GPU_FUSION_PTX
        shuffle_word = SHFL_UP_SYNC((unsigned int)input_alias[WORD], src_offset, first_thread, LOGICAL_WARP_THREADS, member_mask);
#endif //USE_GPU_FUSION_PTX
        output_alias[WORD] = shuffle_word;
    }

    return output;
}


/**
 * \brief Shuffle-down for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><em>i</em>+<tt>src_offset</tt></sub>.  For thread lanes \e i >= WARP_THREADS, the thread's own \p input is returned to the thread.  ![](shfl_down_logo.png)
 * \ingroup WarpModule
 *
 * \tparam LOGICAL_WARP_THREADS     The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 * \tparam T                        <b>[inferred]</b> The input/output element type
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * successor of its successor.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleDown<32>(thread_data, 2, 31, 0xffffffff);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{3.0, 4.0, 5.0, 6.0, 7.0, ..., 32.0}</tt>.
 *
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleDown(
    T               input,              ///< [in] The value to broadcast
    int             src_offset,         ///< [in] The relative up-offset of the peer to read from
    int             last_thread,        ///< [in] Index of last thread in logical warp (typically 31 for a 32-thread warp)
    unsigned long long    member_mask)        ///< [in] 32-bit mask of participating warp lanes
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        //SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
        SHFL_C = (64 - LOGICAL_WARP_THREADS) << 8
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
#ifdef USE_GPU_FUSION_PTX
    shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[0], src_offset, last_thread | SHFL_C, member_mask);
#else //USE_GPU_FUSION_PTX
    shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[0], src_offset, last_thread, LOGICAL_WARP_THREADS, member_mask);
#endif //USE_GPU_FUSION_PTX
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
#ifdef USE_GPU_FUSION_PTX
        shuffle_word       = SHFL_DOWN_SYNC((unsigned int)input_alias[WORD], src_offset, last_thread | SHFL_C, member_mask);
#else //USE_GPU_FUSION_PTX
        shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[WORD], src_offset, last_thread, LOGICAL_WARP_THREADS, member_mask);
#endif //USE_GPU_FUSION_PTX
        output_alias[WORD] = shuffle_word;
    }

    return output;
}


/**
 * \brief Shuffle-broadcast for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input
 * contributed by <em>warp-lane</em><sub><tt>src_lane</tt></sub>.  For \p src_lane < 0 or \p src_lane >= WARP_THREADS,
 * then the thread's own \p input is returned to the thread. ![](shfl_broadcast_logo.png)
 *
 * \tparam LOGICAL_WARP_THREADS     The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 * \tparam T                        <b>[inferred]</b> The input/output element type
 *
 * \ingroup WarpModule
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from <em>warp-lane</em><sub>0</sub>.
 *
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from thread 0
 *     double peer_data = ShuffleIndex<32>(thread_data, 0, 0xffffffff);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0}</tt>.
 *
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleIndex(
    T               input,                  ///< [in] The value to broadcast
    int             src_lane,               ///< [in] Which warp lane is to do the broadcasting
    unsigned long long   member_mask)            ///< [in] 32-bit mask of participating warp lanes
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        //SHFL_C = ((32 - LOGICAL_WARP_THREADS) << 8) | (LOGICAL_WARP_THREADS - 1)
        SHFL_C = ((64 - LOGICAL_WARP_THREADS) << 8) | (LOGICAL_WARP_THREADS - 1)
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
#ifdef USE_GPU_FUSION_PTX
    shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[0],
                                 src_lane,
                                 SHFL_C,
                                 member_mask);
#else //USE_GPU_FUSION_PTX
    shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[0],
                                src_lane,
                                LOGICAL_WARP_THREADS,
                                member_mask);
#endif //USE_GPU_FUSION_PTX
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
#ifdef USE_GPU_FUSION_PTX
        shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[WORD],
                                     src_lane,
                                     SHFL_C,
                                     member_mask);
#else //USE_GPU_FUSION_PTX
        shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[WORD],
                                     src_lane,
                                     LOGICAL_WARP_THREADS,
                                     member_mask);
#endif //USE_GPU_FUSION_PTX
        output_alias[WORD] = shuffle_word;
    }

    return output;
}



/**
 * Compute a 32b mask of threads having the same least-significant
 * LABEL_BITS of \p label as the calling thread.
 */
#if USE_GPU_FUSION_PTX
template <int LABEL_BITS>
inline __device__ unsigned int MatchAny(unsigned int label)
{
    unsigned int retval;
    // Extract masks of common threads for each bit
    #pragma unroll
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
            long long  mask;
            unsigned int current_bit = 1 << BIT;
            asm (".reg .pred p;"
            "and.b32 %0, %1, %2;"
            "setp.eq.u32 p, %0, %2;"
            "vote.ballot.sync.b32 %0, p, 0xffffffffffffffff;"
            "@!p not.b32 %0, %0;"
            : "=l"(mask) : "r"(label), "r"(current_bit));
            retval = (BIT == 0) ? mask : retval & mask;
    }

    return retval;
}
#else //USE_GPU_FUSION_PTX
template <int LABEL_BITS>
__forceinline__ __device__ unsigned long long MatchAny(unsigned int label)
{
    unsigned long long retval;
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
        unsigned long long current_bit = 1 << BIT;
        unsigned long long mask = current_bit & label;
        bool p = mask == current_bit;
        mask = __ballot_sync(0xfffffffffffffffful, p);
        mask = p ? mask : ~mask;
        retval = (BIT == 0) ? mask : retval & mask;
    }
    return retval;
}
#endif //USE_GPU_FUSION_PTX
CUB_NAMESPACE_END
