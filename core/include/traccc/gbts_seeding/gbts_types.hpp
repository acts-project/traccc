/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"      // traccc::constant
#include "traccc/definitions/qualifiers.hpp"  // TRACCC_HOST_DEVICE, TRACCC_ALIGN

// System include(s).
#include <cmath>

// ---------------------------------------------------------------------------
// Edge precision toggle
// ---------------------------------------------------------------------------
// 1 = use a fp16 edge type where a device compiler provides one (CUDA and
// SYCL); 0 = use float everywhere. Flip this 1 -> 0 to fall back to float and
// test whether fp16 is needed. A host / CPU build is built with float
#define TRACCC_GBTS_EDGE_HALF 1

#if TRACCC_GBTS_EDGE_HALF && defined(__CUDACC__)
#include <cuda_fp16.h>
#elif TRACCC_GBTS_EDGE_HALF && \
    (defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__))
#include <sycl/sycl.hpp>
#endif

namespace traccc {

// ---------------------------------------------------------------------------
// Minimal fixed-size vector storage types, and constructors for them.
// ---------------------------------------------------------------------------
struct TRACCC_ALIGN(8) float2 {
    float x, y;
};
struct TRACCC_ALIGN(16) float4 {
    float x, y, z, w;
};
struct TRACCC_ALIGN(8) int2 {
    int x, y;
};
struct TRACCC_ALIGN(8) uint2 {
    unsigned int x, y;
};
struct TRACCC_ALIGN(16) uint4 {
    unsigned int x, y, z, w;
};
struct TRACCC_ALIGN(4) short2 {
    short x, y;
};

inline TRACCC_HOST_DEVICE float2 make_float2(const float x, const float y) {
    return {x, y};
}
inline TRACCC_HOST_DEVICE float4 make_float4(const float x, const float y,
                                             const float z, const float w) {
    return {x, y, z, w};
}
inline TRACCC_HOST_DEVICE int2 make_int2(const int x, const int y) {
    return {x, y};
}
inline TRACCC_HOST_DEVICE uint2 make_uint2(const unsigned int x,
                                           const unsigned int y) {
    return {x, y};
}
inline TRACCC_HOST_DEVICE uint4 make_uint4(const unsigned int x,
                                           const unsigned int y,
                                           const unsigned int z,
                                           const unsigned int w) {
    return {x, y, z, w};
}
inline TRACCC_HOST_DEVICE short2 make_short2(const short x, const short y) {
    return {x, y};
}

// ---------------------------------------------------------------------------
// Edge type: fp16 where available (see TRACCC_GBTS_EDGE_HALF), float on CPU.
// ---------------------------------------------------------------------------
#if TRACCC_GBTS_EDGE_HALF && defined(__CUDACC__)

using gbts_edge_t = __half;
struct TRACCC_ALIGN(8) gbts_edge4 {
    gbts_edge_t x, y, z, w;
};
inline TRACCC_HOST_DEVICE gbts_edge_t gbts_edge_from_float(const float f) {
    return __float2half(f);
}
inline TRACCC_HOST_DEVICE float gbts_edge_to_float(const gbts_edge_t r) {
    return __half2float(r);
}

#elif TRACCC_GBTS_EDGE_HALF && \
    (defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__))

using gbts_edge_t = ::sycl::half;
struct TRACCC_ALIGN(8) gbts_edge4 {
    gbts_edge_t x, y, z, w;
};
inline TRACCC_HOST_DEVICE gbts_edge_t gbts_edge_from_float(const float f) {
    return static_cast< ::sycl::half>(f);
}
inline TRACCC_HOST_DEVICE float gbts_edge_to_float(const gbts_edge_t r) {
    return static_cast<float>(r);
}

#else

using gbts_edge_t = float;
using gbts_edge4 = float4;
inline TRACCC_HOST_DEVICE gbts_edge_t gbts_edge_from_float(const float f) {
    return f;
}
inline TRACCC_HOST_DEVICE float gbts_edge_to_float(const gbts_edge_t r) {
    return r;
}

#endif

inline TRACCC_HOST_DEVICE gbts_edge4 gbts_make_edge4(const float x,
                                                     const float y,
                                                     const float z,
                                                     const float w) {
    return {gbts_edge_from_float(x), gbts_edge_from_float(y),
            gbts_edge_from_float(z), gbts_edge_from_float(w)};
}

namespace device {

inline constexpr float PI_F = traccc::constant<float>::pi;
inline constexpr float TWO_PI_F = 2.0f * traccc::constant<float>::pi;

// Wrap an angle into (-pi, pi], matching the reference (round-to-nearest).
// This can be generalised using if and floor which is in
// trigonometric helper functions.
// TODO: Test whether this is faster than the trigometric helper function.
TRACCC_HOST_DEVICE inline float phi_wrap(const float phi) {
    return phi - TWO_PI_F * rintf(phi * (1.0f / TWO_PI_F));
}

#if TRACCC_GBTS_EDGE_HALF && defined(__CUDACC__)
// Half-precision phi wrap (device only); kept so half edge math still works.
__device__ inline gbts_edge_t phi_wrap(const gbts_edge_t phi) {
    const __half two_pi_h = __float2half(TWO_PI_F);
    const __half one_h = __float2half(1.0f);
    return phi - two_pi_h * hrint(phi * (one_h / two_pi_h));
}
#elif TRACCC_GBTS_EDGE_HALF && \
    (defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__))
// Half-precision phi wrap; kept so half edge math still works.
inline ::sycl::half phi_wrap(const ::sycl::half phi) {
    const ::sycl::half two_pi_h = static_cast< ::sycl::half>(TWO_PI_F);
    return phi - two_pi_h * ::sycl::rint(phi / two_pi_h);
}
#endif

}  // namespace device

}  // namespace traccc
