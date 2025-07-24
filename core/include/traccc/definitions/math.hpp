/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL include(s).
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

#include "traccc/definitions/qualifiers.hpp"

// System include(s).
#include <cmath>

#ifdef __CUDA_ARCH__
#define TRACCC_MATH_ALWAYS_INLINE __attribute__((always_inline))
#else
#define TRACCC_MATH_ALWAYS_INLINE
#endif

namespace traccc {

/// Namespace to pick up math functions from
namespace math {
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
using ::sycl::abs;
using ::sycl::acos;
using ::sycl::asin;
using ::sycl::atan;
using ::sycl::atan2;
using ::sycl::cos;
using ::sycl::exp;
using ::sycl::fabs;
using ::sycl::floor;
using ::sycl::fmod;
using ::sycl::log;
using ::sycl::max;
using ::sycl::min;
using ::sycl::pow;
using ::sycl::sin;
using ::sycl::sqrt;
using ::sycl::tan;
#else
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::exp;
using std::fabs;
using std::floor;
using std::fmod;
using std::log;
using std::max;
using std::min;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;
#endif  // SYCL

namespace fast {
TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float sqrt(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __fsqrt_rn(v);
#else
    return ::traccc::math::sqrt(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double sqrt(double v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __dsqrt_rn(v);
#else
    return ::traccc::math::sqrt(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float sin(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __sinf(v);
#else
    return ::traccc::math::sin(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double sin(double v) {
    return ::traccc::math::sin(v);
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float asin(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return asinf(v);
#else
    return ::traccc::math::asin(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double asin(double v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return asin(v);
#else
    return ::traccc::math::asin(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float cos(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __cosf(v);
#else
    return ::traccc::math::cos(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double cos(double v) {
    return ::traccc::math::cos(v);
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float acos(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return acosf(v);
#else
    return ::traccc::math::acos(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double acos(double v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return acos(v);
#else
    return ::traccc::math::acos(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float tan(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __tanf(v);
#else
    return ::traccc::math::tan(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double tan(double v) {
    return ::traccc::math::tan(v);
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float atan(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return atanf(v);
#else
    return ::traccc::math::atan(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double atan(double v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return atan(v);
#else
    return ::traccc::math::atan(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float atan2(float y,
                                                                float x) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return atan2f(y, x);
#else
    return ::traccc::math::atan2(y, x);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double atan2(double y,
                                                                 double x) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return atan2(y, x);
#else
    return ::traccc::math::atan2(y, x);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float log(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __logf(v);
#else
    return ::traccc::math::log(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double log(double v) {
    return ::traccc::math::log(v);
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline float exp(float v) {
#if defined(__CUDA_ARCH__) && !defined(TRACCC_FORCE_SLOW_MATH)
    return __expf(v);
#else
    return ::traccc::math::exp(v);
#endif
}

TRACCC_HOST_DEVICE TRACCC_MATH_ALWAYS_INLINE inline double exp(double v) {
    return ::traccc::math::exp(v);
}
}  // namespace fast
}  // namespace math
}  // namespace traccc
