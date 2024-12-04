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

// System include(s).
#include <cmath>

namespace traccc {

/// Namespace to pick up math functions from
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
namespace math = ::sycl;
#else
namespace math = std;
#endif  // SYCL

}  // namespace traccc
