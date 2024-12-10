/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <atomic>

#if (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) && \
    defined(VECMEM_HAVE_SYCL_ATOMIC_REF)
// SYCL include(s).
#include <sycl/sycl.hpp>
#endif

namespace vecmem {
#if (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) && \
    defined(VECMEM_HAVE_SYCL_ATOMIC_REF)
/// Define @c vecmem::memory_order as @c sycl::memory_order
using memory_order = ::sycl::memory_order;
#elif ((!defined(__CUDA_ARCH__)) && (!defined(__HIP_DEVICE_COMPILE__)) && \
       (!defined(CL_SYCL_LANGUAGE_VERSION)) &&                            \
       (!defined(SYCL_LANGUAGE_VERSION)) && __cpp_lib_atomic_ref)
using memory_order = std::memory_order;
#else
/// Custom (dummy) definition for the memory order
enum class memory_order {
    relaxed = 0,
    consume = 1,
    acquire = 2,
    release = 3,
    acq_rel = 4,
    seq_cst = 5
};
#endif
}  // namespace vecmem
