/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// HIP include(s).
#include <hip/hip_runtime.h>

// Set up __VECMEM_THREADFENCE correctly for the vecmem::hip::device_atomic_ref
// code.
#ifdef __HIP_DEVICE_COMPILE__
#define __VECMEM_THREADFENCE __threadfence()
#else
#define __VECMEM_THREADFENCE
#endif  // defined(__HIP_DEVICE_COMPILE__)

// Local include(s).
#include "vecmem/memory/details/cuda_device_atomic_ref.hpp"

namespace vecmem {
namespace hip {

/// Use @c vecmem::cuda::device_atomic_ref for HIP code as well
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = cuda::device_atomic_ref<T, address>;

}  // namespace hip
}  // namespace vecmem
