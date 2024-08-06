/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>

#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>

#else
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>
#endif

namespace traccc::alpaka::vecmem {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
/// Device memory resource
typedef ::vecmem::cuda::device_memory_resource device_memory_resource;
/// Host memory resource
typedef ::vecmem::cuda::host_memory_resource host_memory_resource;
/// Memory copy object
typedef ::vecmem::cuda::copy device_copy;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
/// Device memory resource
typedef ::vecmem::hip::device_memory_resource device_memory_resource;
/// Host memory resource
typedef ::vecmem::hip::host_memory_resource host_memory_resource;
/// Memory copy object
typedef ::vecmem::hip::copy device_copy;
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
/// Device memory resource
typedef ::vecmem::sycl::device_memory_resource device_memory_resource;
/// Host memory resource
typedef ::vecmem::sycl::host_memory_resource host_memory_resource;
/// Memory copy object
typedef ::vecmem::sycl::copy device_copy;
#else  // host-only
/// Device memory resource
typedef ::vecmem::memory_resource device_memory_resource;
typedef ::vecmem::copy device_copy;
#endif

}  // namespace traccc::alpaka::vecmem
